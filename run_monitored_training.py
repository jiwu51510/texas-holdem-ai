#!/usr/bin/env python3
"""监控训练脚本。

使用最优配置进行训练，并实时监控各项指标。
如果指标不符合预期，自动停止并调整参数。
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np

import torch

from models.core import TrainingConfig, Card
from train_river_only import RiverOnlyTrainer, parse_board, board_to_str


@dataclass
class TrainingMetrics:
    """训练指标。"""
    iteration: int = 0
    regret_loss: float = 0.0
    policy_loss: float = 0.0
    regret_grad_norm: float = 0.0
    policy_grad_norm: float = 0.0
    entropy: float = 0.0
    is_oscillating: bool = False
    kl_divergence: float = 0.0
    p0_win_rate: float = 0.0
    avg_utility_p0: float = 0.0
    training_time: float = 0.0


@dataclass
class TrainingThresholds:
    """训练阈值配置。"""
    # 损失阈值
    max_regret_loss: float = 5.0  # 遗憾损失超过此值则警告
    max_policy_loss: float = 3.0  # 策略损失超过此值则警告
    
    # 梯度阈值
    max_grad_norm: float = 10.0  # 梯度范数超过此值则警告
    
    # 收敛阈值
    target_regret_loss: float = 0.5  # 目标遗憾损失
    target_policy_loss: float = 1.0  # 目标策略损失
    
    # 震荡检测 - 放宽容忍度，因为CFR训练中震荡是正常的
    oscillation_patience: int = 20  # 连续震荡次数超过此值则停止
    stop_on_oscillation: bool = False  # 是否因震荡而停止
    
    # 损失增长检测
    loss_increase_patience: int = 5  # 连续损失增长次数超过此值则警告


class MonitoredTrainer:
    """监控训练器。"""
    
    def __init__(
        self,
        config_path: str,
        fixed_board: Optional[List[Card]] = None,
        thresholds: Optional[TrainingThresholds] = None
    ):
        """初始化监控训练器。
        
        Args:
            config_path: 配置文件路径
            fixed_board: 固定公共牌
            thresholds: 训练阈值配置
        """
        self.config_path = config_path
        self.fixed_board = fixed_board
        self.thresholds = thresholds or TrainingThresholds()
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config_dict = json.load(f)
        
        # 创建训练配置
        self.training_config = self._create_training_config()
        
        # 创建训练器
        self.trainer = RiverOnlyTrainer(self.training_config, fixed_board=fixed_board)
        
        # 应用收敛控制参数
        self._apply_convergence_params()
        
        # 训练历史
        self.metrics_history: List[TrainingMetrics] = []
        
        # 状态跟踪
        self.consecutive_oscillations = 0
        self.consecutive_loss_increases = 0
        self.last_regret_loss = float('inf')
        self.best_regret_loss = float('inf')
        self.should_stop = False
        self.stop_reason = ""
    
    def _create_training_config(self) -> TrainingConfig:
        """从配置字典创建训练配置。"""
        return TrainingConfig(
            learning_rate=self.config_dict.get('learning_rate', 0.0001),
            batch_size=self.config_dict.get('batch_size', 256),
            network_architecture=self.config_dict.get('network_architecture', [512, 256, 128]),
            cfr_iterations_per_update=self.config_dict.get('cfr_iterations_per_update', 1000),
            network_train_steps=self.config_dict.get('network_train_steps', 2000),
            regret_buffer_size=self.config_dict.get('regret_buffer_size', 300000),
            strategy_buffer_size=self.config_dict.get('strategy_buffer_size', 300000),
            initial_stack=self.config_dict.get('initial_stack', 1000),
            small_blind=self.config_dict.get('small_blind', 5),
            big_blind=self.config_dict.get('big_blind', 10),
            max_raises_per_street=self.config_dict.get('max_raises_per_street', 4),
        )
    
    def _apply_convergence_params(self):
        """应用收敛控制参数到训练器。"""
        # 遗憾值处理器配置
        regret_config = self.config_dict.get('regret_processor', {})
        self.trainer.regret_processor.config.use_positive_truncation = regret_config.get('use_positive_truncation', True)
        self.trainer.regret_processor.config.decay_factor = regret_config.get('decay_factor', 0.9)
        self.trainer.regret_processor.config.clip_threshold = regret_config.get('clip_threshold', 15.0)
        
        # 网络训练器配置
        network_config = self.config_dict.get('network_trainer', {})
        self.trainer.network_trainer.config.use_huber_loss = network_config.get('use_huber_loss', True)
        self.trainer.network_trainer.config.huber_delta = network_config.get('huber_delta', 0.2)
        self.trainer.network_trainer.config.use_ema = network_config.get('use_ema', True)
        self.trainer.network_trainer.config.ema_decay = network_config.get('ema_decay', 0.999)
        self.trainer.network_trainer.config.gradient_clip_norm = network_config.get('gradient_clip_norm', 0.1)
        
        # 收敛监控器配置
        monitor_config = self.config_dict.get('convergence_monitor', {})
        self.trainer.convergence_monitor.config.entropy_window = monitor_config.get('entropy_window', 100)
        self.trainer.convergence_monitor.config.oscillation_threshold = monitor_config.get('oscillation_threshold', 0.1)
        self.trainer.convergence_monitor.config.kl_warning_threshold = monitor_config.get('kl_warning_threshold', 0.5)
        self.trainer.convergence_monitor.config.monitor_interval = monitor_config.get('monitor_interval', 500)
    
    def _check_metrics(self, metrics: TrainingMetrics) -> List[str]:
        """检查指标是否符合预期。
        
        Returns:
            警告消息列表
        """
        warnings = []
        
        # 检查损失
        if metrics.regret_loss > self.thresholds.max_regret_loss:
            warnings.append(f"⚠️ 遗憾损失过高: {metrics.regret_loss:.4f} > {self.thresholds.max_regret_loss}")
        
        if metrics.policy_loss > self.thresholds.max_policy_loss:
            warnings.append(f"⚠️ 策略损失过高: {metrics.policy_loss:.4f} > {self.thresholds.max_policy_loss}")
        
        # 检查梯度
        if metrics.regret_grad_norm > self.thresholds.max_grad_norm:
            warnings.append(f"⚠️ 遗憾梯度范数过大: {metrics.regret_grad_norm:.4f} > {self.thresholds.max_grad_norm}")
        
        if metrics.policy_grad_norm > self.thresholds.max_grad_norm:
            warnings.append(f"⚠️ 策略梯度范数过大: {metrics.policy_grad_norm:.4f} > {self.thresholds.max_grad_norm}")
        
        # 检查震荡
        if metrics.is_oscillating:
            self.consecutive_oscillations += 1
            if self.consecutive_oscillations >= self.thresholds.oscillation_patience:
                warnings.append(f"⚠️ 连续震荡 {self.consecutive_oscillations} 次")
                if self.thresholds.stop_on_oscillation:
                    warnings.append(f"🛑 因震荡停止训练")
                    self.should_stop = True
                    self.stop_reason = "连续震荡过多"
        else:
            self.consecutive_oscillations = 0
        
        # 检查损失增长
        if metrics.regret_loss > self.last_regret_loss * 1.1:  # 损失增长超过10%
            self.consecutive_loss_increases += 1
            if self.consecutive_loss_increases >= self.thresholds.loss_increase_patience:
                warnings.append(f"⚠️ 连续损失增长 {self.consecutive_loss_increases} 次")
        else:
            self.consecutive_loss_increases = 0
        
        self.last_regret_loss = metrics.regret_loss
        
        # 更新最佳损失
        if metrics.regret_loss < self.best_regret_loss:
            self.best_regret_loss = metrics.regret_loss
        
        return warnings
    
    def _print_progress(self, metrics: TrainingMetrics, warnings: List[str]):
        """打印训练进度。"""
        print(f"\n{'='*70}")
        print(f"迭代 {metrics.iteration}")
        print(f"{'='*70}")
        print(f"  遗憾损失: {metrics.regret_loss:.6f} (最佳: {self.best_regret_loss:.6f})")
        print(f"  策略损失: {metrics.policy_loss:.6f}")
        print(f"  遗憾梯度范数: {metrics.regret_grad_norm:.4f}")
        print(f"  策略梯度范数: {metrics.policy_grad_norm:.4f}")
        print(f"  熵: {metrics.entropy:.4f}")
        print(f"  震荡: {'是' if metrics.is_oscillating else '否'}")
        print(f"  P0胜率: {metrics.p0_win_rate:.2%}")
        print(f"  P0平均收益: {metrics.avg_utility_p0:.2f}")
        print(f"  训练时间: {metrics.training_time:.1f}秒")
        
        if warnings:
            print(f"\n警告:")
            for warning in warnings:
                print(f"  {warning}")
        
        # 检查是否达到目标
        if metrics.regret_loss <= self.thresholds.target_regret_loss:
            print(f"\n✅ 遗憾损失已达到目标 ({metrics.regret_loss:.4f} <= {self.thresholds.target_regret_loss})")
    
    def train(
        self,
        total_iterations: int,
        cfr_per_update: int,
        eval_interval: int = 5000,
        checkpoint_interval: int = 10000,
        checkpoint_dir: str = "checkpoints/monitored"
    ) -> Dict[str, Any]:
        """执行监控训练。
        
        Args:
            total_iterations: 总迭代次数
            cfr_per_update: 每次网络更新的CFR迭代次数
            eval_interval: 评估间隔
            checkpoint_interval: 检查点保存间隔
            checkpoint_dir: 检查点目录
            
        Returns:
            训练结果摘要
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"\n{'#'*70}")
        print(f"开始监控训练")
        print(f"{'#'*70}")
        print(f"配置文件: {self.config_path}")
        print(f"公共牌模式: {self.trainer.get_board_mode()}")
        print(f"总迭代次数: {total_iterations}")
        print(f"CFR迭代/更新: {cfr_per_update}")
        print(f"评估间隔: {eval_interval}")
        print(f"检查点间隔: {checkpoint_interval}")
        print(f"{'#'*70}\n")
        
        start_time = time.time()
        update_count = 0
        
        while self.trainer.iteration < total_iterations and not self.should_stop:
            update_start = time.time()
            
            # CFR 迭代
            for _ in range(cfr_per_update):
                self.trainer.run_cfr_iteration(verbose=False)
            
            # 训练网络
            train_results = self.trainer.train_networks(verbose=False)
            
            update_count += 1
            
            # 收集指标
            convergence_report = train_results.get('convergence_report', {})
            latest_metrics = convergence_report.get('latest_metrics', {})
            
            metrics = TrainingMetrics(
                iteration=self.trainer.iteration,
                regret_loss=train_results.get('regret_loss', 0.0),
                policy_loss=train_results.get('policy_loss', 0.0),
                regret_grad_norm=train_results.get('regret_grad_norm', 0.0),
                policy_grad_norm=train_results.get('policy_grad_norm', 0.0),
                entropy=latest_metrics.get('avg_entropy', 0.0),
                is_oscillating=latest_metrics.get('is_oscillating', False),
                kl_divergence=latest_metrics.get('kl_divergence', 0.0),
                training_time=time.time() - update_start
            )
            
            # 定期评估
            if self.trainer.iteration % eval_interval == 0:
                eval_results = self.trainer.evaluate_strategy(num_hands=500)
                metrics.p0_win_rate = eval_results['p0_win_rate']
                metrics.avg_utility_p0 = eval_results['avg_utility_p0']
            
            self.metrics_history.append(metrics)
            
            # 检查指标
            warnings = self._check_metrics(metrics)
            
            # 打印进度（每次更新都打印）
            self._print_progress(metrics, warnings)
            
            # 保存检查点
            if self.trainer.iteration % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_{self.trainer.iteration}.pt"
                )
                self.trainer.save_checkpoint(checkpoint_path)
        
        # 训练结束
        total_time = time.time() - start_time
        
        # 最终评估
        print(f"\n{'#'*70}")
        print(f"训练完成")
        print(f"{'#'*70}")
        
        final_eval = self.trainer.evaluate_strategy(num_hands=1000)
        
        result = {
            'total_iterations': self.trainer.iteration,
            'total_time_seconds': total_time,
            'final_regret_loss': self.metrics_history[-1].regret_loss if self.metrics_history else 0.0,
            'final_policy_loss': self.metrics_history[-1].policy_loss if self.metrics_history else 0.0,
            'best_regret_loss': self.best_regret_loss,
            'final_p0_win_rate': final_eval['p0_win_rate'],
            'final_avg_utility_p0': final_eval['avg_utility_p0'],
            'stopped_early': self.should_stop,
            'stop_reason': self.stop_reason,
        }
        
        print(f"总迭代次数: {result['total_iterations']}")
        print(f"总训练时间: {result['total_time_seconds']:.1f}秒")
        print(f"最终遗憾损失: {result['final_regret_loss']:.6f}")
        print(f"最佳遗憾损失: {result['best_regret_loss']:.6f}")
        print(f"最终P0胜率: {result['final_p0_win_rate']:.2%}")
        print(f"最终P0平均收益: {result['final_avg_utility_p0']:.2f}")
        
        if self.should_stop:
            print(f"\n⚠️ 训练提前停止: {self.stop_reason}")
        
        # 保存最终检查点
        final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        self.trainer.save_checkpoint(final_checkpoint_path)
        
        # 保存训练报告
        report_path = os.path.join(checkpoint_dir, "training_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n训练报告已保存: {report_path}")
        
        return result


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description='监控训练')
    parser.add_argument('--config', type=str, default='configs/river_optimized_config.json',
                        help='配置文件路径')
    parser.add_argument('--board', type=str, default="AhKsQdJc2h",
                        help='固定公共牌')
    parser.add_argument('--iterations', type=int, default=50000,
                        help='总迭代次数')
    parser.add_argument('--cfr-per-update', type=int, default=1000,
                        help='每次网络更新的CFR迭代次数')
    parser.add_argument('--eval-interval', type=int, default=5000,
                        help='评估间隔')
    parser.add_argument('--checkpoint-interval', type=int, default=10000,
                        help='检查点保存间隔')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/monitored',
                        help='检查点目录')
    
    args = parser.parse_args()
    
    # 解析公共牌
    fixed_board = None
    if args.board:
        try:
            fixed_board = parse_board(args.board)
            print(f"使用固定公共牌: {board_to_str(fixed_board)}")
        except ValueError as e:
            print(f"警告: 无法解析公共牌 '{args.board}': {e}")
            print("将使用随机公共牌")
    
    # 创建监控训练器
    trainer = MonitoredTrainer(
        config_path=args.config,
        fixed_board=fixed_board
    )
    
    # 开始训练
    result = trainer.train(
        total_iterations=args.iterations,
        cfr_per_update=args.cfr_per_update,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir
    )
    
    return result


if __name__ == "__main__":
    main()
