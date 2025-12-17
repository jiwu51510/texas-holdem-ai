#!/usr/bin/env python3
"""收敛控制参数验证脚本。

这个脚本用于验证新的收敛控制参数的有效性，并自动分析训练数据。

功能：
1. 使用不同参数配置进行短期训练
2. 自动收集和分析训练指标
3. 比较不同配置的效果
4. 提供参数调优建议
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import numpy as np

import torch

from models.core import TrainingConfig, Card
from train_river_only import RiverOnlyTrainer, parse_board, board_to_str
from training.regret_processor import RegretProcessorConfig
from training.network_trainer import NetworkTrainerConfig
from training.convergence_monitor import ConvergenceMonitorConfig


@dataclass
class ExperimentConfig:
    """实验配置。"""
    name: str
    description: str
    # 遗憾值处理器配置
    use_positive_truncation: bool = True
    regret_decay_factor: float = 0.99
    regret_clip_threshold: float = 100.0
    # 网络训练器配置
    use_huber_loss: bool = True
    huber_delta: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.995
    gradient_clip_norm: float = 1.0
    # 训练配置
    iterations: int = 5000
    cfr_per_update: int = 500
    train_steps: int = 1000
    batch_size: int = 256
    learning_rate: float = 0.001
    # 基础训练参数（第三轮优化）
    network_architecture: List[int] = None  # None表示使用默认[512, 256, 128]
    regret_buffer_size: int = 200000
    strategy_buffer_size: int = 200000


@dataclass
class ExperimentResult:
    """实验结果。"""
    config_name: str
    # 训练指标
    final_regret_loss: float = 0.0
    final_policy_loss: float = 0.0
    avg_regret_loss: float = 0.0
    avg_policy_loss: float = 0.0
    # 收敛指标
    final_entropy: float = 0.0
    entropy_trend: str = ""  # "decreasing", "stable", "oscillating"
    regret_mean: float = 0.0
    regret_std: float = 0.0
    oscillation_detected: bool = False
    # 评估指标
    p0_win_rate: float = 0.0
    p1_win_rate: float = 0.0
    avg_utility_p0: float = 0.0
    # 训练时间
    training_time_seconds: float = 0.0
    # 历史数据
    loss_history: List[Dict[str, float]] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)


class ConvergenceValidator:
    """收敛控制参数验证器。"""
    
    def __init__(self, fixed_board: Optional[List[Card]] = None):
        """初始化验证器。
        
        Args:
            fixed_board: 固定的公共牌（用于一致性比较）
        """
        self.fixed_board = fixed_board
        self.results: List[ExperimentResult] = []
    
    def create_trainer(self, exp_config: ExperimentConfig) -> RiverOnlyTrainer:
        """根据实验配置创建训练器。"""
        # 网络架构
        network_arch = exp_config.network_architecture if exp_config.network_architecture else [512, 256, 128]
        
        # 创建基础训练配置
        config = TrainingConfig(
            learning_rate=exp_config.learning_rate,
            batch_size=exp_config.batch_size,
            network_architecture=network_arch,
            cfr_iterations_per_update=exp_config.cfr_per_update,
            network_train_steps=exp_config.train_steps,
            regret_buffer_size=exp_config.regret_buffer_size,
            strategy_buffer_size=exp_config.strategy_buffer_size,
            initial_stack=1000,
            small_blind=5,
            big_blind=10,
            max_raises_per_street=4,
        )
        
        # 创建训练器
        trainer = RiverOnlyTrainer(config, fixed_board=self.fixed_board)
        
        # 更新遗憾值处理器配置
        trainer.regret_processor.config.use_positive_truncation = exp_config.use_positive_truncation
        trainer.regret_processor.config.decay_factor = exp_config.regret_decay_factor
        trainer.regret_processor.config.clip_threshold = exp_config.regret_clip_threshold
        
        # 更新网络训练器配置
        trainer.network_trainer.config.use_huber_loss = exp_config.use_huber_loss
        trainer.network_trainer.config.huber_delta = exp_config.huber_delta
        trainer.network_trainer.config.use_ema = exp_config.use_ema
        trainer.network_trainer.config.ema_decay = exp_config.ema_decay
        trainer.network_trainer.config.gradient_clip_norm = exp_config.gradient_clip_norm
        
        return trainer
    
    def run_experiment(self, exp_config: ExperimentConfig, verbose: bool = True) -> ExperimentResult:
        """运行单个实验。"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"实验: {exp_config.name}")
            print(f"描述: {exp_config.description}")
            print(f"{'='*60}")
        
        result = ExperimentResult(config_name=exp_config.name)
        start_time = time.time()
        
        # 创建训练器
        trainer = self.create_trainer(exp_config)
        
        # 训练循环
        update_count = 0
        total_regret_loss = 0.0
        total_policy_loss = 0.0
        loss_count = 0
        
        while trainer.iteration < exp_config.iterations:
            # CFR 迭代
            for _ in range(exp_config.cfr_per_update):
                trainer.run_cfr_iteration(verbose=False)
            
            # 训练网络
            train_results = trainer.train_networks(verbose=False)
            
            # 记录损失
            regret_loss = train_results.get('regret_loss', 0.0)
            policy_loss = train_results.get('policy_loss', 0.0)
            total_regret_loss += regret_loss
            total_policy_loss += policy_loss
            loss_count += 1
            
            result.loss_history.append({
                'iteration': trainer.iteration,
                'regret_loss': regret_loss,
                'policy_loss': policy_loss,
            })
            
            # 记录收敛指标
            convergence_report = train_results.get('convergence_report', {})
            latest_metrics = convergence_report.get('latest_metrics', {})
            if latest_metrics:
                entropy = latest_metrics.get('avg_entropy', 0.0)
                result.entropy_history.append(entropy)
                result.regret_mean = latest_metrics.get('regret_mean', 0.0)
                result.regret_std = latest_metrics.get('regret_std', 0.0)
                if latest_metrics.get('is_oscillating', False):
                    result.oscillation_detected = True
            
            update_count += 1
            
            if verbose and update_count % 2 == 0:
                progress = trainer.iteration / exp_config.iterations
                print(f"  进度: {trainer.iteration}/{exp_config.iterations} ({progress:.0%}) "
                      f"| 遗憾损失: {regret_loss:.6f} | 策略损失: {policy_loss:.6f}")
        
        # 计算最终指标
        result.final_regret_loss = result.loss_history[-1]['regret_loss'] if result.loss_history else 0.0
        result.final_policy_loss = result.loss_history[-1]['policy_loss'] if result.loss_history else 0.0
        result.avg_regret_loss = total_regret_loss / max(loss_count, 1)
        result.avg_policy_loss = total_policy_loss / max(loss_count, 1)
        
        # 分析熵趋势
        if len(result.entropy_history) >= 3:
            result.final_entropy = result.entropy_history[-1]
            result.entropy_trend = self._analyze_trend(result.entropy_history)
        
        # 评估策略
        if verbose:
            print("  评估策略...")
        eval_results = trainer.evaluate_strategy(num_hands=500)
        result.p0_win_rate = eval_results['p0_win_rate']
        result.p1_win_rate = eval_results['p1_win_rate']
        result.avg_utility_p0 = eval_results['avg_utility_p0']
        
        result.training_time_seconds = time.time() - start_time
        
        if verbose:
            print(f"\n  结果:")
            print(f"    最终遗憾损失: {result.final_regret_loss:.6f}")
            print(f"    最终策略损失: {result.final_policy_loss:.6f}")
            print(f"    熵趋势: {result.entropy_trend}")
            print(f"    震荡检测: {'是' if result.oscillation_detected else '否'}")
            print(f"    P0胜率: {result.p0_win_rate:.2%}")
            print(f"    训练时间: {result.training_time_seconds:.1f}秒")
        
        self.results.append(result)
        return result
    
    def _analyze_trend(self, values: List[float]) -> str:
        """分析数值趋势。"""
        if len(values) < 3:
            return "insufficient_data"
        
        # 计算差分
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        
        # 计算符号变化次数
        sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
        
        # 判断趋势
        avg_diff = np.mean(diffs)
        if sign_changes > len(diffs) * 0.4:
            return "oscillating"
        elif avg_diff < -0.01:
            return "decreasing"
        elif avg_diff > 0.01:
            return "increasing"
        else:
            return "stable"
    
    def compare_results(self) -> Dict[str, Any]:
        """比较所有实验结果。"""
        if not self.results:
            return {}
        
        comparison = {
            'best_regret_loss': None,
            'best_policy_loss': None,
            'most_stable': None,
            'fastest_convergence': None,
            'recommendations': [],
        }
        
        # 找出最佳配置
        best_regret = min(self.results, key=lambda r: r.final_regret_loss)
        best_policy = min(self.results, key=lambda r: r.final_policy_loss)
        
        # 找出最稳定的配置（无震荡且熵趋势稳定或下降）
        stable_results = [r for r in self.results 
                         if not r.oscillation_detected 
                         and r.entropy_trend in ['stable', 'decreasing']]
        if stable_results:
            most_stable = min(stable_results, key=lambda r: r.avg_regret_loss)
            comparison['most_stable'] = most_stable.config_name
        
        comparison['best_regret_loss'] = best_regret.config_name
        comparison['best_policy_loss'] = best_policy.config_name
        
        # 生成建议
        recommendations = []
        
        # 检查震荡
        oscillating_configs = [r.config_name for r in self.results if r.oscillation_detected]
        if oscillating_configs:
            recommendations.append(
                f"配置 {', '.join(oscillating_configs)} 检测到震荡，建议降低学习率或增加EMA衰减率"
            )
        
        # 检查损失是否收敛
        high_loss_configs = [r.config_name for r in self.results if r.final_regret_loss > 1.0]
        if high_loss_configs:
            recommendations.append(
                f"配置 {', '.join(high_loss_configs)} 损失较高，建议增加训练步数或调整网络架构"
            )
        
        # 推荐最佳配置
        if comparison['most_stable']:
            recommendations.append(
                f"推荐使用配置 '{comparison['most_stable']}'，它在稳定性和损失方面表现最佳"
            )
        else:
            recommendations.append(
                f"推荐使用配置 '{best_regret.config_name}'，它的遗憾损失最低"
            )
        
        comparison['recommendations'] = recommendations
        return comparison
    
    def generate_report(self) -> str:
        """生成详细报告。"""
        lines = []
        lines.append("=" * 70)
        lines.append("收敛控制参数验证报告")
        lines.append("=" * 70)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 实验结果摘要
        lines.append("实验结果摘要:")
        lines.append("-" * 70)
        lines.append(f"{'配置名称':<20} {'遗憾损失':<12} {'策略损失':<12} {'熵趋势':<12} {'震荡':<6}")
        lines.append("-" * 70)
        
        for result in self.results:
            oscillation = "是" if result.oscillation_detected else "否"
            lines.append(
                f"{result.config_name:<20} "
                f"{result.final_regret_loss:<12.6f} "
                f"{result.final_policy_loss:<12.6f} "
                f"{result.entropy_trend:<12} "
                f"{oscillation:<6}"
            )
        
        lines.append("")
        
        # 比较分析
        comparison = self.compare_results()
        lines.append("比较分析:")
        lines.append("-" * 70)
        lines.append(f"最佳遗憾损失: {comparison.get('best_regret_loss', 'N/A')}")
        lines.append(f"最佳策略损失: {comparison.get('best_policy_loss', 'N/A')}")
        lines.append(f"最稳定配置: {comparison.get('most_stable', 'N/A')}")
        lines.append("")
        
        # 建议
        lines.append("调优建议:")
        lines.append("-" * 70)
        for rec in comparison.get('recommendations', []):
            lines.append(f"• {rec}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


def get_default_experiments() -> List[ExperimentConfig]:
    """获取默认实验配置列表。"""
    return [
        # 基准配置（无收敛控制）
        ExperimentConfig(
            name="baseline",
            description="基准配置（无收敛控制增强）",
            use_positive_truncation=False,
            regret_decay_factor=1.0,  # 无衰减
            regret_clip_threshold=1000.0,  # 高阈值
            use_huber_loss=False,
            use_ema=False,
            gradient_clip_norm=10.0,  # 宽松裁剪
        ),
        # CFR+ 配置
        ExperimentConfig(
            name="cfr_plus",
            description="CFR+ 正遗憾值截断",
            use_positive_truncation=True,
            regret_decay_factor=1.0,
            regret_clip_threshold=100.0,
            use_huber_loss=False,
            use_ema=False,
            gradient_clip_norm=1.0,
        ),
        # 完整收敛控制配置
        ExperimentConfig(
            name="full_convergence",
            description="完整收敛控制（推荐配置）",
            use_positive_truncation=True,
            regret_decay_factor=0.995,
            regret_clip_threshold=50.0,
            use_huber_loss=True,
            huber_delta=1.0,
            use_ema=True,
            ema_decay=0.99,
            gradient_clip_norm=1.0,
        ),
        # 激进衰减配置
        ExperimentConfig(
            name="aggressive_decay",
            description="激进遗憾值衰减",
            use_positive_truncation=True,
            regret_decay_factor=0.95,
            regret_clip_threshold=30.0,
            use_huber_loss=True,
            use_ema=True,
            ema_decay=0.99,
            gradient_clip_norm=0.5,
        ),
        # 高EMA衰减配置
        ExperimentConfig(
            name="high_ema",
            description="高EMA衰减率（更平滑更新）",
            use_positive_truncation=True,
            regret_decay_factor=0.99,
            regret_clip_threshold=50.0,
            use_huber_loss=True,
            use_ema=True,
            ema_decay=0.999,
            gradient_clip_norm=1.0,
        ),
    ]


def get_round4_experiments() -> List[ExperimentConfig]:
    """获取第四轮优化实验配置（解决震荡问题）。
    
    基于第三轮最优配置(more_cfr)，专注于：
    - 解决震荡问题
    - 进一步降低损失
    - 提高训练稳定性
    """
    # 基准：第三轮最优配置
    base_config = {
        'use_positive_truncation': True,
        'regret_decay_factor': 0.95,
        'regret_clip_threshold': 25.0,
        'use_huber_loss': True,
        'huber_delta': 0.5,
        'use_ema': True,
        'ema_decay': 0.995,
        'gradient_clip_norm': 0.3,
        'learning_rate': 0.0003,
        'cfr_per_update': 1000,  # 第三轮最优
    }
    
    return [
        # 基准（第三轮最优）
        ExperimentConfig(
            name="round3_best",
            description="第三轮最优配置（基准）",
            **base_config,
        ),
        # 更高EMA衰减（减少震荡）
        ExperimentConfig(
            name="higher_ema",
            description="更高EMA衰减率 0.998",
            ema_decay=0.998,
            **{k: v for k, v in base_config.items() if k != 'ema_decay'},
        ),
        # 极高EMA衰减
        ExperimentConfig(
            name="ultra_ema",
            description="极高EMA衰减率 0.999",
            ema_decay=0.999,
            **{k: v for k, v in base_config.items() if k != 'ema_decay'},
        ),
        # 更低学习率
        ExperimentConfig(
            name="lower_lr",
            description="更低学习率 0.0001",
            learning_rate=0.0001,
            **{k: v for k, v in base_config.items() if k != 'learning_rate'},
        ),
        # 更严格梯度裁剪
        ExperimentConfig(
            name="tighter_clip",
            description="更严格梯度裁剪 0.1",
            gradient_clip_norm=0.1,
            **{k: v for k, v in base_config.items() if k != 'gradient_clip_norm'},
        ),
        # 更强遗憾衰减
        ExperimentConfig(
            name="stronger_decay",
            description="更强遗憾衰减 0.9",
            regret_decay_factor=0.9,
            **{k: v for k, v in base_config.items() if k != 'regret_decay_factor'},
        ),
        # 更低遗憾裁剪阈值
        ExperimentConfig(
            name="lower_clip_threshold",
            description="更低遗憾裁剪阈值 15.0",
            regret_clip_threshold=15.0,
            **{k: v for k, v in base_config.items() if k != 'regret_clip_threshold'},
        ),
        # 更小Huber delta
        ExperimentConfig(
            name="smaller_huber",
            description="更小Huber delta 0.25",
            huber_delta=0.25,
            **{k: v for k, v in base_config.items() if k != 'huber_delta'},
        ),
        # 综合稳定配置
        ExperimentConfig(
            name="stable_combined",
            description="综合稳定配置",
            use_positive_truncation=True,
            regret_decay_factor=0.92,
            regret_clip_threshold=20.0,
            use_huber_loss=True,
            huber_delta=0.3,
            use_ema=True,
            ema_decay=0.998,
            gradient_clip_norm=0.2,
            learning_rate=0.0002,
            cfr_per_update=1000,
        ),
        # 极端稳定配置
        ExperimentConfig(
            name="ultra_stable",
            description="极端稳定配置",
            use_positive_truncation=True,
            regret_decay_factor=0.9,
            regret_clip_threshold=15.0,
            use_huber_loss=True,
            huber_delta=0.2,
            use_ema=True,
            ema_decay=0.999,
            gradient_clip_norm=0.1,
            learning_rate=0.0001,
            cfr_per_update=1000,
        ),
    ]


def get_round3_experiments() -> List[ExperimentConfig]:
    """获取第三轮优化实验配置（基础训练参数优化）。
    
    基于第二轮最优配置(optimal_combined)，优化：
    - 网络架构
    - 批次大小
    - 缓冲区大小
    - CFR迭代次数
    - 训练步数
    """
    # 基准：第二轮最优配置
    base_config = {
        'use_positive_truncation': True,
        'regret_decay_factor': 0.95,
        'regret_clip_threshold': 25.0,
        'use_huber_loss': True,
        'huber_delta': 0.5,
        'use_ema': True,
        'ema_decay': 0.995,
        'gradient_clip_norm': 0.3,
        'learning_rate': 0.0003,
    }
    
    return [
        # 基准（第二轮最优）
        ExperimentConfig(
            name="round2_best",
            description="第二轮最优配置（基准）",
            **base_config,
        ),
        # 更大网络
        ExperimentConfig(
            name="larger_network",
            description="更大网络架构 [768, 384, 192]",
            network_architecture=[768, 384, 192],
            **base_config,
        ),
        # 更小网络
        ExperimentConfig(
            name="smaller_network",
            description="更小网络架构 [256, 128, 64]",
            network_architecture=[256, 128, 64],
            **base_config,
        ),
        # 更深网络
        ExperimentConfig(
            name="deeper_network",
            description="更深网络架构 [512, 256, 128, 64]",
            network_architecture=[512, 256, 128, 64],
            **base_config,
        ),
        # 更大批次
        ExperimentConfig(
            name="larger_batch",
            description="更大批次大小 512",
            batch_size=512,
            **base_config,
        ),
        # 更小批次
        ExperimentConfig(
            name="smaller_batch",
            description="更小批次大小 128",
            batch_size=128,
            **base_config,
        ),
        # 更大缓冲区
        ExperimentConfig(
            name="larger_buffer",
            description="更大缓冲区 500000",
            regret_buffer_size=500000,
            strategy_buffer_size=500000,
            **base_config,
        ),
        # 更多CFR迭代
        ExperimentConfig(
            name="more_cfr",
            description="更多CFR迭代 1000/更新",
            cfr_per_update=1000,
            **base_config,
        ),
        # 更多训练步数
        ExperimentConfig(
            name="more_train_steps",
            description="更多训练步数 2000",
            train_steps=2000,
            **base_config,
        ),
        # 综合优化
        ExperimentConfig(
            name="round3_optimal",
            description="第三轮综合优化",
            network_architecture=[512, 256, 128],
            batch_size=512,
            regret_buffer_size=300000,
            strategy_buffer_size=300000,
            cfr_per_update=800,
            train_steps=1500,
            **base_config,
        ),
    ]


def get_round2_experiments() -> List[ExperimentConfig]:
    """获取第二轮优化实验配置（针对震荡问题）。"""
    return [
        # 基于aggressive_decay的低学习率版本
        ExperimentConfig(
            name="low_lr_aggressive",
            description="低学习率 + 激进衰减",
            use_positive_truncation=True,
            regret_decay_factor=0.95,
            regret_clip_threshold=30.0,
            use_huber_loss=True,
            use_ema=True,
            ema_decay=0.99,
            gradient_clip_norm=0.5,
            learning_rate=0.0003,  # 降低学习率
        ),
        # 超低学习率配置
        ExperimentConfig(
            name="ultra_low_lr",
            description="超低学习率（0.0001）",
            use_positive_truncation=True,
            regret_decay_factor=0.95,
            regret_clip_threshold=30.0,
            use_huber_loss=True,
            use_ema=True,
            ema_decay=0.995,
            gradient_clip_norm=0.5,
            learning_rate=0.0001,
        ),
        # 超高EMA衰减配置
        ExperimentConfig(
            name="ultra_high_ema",
            description="超高EMA衰减率（0.9995）",
            use_positive_truncation=True,
            regret_decay_factor=0.95,
            regret_clip_threshold=30.0,
            use_huber_loss=True,
            use_ema=True,
            ema_decay=0.9995,
            gradient_clip_norm=0.5,
            learning_rate=0.0005,
        ),
        # 极端梯度裁剪配置
        ExperimentConfig(
            name="tight_grad_clip",
            description="严格梯度裁剪（0.1）",
            use_positive_truncation=True,
            regret_decay_factor=0.95,
            regret_clip_threshold=30.0,
            use_huber_loss=True,
            use_ema=True,
            ema_decay=0.99,
            gradient_clip_norm=0.1,
            learning_rate=0.0005,
        ),
        # 综合最优配置
        ExperimentConfig(
            name="optimal_combined",
            description="综合最优配置",
            use_positive_truncation=True,
            regret_decay_factor=0.95,
            regret_clip_threshold=25.0,
            use_huber_loss=True,
            huber_delta=0.5,  # 更小的delta
            use_ema=True,
            ema_decay=0.995,
            gradient_clip_norm=0.3,
            learning_rate=0.0003,
        ),
    ]


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description='收敛控制参数验证')
    parser.add_argument('--board', type=str, default="AhKsQdJc2h",
                        help='固定公共牌（用于一致性比较）')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='每个实验的迭代次数')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式（减少迭代次数）')
    parser.add_argument('--output', type=str, default=None,
                        help='报告输出文件路径')
    parser.add_argument('--round2', action='store_true',
                        help='运行第二轮优化实验（针对震荡问题）')
    parser.add_argument('--round3', action='store_true',
                        help='运行第三轮优化实验（基础训练参数）')
    parser.add_argument('--round4', action='store_true',
                        help='运行第四轮优化实验（解决震荡问题）')
    
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
    
    # 创建验证器
    validator = ConvergenceValidator(fixed_board=fixed_board)
    
    # 获取实验配置
    if args.round4:
        experiments = get_round4_experiments()
        print("运行第四轮优化实验（解决震荡问题）")
    elif args.round3:
        experiments = get_round3_experiments()
        print("运行第三轮优化实验（基础训练参数）")
    elif args.round2:
        experiments = get_round2_experiments()
        print("运行第二轮优化实验（针对震荡问题）")
    else:
        experiments = get_default_experiments()
    
    # 调整迭代次数
    iterations = 2000 if args.quick else args.iterations
    for exp in experiments:
        exp.iterations = iterations
        if args.quick:
            exp.cfr_per_update = 200
            exp.train_steps = 500
    
    print(f"\n将运行 {len(experiments)} 个实验，每个 {iterations} 次迭代")
    print("=" * 60)
    
    # 运行所有实验
    for exp in experiments:
        validator.run_experiment(exp, verbose=True)
    
    # 生成报告
    report = validator.generate_report()
    print("\n" + report)
    
    # 保存报告
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存到: {args.output}")
    
    # 保存详细结果为JSON
    results_path = args.output.replace('.txt', '_detailed.json') if args.output else 'convergence_validation_results.json'
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'fixed_board': board_to_str(fixed_board) if fixed_board else None,
        'experiments': [
            {
                'config_name': r.config_name,
                'final_regret_loss': r.final_regret_loss,
                'final_policy_loss': r.final_policy_loss,
                'avg_regret_loss': r.avg_regret_loss,
                'avg_policy_loss': r.avg_policy_loss,
                'final_entropy': r.final_entropy,
                'entropy_trend': r.entropy_trend,
                'oscillation_detected': r.oscillation_detected,
                'p0_win_rate': r.p0_win_rate,
                'p1_win_rate': r.p1_win_rate,
                'avg_utility_p0': r.avg_utility_p0,
                'training_time_seconds': r.training_time_seconds,
            }
            for r in validator.results
        ],
        'comparison': validator.compare_results(),
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"详细结果已保存到: {results_path}")


if __name__ == "__main__":
    main()
