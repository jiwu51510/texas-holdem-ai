#!/usr/bin/env python3
"""带监控的训练脚本 - 演示如何使用训练引擎和监控系统。

使用方法：
    python3 run_training_with_monitor.py [--episodes N] [--checkpoint-dir DIR]
    
TensorBoard查看：
    tensorboard --logdir=runs
    然后在浏览器中打开 http://localhost:6006
"""

import os
import sys
import argparse
import signal
from datetime import datetime

from models.core import TrainingConfig
from training.training_engine import TrainingEngine
from monitoring.metrics_collector import MetricsCollector
from monitoring.training_monitor import TrainingMonitor
from utils.checkpoint_manager import CheckpointManager

# 检查TensorBoard是否可用
try:
    from monitoring.tensorboard_logger import is_tensorboard_available
    TENSORBOARD_AVAILABLE = is_tensorboard_available()
except ImportError:
    TENSORBOARD_AVAILABLE = False


def signal_handler(signum, frame):
    """处理中断信号，优雅退出。"""
    print("\n\n收到中断信号，正在保存模型...")
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description='德州扑克AI训练（带监控）')
    parser.add_argument('--episodes', '-e', type=int, default=None,
                        help='训练回合数（默认使用配置文件中的值）')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='配置文件路径（JSON格式）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='从指定检查点恢复训练')
    parser.add_argument('--checkpoint-dir', '-d', type=str, default='checkpoints/training',
                        help='检查点保存目录（默认: checkpoints/training）')
    parser.add_argument('--checkpoint-interval', '-i', type=int, default=None,
                        help='检查点保存间隔（默认使用配置文件中的值）')
    parser.add_argument('--log-dir', '-l', type=str, default='logs',
                        help='日志目录（默认: logs）')
    parser.add_argument('--tensorboard-dir', '-t', type=str, default='runs',
                        help='TensorBoard日志目录（默认: runs）')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='禁用TensorBoard')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='实验名称（用于TensorBoard）')
    args = parser.parse_args()
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 70)
    print("德州扑克AI训练系统 - 带监控训练")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 加载配置文件或使用默认配置
    from utils.config_manager import ConfigManager
    config_manager = ConfigManager()
    
    if args.config:
        print(f"加载配置文件: {args.config}")
        config = config_manager.load_config(args.config)
        
        # 命令行参数覆盖配置文件
        if args.episodes is not None or args.checkpoint_interval is not None:
            # 需要重新创建配置对象
            from dataclasses import asdict
            config_dict = asdict(config)
            if args.episodes is not None:
                config_dict['num_episodes'] = args.episodes
            if args.checkpoint_interval is not None:
                config_dict['checkpoint_interval'] = args.checkpoint_interval
            config = TrainingConfig(**config_dict)
    else:
        # 尝试加载默认配置文件
        default_config_path = 'configs/default_config.json'
        if os.path.exists(default_config_path):
            print(f"加载默认配置文件: {default_config_path}")
            config = config_manager.load_config(default_config_path)
            
            # 命令行参数覆盖
            if args.episodes is not None or args.checkpoint_interval is not None:
                from dataclasses import asdict
                config_dict = asdict(config)
                if args.episodes is not None:
                    config_dict['num_episodes'] = args.episodes
                if args.checkpoint_interval is not None:
                    config_dict['checkpoint_interval'] = args.checkpoint_interval
                config = TrainingConfig(**config_dict)
        else:
            # 使用代码中的默认配置
            print("使用内置默认配置")
            config = config_manager.get_default_config()
            if args.episodes is not None:
                from dataclasses import asdict
                config_dict = asdict(config)
                config_dict['num_episodes'] = args.episodes
                if args.checkpoint_interval is not None:
                    config_dict['checkpoint_interval'] = args.checkpoint_interval
                config = TrainingConfig(**config_dict)
    
    print("训练配置:")
    print(f"  训练回合数: {config.num_episodes}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  网络架构: {config.network_architecture}")
    print(f"  检查点间隔: {config.checkpoint_interval}")
    print(f"  初始筹码: {config.initial_stack}")
    print(f"  盲注: {config.small_blind}/{config.big_blind}")
    print("Deep CFR 参数:")
    print(f"  遗憾缓冲区大小: {config.regret_buffer_size}")
    print(f"  策略缓冲区大小: {config.strategy_buffer_size}")
    print(f"  CFR迭代次数/更新: {config.cfr_iterations_per_update}")
    print(f"  网络训练步数: {config.network_train_steps}")
    print()
    
    # TensorBoard设置
    enable_tensorboard = not args.no_tensorboard and TENSORBOARD_AVAILABLE
    if enable_tensorboard:
        print(f"TensorBoard: 已启用")
        print(f"  日志目录: {args.tensorboard_dir}")
        print(f"  查看命令: tensorboard --logdir={args.tensorboard_dir}")
        print(f"  浏览器打开: http://localhost:6006")
    else:
        if args.no_tensorboard:
            print("TensorBoard: 已禁用（用户选择）")
        else:
            print("TensorBoard: 未安装（pip install tensorboard）")
    print()
    
    # 创建训练引擎
    print("初始化训练引擎...")
    engine = TrainingEngine(
        config, 
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir,
        enable_tensorboard=enable_tensorboard,
        experiment_name=args.experiment_name
    )
    
    # 从检查点恢复训练
    if args.checkpoint:
        print(f"从检查点恢复: {args.checkpoint}")
        engine.load_checkpoint(args.checkpoint)
        print()
    
    # 创建监控系统
    log_path = os.path.join(args.log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    metrics_collector = MetricsCollector(window_size=100)
    monitor = TrainingMonitor(
        metrics_collector=metrics_collector,
        update_interval=5.0,
        log_file_path=log_path
    )
    
    # 设置监控回调
    def on_update(metrics):
        """监控更新回调。"""
        print(f"\r[回合 {metrics.get('episodes_completed', 0):4d}] "
              f"胜率: {metrics.get('win_rate', 0):.1%} | "
              f"平均奖励: {metrics.get('avg_reward', 0):+.1f} | "
              f"损失: {metrics.get('loss', 0):.4f}", end='', flush=True)
    
    def on_anomaly(anomalies):
        """异常检测回调。"""
        print(f"\n⚠️  检测到异常: {', '.join(anomalies)}")
    
    monitor.set_on_update_callback(on_update)
    monitor.set_on_anomaly_callback(on_anomaly)
    
    # 启动监控
    monitor.start()
    print(f"监控已启动，日志保存到: {log_path}")
    print()
    print("-" * 70)
    print("开始训练... (按 Ctrl+C 可安全中断)")
    print("-" * 70)
    
    try:
        # 执行训练
        results = engine.train(num_episodes=config.num_episodes)
        
        print("\n" + "-" * 70)
        print("训练完成!")
        print("-" * 70)
        print(f"  总回合数: {results['total_episodes']}")
        print(f"  最终胜率: {results['win_rate']:.2%}")
        print(f"  平均奖励: {results['avg_reward']:.2f}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        # 保存当前状态
        engine.save_checkpoint()
        print("已保存当前模型状态")
    
    finally:
        # 停止监控
        monitor.stop()
    
    # 显示检查点信息
    print()
    print("-" * 70)
    print("可用检查点:")
    print("-" * 70)
    
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if checkpoints:
        for i, cp in enumerate(checkpoints[-5:], 1):  # 显示最近5个
            print(f"  {i}. {os.path.basename(cp.path)}")
            print(f"     回合: {cp.episode_number} | 胜率: {cp.win_rate:.2%} | "
                  f"时间: {cp.timestamp.strftime('%H:%M:%S')}")
    else:
        print("  没有找到检查点")
    
    print()
    print("=" * 70)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 返回最新检查点路径
    if checkpoints:
        return checkpoints[-1].path
    return None


if __name__ == "__main__":
    main()
