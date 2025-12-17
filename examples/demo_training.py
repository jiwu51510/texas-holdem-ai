"""训练示例脚本 - 演示如何使用 Deep CFR 训练德州扑克AI。

本脚本展示了：
- 创建 Deep CFR 训练配置
- 初始化训练引擎（使用遗憾网络和策略网络）
- 执行 Deep CFR 训练循环
- 保存和加载检查点

Deep CFR 架构说明：
- 遗憾网络（Regret Network）：学习每个动作的即时遗憾值
- 策略网络（Policy Network）：学习长期平均策略
- 蓄水池缓冲区：存储训练样本，使用蓄水池采样保持固定大小
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.core import TrainingConfig
from training.training_engine import TrainingEngine
from utils.config_manager import ConfigManager


def demo_basic_training():
    """演示 Deep CFR 基本训练流程。"""
    print("=" * 60)
    print("德州扑克AI训练示例 - Deep CFR 训练")
    print("=" * 60)
    
    # 创建 Deep CFR 训练配置
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=256,  # Deep CFR 通常使用较大批次
        num_episodes=100,  # 示例使用较少回合数
        discount_factor=1.0,  # CFR 通常不使用折扣
        network_architecture=[256, 128, 64],  # 较小的网络用于演示
        checkpoint_interval=50,
        num_parallel_envs=1,
        initial_stack=1000,
        small_blind=5,
        big_blind=10,
        # Deep CFR 特有参数
        regret_buffer_size=10000,  # 演示用较小缓冲区
        strategy_buffer_size=10000,
        cfr_iterations_per_update=10,  # 演示用较少迭代
        network_train_steps=100
    )
    
    print("\n训练配置:")
    print(f"  学习率: {config.learning_rate}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  训练回合数: {config.num_episodes}")
    print(f"  网络架构: {config.network_architecture}")
    print(f"  检查点间隔: {config.checkpoint_interval}")
    print(f"  初始筹码: {config.initial_stack}")
    print(f"  盲注: {config.small_blind}/{config.big_blind}")
    print("\nDeep CFR 参数:")
    print(f"  遗憾缓冲区大小: {config.regret_buffer_size}")
    print(f"  策略缓冲区大小: {config.strategy_buffer_size}")
    print(f"  CFR迭代次数/更新: {config.cfr_iterations_per_update}")
    print(f"  网络训练步数: {config.network_train_steps}")
    
    # 创建检查点目录
    checkpoint_dir = "checkpoints/demo"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化训练引擎
    print("\n初始化训练引擎...")
    engine = TrainingEngine(config, checkpoint_dir=checkpoint_dir)
    
    # 执行训练
    print("\n开始训练...")
    print("-" * 60)
    
    results = engine.train(num_episodes=config.num_episodes)
    
    print("-" * 60)
    print("\n训练完成!")
    print(f"  总回合数: {results['total_episodes']}")
    print(f"  最终胜率: {results['win_rate']:.2%}")
    print(f"  平均奖励: {results['avg_reward']:.2f}")
    
    return engine


def demo_checkpoint_resume():
    """演示从检查点恢复训练。"""
    print("\n" + "=" * 60)
    print("德州扑克AI训练示例 - 检查点恢复")
    print("=" * 60)
    
    checkpoint_dir = "checkpoints/demo"
    
    # 检查是否有可用的检查点
    from utils.checkpoint_manager import CheckpointManager
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        print("\n没有找到可用的检查点，请先运行基本训练示例。")
        return None
    
    # 使用最新的检查点
    latest_checkpoint = checkpoints[-1]
    print(f"\n找到检查点: {latest_checkpoint.path}")
    print(f"  回合数: {latest_checkpoint.episode_number}")
    print(f"  胜率: {latest_checkpoint.win_rate:.2%}")
    print(f"  保存时间: {latest_checkpoint.timestamp}")
    
    # 创建新的训练配置
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=16,
        num_episodes=50,  # 继续训练50回合
        discount_factor=0.99,
        network_architecture=[256, 128, 64],
        checkpoint_interval=25,
        num_parallel_envs=1,
        initial_stack=1000,
        small_blind=5,
        big_blind=10
    )
    
    # 初始化训练引擎并加载检查点
    print("\n从检查点恢复训练...")
    engine = TrainingEngine(config, checkpoint_dir=checkpoint_dir)
    engine.load_checkpoint(latest_checkpoint.path)
    
    # 继续训练
    print("\n继续训练...")
    print("-" * 60)
    
    results = engine.train(num_episodes=50)
    
    print("-" * 60)
    print("\n训练完成!")
    print(f"  总回合数: {results['total_episodes']}")
    print(f"  最终胜率: {results['win_rate']:.2%}")
    print(f"  平均奖励: {results['avg_reward']:.2f}")
    
    return engine


def demo_config_file_training():
    """演示使用配置文件进行训练。"""
    print("\n" + "=" * 60)
    print("德州扑克AI训练示例 - 配置文件训练")
    print("=" * 60)
    
    config_path = "configs/demo_config.json"
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"\n配置文件不存在: {config_path}")
        print("请先创建配置文件，或运行 demo_create_config() 创建示例配置。")
        return None
    
    # 加载配置
    config_manager = ConfigManager()
    print(f"\n从配置文件加载: {config_path}")
    config = config_manager.load_config(config_path)
    
    print("\n加载的配置:")
    print(f"  学习率: {config.learning_rate}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  训练回合数: {config.num_episodes}")
    print(f"  网络架构: {config.network_architecture}")
    
    # 初始化训练引擎
    checkpoint_dir = "checkpoints/config_demo"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n初始化训练引擎...")
    engine = TrainingEngine(config, checkpoint_dir=checkpoint_dir)
    
    # 执行训练
    print("\n开始训练...")
    print("-" * 60)
    
    results = engine.train()
    
    print("-" * 60)
    print("\n训练完成!")
    
    return engine


def demo_create_config():
    """演示创建和保存 Deep CFR 配置文件。"""
    print("\n" + "=" * 60)
    print("德州扑克AI训练示例 - 创建 Deep CFR 配置文件")
    print("=" * 60)
    
    # 创建配置目录
    os.makedirs("configs", exist_ok=True)
    
    # 创建 Deep CFR 配置
    config = TrainingConfig(
        learning_rate=0.0005,
        batch_size=256,  # Deep CFR 通常使用较大批次
        num_episodes=100,
        discount_factor=1.0,  # CFR 通常不使用折扣
        network_architecture=[256, 128, 64],
        checkpoint_interval=50,
        num_parallel_envs=1,
        initial_stack=1000,
        small_blind=5,
        big_blind=10,
        # Deep CFR 特有参数
        regret_buffer_size=10000,
        strategy_buffer_size=10000,
        cfr_iterations_per_update=10,
        network_train_steps=100
    )
    
    # 保存配置
    config_manager = ConfigManager()
    config_path = "configs/demo_config.json"
    config_manager.save_config(config, config_path)
    
    print(f"\n配置文件已保存到: {config_path}")
    print("\n基础配置:")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  num_episodes: {config.num_episodes}")
    print(f"  discount_factor: {config.discount_factor}")
    print(f"  network_architecture: {config.network_architecture}")
    print(f"  checkpoint_interval: {config.checkpoint_interval}")
    print(f"  num_parallel_envs: {config.num_parallel_envs}")
    print(f"  initial_stack: {config.initial_stack}")
    print(f"  small_blind: {config.small_blind}")
    print(f"  big_blind: {config.big_blind}")
    print("\nDeep CFR 参数:")
    print(f"  regret_buffer_size: {config.regret_buffer_size}")
    print(f"  strategy_buffer_size: {config.strategy_buffer_size}")
    print(f"  cfr_iterations_per_update: {config.cfr_iterations_per_update}")
    print(f"  network_train_steps: {config.network_train_steps}")


if __name__ == "__main__":
    print("德州扑克AI训练系统 - Deep CFR 训练示例\n")
    
    # 创建示例配置文件
    demo_create_config()
    
    # 运行 Deep CFR 训练
    engine = demo_basic_training()
    
    # 演示从检查点恢复
    if engine:
        demo_checkpoint_resume()
    
    print("\n" + "=" * 60)
    print("所有 Deep CFR 训练示例完成!")
    print("=" * 60)
