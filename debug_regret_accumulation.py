#!/usr/bin/env python3
"""诊断遗憾值累积是否正确工作。

运行几次 CFR 迭代，检查遗憾值是否在累积。
"""

import torch
import numpy as np
from typing import List, Dict

from models.core import GameState, GameStage, Card, TrainingConfig
from train_river_only import RiverOnlyTrainer, parse_card


def print_separator(title: str):
    """打印分隔线。"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    """主函数。"""
    print("遗憾值累积诊断")
    print("=" * 70)
    
    # 创建配置
    config = TrainingConfig(
        initial_stack=30,
        small_blind=5,
        big_blind=10,
        max_raises_per_street=4,
        network_architecture=[512, 256, 128],
        learning_rate=0.001,
        batch_size=256,
        regret_buffer_size=100000,
        strategy_buffer_size=100000,
        network_train_steps=100,
    )
    
    # 固定公共牌
    board = [parse_card(c) for c in ['Ah', 'As', 'Qd', '7c', '2h']]
    
    # 创建训练器
    trainer = RiverOnlyTrainer(config, fixed_board=board)
    
    print(f"初始筹码: {config.initial_stack}")
    print(f"大盲注: {config.big_blind}")
    print(f"公共牌: {' '.join(str(c) for c in board)}")
    
    print_separator("初始网络输出")
    
    # 测试状态
    test_hand = [parse_card('Ac'), parse_card('Ad')]
    state = trainer._create_river_state(
        [[test_hand[0], test_hand[1]], [parse_card('Kc'), parse_card('Qc')]],
        board
    )
    
    # 获取初始策略
    strategy = trainer._get_strategy(state, 0)
    print(f"初始策略: {strategy}")
    print(f"最大动作: {np.argmax(strategy)}")
    
    # 获取初始遗憾值
    state_encoding = trainer.state_encoder.encode(state, 0)
    state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
    with torch.no_grad():
        regrets = trainer.regret_network(state_tensor).squeeze(0).numpy()
    print(f"初始遗憾值: {regrets}")
    
    print_separator("运行 CFR 迭代")
    
    # 运行一些 CFR 迭代
    for i in range(10):
        result = trainer.run_cfr_iteration(verbose=False)
        if i < 3 or i == 9:
            print(f"迭代 {i+1}:")
            print(f"  缓冲区大小: regret={result['regret_buffer_size']}, strategy={result['strategy_buffer_size']}")
    
    print_separator("CFR 迭代后的网络输出（训练前）")
    
    # 获取 CFR 迭代后的策略（网络还没训练）
    strategy = trainer._get_strategy(state, 0)
    print(f"策略: {strategy}")
    print(f"最大动作: {np.argmax(strategy)}")
    
    with torch.no_grad():
        regrets = trainer.regret_network(state_tensor).squeeze(0).numpy()
    print(f"遗憾值: {regrets}")
    
    print_separator("检查缓冲区内容")
    
    # 检查缓冲区中的遗憾值
    if len(trainer.regret_buffer) > 0:
        states, targets, iterations = trainer.regret_buffer.sample(min(10, len(trainer.regret_buffer)))
        print(f"采样 {len(states)} 个样本")
        print(f"遗憾值目标示例:")
        for i, target in enumerate(targets[:5]):
            print(f"  样本 {i}: {target}")
            print(f"    正遗憾: {np.maximum(target, 0)}")
            print(f"    最大动作: {np.argmax(target)}")
    
    print_separator("训练网络")
    
    # 训练网络
    train_result = trainer.train_networks(verbose=True)
    print(f"遗憾网络损失: {train_result['regret_loss']:.6f}")
    print(f"策略网络损失: {train_result['policy_loss']:.6f}")
    
    print_separator("训练后的网络输出")
    
    # 获取训练后的策略
    strategy = trainer._get_strategy(state, 0)
    print(f"策略: {strategy}")
    print(f"最大动作: {np.argmax(strategy)}")
    
    with torch.no_grad():
        regrets = trainer.regret_network(state_tensor).squeeze(0).numpy()
    print(f"遗憾值: {regrets}")
    
    print_separator("多轮训练")
    
    # 多轮训练
    for round_num in range(5):
        # 运行 CFR 迭代
        for _ in range(100):
            trainer.run_cfr_iteration(verbose=False)
        
        # 训练网络
        trainer.train_networks(verbose=False)
        
        # 检查策略
        strategy = trainer._get_strategy(state, 0)
        with torch.no_grad():
            regrets = trainer.regret_network(state_tensor).squeeze(0).numpy()
        
        print(f"轮次 {round_num + 1} (迭代 {trainer.iteration}):")
        print(f"  策略: {strategy}")
        print(f"  遗憾值: {regrets}")
        print(f"  最大动作: {np.argmax(strategy)} ({['FOLD', 'CHECK', 'CALL', 'RAISE_S', 'RAISE_B', 'ALL_IN'][np.argmax(strategy)]})")


if __name__ == '__main__':
    main()
