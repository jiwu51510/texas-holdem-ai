#!/usr/bin/env python3
"""诊断 CFR 遗憾值累积问题。

Deep CFR 的正确实现应该累积遗憾值，而不是只存储即时遗憾值。

标准 CFR 算法：
- 累积遗憾[t] = 累积遗憾[t-1] + 即时遗憾[t]
- 策略 = Regret Matching(累积遗憾)

Deep CFR 的正确做法：
- 网络预测累积遗憾值
- 新遗憾 = 网络预测值 + 即时遗憾
- 存储新遗憾到缓冲区
"""

import numpy as np
import torch
from typing import List, Dict, Tuple

from models.core import TrainingConfig, GameState, Action, ActionType, GameStage, Card
from models.networks import RegretNetwork
from environment.poker_environment import PokerEnvironment
from environment.state_encoder import StateEncoder


def parse_card(card_str: str) -> Card:
    """解析牌的字符串表示。"""
    rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
    rank_str = card_str[:-1].upper()
    suit_str = card_str[-1].lower()
    if rank_str in rank_map:
        rank = rank_map[rank_str]
    else:
        rank = int(rank_str)
    return Card(rank, suit_str)


def card_to_str(card: Card) -> str:
    """将牌转换为字符串。"""
    rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
    rank_str = rank_map.get(card.rank, str(card.rank))
    return f"{rank_str}{card.suit}"


def print_separator(title: str):
    """打印分隔线。"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demonstrate_cfr_accumulation():
    """演示 CFR 遗憾值累积的重要性。"""
    print_separator("CFR 遗憾值累积演示")
    
    print("假设一个简单的博弈场景：")
    print("- 3 个动作：A, B, C")
    print("- 动作 A 的真实价值最高")
    print()
    
    # 模拟多轮迭代
    num_iterations = 10
    
    # 方法1：不累积遗憾值（当前实现的问题）
    print("方法1：不累积遗憾值（错误）")
    print("-" * 40)
    
    regrets_no_accumulation = np.zeros(3)
    for i in range(num_iterations):
        # 模拟即时遗憾值（动作 A 总是最好）
        instant_regrets = np.array([2.0, -1.0, -1.0])  # A 比平均好 2，B 和 C 比平均差 1
        
        # 不累积，直接使用即时遗憾值
        regrets_no_accumulation = instant_regrets  # 错误：每次都覆盖
        
        # 计算策略
        positive = np.maximum(regrets_no_accumulation, 0)
        total = positive.sum()
        if total > 0:
            strategy = positive / total
        else:
            strategy = np.ones(3) / 3
        
        if i < 3 or i == num_iterations - 1:
            print(f"  迭代 {i+1}: 遗憾值={regrets_no_accumulation}, 策略={strategy}")
    
    print()
    
    # 方法2：累积遗憾值（正确实现）
    print("方法2：累积遗憾值（正确）")
    print("-" * 40)
    
    regrets_with_accumulation = np.zeros(3)
    for i in range(num_iterations):
        # 模拟即时遗憾值
        instant_regrets = np.array([2.0, -1.0, -1.0])
        
        # 累积遗憾值
        regrets_with_accumulation += instant_regrets  # 正确：累加
        
        # 计算策略
        positive = np.maximum(regrets_with_accumulation, 0)
        total = positive.sum()
        if total > 0:
            strategy = positive / total
        else:
            strategy = np.ones(3) / 3
        
        if i < 3 or i == num_iterations - 1:
            print(f"  迭代 {i+1}: 累积遗憾={regrets_with_accumulation}, 策略={strategy}")
    
    print()
    print("结论：")
    print("- 不累积时，策略始终是 [1, 0, 0]（只看即时遗憾）")
    print("- 累积时，策略也是 [1, 0, 0]，但累积遗憾值更大，更稳定")
    print("- 关键区别在于：累积遗憾值能够记住历史信息")


def check_current_implementation():
    """检查当前实现是否累积遗憾值。"""
    print_separator("检查当前实现")
    
    # 读取 train_river_only.py 中的关键代码
    print("当前 train_river_only.py 中的遗憾值计算：")
    print()
    print("```python")
    print("# 计算遗憾值（只为合法行动计算，非法行动保持为0）")
    print("regrets = np.zeros(6, dtype=np.float32)")
    print("for action_idx, action_value in action_values.items():")
    print("    regrets[action_idx] = action_value - expected_value  # 即时遗憾")
    print()
    print("# 存储样本")
    print("weighted_regrets = regrets * opponent_reach")
    print("self.regret_buffer.add(state_encoding, weighted_regrets, self.iteration)")
    print("```")
    print()
    print("问题：存储的是即时遗憾值，没有累积！")
    print()
    print("正确的做法应该是：")
    print()
    print("```python")
    print("# 获取网络预测的累积遗憾值")
    print("with torch.no_grad():")
    print("    predicted_regrets = self.regret_network(state_tensor).squeeze().numpy()")
    print()
    print("# 计算新的累积遗憾值 = 预测值 + 即时遗憾")
    print("accumulated_regrets = predicted_regrets + instant_regrets")
    print()
    print("# 存储累积遗憾值")
    print("self.regret_buffer.add(state_encoding, accumulated_regrets, self.iteration)")
    print("```")


def demonstrate_deep_cfr_correct():
    """演示 Deep CFR 的正确实现。"""
    print_separator("Deep CFR 正确实现演示")
    
    print("Deep CFR 的核心思想：")
    print("1. 使用神经网络近似累积遗憾值函数")
    print("2. 每次迭代：")
    print("   a. 用网络预测当前状态的累积遗憾值")
    print("   b. 计算即时遗憾值")
    print("   c. 新累积遗憾 = 预测值 + 即时遗憾")
    print("   d. 存储 (状态, 新累积遗憾) 到缓冲区")
    print("3. 定期训练网络拟合缓冲区中的数据")
    print()
    
    # 模拟 Deep CFR 的正确流程
    print("模拟 Deep CFR 正确流程：")
    print("-" * 40)
    
    # 简化的网络（用字典模拟）
    network_predictions = {}  # state -> predicted_regrets
    buffer = []  # [(state, accumulated_regrets)]
    
    # 模拟 5 次迭代
    for iteration in range(5):
        # 模拟一个状态
        state = f"state_{iteration % 2}"  # 只有 2 个状态
        
        # 获取网络预测（如果没有，返回 0）
        predicted = network_predictions.get(state, np.zeros(3))
        
        # 计算即时遗憾（模拟）
        instant = np.array([2.0, -1.0, -1.0])
        
        # 累积遗憾 = 预测 + 即时
        accumulated = predicted + instant
        
        # 存储到缓冲区
        buffer.append((state, accumulated.copy()))
        
        print(f"迭代 {iteration + 1}:")
        print(f"  状态: {state}")
        print(f"  网络预测: {predicted}")
        print(f"  即时遗憾: {instant}")
        print(f"  累积遗憾: {accumulated}")
        
        # 模拟网络训练（简化：直接更新预测值）
        if iteration % 2 == 1:  # 每 2 次迭代训练一次
            print("  [训练网络]")
            for s, r in buffer[-2:]:  # 用最近的样本更新
                network_predictions[s] = r
            print(f"  网络更新后: {network_predictions}")
        print()


def main():
    """主函数。"""
    print("CFR 遗憾值累积问题诊断")
    print("=" * 70)
    
    # 演示累积的重要性
    demonstrate_cfr_accumulation()
    
    # 检查当前实现
    check_current_implementation()
    
    # 演示正确实现
    demonstrate_deep_cfr_correct()
    
    print_separator("修复建议")
    print("需要修改 train_river_only.py 中的 traverse_river 方法：")
    print()
    print("1. 在计算即时遗憾值后，获取网络预测的累积遗憾值")
    print("2. 将即时遗憾值加到预测值上，得到新的累积遗憾值")
    print("3. 存储新的累积遗憾值到缓冲区")
    print()
    print("这样，网络就能学习到累积遗憾值，策略才能正确收敛。")


if __name__ == '__main__':
    main()
