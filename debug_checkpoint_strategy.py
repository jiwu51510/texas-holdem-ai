#!/usr/bin/env python3
"""诊断检查点策略问题。

分析为什么策略全部是 100% 大加注。
"""

import torch
import numpy as np
from typing import List, Dict

from models.core import GameState, GameStage, Card
from models.networks import RegretNetwork, PolicyNetwork
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


def load_checkpoint(path: str):
    """加载检查点。"""
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


def analyze_regret_network(checkpoint: Dict):
    """分析遗憾网络的输出。"""
    print_separator("遗憾网络分析")
    
    # 创建网络
    state_encoder = StateEncoder()
    input_dim = state_encoder.encoding_dim
    action_dim = 6
    
    regret_network = RegretNetwork(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        action_dim=action_dim
    )
    
    # 加载权重
    if 'regret_network_state_dict' in checkpoint:
        regret_network.load_state_dict(checkpoint['regret_network_state_dict'])
    elif 'regret_network' in checkpoint:
        regret_network.load_state_dict(checkpoint['regret_network'])
    else:
        print("错误：找不到遗憾网络权重")
        return
    
    regret_network.eval()
    
    # 创建测试状态
    board = [parse_card(c) for c in ['Ah', 'As', 'Qd', '7c', '2h']]
    
    # 测试不同的手牌
    test_hands = [
        ([parse_card('Ac'), parse_card('Ad')], "AA（坚果牌）"),
        ([parse_card('Kh'), parse_card('Ks')], "KK（强牌）"),
        ([parse_card('Jh'), parse_card('Js')], "JJ（中等牌）"),
        ([parse_card('5h'), parse_card('5s')], "55（弱牌）"),
        ([parse_card('3h'), parse_card('4s')], "34o（空气牌）"),
    ]
    
    print("动作索引映射：")
    print("  0: FOLD, 1: CHECK, 2: CALL, 3: RAISE_SMALL, 4: RAISE_BIG, 5: ALL_IN")
    print()
    
    for hand, desc in test_hands:
        # 创建游戏状态
        state = GameState(
            player_hands=[(hand[0], hand[1]), (parse_card('Kc'), parse_card('Qc'))],
            community_cards=board,
            pot=20,
            player_stacks=[990, 990],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.RIVER,
            action_history=[],
            current_player=1
        )
        
        # 编码状态 - 使用玩家0的视角（手牌在 player_hands[0]）
        state_encoding = state_encoder.encode(state, 0)
        state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
        
        # 获取遗憾值
        with torch.no_grad():
            regrets = regret_network(state_tensor).squeeze(0).numpy()
            strategy = regret_network.get_strategy(state_tensor).squeeze(0).numpy()
        
        print(f"{desc}:")
        print(f"  遗憾值: {regrets}")
        print(f"  策略:   {strategy}")
        print(f"  最大遗憾动作: {np.argmax(regrets)} ({['FOLD', 'CHECK', 'CALL', 'RAISE_S', 'RAISE_B', 'ALL_IN'][np.argmax(regrets)]})")
        print(f"  最大策略动作: {np.argmax(strategy)} ({['FOLD', 'CHECK', 'CALL', 'RAISE_S', 'RAISE_B', 'ALL_IN'][np.argmax(strategy)]})")
        print()


def analyze_network_weights(checkpoint: Dict):
    """分析网络权重的分布。"""
    print_separator("网络权重分析")
    
    if 'regret_network_state_dict' in checkpoint:
        state_dict = checkpoint['regret_network_state_dict']
    elif 'regret_network' in checkpoint:
        state_dict = checkpoint['regret_network']
    else:
        print("错误：找不到遗憾网络权重")
        return
    
    print("遗憾网络各层权重统计：")
    for name, param in state_dict.items():
        if 'weight' in name or 'bias' in name:
            data = param.numpy()
            print(f"  {name}:")
            print(f"    形状: {data.shape}")
            print(f"    均值: {data.mean():.6f}")
            print(f"    标准差: {data.std():.6f}")
            print(f"    最小值: {data.min():.6f}")
            print(f"    最大值: {data.max():.6f}")
    
    # 特别关注输出层
    print("\n输出层偏置（决定各动作的基础倾向）：")
    for name, param in state_dict.items():
        if 'bias' in name:
            # 找最后一层
            pass
    
    # 找到最后一层的偏置
    last_bias_name = None
    for name in state_dict.keys():
        if 'bias' in name:
            last_bias_name = name
    
    if last_bias_name:
        bias = state_dict[last_bias_name].numpy()
        print(f"  {last_bias_name}: {bias}")
        print(f"  动作偏好排序: {np.argsort(bias)[::-1]}")


def analyze_training_stats(checkpoint: Dict):
    """分析训练统计信息。"""
    print_separator("训练统计信息")
    
    print(f"迭代次数: {checkpoint.get('iteration', 'N/A')}")
    print(f"Episode: {checkpoint.get('episode_number', 'N/A')}")
    
    stats = checkpoint.get('stats', {})
    if stats:
        print(f"统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # 检查固定公共牌
    fixed_board = checkpoint.get('fixed_board')
    if fixed_board:
        print(f"固定公共牌: {fixed_board}")


def check_regret_matching():
    """检查 Regret Matching 的实现。"""
    print_separator("Regret Matching 检查")
    
    print("Regret Matching 算法：")
    print("  1. 取正遗憾值：positive = max(regrets, 0)")
    print("  2. 如果 sum(positive) > 0：strategy = positive / sum(positive)")
    print("  3. 否则：strategy = uniform distribution")
    print()
    
    # 测试不同的遗憾值分布
    test_cases = [
        np.array([0, 0, 0, 0, 100, 0]),  # 只有 RAISE_BIG 有正遗憾
        np.array([-10, -5, -5, -2, 10, -3]),  # RAISE_BIG 最高
        np.array([5, 3, 2, 1, 10, 0]),  # 多个正遗憾
        np.array([-1, -1, -1, -1, -1, -1]),  # 全负
    ]
    
    for i, regrets in enumerate(test_cases):
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        if total > 0:
            strategy = positive / total
        else:
            strategy = np.ones(6) / 6
        
        print(f"测试 {i+1}:")
        print(f"  遗憾值: {regrets}")
        print(f"  正遗憾: {positive}")
        print(f"  策略:   {strategy}")
        print()


def main():
    """主函数。"""
    checkpoint_path = "checkpoints/river_only_fixed/checkpoint_70000.pt"
    
    print("检查点策略诊断")
    print("=" * 70)
    print(f"检查点路径: {checkpoint_path}")
    
    try:
        checkpoint = load_checkpoint(checkpoint_path)
        print(f"检查点加载成功")
        print(f"检查点格式: {checkpoint.get('checkpoint_format', '未知')}")
    except Exception as e:
        print(f"加载检查点失败: {e}")
        return
    
    # 分析训练统计
    analyze_training_stats(checkpoint)
    
    # 分析网络权重
    analyze_network_weights(checkpoint)
    
    # 分析遗憾网络输出
    analyze_regret_network(checkpoint)
    
    # 检查 Regret Matching
    check_regret_matching()
    
    print_separator("诊断结论")
    print("如果所有手牌的策略都是 100% RAISE_BIG，可能的原因：")
    print("1. 遗憾网络的输出偏向 RAISE_BIG（输出层偏置问题）")
    print("2. 训练数据不平衡（RAISE_BIG 的遗憾值累积过高）")
    print("3. 遗憾值累积逻辑有问题")
    print("4. 训练迭代次数不足")


if __name__ == '__main__':
    main()
