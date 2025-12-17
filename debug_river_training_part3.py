#!/usr/bin/env python3
"""河牌训练诊断脚本 - 第三部分。

深入检查为什么网络输出相同的值。
"""

import numpy as np
import torch
from typing import List

from models.core import GameState, GameStage, Card
from models.networks import RegretNetwork
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
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_encoding_difference():
    """检查不同手牌的编码是否不同。"""
    print_separator("检查编码差异")
    
    encoder = StateEncoder()
    board = [parse_card(c) for c in ['Ah', 'As', 'Qd', '7c', '2h']]
    
    test_hands = [
        [parse_card('Ad'), parse_card('Kh')],  # 三条A
        [parse_card('Kd'), parse_card('Ks')],  # 两对
        [parse_card('3d'), parse_card('4s')],  # 空气
    ]
    
    encodings = []
    
    for i, hand in enumerate(test_hands):
        state = GameState(
            player_hands=[(hand[0], hand[1]), (parse_card('5d'), parse_card('6d'))],
            community_cards=board,
            pot=20,
            player_stacks=[20, 20],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.RIVER,
            action_history=[],
            current_player=1
        )
        
        encoding = encoder.encode(state, player_id=1)
        encodings.append(encoding)
        
        print(f"手牌 {i+1}: {card_to_str(hand[0])} {card_to_str(hand[1])}")
        print(f"  编码非零元素数: {np.count_nonzero(encoding)}")
        print(f"  编码和: {encoding.sum():.4f}")
        
        # 打印私牌部分的编码（前104维是私牌）
        hole_cards_encoding = encoding[:104]
        print(f"  私牌编码非零位置: {np.where(hole_cards_encoding > 0)[0].tolist()}")
    
    # 比较编码差异
    print(f"\n编码差异:")
    for i in range(len(encodings)):
        for j in range(i+1, len(encodings)):
            diff = np.abs(encodings[i] - encodings[j]).sum()
            print(f"  手牌{i+1} vs 手牌{j+1}: {diff:.4f}")


def check_network_weights():
    """检查网络权重。"""
    print_separator("检查网络权重")
    
    import os
    checkpoint_path = "checkpoints/river_only_fixed/checkpoint_15000.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"检查点不存在: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    encoder = StateEncoder()
    regret_net = RegretNetwork(
        input_dim=encoder.encoding_dim,
        hidden_dims=[512, 256, 128],
        action_dim=6
    )
    regret_net.load_state_dict(checkpoint['regret_network_state_dict'])
    
    print("网络层权重统计:")
    for name, param in regret_net.named_parameters():
        print(f"  {name}:")
        print(f"    形状: {param.shape}")
        print(f"    均值: {param.mean().item():.6f}")
        print(f"    标准差: {param.std().item():.6f}")
        print(f"    最小值: {param.min().item():.6f}")
        print(f"    最大值: {param.max().item():.6f}")


def check_forward_pass_details():
    """检查前向传播的详细过程。"""
    print_separator("检查前向传播详细过程")
    
    import os
    checkpoint_path = "checkpoints/river_only_fixed/checkpoint_15000.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"检查点不存在: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    encoder = StateEncoder()
    regret_net = RegretNetwork(
        input_dim=encoder.encoding_dim,
        hidden_dims=[512, 256, 128],
        action_dim=6
    )
    regret_net.load_state_dict(checkpoint['regret_network_state_dict'])
    regret_net.eval()
    
    board = [parse_card(c) for c in ['Ah', 'As', 'Qd', '7c', '2h']]
    
    # 测试两个不同的手牌
    hands = [
        [parse_card('Ad'), parse_card('Kh')],
        [parse_card('3d'), parse_card('4s')],
    ]
    
    for hand in hands:
        state = GameState(
            player_hands=[(hand[0], hand[1]), (parse_card('5d'), parse_card('6d'))],
            community_cards=board,
            pot=20,
            player_stacks=[20, 20],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.RIVER,
            action_history=[],
            current_player=1
        )
        
        encoding = encoder.encode(state, player_id=1)
        state_tensor = torch.FloatTensor(encoding).unsqueeze(0)
        
        print(f"\n手牌: {card_to_str(hand[0])} {card_to_str(hand[1])}")
        print(f"输入张量形状: {state_tensor.shape}")
        print(f"输入非零元素: {(state_tensor != 0).sum().item()}")
        
        # 逐层检查
        x = state_tensor
        for i, layer in enumerate(regret_net.network):
            x = layer(x)
            print(f"  层 {i} ({type(layer).__name__}): 输出形状={x.shape}, "
                  f"均值={x.mean().item():.4f}, 标准差={x.std().item():.4f}")


def check_buffer_content():
    """检查训练缓冲区的内容（如果可以访问）。"""
    print_separator("检查训练过程中的样本")
    
    # 模拟一次 CFR 遍历，检查生成的样本
    from train_river_only import RiverOnlyTrainer
    from models.core import TrainingConfig
    
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=256,
        network_architecture=[512, 256, 128],
        cfr_iterations_per_update=500,
        network_train_steps=300,
        regret_buffer_size=2000000,
        strategy_buffer_size=2000000,
        initial_stack=30,
        small_blind=5,
        big_blind=10,
        max_raises_per_street=4,
    )
    
    board = [parse_card(c) for c in ['Ah', 'As', 'Qd', '7c', '2h']]
    trainer = RiverOnlyTrainer(config, fixed_board=board)
    
    # 运行几次 CFR 迭代
    print("运行 10 次 CFR 迭代...")
    for i in range(10):
        trainer.run_cfr_iteration(verbose=False)
    
    print(f"\n缓冲区状态:")
    print(f"  遗憾缓冲区大小: {len(trainer.regret_buffer)}")
    print(f"  策略缓冲区大小: {len(trainer.strategy_buffer)}")
    
    # 检查遗憾缓冲区中的样本
    if len(trainer.regret_buffer) > 0:
        states, targets, iterations = trainer.regret_buffer.sample(min(10, len(trainer.regret_buffer)))
        
        print(f"\n遗憾缓冲区样本:")
        for i in range(min(5, len(states))):
            print(f"  样本 {i+1}:")
            print(f"    状态非零元素: {np.count_nonzero(states[i])}")
            print(f"    遗憾值: {targets[i]}")
            print(f"    遗憾值范围: [{targets[i].min():.4f}, {targets[i].max():.4f}]")


def check_regret_matching():
    """检查 Regret Matching 的实现。"""
    print_separator("检查 Regret Matching 实现")
    
    from models.networks import RegretNetwork
    
    # 创建一个简单的网络
    net = RegretNetwork(input_dim=10, hidden_dims=[32], action_dim=6)
    
    # 测试不同的遗憾值输入
    test_regrets = [
        torch.tensor([[1.0, 2.0, 3.0, 0.0, -1.0, -2.0]]),  # 混合正负
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),    # 全零
        torch.tensor([[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]]),  # 全负
        torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),   # 只有一个正
    ]
    
    print("Regret Matching 测试:")
    for i, regrets in enumerate(test_regrets):
        # 手动计算 Regret Matching
        positive_regrets = torch.clamp(regrets, min=0)
        total = positive_regrets.sum(dim=-1, keepdim=True)
        
        if total.item() > 0:
            strategy = positive_regrets / total
        else:
            strategy = torch.ones_like(regrets) / regrets.shape[-1]
        
        print(f"\n  测试 {i+1}:")
        print(f"    遗憾值: {regrets.squeeze().tolist()}")
        print(f"    正遗憾值: {positive_regrets.squeeze().tolist()}")
        print(f"    策略: {strategy.squeeze().tolist()}")


def main():
    """主函数。"""
    print("河牌训练诊断 - 第三部分（深入分析）")
    print("="*60)
    
    # 检查编码差异
    check_encoding_difference()
    
    # 检查网络权重
    check_network_weights()
    
    # 检查前向传播
    check_forward_pass_details()
    
    # 检查 Regret Matching
    check_regret_matching()
    
    # 检查缓冲区内容
    check_buffer_content()
    
    print_separator("诊断完成")


if __name__ == '__main__':
    main()
