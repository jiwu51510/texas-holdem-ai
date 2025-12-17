#!/usr/bin/env python3
"""诊断合法动作问题。

检查在不同状态下的合法动作。
"""

import torch
import numpy as np
from typing import List

from models.core import GameState, GameStage, Card, TrainingConfig
from models.networks import RegretNetwork
from environment.state_encoder import StateEncoder
from environment.poker_environment import PokerEnvironment


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


def main():
    """主函数。"""
    print("合法动作诊断")
    print("=" * 70)
    
    # 创建环境
    env = PokerEnvironment(
        initial_stack=30,
        small_blind=5,
        big_blind=10,
        max_raises_per_street=4
    )
    
    # 固定公共牌
    board = [parse_card(c) for c in ['Ah', 'As', 'Qd', '7c', '2h']]
    
    # 创建河牌初始状态
    state = GameState(
        player_hands=[
            (parse_card('Ac'), parse_card('Ad')),
            (parse_card('Kc'), parse_card('Qc'))
        ],
        community_cards=board,
        pot=20,  # 2 * big_blind
        player_stacks=[20, 20],  # initial_stack - big_blind
        current_bets=[0, 0],
        button_position=0,
        stage=GameStage.RIVER,
        action_history=[],
        current_player=1  # 大盲位先行动
    )
    
    print("河牌初始状态（大盲位先行动）：")
    print(f"  底池: {state.pot}")
    print(f"  筹码: P0={state.player_stacks[0]}, P1={state.player_stacks[1]}")
    print(f"  当前下注: {state.current_bets}")
    print(f"  当前玩家: {state.current_player}")
    
    legal_actions = env.get_legal_actions(state)
    print(f"\n合法动作:")
    for action in legal_actions:
        print(f"  {action.action_type.name}: amount={action.amount}")
    
    print("\n" + "=" * 70)
    print("注意：在河牌初始状态，没有 CALL 动作（因为没有人下注）")
    print("所以 CHECK 和 CALL 在不同状态下是互斥的")
    print("=" * 70)
    
    # 加载检查点
    checkpoint_path = "checkpoints/river_only_fixed/checkpoint_70000.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"\n加载检查点: {checkpoint_path}")
        
        # 创建网络
        state_encoder = StateEncoder()
        regret_network = RegretNetwork(
            input_dim=state_encoder.encoding_dim,
            hidden_dims=[512, 256, 128],
            action_dim=6
        )
        regret_network.load_state_dict(checkpoint['regret_network_state_dict'])
        regret_network.eval()
        
        # 测试不同状态
        print("\n" + "=" * 70)
        print("测试不同状态的策略")
        print("=" * 70)
        
        # 状态1：河牌初始（没有下注）
        state1 = GameState(
            player_hands=[
                (parse_card('Ac'), parse_card('Ad')),
                (parse_card('Kc'), parse_card('Qc'))
            ],
            community_cards=board,
            pot=20,
            player_stacks=[20, 20],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.RIVER,
            action_history=[],
            current_player=1
        )
        
        encoding1 = state_encoder.encode(state1, 1)
        tensor1 = torch.FloatTensor(encoding1).unsqueeze(0)
        with torch.no_grad():
            regrets1 = regret_network(tensor1).squeeze(0).numpy()
            strategy1 = regret_network.get_strategy(tensor1).squeeze(0).numpy()
        
        print("\n状态1：河牌初始（P1 先行动，无下注）")
        print(f"  合法动作: CHECK, RAISE_SMALL, RAISE_BIG, ALL_IN")
        print(f"  遗憾值: {regrets1}")
        print(f"  策略: {strategy1}")
        print(f"  最大策略动作: {np.argmax(strategy1)} ({['FOLD', 'CHECK', 'CALL', 'RAISE_S', 'RAISE_B', 'ALL_IN'][np.argmax(strategy1)]})")
        
        # 状态2：对手已下注
        state2 = GameState(
            player_hands=[
                (parse_card('Ac'), parse_card('Ad')),
                (parse_card('Kc'), parse_card('Qc'))
            ],
            community_cards=board,
            pot=40,  # 20 + 20 (对手下注)
            player_stacks=[20, 0],  # P1 已全下
            current_bets=[0, 20],
            button_position=0,
            stage=GameStage.RIVER,
            action_history=[],
            current_player=0
        )
        
        encoding2 = state_encoder.encode(state2, 0)
        tensor2 = torch.FloatTensor(encoding2).unsqueeze(0)
        with torch.no_grad():
            regrets2 = regret_network(tensor2).squeeze(0).numpy()
            strategy2 = regret_network.get_strategy(tensor2).squeeze(0).numpy()
        
        print("\n状态2：对手已全下（P0 行动）")
        print(f"  合法动作: FOLD, CALL")
        print(f"  遗憾值: {regrets2}")
        print(f"  策略: {strategy2}")
        print(f"  最大策略动作: {np.argmax(strategy2)} ({['FOLD', 'CHECK', 'CALL', 'RAISE_S', 'RAISE_B', 'ALL_IN'][np.argmax(strategy2)]})")
        
    except Exception as e:
        print(f"加载检查点失败: {e}")


if __name__ == '__main__':
    main()
