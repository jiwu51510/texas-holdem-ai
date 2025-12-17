#!/usr/bin/env python3
"""诊断手牌编码问题。"""

import numpy as np
from models.core import GameState, GameStage, Card
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


def main():
    """主函数。"""
    print("手牌编码诊断")
    print("=" * 70)
    
    state_encoder = StateEncoder()
    
    # 测试 encode_cards 方法
    card1 = parse_card('Ac')
    card2 = parse_card('Ad')
    
    print(f"测试牌: {card1}, {card2}")
    print(f"card1.rank = {card1.rank}, card1.suit = {card1.suit}")
    print(f"card2.rank = {card2.rank}, card2.suit = {card2.suit}")
    
    # 直接调用 encode_cards
    encoding = state_encoder.encode_cards([card1, card2])
    print(f"\nencode_cards 结果:")
    print(f"  长度: {len(encoding)}")
    print(f"  非零元素数: {np.count_nonzero(encoding)}")
    print(f"  非零位置: {np.where(encoding > 0)[0]}")
    
    # 检查 GameState 中的 player_hands 格式
    board = [parse_card(c) for c in ['Ah', 'As', 'Qd', '7c', '2h']]
    
    state = GameState(
        player_hands=[(card1, card2), (parse_card('Kc'), parse_card('Qc'))],
        community_cards=board,
        pot=20,
        player_stacks=[990, 990],
        current_bets=[0, 0],
        button_position=0,
        stage=GameStage.RIVER,
        action_history=[],
        current_player=1
    )
    
    print(f"\nGameState.player_hands[0] 类型: {type(state.player_hands[0])}")
    print(f"GameState.player_hands[0] 内容: {state.player_hands[0]}")
    
    hand = state.player_hands[0]
    print(f"\nhand[0] 类型: {type(hand[0])}")
    print(f"hand[1] 类型: {type(hand[1])}")
    
    # 测试 _encode_full 中的逻辑
    print("\n测试 _encode_full 中的手牌编码逻辑:")
    hand_encoding = state_encoder.encode_cards([hand[0], hand[1]])
    print(f"  长度: {len(hand_encoding)}")
    print(f"  非零元素数: {np.count_nonzero(hand_encoding)}")
    print(f"  非零位置: {np.where(hand_encoding > 0)[0]}")
    
    # 完整编码
    print("\n完整状态编码:")
    full_encoding = state_encoder.encode(state, 0)
    print(f"  长度: {len(full_encoding)}")
    print(f"  非零元素数: {np.count_nonzero(full_encoding)}")
    print(f"  前104维（手牌）非零位置: {np.where(full_encoding[:104] > 0)[0]}")
    print(f"  104-364维（公共牌）非零位置: {np.where(full_encoding[104:364] > 0)[0]}")
    
    # 测试不同手牌
    print("\n" + "=" * 70)
    print("测试不同手牌的编码")
    print("=" * 70)
    
    test_hands = [
        ([parse_card('Ac'), parse_card('Ad')], "AA"),
        ([parse_card('Kh'), parse_card('Ks')], "KK"),
        ([parse_card('3h'), parse_card('4s')], "34o"),
    ]
    
    for hand_cards, desc in test_hands:
        state = GameState(
            player_hands=[(hand_cards[0], hand_cards[1]), (parse_card('Kc'), parse_card('Qc'))],
            community_cards=board,
            pot=20,
            player_stacks=[990, 990],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.RIVER,
            action_history=[],
            current_player=0  # 玩家0的视角
        )
        
        encoding = state_encoder.encode(state, 0)
        hand_part = encoding[:104]
        
        print(f"\n{desc}:")
        print(f"  手牌部分非零位置: {np.where(hand_part > 0)[0]}")
        print(f"  手牌部分非零数量: {np.count_nonzero(hand_part)}")


if __name__ == '__main__':
    main()
