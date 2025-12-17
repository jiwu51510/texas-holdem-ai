#!/usr/bin/env python3
"""诊断状态编码问题。

检查不同手牌的状态编码是否不同。
"""

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
    print("状态编码诊断")
    print("=" * 70)
    
    state_encoder = StateEncoder()
    print(f"编码维度: {state_encoder.encoding_dim}")
    
    # 固定公共牌
    board = [parse_card(c) for c in ['Ah', 'As', 'Qd', '7c', '2h']]
    
    # 测试不同的手牌
    test_hands = [
        ([parse_card('Ac'), parse_card('Ad')], "AA"),
        ([parse_card('Kh'), parse_card('Ks')], "KK"),
        ([parse_card('3h'), parse_card('4s')], "34o"),
    ]
    
    encodings = []
    
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
        
        # 编码状态 - 注意：我们要编码玩家0的视角，因为手牌在 player_hands[0]
        encoding = state_encoder.encode(state, 0)
        encodings.append((desc, encoding))
        
        print(f"\n{desc} 手牌编码:")
        print(f"  编码长度: {len(encoding)}")
        print(f"  非零元素数: {np.count_nonzero(encoding)}")
        print(f"  前20个元素: {encoding[:20]}")
        print(f"  编码和: {encoding.sum():.4f}")
    
    # 比较编码差异
    print("\n" + "=" * 70)
    print("编码差异分析")
    print("=" * 70)
    
    for i in range(len(encodings)):
        for j in range(i + 1, len(encodings)):
            desc1, enc1 = encodings[i]
            desc2, enc2 = encodings[j]
            diff = np.abs(enc1 - enc2)
            diff_count = np.count_nonzero(diff > 0.001)
            diff_sum = diff.sum()
            
            print(f"\n{desc1} vs {desc2}:")
            print(f"  不同元素数: {diff_count}")
            print(f"  差异总和: {diff_sum:.4f}")
            
            if diff_count > 0:
                diff_indices = np.where(diff > 0.001)[0]
                print(f"  差异位置: {diff_indices[:10]}...")
            else:
                print("  警告：两个编码完全相同！")


if __name__ == '__main__':
    main()
