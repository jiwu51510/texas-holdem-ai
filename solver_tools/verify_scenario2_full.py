#!/usr/bin/env python3
"""
使用OMPEval完整枚举验证场景2的范围胜率
公共牌: 2h 6h 3h Ad 8c
OOP手牌: AcAs (作为死牌)
报告中的范围胜率: 40.922%

关键：需要展开OOP范围（排除死牌后），然后计算所有组合对的胜率
"""

import subprocess
import json
import os

OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'OMPEval', 'equity_calc')

# 公共牌
BOARD = "2h6h3hAd8c"
BOARD_CARDS = ['2h', '6h', '3h', 'Ad', '8c']

# 固定OOP手牌（作为死牌）
FIXED_OOP_HAND = ['Ac', 'As']

# 死牌 = 公共牌 + 固定OOP手牌
DEAD_CARDS = set(BOARD_CARDS + FIXED_OOP_HAND)

# IP有效范围（从报告中复制，169个组合）
IP_RANGE_STR = 'AhKc,AhKd,AhKh,AhKs,AhQc,AhQd,AhQh,AhQs,AhJc,AhJd,AhJh,AhJs,AhTc,AhTd,AhTh,AhTs,Ah9h,Ah8h,Ah7h,Ah5h,Ah4h,KcKd,KcKh,KcKs,KdKh,KdKs,KhKs,KcQc,KcQd,KcQh,KcQs,KdQc,KdQd,KdQh,KdQs,KhQc,KhQd,KhQh,KhQs,KsQc,KsQd,KsQh,KsQs,KcJc,KcJd,KcJh,KcJs,KdJc,KdJd,KdJh,KdJs,KhJc,KhJd,KhJh,KhJs,KsJc,KsJd,KsJh,KsJs,KcTc,KdTd,KhTh,KsTs,Kc9c,Kd9d,Kh9h,Ks9s,QcQd,QcQh,QcQs,QdQh,QdQs,QhQs,QcJc,QcJd,QcJh,QcJs,QdJc,QdJd,QdJh,QdJs,QhJc,QhJd,QhJh,QhJs,QsJc,QsJd,QsJh,QsJs,QcTc,QdTd,QhTh,QsTs,Qc9c,Qd9d,Qh9h,Qs9s,JcJd,JcJh,JcJs,JdJh,JdJs,JhJs,JcTc,JdTd,JhTh,JsTs,Jc9c,Jd9d,Jh9h,Js9s,TcTd,TcTh,TcTs,TdTh,TdTs,ThTs,Tc9c,Td9d,Th9h,Ts9s,9c9d,9c9h,9c9s,9d9h,9d9s,9h9s,9d8d,9h8h,9s8s,8d8h,8d8s,8h8s,8d7d,8h7h,8s7s,7c7d,7c7h,7c7s,7d7h,7d7s,7h7s,7c6c,7d6d,7s6s,6c6d,6c6s,6d6s,6c5c,6d5d,6s5s,5c5d,5c5h,5c5s,5d5h,5d5s,5h5s,4c4d,4c4h,4c4s,4d4h,4d4s,4h4s,3c3d,3c3s,3d3s,2c2d,2c2s,2d2s'


def calculate_equity(hand1, hand2, board):
    """使用OMPEval计算单个手牌对的胜率"""
    cmd = [OMPEVAL_PATH, hand1, hand2, board]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return None
    
    output = result.stdout.replace('nan', '0')
    return json.loads(output)


def expand_oop_range(dead_cards):
    """展开OOP范围（排除死牌）"""
    # OOP范围: AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o
    
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    hands = []
    
    # 对子 AA-22
    for r in range(13):
        rank = ranks[r]
        for s1 in range(4):
            for s2 in range(s1+1, 4):
                card1 = rank + suits[s1]
                card2 = rank + suits[s2]
                if card1 not in dead_cards and card2 not in dead_cards:
                    hands.append(card1 + card2)
    
    # 同花 AKs-A2s, KQs-K2s, QJs-Q2s, JTs-J6s, T9s-T6s, 98s-96s, 87s-86s, 76s-75s, 65s-64s, 54s
    suited_ranges = [
        (12, 0, 11),   # AKs-A2s: A with K down to 2
        (11, 0, 10),   # KQs-K2s
        (10, 0, 9),    # QJs-Q2s
        (9, 4, 8),     # JTs-J6s
        (8, 4, 7),     # T9s-T6s
        (7, 4, 6),     # 98s-96s
        (6, 4, 5),     # 87s-86s
        (5, 3, 4),     # 76s-75s
        (4, 2, 3),     # 65s-64s
        (3, 2, 2),     # 54s
    ]
    
    for high, low_start, low_end in suited_ranges:
        for low in range(low_start, low_end + 1):
            if low >= high:
                continue
            for s in range(4):
                card1 = ranks[high] + suits[s]
                card2 = ranks[low] + suits[s]
                if card1 not in dead_cards and card2 not in dead_cards:
                    hands.append(card1 + card2)
    
    # 非同花 AKo-A2o, KQo-K9o, QJo-Q9o, JTo-J9o, T9o
    offsuit_ranges = [
        (12, 0, 11),   # AKo-A2o
        (11, 7, 10),   # KQo-K9o
        (10, 7, 9),    # QJo-Q9o
        (9, 7, 8),     # JTo-J9o
        (8, 7, 7),     # T9o
    ]
    
    for high, low_start, low_end in offsuit_ranges:
        for low in range(low_start, low_end + 1):
            if low >= high:
                continue
            for s1 in range(4):
                for s2 in range(4):
                    if s1 != s2:
                        card1 = ranks[high] + suits[s1]
                        card2 = ranks[low] + suits[s2]
                        if card1 not in dead_cards and card2 not in dead_cards:
                            hands.append(card1 + card2)
    
    return hands


def main():
    print("=" * 70)
    print("使用OMPEval完整枚举验证场景2的范围胜率")
    print("=" * 70)
    print(f"\n公共牌: {BOARD}")
    print(f"固定OOP手牌: {''.join(FIXED_OOP_HAND)} (作为死牌)")
    print(f"死牌: {DEAD_CARDS}")
    print("报告中的范围胜率: 40.922%")
    
    # 展开OOP范围（排除死牌）
    oop_hands = expand_oop_range(DEAD_CARDS)
    print(f"\nOOP范围展开后组合数: {len(oop_hands)}")
    
    # IP有效范围
    ip_hands = IP_RANGE_STR.split(',')
    print(f"IP有效范围组合数: {len(ip_hands)}")
    
    # 计算所有组合对的胜率
    print("\n计算所有组合对的胜率（这可能需要一些时间）...")
    
    total_wins = 0
    total_ties = 0
    total_combos = 0
    
    for i, oop_hand in enumerate(oop_hands):
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(oop_hands)}")
        
        oop_cards = set([oop_hand[:2], oop_hand[2:]])
        
        for ip_hand in ip_hands:
            ip_cards = set([ip_hand[:2], ip_hand[2:]])
            
            # 检查是否有冲突
            if oop_cards & ip_cards:
                continue
            
            # 计算胜率
            result = calculate_equity(oop_hand, ip_hand, BOARD)
            if result is None:
                continue
            
            total_wins += result['wins'][0]
            total_ties += result['tieCount']
            total_combos += 1
    
    print(f"\n总组合对数: {total_combos}")
    print(f"OOP胜: {total_wins}")
    print(f"平局: {total_ties}")
    print(f"IP胜: {total_combos - total_wins - total_ties}")
    
    if total_combos > 0:
        oop_equity = (total_wins + total_ties / 2) / total_combos * 100
        print(f"\nOOP范围胜率: {oop_equity:.3f}%")
        print(f"报告中的范围胜率: 40.922%")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
