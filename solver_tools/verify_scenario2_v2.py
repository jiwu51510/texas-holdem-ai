#!/usr/bin/env python3
"""
清晰验证场景2的范围胜率计算

场景2参数：
- 公共牌: 2h 6h 3h Ad 8c
- OOP手牌: AcAs (固定手牌)
- IP有效范围: 169个组合（IP原始范围 - 公共牌 - 固定OOP手牌）

范围胜率定义：
- OOP原始范围 vs IP有效范围
- OOP范围不变，只有IP范围需要排除死牌
- 计算所有OOP手牌 vs IP手牌组合对的整体胜率
"""

import subprocess
import json
import os

OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'OMPEval', 'equity_calc')

# 公共牌
BOARD = "2h6h3hAd8c"
BOARD_CARDS = ['2h', '6h', '3h', 'Ad', '8c']

# 固定OOP手牌
FIXED_OOP_HAND = ['Ac', 'As']

# OOP原始范围定义
OOP_RANGE_STR = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o'

# IP有效范围（IP原始范围 - 公共牌 - 固定OOP手牌，169个组合）
IP_EFFECTIVE_RANGE_STR = 'AhKc,AhKd,AhKh,AhKs,AhQc,AhQd,AhQh,AhQs,AhJc,AhJd,AhJh,AhJs,AhTc,AhTd,AhTh,AhTs,Ah9h,Ah8h,Ah7h,Ah5h,Ah4h,KcKd,KcKh,KcKs,KdKh,KdKs,KhKs,KcQc,KcQd,KcQh,KcQs,KdQc,KdQd,KdQh,KdQs,KhQc,KhQd,KhQh,KhQs,KsQc,KsQd,KsQh,KsQs,KcJc,KcJd,KcJh,KcJs,KdJc,KdJd,KdJh,KdJs,KhJc,KhJd,KhJh,KhJs,KsJc,KsJd,KsJh,KsJs,KcTc,KdTd,KhTh,KsTs,Kc9c,Kd9d,Kh9h,Ks9s,QcQd,QcQh,QcQs,QdQh,QdQs,QhQs,QcJc,QcJd,QcJh,QcJs,QdJc,QdJd,QdJh,QdJs,QhJc,QhJd,QhJh,QhJs,QsJc,QsJd,QsJh,QsJs,QcTc,QdTd,QhTh,QsTs,Qc9c,Qd9d,Qh9h,Qs9s,JcJd,JcJh,JcJs,JdJh,JdJs,JhJs,JcTc,JdTd,JhTh,JsTs,Jc9c,Jd9d,Jh9h,Js9s,TcTd,TcTh,TcTs,TdTh,TdTs,ThTs,Tc9c,Td9d,Th9h,Ts9s,9c9d,9c9h,9c9s,9d9h,9d9s,9h9s,9d8d,9h8h,9s8s,8d8h,8d8s,8h8s,8d7d,8h7h,8s7s,7c7d,7c7h,7c7s,7d7h,7d7s,7h7s,7c6c,7d6d,7s6s,6c6d,6c6s,6d6s,6c5c,6d5d,6s5s,5c5d,5c5h,5c5s,5d5h,5d5s,5h5s,4c4d,4c4h,4c4s,4d4h,4d4s,4h4s,3c3d,3c3s,3d3s,2c2d,2c2s,2d2s'


def calculate_equity(hand1, hand2, board):
    """使用OMPEval计算单个手牌对的胜率"""
    cmd = [OMPEVAL_PATH, hand1, hand2, board]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    output = result.stdout.replace('nan', '0')
    return json.loads(output)


def expand_oop_range_original():
    """展开OOP原始范围（不排除任何死牌）"""
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    hands = []
    
    # 对子 AA-22
    for r in range(13):
        rank = ranks[r]
        for s1 in range(4):
            for s2 in range(s1+1, 4):
                hands.append(rank + suits[s1] + rank + suits[s2])
    
    # 同花
    suited_ranges = [(12,0,11),(11,0,10),(10,0,9),(9,4,8),(8,4,7),(7,4,6),(6,4,5),(5,3,4),(4,2,3),(3,2,2)]
    for high, low_start, low_end in suited_ranges:
        for low in range(low_start, low_end + 1):
            if low >= high: continue
            for s in range(4):
                hands.append(ranks[high] + suits[s] + ranks[low] + suits[s])
    
    # 非同花
    offsuit_ranges = [(12,0,11),(11,7,10),(10,7,9),(9,7,8),(8,7,7)]
    for high, low_start, low_end in offsuit_ranges:
        for low in range(low_start, low_end + 1):
            if low >= high: continue
            for s1 in range(4):
                for s2 in range(4):
                    if s1 != s2:
                        hands.append(ranks[high] + suits[s1] + ranks[low] + suits[s2])
    return hands


def main():
    print("=" * 80)
    print("场景2 范围胜率验证")
    print("=" * 80)
    
    print("\n【输入参数】")
    print(f"  公共牌: {' '.join(BOARD_CARDS)}")
    print(f"  固定OOP手牌: {FIXED_OOP_HAND[0]} {FIXED_OOP_HAND[1]}")
    print(f"  OOP范围: {OOP_RANGE_STR}")
    
    oop_hands = expand_oop_range_original()
    ip_hands = IP_EFFECTIVE_RANGE_STR.split(',')
    
    print(f"\n【OOP原始范围】组合数: {len(oop_hands)}")
    print(f"【IP有效范围】组合数: {len(ip_hands)}")
    
    print("\n【计算方法】")
    print("  OOP原始范围 vs IP有效范围")
    print("  范围胜率 = (OOP胜 + 平局/2) / 有效组合对数")
    
    print("\n【开始计算】")
    
    total_wins = 0
    total_ties = 0
    total_combos = 0
    board_set = set(BOARD_CARDS)
    
    for i, oop_hand in enumerate(oop_hands):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(oop_hands)}")
        
        oop_cards = set([oop_hand[:2], oop_hand[2:]])
        if oop_cards & board_set: continue
        
        for ip_hand in ip_hands:
            ip_cards = set([ip_hand[:2], ip_hand[2:]])
            if oop_cards & ip_cards: continue
            
            result = calculate_equity(oop_hand, ip_hand, BOARD)
            if result is None: continue
            
            total_wins += result['wins'][0]
            total_ties += result['tieCount']
            total_combos += 1
    
    print(f"\n【结果】")
    print(f"  有效组合对数: {total_combos}")
    print(f"  OOP胜: {total_wins}")
    print(f"  平局: {total_ties}")
    print(f"  IP胜: {total_combos - total_wins - total_ties}")
    
    if total_combos > 0:
        win_rate = total_wins / total_combos * 100
        tie_rate = total_ties / total_combos * 100
        lose_rate = (total_combos - total_wins - total_ties) / total_combos * 100
        equity = (total_wins + total_ties / 2) / total_combos * 100
        
        print(f"\n【胜率分解】")
        print(f"  OOP胜率: {win_rate:.3f}%")
        print(f"  平局率: {tie_rate:.3f}%")
        print(f"  IP胜率: {lose_rate:.3f}%")
        print(f"\n【综合胜率】")
        print(f"  OOP范围胜率 = (胜 + 平/2) / 总数 = ({total_wins} + {total_ties}/2) / {total_combos}")
        print(f"  OOP范围胜率 = {equity:.3f}%")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
