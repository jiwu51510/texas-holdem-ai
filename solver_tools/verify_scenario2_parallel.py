#!/usr/bin/env python3
"""
并行验证场景2的范围胜率 - 使用多进程加速
"""

import subprocess
import json
import os
from multiprocessing import Pool, cpu_count

OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'OMPEval', 'equity_calc')

# 公共牌
BOARD = "2h6h3hAd8c"
BOARD_CARDS = ['2h', '6h', '3h', 'Ad', '8c']

# IP有效范围（169个组合）
IP_EFFECTIVE_RANGE_STR = 'AhKc,AhKd,AhKh,AhKs,AhQc,AhQd,AhQh,AhQs,AhJc,AhJd,AhJh,AhJs,AhTc,AhTd,AhTh,AhTs,Ah9h,Ah8h,Ah7h,Ah5h,Ah4h,KcKd,KcKh,KcKs,KdKh,KdKs,KhKs,KcQc,KcQd,KcQh,KcQs,KdQc,KdQd,KdQh,KdQs,KhQc,KhQd,KhQh,KhQs,KsQc,KsQd,KsQh,KsQs,KcJc,KcJd,KcJh,KcJs,KdJc,KdJd,KdJh,KdJs,KhJc,KhJd,KhJh,KhJs,KsJc,KsJd,KsJh,KsJs,KcTc,KdTd,KhTh,KsTs,Kc9c,Kd9d,Kh9h,Ks9s,QcQd,QcQh,QcQs,QdQh,QdQs,QhQs,QcJc,QcJd,QcJh,QcJs,QdJc,QdJd,QdJh,QdJs,QhJc,QhJd,QhJh,QhJs,QsJc,QsJd,QsJh,QsJs,QcTc,QdTd,QhTh,QsTs,Qc9c,Qd9d,Qh9h,Qs9s,JcJd,JcJh,JcJs,JdJh,JdJs,JhJs,JcTc,JdTd,JhTh,JsTs,Jc9c,Jd9d,Jh9h,Js9s,TcTd,TcTh,TcTs,TdTh,TdTs,ThTs,Tc9c,Td9d,Th9h,Ts9s,9c9d,9c9h,9c9s,9d9h,9d9s,9h9s,9d8d,9h8h,9s8s,8d8h,8d8s,8h8s,8d7d,8h7h,8s7s,7c7d,7c7h,7c7s,7d7h,7d7s,7h7s,7c6c,7d6d,7s6s,6c6d,6c6s,6d6s,6c5c,6d5d,6s5s,5c5d,5c5h,5c5s,5d5h,5d5s,5h5s,4c4d,4c4h,4c4s,4d4h,4d4s,4h4s,3c3d,3c3s,3d3s,2c2d,2c2s,2d2s'


def expand_oop_range_original():
    """展开OOP原始范围"""
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


def calc_one_oop_hand(oop_hand):
    """计算单个OOP手牌对所有IP手牌的胜率"""
    board_set = set(BOARD_CARDS)
    ip_hands = IP_EFFECTIVE_RANGE_STR.split(',')
    
    oop_cards = set([oop_hand[:2], oop_hand[2:]])
    if oop_cards & board_set:
        return (0, 0, 0)
    
    wins = 0
    ties = 0
    combos = 0
    
    for ip_hand in ip_hands:
        ip_cards = set([ip_hand[:2], ip_hand[2:]])
        if oop_cards & ip_cards:
            continue
        
        cmd = [OMPEVAL_PATH, oop_hand, ip_hand, BOARD]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            continue
        
        output = result.stdout.replace('nan', '0')
        data = json.loads(output)
        wins += data['wins'][0]
        ties += data['tieCount']
        combos += 1
    
    return (wins, ties, combos)


def main():
    print("=" * 80)
    print("场景2 范围胜率验证（并行版）")
    print("=" * 80)
    
    print("\n【输入参数】")
    print(f"  公共牌: {' '.join(BOARD_CARDS)}")
    print(f"  CPU核心数: {cpu_count()}")
    
    oop_hands = expand_oop_range_original()
    print(f"  OOP原始范围组合数: {len(oop_hands)}")
    print(f"  IP有效范围组合数: {len(IP_EFFECTIVE_RANGE_STR.split(','))}")
    
    print("\n【开始并行计算】")
    
    with Pool(cpu_count()) as pool:
        results = pool.map(calc_one_oop_hand, oop_hands)
    
    total_wins = sum(r[0] for r in results)
    total_ties = sum(r[1] for r in results)
    total_combos = sum(r[2] for r in results)
    
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
        print(f"  OOP范围胜率 = {equity:.3f}%")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
