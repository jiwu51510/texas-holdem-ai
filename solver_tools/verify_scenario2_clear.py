#!/usr/bin/env python3
"""
清晰验证场景2的范围胜率计算

场景2参数：
- 公共牌: 2h 6h 3h Ad 8c
- OOP手牌: AcAs (固定手牌)
- IP有效范围: 169个组合（IP原始范围 牌）

范围胜率定义：
- OOP原始范围 vs IP有效范围
- OOP范围不变，只有IP范围需要排除死牌
- 计算所有OOP手牌 vs IP手牌组合对的整体胜率
"""

import subprocess
import json
import os

OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'OMPEval', 'equity_calc')

# ============================================================
# 场景2参数
# ============================================================

# 公共牌
BOARD = "2h6h3hAd8c"
BOARD_CARDS = ['2h', '6h', '3h', 'Ad', '8c']

# 固定OOP手牌
FIXED_OOP_HAND = ['Ac', 'As']

# OOP原始范围定义
OOP_RANGE_STR = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s'

# IP有效范围（IP原始范围 - 
IP_EFFECTIVE_RANGE_STR = 'AhKc,AhKd,AhKh,AhKs,AhQc,AhQd,AhQh,AhQs,AhJc,AhJd,AhJh,AhJs,AhTc,AhTd,AhTh,AhTs,Ah9h,Ah8h,Ah7h,Ah5h,Ah4h,KcKd


def calculate_equity(hand1, hand2,):
    """使用OMPEval计算单个手牌对的胜率"""
rd]

    
    if result.returncode != 0:
        return None
    
    0')
    return json.loads(output)


def expand_oop_range_original():
    """
除任何死牌）
 
    OOP范围: AA-22,AKs-A2s,KQs-K2s,T9o
    """
    ranks = '2345
    hs'
    hands = []
    
    # 对子 AA-22
    for r in range(13):
        rank =ks[r]
    :
            fo):
                card1 =
                card2 =uits[s2]
                hands.appen2)
    
    # 同花 AKs-A2s, KQs-K2s, QJs-Q2s, JTs-
    suited_ranges = [
        (12, 0, 11),   # AKs-A2s
        (11, 0, 10),   # KQs-K2s
     QJs-Q2s
        (9, 4, 8),     # JTs-J6s
        (8, 4, 7),   6s
        (7, 4, 6),     # 98s-96s
        (6, 4, 5),     # 87s-86s
        (5, 3, 4),     # 76s-75s
        (4, 2, 3),     # 65s-64s
        (3, 2, 2),     # 54s
    ]
    
    for high, low_start, low_end
        for low in range(low_sta1):
            if low >= high:
     inue
    
                card1 = ranks[high] + suits[s]
                card2 = ranks[low] + suits[s]
                hands.appen
    
    # 非同花 AKo-A2o, KQo-K9o, QJT9o
    offsuit_ranges = [
        (12, 0, 11),   # AKo-A2o
        (11, 7, 10),   # KQo-K9o
        (10, 7, 9),    # QJo-Q9o
     JTo-J9o
        (8, 7, 7),     # T9o
    ]
    
    for high, low_start, low_end
        for low in range(low_sta
            if low >= high:
                continue
     ange(4):
    ange(4):
                    if s1 != s2:
                        card1 = ranks[high] + suits[s1]
                        car
                         + card2)
    
    return hands


def main():
    print("=" * 80)
    print("场景2 范围胜率验证")
    )
    
【输入参数】")
RDS)}")
    print(f")
    print(f"  OOP范围")
    print(f"  IP原始范围: Ao")
    
    
    oop_hands = expanriginal()
    print(f"\n【OOP原始范围】")
    print(f"  组合数: {len(oop_hands)}")
    
    # IP有效范围（已排除公共牌和固定OOP手牌）
    ,')
    print(f"\n【IP有效)
    print(f"  组合数: {len(ip_hands)}")
    
    print("\n【计算方法】")
    print("  范围胜率 = 所有有效(OOP手牌 vs IP手牌率")
    ")
    print(" )
    
    print("\n【开始计算】")
    
    total_wins = 0
    0
    total_combos = 0
    skipped_board_conflict = 0
    skipped_hand_conflict = 0
    
    board_set = set(BDS)
    
    ):
        if (i + 1)
            print(
        
        oop_card1 = oop_h
    [2:]
        oop_cards = set([oop_card1, oop_card2])
        
        # 检查OOP手牌是否与公共牌冲突
        
            skipped_board_conflict += 1
        tinue
        
        for ip_hand in ip_hands:
            _hand[:2]
            ip_card2 = ip_hand[2:]
            ip_cards = set([ip_card1_card2])
            
            # 检查OOP手牌和IP
            s:
                sk1
                continue
            
            # 计算胜率
            
            if result is None:
                continue
            
    0]
            total_tie']
            total_combos += 1
    
    print("\n【计算结果】")
    print(f"  OOP原始范围组合数: {len(o")
    print(f"  IP有效范围组合数: {len(ip_hands)}")
    )
    print(f"  跳过的OOP与IP手")
    print(f"  有效组合对数: {total_combos}")
    print(f"  OOP胜: {total}")
    print(f"  平局: {total_ties}")
    print(f"  IP胜: {total_combos - total_wins -es}")
    
    if t:
        oop_equity = (total_wins + tota
        print(f"\n【范围胜率】")
        print(f"  计算公式: (OOP胜 + 平局/2) / 有效组合对数")
        print")
        print(f"  = {oop_equity:.3f}%")
    ")
        

1:
            print(f"  ✓ 结果")
        else:
n()
ai min__':
   e__ == '__ma__nam
if  * 80)

 "="n" +rint("\   
    p")
  {diff:.3f}%果不一致！差异:f"  ✗ 结print(            