#!/usr/bin/env python3
"""
快速验证场景2的范围胜率 - 使用OMPEval的范围对范围功能
"""

import subprocess
import json
import os

OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'OMPEval', 'equity_calc')

# 公共牌
BOARD = "2h6h3hAd8c"
BOARD_CARDS = ['2h', '6h', '3h', 'Ad', '8c']

# OOP原始范围
OOP_RANGE_STR = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o'

# IP有效范围（169个组合）
IP_EFFECTIVE_RANGE_STR = 'AhKc,AhKd,AhKh,AhKs,AhQc,AhQd,AhQh,AhQs,AhJc,AhJd,AhJh,AhJs,AhTc,AhTd,AhTh,AhTs,Ah9h,Ah8h,Ah7h,Ah5h,Ah4h,KcKd,KcKh,KcKs,KdKh,KdKs,KhKs,KcQc,KcQd,KcQh,KcQs,KdQc,KdQd,KdQh,KdQs,KhQc,KhQd,KhQh,KhQs,KsQc,KsQd,KsQh,KsQs,KcJc,KcJd,KcJh,KcJs,KdJc,KdJd,KdJh,KdJs,KhJc,KhJd,KhJh,KhJs,KsJc,KsJd,KsJh,KsJs,KcTc,KdTd,KhTh,KsTs,Kc9c,Kd9d,Kh9h,Ks9s,QcQd,QcQh,QcQs,QdQh,QdQs,QhQs,QcJc,QcJd,QcJh,QcJs,QdJc,QdJd,QdJh,QdJs,QhJc,QhJd,QhJh,QhJs,QsJc,QsJd,QsJh,QsJs,QcTc,QdTd,QhTh,QsTs,Qc9c,Qd9d,Qh9h,Qs9s,JcJd,JcJh,JcJs,JdJh,JdJs,JhJs,JcTc,JdTd,JhTh,JsTs,Jc9c,Jd9d,Jh9h,Js9s,TcTd,TcTh,TcTs,TdTh,TdTs,ThTs,Tc9c,Td9d,Th9h,Ts9s,9c9d,9c9h,9c9s,9d9h,9d9s,9h9s,9d8d,9h8h,9s8s,8d8h,8d8s,8h8s,8d7d,8h7h,8s7s,7c7d,7c7h,7c7s,7d7h,7d7s,7h7s,7c6c,7d6d,7s6s,6c6d,6c6s,6d6s,6c5c,6d5d,6s5s,5c5d,5c5h,5c5s,5d5h,5d5s,5h5s,4c4d,4c4h,4c4s,4d4h,4d4s,4h4s,3c3d,3c3s,3d3s,2c2d,2c2s,2d2s'


def calculate_range_vs_range(range1, range2, board):
    """使用OMPEval计算范围对范围的胜率"""
    cmd = [OMPEVAL_PATH, range1, range2, board]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return None
    output = result.stdout.replace('nan', '0')
    return json.loads(output)


def main():
    print("=" * 80)
    print("场景2 范围胜率验证（快速版）")
    print("=" * 80)
    
    print("\n【输入参数】")
    print(f"  公共牌: {' '.join(BOARD_CARDS)}")
    print(f"  OOP范围: {OOP_RANGE_STR}")
    print(f"  IP有效范围: {len(IP_EFFECTIVE_RANGE_STR.split(','))}个组合")
    
    print("\n【计算方法】")
    print("  使用OMPEval的范围对范围功能，一次性计算")
    
    print("\n【开始计算】")
    result = calculate_range_vs_range(OOP_RANGE_STR, IP_EFFECTIVE_RANGE_STR, BOARD)
    
    if result:
        print(f"\n【OMPEval原始输出】")
        print(f"  {result}")
        
        wins = result['wins']
        tie_count = result['tieCount']
        total = sum(wins) + tie_count
        
        print(f"\n【结果】")
        print(f"  OOP胜: {wins[0]}")
        print(f"  IP胜: {wins[1]}")
        print(f"  平局: {tie_count}")
        print(f"  总数: {total}")
        
        if total > 0:
            win_rate = wins[0] / total * 100
            tie_rate = tie_count / total * 100
            lose_rate = wins[1] / total * 100
            equity = (wins[0] + tie_count / 2) / total * 100
            
            print(f"\n【胜率分解】")
            print(f"  OOP胜率: {win_rate:.3f}%")
            print(f"  平局率: {tie_rate:.3f}%")
            print(f"  IP胜率: {lose_rate:.3f}%")
            print(f"\n【综合胜率】")
            print(f"  OOP范围胜率 = {equity:.3f}%")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
