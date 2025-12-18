#!/usr/bin/env python3
"""
使用OMPEval验证场景2的范围胜率
公共牌: 2h 6h 3h Ad 8c
OOP手牌: AcAs
报告中的范围胜率: 40.922%
"""

import subprocess
import json
import os

OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'OMPEval', 'equity_calc')

# 公共牌
BOARD = "2h6h3hAd8c"

# OOP范围定义
OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o'

# IP有效范围（从报告中复制，169个组合）
IP_RANGE = 'AhKc,AhKd,AhKh,AhKs,AhQc,AhQd,AhQh,AhQs,AhJc,AhJd,AhJh,AhJs,AhTc,AhTd,AhTh,AhTs,Ah9h,Ah8h,Ah7h,Ah5h,Ah4h,KcKd,KcKh,KcKs,KdKh,KdKs,KhKs,KcQc,KcQd,KcQh,KcQs,KdQc,KdQd,KdQh,KdQs,KhQc,KhQd,KhQh,KhQs,KsQc,KsQd,KsQh,KsQs,KcJc,KcJd,KcJh,KcJs,KdJc,KdJd,KdJh,KdJs,KhJc,KhJd,KhJh,KhJs,KsJc,KsJd,KsJh,KsJs,KcTc,KdTd,KhTh,KsTs,Kc9c,Kd9d,Kh9h,Ks9s,QcQd,QcQh,QcQs,QdQh,QdQs,QhQs,QcJc,QcJd,QcJh,QcJs,QdJc,QdJd,QdJh,QdJs,QhJc,QhJd,QhJh,QhJs,QsJc,QsJd,QsJh,QsJs,QcTc,QdTd,QhTh,QsTs,Qc9c,Qd9d,Qh9h,Qs9s,JcJd,JcJh,JcJs,JdJh,JdJs,JhJs,JcTc,JdTd,JhTh,JsTs,Jc9c,Jd9d,Jh9h,Js9s,TcTd,TcTh,TcTs,TdTh,TdTs,ThTs,Tc9c,Td9d,Th9h,Ts9s,9c9d,9c9h,9c9s,9d9h,9d9s,9h9s,9d8d,9h8h,9s8s,8d8h,8d8s,8h8s,8d7d,8h7h,8s7s,7c7d,7c7h,7c7s,7d7h,7d7s,7h7s,7c6c,7d6d,7s6s,6c6d,6c6s,6d6s,6c5c,6d5d,6s5s,5c5d,5c5h,5c5s,5d5h,5d5s,5h5s,4c4d,4c4h,4c4s,4d4h,4d4s,4h4s,3c3d,3c3s,3d3s,2c2d,2c2s,2d2s'


def calculate_equity(range1, range2, board=''):
    """使用OMPEval计算范围vs范围胜率"""
    cmd = [OMPEVAL_PATH, range1, range2]
    if board:
        cmd.append(board)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"OMPEval error: {result.stderr}")
    
    output = result.stdout.replace('nan', '0')
    return json.loads(output)


def main():
    print("=" * 70)
    print("使用OMPEval验证场景2的范围胜率")
    print("=" * 70)
    print(f"\n公共牌: {BOARD}")
    print("OOP手牌: AcAs (作为死牌)")
    print("报告中的范围胜率: 40.922%")
    
    print(f"\nIP有效范围组合数: {len(IP_RANGE.split(','))}")
    
    # 使用OMPEval计算 OOP范围 vs IP有效范围
    print("\n计算 OOP范围 vs IP有效范围...")
    try:
        result = calculate_equity(OOP_RANGE, IP_RANGE, BOARD)
        print(f"\nOMPEval计算结果:")
        print(f"  OOP Equity: {result['equity'][0] * 100:.3f}%")
        print(f"  IP Equity: {result['equity'][1] * 100:.3f}%")
        print(f"  总手数: {result['hands']}")
        print(f"  OOP胜: {result['wins'][0]}")
        print(f"  IP胜: {result['wins'][1]}")
        print(f"  平局: {result['tieCount']}")
        
        # 手动计算equity
        total = result['wins'][0] + result['wins'][1] + result['tieCount']
        if total > 0:
            oop_equity = (result['wins'][0] + result['tieCount'] / 2) / total * 100
            print(f"\n  手动计算OOP Equity: {oop_equity:.3f}%")
    except Exception as e:
        print(f"计算失败: {e}")
    
    print("\n" + "=" * 70)
    print("你的计算结果是多少？请告诉我以便对比。")
    print("=" * 70)


if __name__ == '__main__':
    main()
