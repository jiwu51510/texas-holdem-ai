#!/usr/bin/env python3
"""
验证新报告中反例1的范围胜率计算
场景1: 公共牌 5h Td Jc 3h 4d, OOP手牌 AdAh
场景2: 公共牌 2h 6h 3h Ad 8c, OOP手牌 AcAs
"""

import subprocess
import json
import os

OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'OMPEval', 'equity_calc')

# OOP范围定义
OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o'

# IP范围定义
IP_RANGE = 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo'


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
    print("验证新报告中反例1的范围胜率计算")
    print("=" * 70)
    
    # 场景1: 公共牌 5h Td Jc 3h 4d, OOP手牌 AdAh
    print("\n【场景1】")
    print("公共牌: 5h Td Jc 3h 4d")
    print("OOP手牌: AdAh")
    print("报告中的范围胜率: 41.669%")
    
    # IP有效范围（从报告中复制，179个组合）
    ip_range_1 = 'AcAs,AcKc,AcKd,AcKh,AcKs,AsKc,AsKd,AsKh,AsKs,AcQc,AcQd,AcQh,AcQs,AsQc,AsQd,AsQh,AsQs,AcJd,AcJh,AcJs,AsJd,AsJh,AsJs,AcTc,AcTh,AcTs,AsTc,AsTh,AsTs,Ac9c,As9s,Ac8c,As8s,Ac7c,As7s,Ac6c,As6s,Ac5c,As5s,Ac4c,As4s,Ac3c,As3s,Ac2c,As2s,KcKd,KcKh,KcKs,KdKh,KdKs,KhKs,KcQc,KcQd,KcQh,KcQs,KdQc,KdQd,KdQh,KdQs,KhQc,KhQd,KhQh,KhQs,KsQc,KsQd,KsQh,KsQs,KcJd,KcJh,KcJs,KdJd,KdJh,KdJs,KhJd,KhJh,KhJs,KsJd,KsJh,KsJs,KcTc,KhTh,KsTs,Kc9c,Kd9d,Kh9h,Ks9s,QcQd,QcQh,QcQs,QdQh,QdQs,QhQs,QcJd,QcJh,QcJs,QdJd,QdJh,QdJs,QhJd,QhJh,QhJs,QsJd,QsJh,QsJs,QcTc,QhTh,QsTs,Qc9c,Qd9d,Qh9h,Qs9s,JdJh,JdJs,JhJs,JhTh,JsTs,Jd9d,Jh9h,Js9s,TcTh,TcTs,ThTs,Tc9c,Th9h,Ts9s,9c9d,9c9h,9c9s,9d9h,9d9s,9h9s,9c8c,9d8d,9h8h,9s8s,8c8d,8c8h,8c8s,8d8h,8d8s,8h8s,8c7c,8d7d,8h7h,8s7s,7c7d,7c7h,7c7s,7d7h,7d7s,7h7s,7c6c,7d6d,7h6h,7s6s,6c6d,6c6h,6c6s,6d6h,6d6s,6h6s,6c5c,6d5d,6s5s,5c5d,5c5s,5d5s,4c4h,4c4s,4h4s,3c3d,3c3s,3d3s,2c2d,2c2h,2c2s,2d2h,2d2s,2h2s'
    
    board_1 = '5hTdJc3h4d'
    
    # 计算OOP范围（排除死牌：公共牌 + OOP手牌）
    # 死牌: 5h, Td, Jc, 3h, 4d, Ad, Ah
    # 需要从OOP_RANGE中排除这些牌
    
    print(f"\nIP有效范围组合数: {len(ip_range_1.split(','))}")
    
    # 使用OMPEval计算
    try:
        result = calculate_equity(OOP_RANGE, ip_range_1, board_1)
        print(f"\nOMPEval计算结果:")
        print(f"  OOP Equity: {result['equity'][0] * 100:.3f}%")
        print(f"  总手数: {result['hands']}")
        
        # 计算纯胜率
        total = result['wins'][0] + result['wins'][1] + result['tieCount']
        if total > 0:
            equity = (result['wins'][0] + result['tieCount'] / 2) / total * 100
            print(f"  OOP Equity (胜+平/2): {equity:.3f}%")
    except Exception as e:
        print(f"计算失败: {e}")
    
    # 场景2: 公共牌 2h 6h 3h Ad 8c, OOP手牌 AcAs
    print("\n" + "=" * 70)
    print("\n【场景2】")
    print("公共牌: 2h 6h 3h Ad 8c")
    print("OOP手牌: AcAs")
    print("报告中的范围胜率: 40.922%")
    
    # IP有效范围（从报告中复制，169个组合）
    ip_range_2 = 'AhKc,AhKd,AhKh,AhKs,AhQc,AhQd,AhQh,AhQs,AhJc,AhJd,AhJh,AhJs,AhTc,AhTd,AhTh,AhTs,Ah9h,Ah8h,Ah7h,Ah5h,Ah4h,KcKd,KcKh,KcKs,KdKh,KdKs,KhKs,KcQc,KcQd,KcQh,KcQs,KdQc,KdQd,KdQh,KdQs,KhQc,KhQd,KhQh,KhQs,KsQc,KsQd,KsQh,KsQs,KcJc,KcJd,KcJh,KcJs,KdJc,KdJd,KdJh,KdJs,KhJc,KhJd,KhJh,KhJs,KsJc,KsJd,KsJh,KsJs,KcTc,KdTd,KhTh,KsTs,Kc9c,Kd9d,Kh9h,Ks9s,QcQd,QcQh,QcQs,QdQh,QdQs,QhQs,QcJc,QcJd,QcJh,QcJs,QdJc,QdJd,QdJh,QdJs,QhJc,QhJd,QhJh,QhJs,QsJc,QsJd,QsJh,QsJs,QcTc,QdTd,QhTh,QsTs,Qc9c,Qd9d,Qh9h,Qs9s,JcJd,JcJh,JcJs,JdJh,JdJs,JhJs,JcTc,JdTd,JhTh,JsTs,Jc9c,Jd9d,Jh9h,Js9s,TcTd,TcTh,TcTs,TdTh,TdTs,ThTs,Tc9c,Td9d,Th9h,Ts9s,9c9d,9c9h,9c9s,9d9h,9d9s,9h9s,9d8d,9h8h,9s8s,8d8h,8d8s,8h8s,8d7d,8h7h,8s7s,7c7d,7c7h,7c7s,7d7h,7d7s,7h7s,7c6c,7d6d,7s6s,6c6d,6c6s,6d6s,6c5c,6d5d,6s5s,5c5d,5c5h,5c5s,5d5h,5d5s,5h5s,4c4d,4c4h,4c4s,4d4h,4d4s,4h4s,3c3d,3c3s,3d3s,2c2d,2c2s,2d2s'
    
    board_2 = '2h6h3hAd8c'
    
    print(f"\nIP有效范围组合数: {len(ip_range_2.split(','))}")
    
    try:
        result = calculate_equity(OOP_RANGE, ip_range_2, board_2)
        print(f"\nOMPEval计算结果:")
        print(f"  OOP Equity: {result['equity'][0] * 100:.3f}%")
        print(f"  总手数: {result['hands']}")
        
        total = result['wins'][0] + result['wins'][1] + result['tieCount']
        if total > 0:
            equity = (result['wins'][0] + result['tieCount'] / 2) / total * 100
            print(f"  OOP Equity (胜+平/2): {equity:.3f}%")
    except Exception as e:
        print(f"计算失败: {e}")


if __name__ == '__main__':
    main()
