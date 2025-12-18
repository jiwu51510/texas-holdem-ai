#!/usr/bin/env python3
"""
使用报告中的精确范围验证Case 2的范围胜率
"""

import subprocess
import json
import os

OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'OMPEval', 'equity_calc')

# Case 2 公共牌: 2h 8d 2c Jc Ts
BOARD = "2h8d2cJcTs"

# 报告中的OOP范围（468个组合）- 从simplify_range.py生成
OOP_RANGE = '2d2s,3h3d,3h3s,3h3c,3d3s,3d3c,3s3c,4h4d,4h4s,4h4c,4d4s,4d4c,4s4c,5h5d,5h5s,5h5c,5d5s,5d5c,5s5c,6h6d,6h6s,6h6c,6d6s,6d6c,6s6c,7h7d,7h7s,7h7c,7d7s,7d7c,7s7c,8h8s,8h8c,8s8c,9h9d,9h9s,9h9c,9d9s,9d9c,9s9c,ThTd,ThTc,TdTc,JhJd,JhJs,JdJs,QhQd,QhQs,QhQc,QdQs,QdQc,QsQc,KhKd,KhKs,KhKc,KdKs,KdKc,KsKc,AhAd,AhAs,AhAc,AdAs,AdAc,AsAc,Ad2d,As2s,Ah3h,Ad3d,As3s,Ac3c,Ah4h,Ad4d,As4s,Ac4c,Ah5h,Ad5d,As5s,Ac5c,Ah6h,Ad6d,As6s,Ac6c,Ah7h,Ad7d,As7s,Ac7c,Ah8h,As8s,Ac8c,Ah9h,Ad9d,As9s,Ac9c,AhTh,AdTd,AcTc,AhJh,AdJd,AsJs,AhQh,AdQd,AsQs,AcQc,AhKh,AdKd,AsKs,AcKc,Kd2d,Ks2s,Kh3h,Kd3d,Ks3s,Kc3c,Kh4h,Kd4d,Ks4s,Kc4c,Kh5h,Kd5d,Ks5s,Kc5c,Kh6h,Kd6d,Ks6s,Kc6c,Kh7h,Kd7d,Ks7s,Kc7c,Kh8h,Ks8s,Kc8c,Kh9h,Kd9d,Ks9s,Kc9c,KhTh,KdTd,KcTc,KhJh,KdJd,KsJs,KhQh,KdQd,KsQs,KcQc,Qd2d,Qs2s,Qh3h,Qd3d,Qs3s,Qc3c,Qh4h,Qd4d,Qs4s,Qc4c,Qh5h,Qd5d,Qs5s,Qc5c,Qh6h,Qd6d,Qs6s,Qc6c,Qh7h,Qd7d,Qs7s,Qc7c,Qh8h,Qs8s,Qc8c,Qh9h,Qd9d,Qs9s,Qc9c,QhTh,QdTd,QcTc,QhJh,QdJd,QsJs,Jh6h,Jd6d,Js6s,Jh7h,Jd7d,Js7s,Jh8h,Js8s,Jh9h,Jd9d,Js9s,JhTh,JdTd,Th6h,Td6d,Tc6c,Th7h,Td7d,Tc7c,Th8h,Tc8c,Th9h,Td9d,Tc9c,9h6h,9d6d,9s6s,9c6c,9h7h,9d7d,9s7s,9c7c,9h8h,9s8s,9c8c,8h6h,8s6s,8c6c,8h7h,8s7s,8c7c,7h5h,7d5d,7s5s,7c5c,7h6h,7d6d,7s6s,7c6c,6h4h,6d4d,6s4s,6c4c,6h5h,6d5d,6s5s,6c5c,5h4h,5d4d,5s4s,5c4c,Ah2d,Ah2s,Ad2s,As2d,Ac2d,Ac2s,Ah3d,Ah3s,Ah3c,Ad3h,Ad3s,Ad3c,As3h,As3d,As3c,Ac3h,Ac3d,Ac3s,Ah4d,Ah4s,Ah4c,Ad4h,Ad4s,Ad4c,As4h,As4d,As4c,Ac4h,Ac4d,Ac4s,Ah5d,Ah5s,Ah5c,Ad5h,Ad5s,Ad5c,As5h,As5d,As5c,Ac5h,Ac5d,Ac5s,Ah6d,Ah6s,Ah6c,Ad6h,Ad6s,Ad6c,As6h,As6d,As6c,Ac6h,Ac6d,Ac6s,Ah7d,Ah7s,Ah7c,Ad7h,Ad7s,Ad7c,As7h,As7d,As7c,Ac7h,Ac7d,Ac7s,Ah8s,Ah8c,Ad8h,Ad8s,Ad8c,As8h,As8c,Ac8h,Ac8s,Ah9d,Ah9s,Ah9c,Ad9h,Ad9s,Ad9c,As9h,As9d,As9c,Ac9h,Ac9d,Ac9s,AhTd,AhTc,AdTh,AdTc,AsTh,AsTd,AsTc,AcTh,AcTd,AhJd,AhJs,AdJh,AdJs,AsJh,AsJd,AcJh,AcJd,AcJs,AhQd,AhQs,AhQc,AdQh,AdQs,AdQc,AsQh,AsQd,AsQc,AcQh,AcQd,AcQs,AhKd,AhKs,AhKc,AdKh,AdKs,AdKc,AsKh,AsKd,AsKc,AcKh,AcKd,AcKs,Kh9d,Kh9s,Kh9c,Kd9h,Kd9s,Kd9c,Ks9h,Ks9d,Ks9c,Kc9h,Kc9d,Kc9s,KhTd,KhTc,KdTh,KdTc,KsTh,KsTd,KsTc,KcTh,KcTd,KhJd,KhJs,KdJh,KdJs,KsJh,KsJd,KcJh,KcJd,KcJs,KhQd,KhQs,KhQc,KdQh,KdQs,KdQc,KsQh,KsQd,KsQc,KcQh,KcQd,KcQs,Qh9d,Qh9s,Qh9c,Qd9h,Qd9s,Qd9c,Qs9h,Qs9d,Qs9c,Qc9h,Qc9d,Qc9s,QhTd,QhTc,QdTh,QdTc,QsTh,QsTd,QsTc,QcTh,QcTd,QhJd,QhJs,QdJh,QdJs,QsJh,QsJd,QcJh,QcJd,QcJs,Jh9d,Jh9s,Jh9c,Jd9h,Jd9s,Jd9c,Js9h,Js9d,Js9c,JhTd,JhTc,JdTh,JdTc,JsTh,JsTd,JsTc,Th9d,Th9s,Th9c,Td9h,Td9s,Td9c,Tc9h,Tc9d,Tc9s'

# 报告中的IP范围（195个组合）
IP_RANGE = 'AcAd,AcAh,AcAs,AdAh,AdAs,AhAs,AcKc,AcKh,AdKc,AdKh,AhKc,AhKh,AsKc,AsKh,AcQc,AcQd,AcQh,AcQs,AdQc,AdQd,AdQh,AdQs,AhQc,AhQd,AhQh,AhQs,AsQc,AsQd,AsQh,AsQs,AcJd,AcJh,AcJs,AdJd,AdJh,AdJs,AhJd,AhJh,AhJs,AsJd,AsJh,AsJs,AcTc,AcTd,AcTh,AdTc,AdTd,AdTh,AhTc,AhTd,AhTh,AsTc,AsTd,AsTh,Ac9c,Ad9d,Ah9h,As9s,Ac8c,Ah8h,As8s,Ac7c,Ad7d,Ah7h,As7s,Ac6c,Ad6d,Ah6h,As6s,Ac5c,Ad5d,Ah5h,As5s,Ac4c,Ad4d,Ah4h,As4s,Ac3c,Ad3d,Ah3h,As3s,Ad2d,As2s,KcKh,KcQc,KcQd,KcQh,KcQs,KhQc,KhQd,KhQh,KhQs,KcJd,KcJh,KcJs,KhJd,KhJh,KhJs,KcTc,KhTh,Kc9c,Kh9h,QcQd,QcQh,QcQs,QdQh,QdQs,QhQs,QcJd,QcJh,QcJs,QdJd,QdJh,QdJs,QhJd,QhJh,QhJs,QsJd,QsJh,QsJs,QcTc,QdTd,QhTh,Qc9c,Qd9d,Qh9h,Qs9s,JdJh,JdJs,JhJs,JdTd,JhTh,Jd9d,Jh9h,Js9s,TcTd,TcTh,TdTh,Tc9c,Td9d,Th9h,9c9d,9c9h,9c9s,9d9h,9d9s,9h9s,9c8c,9h8h,9s8s,8c8h,8c8s,8h8s,8c7c,8h7h,8s7s,7c7d,7c7h,7c7s,7d7h,7d7s,7h7s,7c6c,7d6d,7h6h,7s6s,6c6d,6c6h,6c6s,6d6h,6d6s,6h6s,6c5c,6d5d,6h5h,6s5s,5c5d,5c5h,5c5s,5d5h,5d5s,5h5s,4c4d,4c4h,4c4s,4d4h,4d4s,4h4s,3c3d,3c3h,3c3s,3d3h,3d3s,3h3s,2d2s'


def calculate_equity(range1, range2, board=''):
    cmd = [OMPEVAL_PATH, range1, range2]
    if board:
        cmd.append(board)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"OMPEval error: {result.stderr}")
    
    # 处理nan值
    output = result.stdout.replace('nan', '0')
    return json.loads(output)


def main():
    print("使用报告中的精确范围验证Case 2")
    print(f"\n公共牌: {BOARD}")
    
    print(f"\nOOP组合数: {len(OOP_RANGE.split(','))}")
    print(f"IP组合数: {len(IP_RANGE.split(','))}")
    
    # 计算胜率
    result = calculate_equity(OOP_RANGE, IP_RANGE, BOARD)
    
    print(f"\n{'='*60}")
    print(f"结果:")
    print(f"{'='*60}")
    print(f"  OOP胜率(equity): {result['equity'][0] * 100:.3f}%")
    print(f"  IP胜率(equity): {result['equity'][1] * 100:.3f}%")
    print(f"  OOP胜: {result['wins'][0]}")
    print(f"  IP胜: {result['wins'][1]}")
    print(f"  平局次数: {result['tieCount']}")
    print(f"  总手数: {result['hands']}")
    
    total = result['wins'][0] + result['wins'][1] + result['tieCount']
    if total > 0:
        win_rate = result['wins'][0] / total * 100
        tie_rate = result['tieCount'] / total * 100
        loss_rate = result['wins'][1] / total * 100
        print(f"\n  OOP纯胜率: {win_rate:.3f}%")
        print(f"  平局率: {tie_rate:.3f}%")
        print(f"  OOP负率: {loss_rate:.3f}%")
        
        # 计算equity方式的胜率
        equity = (result['wins'][0] + result['tieCount'] / 2) / total * 100
        print(f"\n  OOP Equity (胜+平/2): {equity:.3f}%")


if __name__ == '__main__':
    main()
