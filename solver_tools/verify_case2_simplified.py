#!/usr/bin/env python3
"""
使用简化后的IP范围验证Case 2的范围胜率
"""

import subprocess
import json
import os

OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'OMPEval', 'equity_calc')

# 原始OOP范围
OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o'

# 用户提供的简化后IP范围（已排除与公共牌冲突的手牌）
IP_RANGE_SIMPLIFIED = 'AA,QQ,99,77-33,KhKc,JdJh,JdJs,JhJs,TdTh,TdTc,ThTc,8h8s,8h8c,8s8c,2d2s,AQs,A9s,A7s-A3s,Q9s,76s,65s,AhKh,AcKc,KhQh,KcQc,AdJd,AhJh,AsJs,KhJh,QdJd,QhJh,QsJs,AdTd,AhTh,AcTc,KhTh,KcTc,QdTd,QhTh,QcTc,JdTd,JhTh,Kh9h,Kc9c,Jd9d,Jh9h,Js9s,Td9d,Th9h,Tc9c,Ah8h,As8s,Ac8c,9h8h,9s8s,9c8c,8h7h,8s7s,8c7c,Ad2d,As2s,AQo,AdKh,AdKc,AhKc,AsKh,AsKc,AcKh,AdJh,AdJs,AhJd,AhJs,AsJd,AsJh,AcJd,AcJh,AcJs,AdTh,AdTc,AhTd,AhTc,AsTd,AsTh,AsTc,AcTd,AcTh,KhQd,KhQs,KhQc,KcQd,KcQh,KcQs,KhJd,KhJs,KcJd,KcJh,KcJs,QdJh,QdJs,QhJd,QhJs,QsJd,QsJh,QcJd,QcJh,QcJs'

# Case 2 公共牌
BOARD = "2h8d2cJcTs"


def expand_range_to_ompeval(range_str):
    """将范围字符串展开为OMPEval支持的格式"""
    ranks = '23456789TJQKA'
    result = []
    
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if '-' in part:
            start, end = part.split('-')
            
            if len(start) == 2 and start[0] == start[1]:
                start_rank = ranks.index(start[0])
                end_rank = ranks.index(end[0])
                min_rank = min(start_rank, end_rank)
                max_rank = max(start_rank, end_rank)
                for r in range(min_rank, max_rank + 1):
                    result.append(ranks[r] + ranks[r])
            elif start.endswith('s'):
                high_rank = ranks.index(start[0])
                start_low = ranks.index(start[1])
                end_low = ranks.index(end[1])
                min_low = min(start_low, end_low)
                max_low = max(start_low, end_low)
                for r in range(min_low, max_low + 1):
                    if r != high_rank:
                        result.append(ranks[high_rank] + ranks[r] + 's')
            elif start.endswith('o'):
                high_rank = ranks.index(start[0])
                start_low = ranks.index(start[1])
                end_low = ranks.index(end[1])
                min_low = min(start_low, end_low)
                max_low = max(start_low, end_low)
                for r in range(min_low, max_low + 1):
                    if r != high_rank:
                        result.append(ranks[high_rank] + ranks[r] + 'o')
        else:
            result.append(part)
    
    return ','.join(result)


def calculate_equity(range1, range2, board='', dead='', expand=True):
    # 展开范围（如果需要）
    if expand:
        range1 = expand_range_to_ompeval(range1)
        range2 = expand_range_to_ompeval(range2)
    
    cmd = [OMPEVAL_PATH, range1, range2]
    if board:
        cmd.append(board)
    if dead:
        cmd.append(dead)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"OMPEval error: {result.stderr}")
    
    try:
        # 处理nan值
        output = result.stdout.replace('nan', '0')
        return json.loads(output)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"原始输出: {result.stdout}")
        raise


def main():
    print("使用简化后的IP范围验证Case 2")
    print(f"\n公共牌: {BOARD}")
    print(f"\nOOP范围（原始）: {OOP_RANGE}")
    print(f"\nIP范围（简化后）: {IP_RANGE_SIMPLIFIED}")
    
    # 使用简化后的IP范围（IP范围已经是展开的，不需要再展开）
    # OOP范围需要展开
    result = calculate_equity(OOP_RANGE, IP_RANGE_SIMPLIFIED, BOARD, expand=False)
    # 手动展开OOP范围
    oop_expanded = expand_range_to_ompeval(OOP_RANGE)
    result = calculate_equity(oop_expanded, IP_RANGE_SIMPLIFIED, BOARD, '', expand=False)
    
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
