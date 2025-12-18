#!/usr/bin/env python3
"""
使用OMPEval计算范围对范围的胜率
"""

import subprocess
import json
import os

# OMPEval可执行文件路径
OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'OMPEval', 'equity_calc')

# 范围定义
OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o'
IP_RANGE = 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo'


def expand_range_to_ompeval(range_str):
    """
    将范围字符串展开为OMPEval支持的格式
    OMPEval支持: AA, KK, AKs, AKo, K4+, 44+
    不支持: AKs-A2s 这种范围表示法
    """
    ranks = '23456789TJQKA'
    result = []
    
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if '-' in part:
            # 范围表示法: AA-22, AKs-A2s, AKo-A2o
            start, end = part.split('-')
            
            if len(start) == 2 and start[0] == start[1]:
                # 对子范围: AA-22
                start_rank = ranks.index(start[0])
                end_rank = ranks.index(end[0])
                min_rank = min(start_rank, end_rank)
                max_rank = max(start_rank, end_rank)
                for r in range(min_rank, max_rank + 1):
                    result.append(ranks[r] + ranks[r])
            elif start.endswith('s'):
                # 同花范围: AKs-A2s
                high_rank = ranks.index(start[0])
                start_low = ranks.index(start[1])
                end_low = ranks.index(end[1])
                min_low = min(start_low, end_low)
                max_low = max(start_low, end_low)
                for r in range(min_low, max_low + 1):
                    if r != high_rank:
                        result.append(ranks[high_rank] + ranks[r] + 's')
            elif start.endswith('o'):
                # 不同花范围: AKo-A2o
                high_rank = ranks.index(start[0])
                start_low = ranks.index(start[1])
                end_low = ranks.index(end[1])
                min_low = min(start_low, end_low)
                max_low = max(start_low, end_low)
                for r in range(min_low, max_low + 1):
                    if r != high_rank:
                        result.append(ranks[high_rank] + ranks[r] + 'o')
        else:
            # 单个手牌: AA, AKs, AKo, 54s
            result.append(part)
    
    return ','.join(result)


def calculate_equity(range1, range2, board='', dead=''):
    """
    使用OMPEval计算范围对范围的胜率
    """
    # 展开范围
    range1_expanded = expand_range_to_ompeval(range1)
    range2_expanded = expand_range_to_ompeval(range2)
    
    # 调用OMPEval
    cmd = [OMPEVAL_PATH, range1_expanded, range2_expanded]
    if board:
        cmd.append(board)
    if dead:
        cmd.append(dead)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"OMPEval error: {result.stderr}")
    
    # 解析JSON结果
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"原始输出: {result.stdout}")
        raise


def main():
    print("使用OMPEval计算范围对范围胜率")
    print(f"\nOOP范围: {OOP_RANGE}")
    print(f"IP范围: {IP_RANGE}")
    
    # Case 2: 公共牌 2h 8d 2c Jc Ts
    board = "2h8d2cJcTs"
    print(f"\n公共牌: {board}")
    
    result = calculate_equity(OOP_RANGE, IP_RANGE, board)
    
    print(f"\n结果:")
    print(f"  OOP胜率(equity): {result['equity'][0] * 100:.3f}%")
    print(f"  IP胜率(equity): {result['equity'][1] * 100:.3f}%")
    print(f"  OOP胜: {result['wins'][0]}")
    print(f"  IP胜: {result['wins'][1]}")
    print(f"  平局次数: {result['tieCount']}")
    print(f"  总手数: {result['hands']}")
    
    # 计算胜率和平局率
    total = result['wins'][0] + result['wins'][1] + result['tieCount']
    if total > 0:
        win_rate = result['wins'][0] / total * 100
        tie_rate = result['tieCount'] / total * 100
        loss_rate = result['wins'][1] / total * 100
        print(f"\n  OOP纯胜率: {win_rate:.3f}%")
        print(f"  平局率: {tie_rate:.3f}%")
        print(f"  OOP负率: {loss_rate:.3f}%")


if __name__ == '__main__':
    main()
