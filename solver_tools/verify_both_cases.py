#!/usr/bin/env python3
"""
使用OMPEval验证两个Case的范围胜率
"""

import subprocess
import json
import os

OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'OMPEval', 'equity_calc')

OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o'
IP_RANGE = 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo'


def expand_range_to_ompeval(range_str):
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


def calculate_equity(range1, range2, board='', dead=''):
    range1_expanded = expand_range_to_ompeval(range1)
    range2_expanded = expand_range_to_ompeval(range2)
    
    cmd = [OMPEVAL_PATH, range1_expanded, range2_expanded]
    if board:
        cmd.append(board)
    if dead:
        cmd.append(dead)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"OMPEval error: {result.stderr}")
    
    return json.loads(result.stdout)


def print_result(case_name, board, result):
    print(f"\n{'='*60}")
    print(f"{case_name}")
    print(f"公共牌: {board}")
    print(f"{'='*60}")
    
    print(f"  OOP胜率(equity): {result['equity'][0] * 100:.3f}%")
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


def main():
    print("使用OMPEval验证范围对范围胜率")
    print(f"\nOOP范围: {OOP_RANGE}")
    print(f"IP范围: {IP_RANGE}")
    
    # Case 1: 公共牌 4c 2h 7s Kd 5c
    board1 = "4c2h7sKd5c"
    result1 = calculate_equity(OOP_RANGE, IP_RANGE, board1)
    print_result("Case 1", board1, result1)
    
    # Case 2: 公共牌 2h 8d 2c Jc Ts
    board2 = "2h8d2cJcTs"
    result2 = calculate_equity(OOP_RANGE, IP_RANGE, board2)
    print_result("Case 2", board2, result2)


if __name__ == '__main__':
    main()
