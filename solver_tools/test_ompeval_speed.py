#!/usr/bin/env python3
"""
测试 OMPEval 范围对范围计算的速度
"""

import subprocess
import json
import os
import time
import random

# OMPEval可执行文件路径
OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'OMPEval', 'equity_calc')

# 范围定义
OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o'
IP_RANGE = 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo'


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


def generate_random_board():
    """生成随机的5张公共牌"""
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    cards = []
    used = set()
    
    while len(cards) < 5:
        r = random.choice(ranks)
        s = random.choice(suits)
        card = r + s
        if card not in used:
            used.add(card)
            cards.append(card)
    
    return ''.join(cards)


def calculate_equity(range1, range2, board='', dead=''):
    """使用OMPEval计算范围对范围的胜率"""
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


def main():
    print("=" * 60)
    print("测试 OMPEval 范围对范围计算速度")
    print("=" * 60)
    
    # 检查 OMPEval 是否存在
    if not os.path.exists(OMPEVAL_PATH):
        print(f"错误: OMPEval 不存在于 {OMPEVAL_PATH}")
        return
    
    print(f"\nOOP范围: {OOP_RANGE}")
    print(f"IP范围: {IP_RANGE}")
    
    # 测试单次计算
    print("\n--- 单次计算测试 ---")
    board = "2h8d2cJcTs"
    print(f"公共牌: {board}")
    
    start = time.time()
    result = calculate_equity(OOP_RANGE, IP_RANGE, board)
    elapsed = time.time() - start
    
    print(f"计算时间: {elapsed*1000:.2f}ms")
    print(f"OOP胜率: {result['equity'][0] * 100:.3f}%")
    print(f"总组合数: {result['hands']}")
    
    # 测试多次计算
    print("\n--- 多次计算测试 (100次) ---")
    times = []
    
    for i in range(100):
        board = generate_random_board()
        
        start = time.time()
        result = calculate_equity(OOP_RANGE, IP_RANGE, board)
        elapsed = time.time() - start
        times.append(elapsed)
        
        if (i + 1) % 20 == 0:
            avg = sum(times) / len(times)
            print(f"  进度: {i+1}/100, 平均时间: {avg*1000:.2f}ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n--- 统计结果 ---")
    print(f"平均时间: {avg_time*1000:.2f}ms")
    print(f"最小时间: {min_time*1000:.2f}ms")
    print(f"最大时间: {max_time*1000:.2f}ms")
    print(f"预计10000次计算总时间: {avg_time * 10000:.1f}s ({avg_time * 10000 / 60:.1f}分钟)")
    
    # 测试带死牌的计算
    print("\n--- 带死牌计算测试 ---")
    board = "2h8d2cJcTs"
    dead = "AsKs"  # 模拟固定的OOP手牌作为死牌
    
    start = time.time()
    result = calculate_equity(OOP_RANGE, IP_RANGE, board, dead)
    elapsed = time.time() - start
    
    print(f"公共牌: {board}, 死牌: {dead}")
    print(f"计算时间: {elapsed*1000:.2f}ms")
    print(f"OOP胜率: {result['equity'][0] * 100:.3f}%")
    print(f"总组合数: {result['hands']}")


if __name__ == '__main__':
    main()
