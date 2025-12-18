#!/usr/bin/env python3
"""
根据公共牌简化范围（排除与公共牌冲突的手牌组合）
"""

# 公共牌: 2h 8d 2c Jc Ts
BOARD = ['2h', '8d', '2c', 'Jc', 'Ts']
BOARD_SET = set(BOARD)

RANKS = '23456789TJQKA'
SUITS = 'hdsc'


def expand_range_to_combos(range_str):
    """将范围字符串展开为所有具体的手牌组合"""
    combos = []
    
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if '-' in part:
            start, end = part.split('-')
            
            if len(start) == 2 and start[0] == start[1]:
                # 对子范围: AA-22
                start_rank = RANKS.index(start[0])
                end_rank = RANKS.index(end[0])
                min_rank = min(start_rank, end_rank)
                max_rank = max(start_rank, end_rank)
                for r in range(min_rank, max_rank + 1):
                    combos.extend(get_pair_combos(RANKS[r]))
            elif start.endswith('s'):
                # 同花范围: AKs-A2s
                high_rank = RANKS.index(start[0])
                start_low = RANKS.index(start[1])
                end_low = RANKS.index(end[1])
                min_low = min(start_low, end_low)
                max_low = max(start_low, end_low)
                for r in range(min_low, max_low + 1):
                    if r != high_rank:
                        combos.extend(get_suited_combos(RANKS[high_rank], RANKS[r]))
            elif start.endswith('o'):
                # 不同花范围: AKo-A2o
                high_rank = RANKS.index(start[0])
                start_low = RANKS.index(start[1])
                end_low = RANKS.index(end[1])
                min_low = min(start_low, end_low)
                max_low = max(start_low, end_low)
                for r in range(min_low, max_low + 1):
                    if r != high_rank:
                        combos.extend(get_offsuit_combos(RANKS[high_rank], RANKS[r]))
        elif len(part) == 2:
            # 对子: AA
            combos.extend(get_pair_combos(part[0]))
        elif len(part) == 3:
            if part.endswith('s'):
                combos.extend(get_suited_combos(part[0], part[1]))
            elif part.endswith('o'):
                combos.extend(get_offsuit_combos(part[0], part[1]))
        elif len(part) == 4:
            # 具体手牌: AhKh
            combos.append(part)
    
    return combos


def get_pair_combos(rank):
    """获取对子的所有组合"""
    combos = []
    for i, s1 in enumerate(SUITS):
        for s2 in SUITS[i+1:]:
            combos.append(f"{rank}{s1}{rank}{s2}")
    return combos


def get_suited_combos(rank1, rank2):
    """获取同花的所有组合"""
    combos = []
    for s in SUITS:
        combos.append(f"{rank1}{s}{rank2}{s}")
    return combos


def get_offsuit_combos(rank1, rank2):
    """获取不同花的所有组合"""
    combos = []
    for s1 in SUITS:
        for s2 in SUITS:
            if s1 != s2:
                combos.append(f"{rank1}{s1}{rank2}{s2}")
    return combos


def filter_combos(combos, board_set):
    """过滤掉与公共牌冲突的组合"""
    valid = []
    for combo in combos:
        card1 = combo[:2]
        card2 = combo[2:]
        if card1 not in board_set and card2 not in board_set:
            valid.append(combo)
    return valid


def main():
    # OOP范围
    OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o'
    
    print(f"公共牌: {' '.join(BOARD)}")
    print(f"\n原始OOP范围: {OOP_RANGE}")
    
    # 展开并过滤
    all_combos = expand_range_to_combos(OOP_RANGE)
    valid_combos = filter_combos(all_combos, BOARD_SET)
    
    print(f"\n原始组合数: {len(all_combos)}")
    print(f"有效组合数: {len(valid_combos)}")
    
    # 输出简化后的范围
    print(f"\n简化后的OOP范围:")
    print(','.join(valid_combos))


if __name__ == '__main__':
    main()
