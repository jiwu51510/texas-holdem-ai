#!/usr/bin/env python3
"""
使用 wasm-postflop + OMPEval 进行跨公共牌四维度胜率-策略验证实验 V3

改进：使用 OMPEval (C++) 计算范围对范围胜率，速度比 poker-odds-calc 快约60倍
"""

import subprocess
import json
import os
import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# OMPEval可执行文件路径
OMPEVAL_PATH = os.path.join(os.path.dirname(__file__), 'solver_tools', 'OMPEval', 'equity_calc')

# 范围定义
OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o'
IP_RANGE = 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo'

@dataclass
class EquityResult:
    """胜率计算结果"""
    equity: float  # 综合胜率 (胜 + 平/2) / 总
    win_rate: float  # 纯胜率
    tie_rate: float  # 平局率
    wins: int
    ties: int
    total: int


@dataclass
class Scenario:
    """场景数据"""
    board: str
    hero_hand: str
    hero_cards: List[str]
    hero_equity: EquityResult
    range_equity: EquityResult
    strategy: Dict[str, float]
    actions: List[str]
    effective_ip_range_count: int


def expand_range_to_ompeval(range_str: str) -> str:
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


def calculate_equity_ompeval(range1: str, range2: str, board: str, dead: str = '') -> EquityResult:
    """使用OMPEval计算范围对范围的胜率"""
    range1_expanded = expand_range_to_ompeval(range1)
    range2_expanded = expand_range_to_ompeval(range2)
    
    # 转换公共牌格式: "As Ks 6d Qc Td" -> "AsKs6dQcTd"
    board_compact = board.replace(' ', '')
    
    cmd = [OMPEVAL_PATH, range1_expanded, range2_expanded, board_compact]
    if dead:
        cmd.append(dead)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"OMPEval error: {result.stderr}")
    
    # 处理 nan 值（OMPEval 在某些情况下返回 nan）
    output = result.stdout.replace('nan', 'null')
    data = json.loads(output)
    
    # 计算胜率和平局率
    wins = data['wins'][0]
    losses = data['wins'][1]
    ties = data['tieCount']
    total = wins + losses + ties
    
    if total == 0:
        return EquityResult(0.5, 0.5, 0, 0, 0, 0)
    
    win_rate = wins / total
    tie_rate = ties / total
    equity = (wins + ties * 0.5) / total
    
    return EquityResult(equity, win_rate, tie_rate, wins, ties, total)


def generate_random_board() -> str:
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
    
    return ' '.join(cards)


def select_random_hero_hand(board: str) -> Optional[Tuple[str, List[str]]]:
    """为给定公共牌随机选择一个不冲突的手牌"""
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    board_cards = set(board.split())
    
    valid_hands = []
    
    # 口袋对
    for r in range(13):
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                card1 = ranks[r] + suits[s1]
                card2 = ranks[r] + suits[s2]
                if card1 not in board_cards and card2 not in board_cards:
                    valid_hands.append((card1 + card2, [card1, card2]))
    
    # 同花连接和高牌
    suited_combos = [
        (12, 11), (11, 10), (10, 9), (9, 8), (8, 7), (7, 6),
        (12, 10), (12, 9), (11, 9),
    ]
    
    for r1, r2 in suited_combos:
        for s in range(4):
            card1 = ranks[r1] + suits[s]
            card2 = ranks[r2] + suits[s]
            if card1 not in board_cards and card2 not in board_cards:
                valid_hands.append((card1 + card2, [card1, card2]))
    
    # 非同花高牌
    offsuit_combos = [
        (12, 11), (12, 10), (12, 9), (11, 10), (11, 9),
    ]
    
    for r1, r2 in offsuit_combos:
        for s1 in range(4):
            for s2 in range(4):
                if s1 != s2:
                    card1 = ranks[r1] + suits[s1]
                    card2 = ranks[r2] + suits[s2]
                    if card1 not in board_cards and card2 not in board_cards:
                        valid_hands.append((card1 + card2, [card1, card2]))
    
    if not valid_hands:
        return None
    
    return random.choice(valid_hands)


def solve_scenario_via_node(board: str, oop_range: str, ip_range: str) -> Optional[Dict]:
    """通过Node.js调用wasm-postflop solver"""
    # 创建临时脚本
    script = f'''
    import {{ solveRiver }} from './solver_tools/postflop_solver.mjs';
    
    async function main() {{
        try {{
            const result = await solveRiver({{
                oopRange: '{oop_range}',
                ipRange: '{ip_range}',
                board: '{board}',
                startingPot: 100,
                effectiveStack: 100,
                oopBetSizes: '50',
                ipBetSizes: '50',
                targetExploitability: 0.3,
                maxIterations: 500,
            }});
            console.log(JSON.stringify(result));
        }} catch (e) {{
            console.error(e.message);
            process.exit(1);
        }}
    }}
    main();
    '''
    
    # 写入临时文件
    temp_file = '/tmp/solve_temp.mjs'
    with open(temp_file, 'w') as f:
        f.write(script)
    
    # 执行
    result = subprocess.run(['node', temp_file], capture_output=True, text=True, cwd=os.getcwd())
    
    if result.returncode != 0:
        return None
    
    try:
        return json.loads(result.stdout)
    except:
        return None


def extract_strategy(solver_result: Dict, hero_cards: List[str]) -> Optional[Dict[str, float]]:
    """从solver结果中提取特定手牌的策略"""
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    
    # 将hero_cards转换为索引
    hero_indices = sorted([
        ranks.index(card[0]) * 4 + suits.index(card[1])
        for card in hero_cards
    ])
    
    # 在oopCards中找到手牌索引
    oop_cards = solver_result['oopCards']
    hero_index = -1
    
    for i, hand_idx in enumerate(oop_cards):
        c1 = hand_idx & 0xFF
        c2 = (hand_idx >> 8) & 0xFF
        cards = sorted([c1, c2])
        if cards[0] == hero_indices[0] and cards[1] == hero_indices[1]:
            hero_index = i
            break
    
    if hero_index == -1:
        return None
    
    # 解析动作
    actions_str = solver_result['actions']
    action_list = []
    for a in actions_str.split('/'):
        import re
        match = re.match(r'(\w+):(\d+)', a)
        if match:
            action_list.append(match.group(1))
        else:
            action_list.append(a)
    
    # 提取策略
    oop_len = len(oop_cards)
    ip_len = len(solver_result['ipCards'])
    num_actions = solver_result['numActions']
    results = solver_result['results']
    
    # 跳过header, weights, normalizer, equity, ev, eqr
    offset = 3
    offset += oop_len + ip_len  # weights
    offset += oop_len + ip_len  # normalizer
    offset += oop_len + ip_len  # equity
    offset += oop_len + ip_len  # ev
    offset += oop_len + ip_len  # eqr
    
    strategy = {}
    for i in range(num_actions):
        strategy[action_list[i]] = results[offset + i * oop_len + hero_index]
    
    return strategy


def calculate_strategy_diff(s1: Dict[str, float], s2: Dict[str, float]) -> float:
    """计算策略差异"""
    keys = set(s1.keys()) | set(s2.keys())
    diff = sum(abs(s1.get(k, 0) - s2.get(k, 0)) for k in keys)
    return diff / len(keys) if keys else 0


def run_experiment():
    """主实验函数"""
    print('=' * 80)
    print('使用 wasm-postflop + OMPEval 进行跨公共牌四维度胜率-策略验证实验 V3')
    print('=' * 80)
    print('\n核心改进: 使用 OMPEval (C++) 计算范围胜率，速度比 poker-odds-calc 快约60倍')
    print('\n核心问题: 在不同的（公共牌+固定手牌）组合下：')
    print('  当手牌胜率和范围胜率都相近时，策略是否相同？')
    print('\n范围定义:')
    print(f'  OOP: {OOP_RANGE}')
    print(f'  IP: {IP_RANGE}')
    
    NUM_SCENARIOS = 10000
    scenarios: List[Scenario] = []
    
    print(f'\n生成 {NUM_SCENARIOS} 个（公共牌+固定手牌）场景...')
    print('使用 OMPEval 计算胜率，预计总时间约 2-3 分钟\n')
    
    start_time = time.time()
    
    for i in range(NUM_SCENARIOS):
        board = generate_random_board()
        hero_info = select_random_hero_hand(board)
        
        if not hero_info:
            continue
        
        hero_hand, hero_cards = hero_info
        
        # 进度输出
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 1
            remaining = (NUM_SCENARIOS - i - 1) / rate
            print(f'进度: {i+1}/{NUM_SCENARIOS} - 已用时: {elapsed:.0f}s - 预计剩余: {remaining:.0f}s')
        
        # 死牌 = 公共牌 + OOP固定手牌
        dead_cards = hero_cards[0] + hero_cards[1]
        
        try:
            # 1. 计算手牌vs IP范围的胜率（使用死牌=公共牌）
            # 注意：这里计算的是固定手牌 vs IP范围
            hero_equity = calculate_equity_ompeval(
                hero_hand,  # 单个手牌
                IP_RANGE,
                board,
                ''  # 无额外死牌
            )
            
            # 2. 计算OOP范围 vs IP范围的胜率（使用死牌=公共牌+固定手牌）
            range_equity = calculate_equity_ompeval(
                OOP_RANGE,
                IP_RANGE,
                board,
                dead_cards  # 固定手牌作为死牌
            )
        except Exception as e:
            print(f'  胜率计算错误: {e}')
            continue

        
        # 3. 使用solver获取策略
        # 将IP范围转换为具体手牌列表（排除死牌）
        solver_result = solve_scenario_via_node(board, OOP_RANGE, IP_RANGE)
        if not solver_result:
            continue
        
        strategy = extract_strategy(solver_result, hero_cards)
        if not strategy:
            continue
        
        # 详细日志只在前10个场景输出
        if i < 10:
            print(f'  场景 {i+1}: {board} + {hero_hand}')
            print(f'    手牌: 胜率={hero_equity.win_rate*100:.3f}%, 平局率={hero_equity.tie_rate*100:.3f}%')
            print(f'    范围: 胜率={range_equity.win_rate*100:.3f}%, 平局率={range_equity.tie_rate*100:.3f}%')
        
        scenarios.append(Scenario(
            board=board,
            hero_hand=hero_hand,
            hero_cards=hero_cards,
            hero_equity=hero_equity,
            range_equity=range_equity,
            strategy=strategy,
            actions=list(strategy.keys()),
            effective_ip_range_count=range_equity.total
        ))
    
    total_time = time.time() - start_time
    print(f'\n\n成功生成 {len(scenarios)} 个场景，总用时: {total_time:.1f}s')
    
    # 分析结果
    analyze_results(scenarios)


def analyze_results(scenarios: List[Scenario]):
    """分析结果"""
    print('\n' + '=' * 80)
    print('分析结果：寻找四维度胜率相近的场景对（阈值: 0.1%）')
    print('条件：手牌胜率、手牌平局率、范围胜率、范围平局率 都相差不超过0.1%')
    print('=' * 80)
    
    threshold = 0.001  # 0.1%
    pairs = []
    
    for i in range(len(scenarios)):
        for j in range(i + 1, len(scenarios)):
            s1 = scenarios[i]
            s2 = scenarios[j]
            
            # 四维度比较
            hero_win_diff = abs(s1.hero_equity.win_rate - s2.hero_equity.win_rate)
            hero_tie_diff = abs(s1.hero_equity.tie_rate - s2.hero_equity.tie_rate)
            range_win_diff = abs(s1.range_equity.win_rate - s2.range_equity.win_rate)
            range_tie_diff = abs(s1.range_equity.tie_rate - s2.range_equity.tie_rate)
            
            if (hero_win_diff < threshold and hero_tie_diff < threshold and
                range_win_diff < threshold and range_tie_diff < threshold):
                strategy_diff = calculate_strategy_diff(s1.strategy, s2.strategy)
                pairs.append({
                    's1': s1, 's2': s2,
                    'hero_win_diff': hero_win_diff,
                    'hero_tie_diff': hero_tie_diff,
                    'range_win_diff': range_win_diff,
                    'range_tie_diff': range_tie_diff,
                    'strategy_diff': strategy_diff
                })
    
    print(f'\n找到 {len(pairs)} 对四维度胜率相近的场景')
    
    significant_pairs = [p for p in pairs if p['strategy_diff'] > 0.15]
    print(f'其中策略差异显著(>15%)的: {len(significant_pairs)} 对')
    
    # 输出反例
    if significant_pairs:
        print('\n【策略差异显著的反例】')
        for p in significant_pairs[:10]:
            s1, s2 = p['s1'], p['s2']
            print(f'\n{"─" * 70}')
            print(f'【场景1】')
            print(f'  公共牌: {s1.board}')
            print(f'  OOP手牌: {s1.hero_hand}')
            print(f'  手牌: 胜率={s1.hero_equity.win_rate*100:.3f}%, 平局率={s1.hero_equity.tie_rate*100:.3f}%')
            print(f'  范围: 胜率={s1.range_equity.win_rate*100:.3f}%, 平局率={s1.range_equity.tie_rate*100:.3f}%')
            print(f'  策略: {s1.strategy}')
            
            print(f'\n【场景2】')
            print(f'  公共牌: {s2.board}')
            print(f'  OOP手牌: {s2.hero_hand}')
            print(f'  手牌: 胜率={s2.hero_equity.win_rate*100:.3f}%, 平局率={s2.hero_equity.tie_rate*100:.3f}%')
            print(f'  范围: 胜率={s2.range_equity.win_rate*100:.3f}%, 平局率={s2.range_equity.tie_rate*100:.3f}%')
            print(f'  策略: {s2.strategy}')
            
            print(f'\n【对比】')
            print(f'  手牌胜率差异: {p["hero_win_diff"]*100:.3f}%, 手牌平局率差异: {p["hero_tie_diff"]*100:.3f}%')
            print(f'  范围胜率差异: {p["range_win_diff"]*100:.3f}%, 范围平局率差异: {p["range_tie_diff"]*100:.3f}%')
            print(f'  策略差异: {p["strategy_diff"]*100:.1f}%')
    
    # 保存结果
    save_results(scenarios, pairs, significant_pairs)


def save_results(scenarios: List[Scenario], pairs: List, significant_pairs: List):
    """保存结果"""
    # 保存JSON
    output_data = {
        'method': 'OMPEval + wasm-postflop',
        'oopRange': OOP_RANGE,
        'ipRange': IP_RANGE,
        'numScenarios': len(scenarios),
        'numPairs': len(pairs),
        'numSignificantPairs': len(significant_pairs),
    }
    
    output_path = 'experiments/results/wasm_postflop_validation_v3.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f'\n结果已保存到: {output_path}')
    
    # 生成报告
    generate_report(scenarios, pairs, significant_pairs)


def generate_report(scenarios: List[Scenario], pairs: List, significant_pairs: List):
    """生成实验报告"""
    report = f'''# 跨公共牌四维度胜率-策略验证实验报告 V3

## 实验改进

**本版本使用 OMPEval (C++) 计算范围胜率，速度比 poker-odds-calc 快约60倍。**

## 实验目的

验证：**在不同的（公共牌+固定手牌）组合下：**
当以下四个条件同时满足时，策略是否相同？
1. 固定手牌vs对手范围的胜率相近（差异<0.1%）
2. 固定手牌vs对手范围的平局率相近（差异<0.1%）
3. 自己范围vs对手范围的胜率相近（差异<0.1%）
4. 自己范围vs对手范围的平局率相近（差异<0.1%）

## 范围定义

- **OOP范围**: {OOP_RANGE}
- **IP范围**: {IP_RANGE}

## 实验规模

- 生成场景数: {len(scenarios)}
- 四维度胜率相近的场景对（差异<0.1%）: {len(pairs)}
- 策略差异显著(>15%)的场景对: {len(significant_pairs)}

## 关键发现

'''
    
    if significant_pairs:
        ratio = len(significant_pairs) / len(pairs) * 100 if pairs else 0
        report += f'''### ⚠️ 四维度胜率标量不足以决定最优策略

在 {len(pairs)} 对四维度胜率相近的场景中，有 {len(significant_pairs)} 对（{ratio:.1f}%）的策略差异显著。

**结论：即使手牌胜率、手牌平局率、范围胜率、范围平局率都精确匹配（差异<0.1%），最优策略仍然可能完全不同。**
'''
    elif pairs:
        report += f'''### ✅ 未发现反例

在找到的 {len(pairs)} 对四维度胜率相近的场景中，未发现策略显著不同的反例。
'''
    else:
        report += f'''### 需要更多数据或调整阈值

在 {len(scenarios)} 个场景中，未找到四维度胜率都相近（差异<0.1%）的场景对。
'''
    
    report_path = 'experiments/results/wasm_postflop_validation_v3_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'报告已保存到: {report_path}')


if __name__ == '__main__':
    run_experiment()
