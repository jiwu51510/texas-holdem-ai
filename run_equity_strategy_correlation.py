#!/usr/bin/env python3
"""胜率-策略相关性实验。

核心问题：当两个手牌具有相同的胜率特征时，它们的最优策略是否相同？

实验设计：
1. 在不同场景中找出具有相同/接近胜率的手牌对
2. 使用CFR计算这些手牌的最优策略
3. 比较策略差异

如果胜率能完全决定策略，那么相同胜率的手牌应该有相同的策略。
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from models.core import Card
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    DeadCardRemover,
    RangeVsRangeCalculator,
)
from experiments.equity_solver_validation.solver_wrapper import SimpleCFRSolver
from experiments.equity_solver_validation.data_models import SolverConfig
from environment.hand_evaluator import HandEvaluator, compare_hands
from models.core import HandRank


def card_to_str(card: Card) -> str:
    rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
                9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'}
    suit_map = {'s': '♠', 'd': '♦', 'c': '♣', 'h': '♥'}
    return f"{rank_map.get(card.rank, str(card.rank))}{suit_map.get(card.suit, card.suit)}"


def cards_to_str(cards: List[Card]) -> str:
    return ' '.join(card_to_str(c) for c in cards)


def parse_hand(hand_str: str) -> Optional[Tuple[Card, Card]]:
    rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
                '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
    if len(hand_str) == 4:
        r1, s1, r2, s2 = hand_str[0], hand_str[1].lower(), hand_str[2], hand_str[3].lower()
        if r1.upper() in rank_map and r2.upper() in rank_map:
            return (Card(rank=rank_map[r1.upper()], suit=s1), 
                    Card(rank=rank_map[r2.upper()], suit=s2))
    return None


def get_hand_type(hole_cards: Tuple[Card, Card], community: List[Card]) -> str:
    """获取手牌类型。"""
    all_cards = list(hole_cards) + community
    hand_rank, _ = HandEvaluator.evaluate_hand(all_cards)
    names = {
        HandRank.HIGH_CARD: "高牌", HandRank.PAIR: "一对", HandRank.TWO_PAIR: "两对",
        HandRank.THREE_OF_A_KIND: "三条", HandRank.STRAIGHT: "顺子",
        HandRank.FLUSH: "同花", HandRank.FULL_HOUSE: "葫芦",
        HandRank.FOUR_OF_A_KIND: "四条", HandRank.STRAIGHT_FLUSH: "同花顺",
    }
    return names.get(hand_rank, str(hand_rank))


@dataclass
class HandEquityProfile:
    """手牌胜率特征。"""
    hand_str: str
    hand_vs_range_equity: float  # 手牌 vs 对手范围
    range_vs_range_equity: float  # 我的范围 vs 对手范围（整体）
    hand_type: str
    solver_strategy: Dict[str, float]  # CFR计算的策略


@dataclass
class EquityPair:
    """具有相似胜率的手牌对。"""
    hand1: HandEquityProfile
    hand2: HandEquityProfile
    equity_diff: float  # 胜率差异
    strategy_diff: float  # 策略差异（TVD）
    same_action: bool  # 最优动作是否相同


def run_experiment(
    name: str,
    description: str,
    community_cards: List[Card],
    oop_range: Dict[str, float],
    ip_range: Dict[str, float],
    pot_size: float = 100.0,
    bet_size: float = 50.0,
    cfr_iterations: int = 500
):
    """运行单个实验场景。"""
    
    print(f"\n{'#'*80}")
    print(f"# {name}")
    print(f"# {description}")
    print(f"{'#'*80}")
    
    print(f"\n【输入】")
    print(f"公共牌: {cards_to_str(community_cards)}")
    print(f"底池: {pot_size}, 下注: {bet_size}")
    
    print(f"\nOOP范围: {list(oop_range.keys())}")
    print(f"IP范围: {list(ip_range.keys())}")
    
    # 初始化
    calculator = RangeVsRangeCalculator()
    remover = DeadCardRemover()
    
    config = SolverConfig(
        pot_size=pot_size,
        effective_stack=200.0,
        oop_bet_sizes=[bet_size / pot_size],
        ip_bet_sizes=[bet_size / pot_size],
        oop_raise_sizes=[],
        ip_raise_sizes=[],
        max_iterations=cfr_iterations,
    )
    solver = SimpleCFRSolver(config)
    
    # 展开OOP范围
    oop_expanded = {}
    for hand, weight in oop_range.items():
        concrete = remover.expand_abstract_hand(hand)
        for c in concrete:
            cards = parse_hand(c)
            if cards:
                board_set = set((card.rank, card.suit) for card in community_cards)
                if not any((card.rank, card.suit) in board_set for card in cards):
                    oop_expanded[c] = weight / len(concrete)
    
    # 计算每个OOP手牌的胜率
    print(f"\n【计算胜率】")
    hand_equities = calculator.calculate_range_vs_range_equity(
        oop_range, ip_range, community_cards
    )
    
    # 计算整体范围vs范围胜率
    range_equity = np.mean(list(hand_equities.values())) if hand_equities else 0.5
    print(f"范围vs范围平均胜率: {range_equity:.4f}")
    
    # 运行CFR求解
    print(f"\n【运行CFR求解器】(迭代次数: {cfr_iterations})")
    solver_result = solver.solve(community_cards, oop_range, ip_range, cfr_iterations)
    solver_strategies = solver_result.strategies.get('root', {})
    
    # 构建手牌特征
    profiles = []
    for hand_str, equity in hand_equities.items():
        cards = parse_hand(hand_str)
        if not cards:
            continue
        
        hand_type = get_hand_type(cards, community_cards)
        strategy = solver_strategies.get(hand_str, {'check': 0.5, 'bet': 0.5})
        
        profiles.append(HandEquityProfile(
            hand_str=hand_str,
            hand_vs_range_equity=equity,
            range_vs_range_equity=range_equity,
            hand_type=hand_type,
            solver_strategy=strategy,
        ))
    
    # 按胜率排序
    profiles.sort(key=lambda x: x.hand_vs_range_equity)
    
    # 打印所有手牌的胜率和策略
    print(f"\n【所有手牌的胜率和策略】")
    print(f"{'手牌':<12} {'牌型':<8} {'手牌vs范围':<12} {'check':<10} {'bet':<10}")
    print("-" * 60)
    
    for p in profiles:
        print(f"{p.hand_str:<12} {p.hand_type:<8} {p.hand_vs_range_equity:<12.4f} "
              f"{p.solver_strategy.get('check', 0):<10.4f} {p.solver_strategy.get('bet', 0):<10.4f}")
    
    # 找出胜率相近但策略不同的手牌对
    print(f"\n{'='*80}")
    print("【关键分析】找出胜率相近的手牌对，比较策略差异")
    print(f"{'='*80}")
    
    equity_pairs = []
    
    for i, p1 in enumerate(profiles):
        for p2 in profiles[i+1:]:
            equity_diff = abs(p1.hand_vs_range_equity - p2.hand_vs_range_equity)
            
            # 只关注胜率差异小于0.1的手牌对
            if equity_diff < 0.1:
                # 计算策略差异（TVD）
                strategy_diff = 0.5 * (
                    abs(p1.solver_strategy.get('check', 0) - p2.solver_strategy.get('check', 0)) +
                    abs(p1.solver_strategy.get('bet', 0) - p2.solver_strategy.get('bet', 0))
                )
                
                # 判断最优动作是否相同
                p1_best = 'bet' if p1.solver_strategy.get('bet', 0) > 0.5 else 'check'
                p2_best = 'bet' if p2.solver_strategy.get('bet', 0) > 0.5 else 'check'
                
                equity_pairs.append(EquityPair(
                    hand1=p1,
                    hand2=p2,
                    equity_diff=equity_diff,
                    strategy_diff=strategy_diff,
                    same_action=(p1_best == p2_best),
                ))
    
    # 按策略差异排序，找出差异最大的
    equity_pairs.sort(key=lambda x: -x.strategy_diff)
    
    if equity_pairs:
        print(f"\n找到 {len(equity_pairs)} 对胜率相近的手牌（胜率差<0.1）")
        print(f"\n【胜率相近但策略差异最大的手牌对】")
        
        for i, pair in enumerate(equity_pairs[:10]):  # 显示前10对
            print(f"\n--- 对比 {i+1} ---")
            print(f"手牌1: {pair.hand1.hand_str} ({pair.hand1.hand_type})")
            print(f"  胜率: {pair.hand1.hand_vs_range_equity:.4f}")
            print(f"  策略: check={pair.hand1.solver_strategy.get('check', 0):.4f}, "
                  f"bet={pair.hand1.solver_strategy.get('bet', 0):.4f}")
            
            print(f"手牌2: {pair.hand2.hand_str} ({pair.hand2.hand_type})")
            print(f"  胜率: {pair.hand2.hand_vs_range_equity:.4f}")
            print(f"  策略: check={pair.hand2.solver_strategy.get('check', 0):.4f}, "
                  f"bet={pair.hand2.solver_strategy.get('bet', 0):.4f}")
            
            print(f"胜率差异: {pair.equity_diff:.4f}")
            print(f"策略差异(TVD): {pair.strategy_diff:.4f}")
            print(f"最优动作相同: {'是' if pair.same_action else '否 ⚠️'}")
    else:
        print("未找到胜率相近的手牌对")
    
    # 统计分析
    print(f"\n{'='*80}")
    print("【统计分析】")
    print(f"{'='*80}")
    
    if equity_pairs:
        avg_strategy_diff = np.mean([p.strategy_diff for p in equity_pairs])
        max_strategy_diff = max(p.strategy_diff for p in equity_pairs)
        same_action_rate = sum(1 for p in equity_pairs if p.same_action) / len(equity_pairs)
        
        print(f"胜率相近手牌对数量: {len(equity_pairs)}")
        print(f"平均策略差异: {avg_strategy_diff:.4f}")
        print(f"最大策略差异: {max_strategy_diff:.4f}")
        print(f"最优动作一致率: {same_action_rate:.2%}")
        
        if max_strategy_diff > 0.3:
            print(f"\n⚠️ 发现显著差异: 胜率相近的手牌存在策略差异 > 0.3")
            print("   这表明胜率标量不足以完全决定最优策略")
        else:
            print(f"\n✓ 胜率相近的手牌策略也相近")
    
    return {
        'name': name,
        'profiles': [asdict(p) for p in profiles],
        'equity_pairs': [
            {
                'hand1': pair.hand1.hand_str,
                'hand2': pair.hand2.hand_str,
                'equity_diff': pair.equity_diff,
                'strategy_diff': pair.strategy_diff,
                'same_action': pair.same_action,
            }
            for pair in equity_pairs
        ],
    }


def main():
    print("=" * 80)
    print("胜率-策略相关性实验")
    print("=" * 80)
    print("\n核心问题: 当两个手牌具有相同的胜率时，它们的最优策略是否相同？")
    print("如果胜率能完全决定策略，那么相同胜率的手牌应该有相同的策略。")
    
    results = []
    
    # 场景1：设计一个有多个相同胜率手牌的场景
    # 使用更大的范围来增加找到相同胜率手牌的概率
    results.append(run_experiment(
        name="场景1_大范围对抗",
        description="使用较大范围，寻找胜率相近的手牌对",
        community_cards=[
            Card(rank=13, suit='s'),  # K♠
            Card(rank=10, suit='d'),  # T♦
            Card(rank=7, suit='c'),   # 7♣
            Card(rank=4, suit='h'),   # 4♥
            Card(rank=2, suit='s'),   # 2♠
        ],
        oop_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0,
            '99': 1.0, '88': 1.0, '77': 1.0, '66': 1.0, '55': 1.0,
            'AKs': 1.0, 'AKo': 1.0, 'AQs': 1.0, 'AQo': 1.0,
            'KQs': 1.0, 'KQo': 1.0, 'KJs': 1.0, 'QJs': 1.0,
            'JTs': 1.0, 'T9s': 1.0, '98s': 1.0, '87s': 1.0,
        },
        ip_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0,
            '99': 1.0, '88': 1.0, '77': 1.0,
            'AKs': 1.0, 'AKo': 1.0, 'AQs': 1.0,
            'KQs': 1.0, 'QJs': 1.0, 'JTs': 1.0,
        },
        cfr_iterations=1000,
    ))
    
    # 场景2：湿润牌面 - 多个顺子可能
    results.append(run_experiment(
        name="场景2_湿润牌面",
        description="连接性高的牌面，多种成牌可能",
        community_cards=[
            Card(rank=11, suit='s'),  # J♠
            Card(rank=10, suit='h'),  # T♥
            Card(rank=9, suit='d'),   # 9♦
            Card(rank=5, suit='c'),   # 5♣
            Card(rank=2, suit='s'),   # 2♠
        ],
        oop_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0,
            'AKs': 1.0, 'AQs': 1.0, 'KQs': 1.0,
            'QJs': 1.0, 'JTs': 1.0, 'T9s': 1.0,
            '98s': 1.0, '87s': 1.0, '76s': 1.0,
            'KJs': 1.0, 'QTs': 1.0, 'J9s': 1.0,
        },
        ip_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0,
            'AKs': 1.0, 'KQs': 1.0, 'QJs': 1.0,
            'JTs': 1.0, 'T9s': 1.0, '98s': 1.0,
            '87s': 1.0, 'KJs': 1.0, 'QTs': 1.0,
        },
        cfr_iterations=1000,
    ))
    
    # 场景3：同花牌面 - 阻断效应
    results.append(run_experiment(
        name="场景3_同花牌面",
        description="四张同花，阻断效应重要",
        community_cards=[
            Card(rank=14, suit='h'),  # A♥
            Card(rank=11, suit='h'),  # J♥
            Card(rank=8, suit='h'),   # 8♥
            Card(rank=5, suit='h'),   # 5♥
            Card(rank=2, suit='d'),   # 2♦
        ],
        oop_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0,
            'AKs': 1.0, 'AKo': 1.0, 'AQs': 1.0,
            'KhQh': 1.0, 'KhTh': 1.0, 'QhTh': 1.0,  # 有♥同花
            'Kh9h': 1.0, 'Th9h': 1.0, '9h7h': 1.0,
            'KsQs': 1.0, 'KsTs': 1.0,  # 无♥
        },
        ip_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0,
            'AKs': 1.0, 'AQs': 1.0,
            'KhQh': 1.0, 'QhTh': 1.0, 'Th9h': 1.0,
            'KsQs': 1.0, 'QsJs': 1.0,
        },
        cfr_iterations=1000,
    ))
    
    # 保存结果
    output_path = 'experiments/results/equity_strategy_correlation.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")
    
    # 总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)
    
    total_pairs = sum(len(r['equity_pairs']) for r in results)
    if total_pairs > 0:
        all_pairs = []
        for r in results:
            all_pairs.extend(r['equity_pairs'])
        
        avg_diff = np.mean([p['strategy_diff'] for p in all_pairs])
        max_diff = max(p['strategy_diff'] for p in all_pairs)
        same_rate = sum(1 for p in all_pairs if p['same_action']) / len(all_pairs)
        
        print(f"\n总共找到 {total_pairs} 对胜率相近的手牌")
        print(f"平均策略差异: {avg_diff:.4f}")
        print(f"最大策略差异: {max_diff:.4f}")
        print(f"最优动作一致率: {same_rate:.2%}")
        
        if max_diff > 0.3 or same_rate < 0.8:
            print("\n⚠️ 结论: 胜率相近的手牌可能有不同的最优策略")
            print("   这表明仅靠胜率标量不足以完全决定策略")
            print("   需要考虑其他因素（如牌型、阻断效应、范围结构等）")
        else:
            print("\n✓ 结论: 在测试场景中，胜率相近的手牌策略也相近")


if __name__ == '__main__':
    main()
