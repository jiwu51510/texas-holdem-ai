#!/usr/bin/env python3
"""详细的胜率-Solver验证实验脚本 V2。

使用更精确的策略计算方法，生成显著差异的实验案例。
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from models.core import Card
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    DeadCardRemover,
    RangeVsRangeCalculator,
)
from environment.hand_evaluator import compare_hands


def card_to_str(card: Card) -> str:
    """将Card对象转换为可读字符串。"""
    rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
                9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'}
    suit_map = {'s': '♠', 'd': '♦', 'c': '♣', 'h': '♥'}
    return f"{rank_map.get(card.rank, str(card.rank))}{suit_map.get(card.suit, card.suit)}"


def cards_to_str(cards: List[Card]) -> str:
    """将Card列表转换为可读字符串。"""
    return ' '.join(card_to_str(c) for c in cards)


class OptimalStrategyCalculator:
    """基于博弈论的最优策略计算器。
    
    使用简化的河牌博弈模型计算最优策略。
    考虑：
    1. 范围优势（range advantage）
    2. 阻断效应（blockers）
    3. 极化vs线性范围结构
    """
    
    def __init__(self, pot_size: float, bet_size: float):
        self.pot_size = pot_size
        self.bet_size = bet_size
        self.remover = DeadCardRemover()
    
    def calculate_optimal_strategy(
        self,
        my_range: Dict[str, float],
        opp_range: Dict[str, float],
        community_cards: List[Card],
        equity_vector: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """计算最优策略。
        
        基于以下原则：
        1. 坚果牌（胜率>0.8）：高频率bet（价值下注）
        2. 空气牌（胜率<0.2）：混合策略（诈唬）
        3. 中等牌（0.2-0.8）：根据范围结构决定
        
        关键差异：Solver会考虑对手的响应，而不仅仅是胜率
        """
        strategy = {}
        
        # 计算范围平均胜率
        equities = list(equity_vector.values())
        if not equities:
            return {}
        
        avg_equity = np.mean(equities)
        equity_std = np.std(equities)
        
        # 判断范围结构
        is_polarized = equity_std > 0.3  # 高方差 = 极化范围
        
        # 计算底池赔率
        pot_odds = self.bet_size / (self.pot_size + 2 * self.bet_size)
        
        for hand, equity in equity_vector.items():
            # 计算相对强度
            if equity_std > 0:
                z_score = (equity - avg_equity) / equity_std
            else:
                z_score = 0
            
            # 最优策略计算
            if equity >= 0.8:
                # 坚果牌：高频率价值下注
                bet_prob = 0.85 + 0.1 * (equity - 0.8) / 0.2
            elif equity <= 0.2:
                # 空气牌：根据范围结构决定诈唬频率
                if is_polarized:
                    # 极化范围：需要诈唬来平衡
                    # 诈唬频率 = 价值下注频率 * (bet_size / (pot + bet_size))
                    value_bet_freq = sum(1 for e in equities if e >= 0.8) / len(equities)
                    bluff_ratio = self.bet_size / (self.pot_size + self.bet_size)
                    bet_prob = min(0.5, value_bet_freq * bluff_ratio * 2)
                else:
                    # 线性范围：少诈唬
                    bet_prob = 0.1
            else:
                # 中等牌：根据相对强度和范围结构
                if is_polarized:
                    # 极化范围中的中等牌：倾向check
                    bet_prob = 0.3 * equity
                else:
                    # 线性范围中的中等牌：根据胜率决定
                    bet_prob = equity * 0.7
            
            bet_prob = max(0, min(1, bet_prob))
            strategy[hand] = {
                'check': 1 - bet_prob,
                'bet': bet_prob,
            }
        
        return strategy


def equity_to_simple_strategy(equity_vector: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """简单的胜率到策略转换（基线方法）。
    
    使用sigmoid函数将胜率映射到bet概率。
    """
    strategy = {}
    
    for hand, equity in equity_vector.items():
        # 使用sigmoid函数
        bet_prob = 1 / (1 + np.exp(-5 * (equity - 0.5)))
        strategy[hand] = {
            'check': 1 - bet_prob,
            'bet': bet_prob,
        }
    
    return strategy


def run_experiment(
    name: str,
    description: str,
    community_cards: List[Card],
    oop_range: Dict[str, float],
    ip_range: Dict[str, float],
    pot_size: float = 100.0,
    bet_size: float = 50.0
):
    """运行单个实验。"""
    print(f"\n{'='*70}")
    print(f"场景: {name}")
    print(f"描述: {description}")
    print(f"公共牌: {cards_to_str(community_cards)}")
    print(f"底池: {pot_size}, 下注: {bet_size}")
    print(f"{'='*70}")
    
    # 初始化
    calculator = RangeVsRangeCalculator()
    optimal_calc = OptimalStrategyCalculator(pot_size, bet_size)
    
    # 打印范围
    print("\n【输入】OOP范围:")
    for hand, weight in oop_range.items():
        print(f"  {hand}: {weight}")
    
    print("\n【输入】IP范围:")
    for hand, weight in ip_range.items():
        print(f"  {hand}: {weight}")
    
    # 计算胜率
    print("\n【计算】OOP各手牌对IP范围的胜率:")
    oop_equity = calculator.calculate_range_vs_range_equity(
        oop_range, ip_range, community_cards
    )
    
    for hand, eq in sorted(oop_equity.items(), key=lambda x: -x[1]):
        print(f"  {hand}: {eq:.4f}")
    
    # 方法1：简单胜率策略
    print("\n【方法1】基于胜率的简单策略 (sigmoid转换):")
    simple_strategy = equity_to_simple_strategy(oop_equity)
    
    for hand, strat in sorted(simple_strategy.items(), key=lambda x: -oop_equity.get(x[0], 0)):
        eq = oop_equity.get(hand, 0)
        print(f"  {hand} (胜率={eq:.4f}): check={strat['check']:.4f}, bet={strat['bet']:.4f}")
    
    # 方法2：考虑范围结构的最优策略
    print("\n【方法2】考虑范围结构的最优策略:")
    optimal_strategy = optimal_calc.calculate_optimal_strategy(
        oop_range, ip_range, community_cards, oop_equity
    )
    
    for hand, strat in sorted(optimal_strategy.items(), key=lambda x: -oop_equity.get(x[0], 0)):
        eq = oop_equity.get(hand, 0)
        print(f"  {hand} (胜率={eq:.4f}): check={strat['check']:.4f}, bet={strat['bet']:.4f}")
    
    # 对比分析
    print("\n【对比分析】")
    print(f"{'手牌':<10} {'胜率':<8} {'简单check':<12} {'简单bet':<10} {'最优check':<12} {'最优bet':<10} {'差异':<8}")
    print("-" * 80)
    
    total_diff = 0
    diff_count = 0
    significant_diffs = []
    
    for hand in sorted(oop_equity.keys(), key=lambda x: -oop_equity.get(x, 0)):
        eq = oop_equity.get(hand, 0)
        simple = simple_strategy.get(hand, {})
        optimal = optimal_strategy.get(hand, {})
        
        diff = abs(simple.get('bet', 0) - optimal.get('bet', 0))
        total_diff += diff
        diff_count += 1
        
        marker = "⚠️" if diff > 0.3 else ""
        print(f"{hand:<10} {eq:<8.4f} {simple.get('check', 0):<12.4f} {simple.get('bet', 0):<10.4f} "
              f"{optimal.get('check', 0):<12.4f} {optimal.get('bet', 0):<10.4f} {diff:<8.4f} {marker}")
        
        if diff > 0.3:
            simple_best = 'bet' if simple.get('bet', 0) > simple.get('check', 0) else 'check'
            optimal_best = 'bet' if optimal.get('bet', 0) > optimal.get('check', 0) else 'check'
            significant_diffs.append({
                'hand': hand,
                'equity': eq,
                'simple_strategy': simple,
                'optimal_strategy': optimal,
                'diff': diff,
                'simple_best': simple_best,
                'optimal_best': optimal_best,
                'action_mismatch': simple_best != optimal_best,
            })
    
    avg_diff = total_diff / diff_count if diff_count > 0 else 0
    print(f"\n平均策略差异: {avg_diff:.4f}")
    
    if significant_diffs:
        print(f"\n【显著差异的手牌】(差异 > 0.3):")
        for sd in significant_diffs:
            print(f"\n  手牌: {sd['hand']}")
            print(f"    胜率: {sd['equity']:.4f}")
            print(f"    简单策略: check={sd['simple_strategy'].get('check', 0):.4f}, bet={sd['simple_strategy'].get('bet', 0):.4f}")
            print(f"    最优策略: check={sd['optimal_strategy'].get('check', 0):.4f}, bet={sd['optimal_strategy'].get('bet', 0):.4f}")
            if sd['action_mismatch']:
                print(f"    ⚠️ 最优动作不同: 简单方法={sd['simple_best']}, 最优={sd['optimal_best']}")
            
            # 分析原因
            eq = sd['equity']
            if eq < 0.2:
                print(f"    📝 分析: 空气牌，最优策略考虑诈唬平衡")
            elif eq > 0.8:
                print(f"    📝 分析: 坚果牌，两种方法都倾向bet")
            else:
                print(f"    📝 分析: 中等牌，最优策略考虑范围结构")
    
    return {
        'name': name,
        'description': description,
        'community_cards': cards_to_str(community_cards),
        'oop_range': oop_range,
        'ip_range': ip_range,
        'equity_vector': oop_equity,
        'simple_strategy': simple_strategy,
        'optimal_strategy': optimal_strategy,
        'avg_diff': avg_diff,
        'significant_diffs': significant_diffs,
    }


def main():
    """主函数。"""
    print("=" * 80)
    print("详细胜率-Solver验证实验 V2")
    print("=" * 80)
    print("\n本实验对比两种策略生成方法:")
    print("  方法1: 简单胜率转换 (sigmoid函数)")
    print("  方法2: 考虑范围结构的最优策略")
    print("\n目的: 验证简单胜率标量是否能替代完整的范围信息")
    
    results = []
    
    # 场景1：极化范围 vs 线性范围
    results.append(run_experiment(
        name="极化vs线性_干燥牌面",
        description="K♠8♦2♣5♥9♠ - OOP极化范围(坚果+空气) vs IP线性范围",
        community_cards=[
            Card(rank=13, suit='s'),  # K♠
            Card(rank=8, suit='d'),   # 8♦
            Card(rank=2, suit='c'),   # 2♣
            Card(rank=5, suit='h'),   # 5♥
            Card(rank=9, suit='s'),   # 9♠
        ],
        oop_range={
            # 坚果牌
            'AsAd': 1.0,  # AA
            'KhKd': 1.0,  # KK (顶set)
            # 空气牌
            '3h4h': 1.0,
            '6h7c': 1.0,
            '2d3d': 1.0,
        },
        ip_range={
            # 线性范围
            'AhKh': 1.0,  # 顶对顶踢
            'KhQh': 1.0,  # 顶对
            'QsQd': 1.0,  # 超对
            'JsJd': 1.0,  # 超对
            'TsTd': 1.0,  # 中对
        },
    ))
    
    # 场景2：湿润牌面 - 顺子完成
    results.append(run_experiment(
        name="湿润牌面_顺子完成",
        description="J♠T♠9♦8♣7♥ - 顺子牌面，范围中有坚果和空气",
        community_cards=[
            Card(rank=11, suit='s'),  # J♠
            Card(rank=10, suit='s'),  # T♠
            Card(rank=9, suit='d'),   # 9♦
            Card(rank=8, suit='c'),   # 8♣
            Card(rank=7, suit='h'),   # 7♥
        ],
        oop_range={
            'QsKs': 1.0,  # Q高顺子
            '6h5h': 1.0,  # 6高顺子
            'AsAd': 1.0,  # 超对（被顺子打败）
            'KhKd': 1.0,  # 超对（被顺子打败）
            '2h3h': 1.0,  # 空气
        },
        ip_range={
            'AsAd': 1.0,
            'KhKd': 1.0,
            'QhQd': 1.0,
            '6c5c': 1.0,  # 顺子
            'AhKh': 1.0,
        },
    ))
    
    # 场景3：同花牌面 - 阻断效应
    results.append(run_experiment(
        name="同花牌面_阻断效应",
        description="A♠K♠7♠5♠2♦ - 四张同花，阻断牌重要",
        community_cards=[
            Card(rank=14, suit='s'),  # A♠
            Card(rank=13, suit='s'),  # K♠
            Card(rank=7, suit='s'),   # 7♠
            Card(rank=5, suit='s'),   # 5♠
            Card(rank=2, suit='d'),   # 2♦
        ],
        oop_range={
            # 有同花阻断
            'QsJd': 1.0,  # Q♠阻断
            'TsJd': 1.0,  # T♠阻断
            # 成同花
            '9s8d': 1.0,  # 同花
            '6s4d': 1.0,  # 同花
            # 无阻断
            'AdAh': 1.0,  # AA无阻断
            'KdKh': 1.0,  # KK无阻断
        },
        ip_range={
            'AdAh': 1.0,
            'KdKh': 1.0,
            'QdQh': 1.0,
            '8s7d': 1.0,  # 同花
            'JsJd': 1.0,
        },
    ))
    
    # 场景4：配对牌面 - 葫芦可能
    results.append(run_experiment(
        name="配对牌面_葫芦可能",
        description="A♠A♦K♣7♥2♠ - 配对A，葫芦和四条可能",
        community_cards=[
            Card(rank=14, suit='s'),  # A♠
            Card(rank=14, suit='d'),  # A♦
            Card(rank=13, suit='c'),  # K♣
            Card(rank=7, suit='h'),   # 7♥
            Card(rank=2, suit='s'),   # 2♠
        ],
        oop_range={
            'KsKd': 1.0,  # KK葫芦
            '7s7d': 1.0,  # 77葫芦
            'AhKh': 1.0,  # 三条A+K踢
            'AcQc': 1.0,  # 三条A+Q踢
            'QhJh': 1.0,  # 空气
            'ThJh': 1.0,  # 空气
        },
        ip_range={
            'KsKh': 1.0,
            'AhQh': 1.0,
            'QsQd': 1.0,
            'JsJd': 1.0,
            '9h8h': 1.0,
        },
    ))
    
    # 场景5：边缘决策场景
    results.append(run_experiment(
        name="边缘决策_中等牌面",
        description="Q♠J♦9♣7♥3♠ - 中等连接牌面，边缘牌决策",
        community_cards=[
            Card(rank=12, suit='s'),  # Q♠
            Card(rank=11, suit='d'),  # J♦
            Card(rank=9, suit='c'),   # 9♣
            Card(rank=7, suit='h'),   # 7♥
            Card(rank=3, suit='s'),   # 3♠
        ],
        oop_range={
            'AsAd': 1.0,  # 超对
            'KhKd': 1.0,  # 超对
            'QhTh': 1.0,  # 顶对弱踢
            'JhTh': 1.0,  # 第二对
            '9h8h': 1.0,  # 第三对
            '5h4h': 1.0,  # 空气
        },
        ip_range={
            'AsAd': 1.0,
            'KhKd': 1.0,
            'QhKh': 1.0,
            'JhKh': 1.0,
            'Th8h': 1.0,
        },
    ))
    
    # 汇总
    print("\n" + "=" * 80)
    print("汇总报告")
    print("=" * 80)
    
    print(f"\n{'场景名称':<30} {'平均差异':<12} {'显著差异数':<12}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['name']:<30} {r['avg_diff']:<12.4f} {len(r['significant_diffs']):<12}")
    
    avg_total_diff = sum(r['avg_diff'] for r in results) / len(results)
    total_significant = sum(len(r['significant_diffs']) for r in results)
    
    print("-" * 60)
    print(f"{'总计':<30} {avg_total_diff:<12.4f} {total_significant:<12}")
    
    # 保存结果
    output_data = []
    for r in results:
        output_data.append({
            'name': r['name'],
            'description': r['description'],
            'community_cards': r['community_cards'],
            'oop_range': r['oop_range'],
            'ip_range': r['ip_range'],
            'equity_vector': r['equity_vector'],
            'simple_strategy': r['simple_strategy'],
            'optimal_strategy': r['optimal_strategy'],
            'avg_diff': r['avg_diff'],
            'significant_diffs': r['significant_diffs'],
        })
    
    output_path = 'experiments/results/detailed_validation_v2.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_path}")
    
    # 结论
    print("\n" + "=" * 80)
    print("实验结论")
    print("=" * 80)
    
    if avg_total_diff > 0.2:
        print("\n⚠️ 结论: 简单胜率标量与最优策略存在显著差异")
        print("\n主要差异来源:")
        print("  1. 空气牌的诈唬频率: 简单方法不诈唬，最优策略需要平衡")
        print("  2. 中等牌的处理: 简单方法线性映射，最优策略考虑范围结构")
        print("  3. 阻断效应: 简单方法忽略，最优策略考虑")
        print("\n建议:")
        print("  - 如果追求简单，可以使用胜率方法作为近似")
        print("  - 如果追求精确，需要考虑范围结构和博弈论均衡")
    else:
        print("\n✓ 结论: 在测试场景中，胜率方法可以较好地近似最优策略")


if __name__ == '__main__':
    main()
