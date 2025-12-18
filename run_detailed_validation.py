#!/usr/bin/env python3
"""详细的胜率-Solver验证实验脚本。

生成显著差异的实验案例，包含完整的输入和输出数据，方便验证。
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from models.core import Card
from experiments.equity_solver_validation.data_models import (
    SolverConfig,
    ExperimentScenario,
)
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    DeadCardRemover,
    RangeVsRangeCalculator,
)
from experiments.equity_solver_validation.solver_wrapper import SimpleCFRSolver
from experiments.equity_solver_validation.strategy_comparator import StrategyComparator
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


@dataclass
class DetailedExperimentResult:
    """详细实验结果。"""
    scenario_name: str
    description: str
    
    # 输入
    community_cards: str
    oop_range: Dict[str, float]
    ip_range: Dict[str, float]
    pot_size: float
    bet_size: float
    
    # 胜率计算结果
    oop_equity_vs_ip: Dict[str, float]  # OOP每手牌对IP范围的胜率
    
    # 基于胜率的策略
    equity_based_strategy: Dict[str, Dict[str, float]]
    
    # Solver策略
    solver_strategy: Dict[str, Dict[str, float]]
    
    # 对比结果
    total_variation_distance: float
    action_agreement_rate: float
    per_hand_diff: Dict[str, float]
    
    # 显著差异的手牌
    significant_diff_hands: List[Dict]


def create_detailed_scenarios() -> List[ExperimentScenario]:
    """创建用于详细分析的实验场景。"""
    config = SolverConfig(
        pot_size=100.0,
        effective_stack=200.0,
        oop_bet_sizes=[0.5],
        ip_bet_sizes=[0.5],
        oop_raise_sizes=[],
        ip_raise_sizes=[],
        max_iterations=200,
    )
    
    scenarios = []
    
    # 场景1：湿润牌面 - 连接性高，策略差异最大
    scenarios.append(ExperimentScenario(
        name="湿润牌面_JT987",
        description="J♠T♠9♦8♣7♥ - 顺子牌面，任何6或Q都成顺子",
        community_cards=[
            Card(rank=11, suit='s'),  # J♠
            Card(rank=10, suit='s'),  # T♠
            Card(rank=9, suit='d'),   # 9♦
            Card(rank=8, suit='c'),   # 8♣
            Card(rank=7, suit='h'),   # 7♥
        ],
        oop_range={
            # 坚果牌
            'QsKs': 1.0,  # 顺子
            '6h5h': 1.0,  # 顺子
            # 中等牌
            'AsAd': 1.0,  # 超对
            'KhKd': 1.0,  # 超对
            # 空气牌
            '2h3h': 1.0,
            '4h5d': 1.0,
        },
        ip_range={
            'AsAd': 1.0,
            'KhKd': 1.0,
            'QhQd': 1.0,
            'AhKh': 1.0,
            '6c5c': 1.0,  # 顺子
        },
        solver_config=config,
        tags=['wet_board', 'straight_board'],
    ))
    
    # 场景2：同花牌面 - 阻断效应明显
    scenarios.append(ExperimentScenario(
        name="同花牌面_AK752s",
        description="A♠K♠7♠5♠2♦ - 四张同花，阻断效应重要",
        community_cards=[
            Card(rank=14, suit='s'),  # A♠
            Card(rank=13, suit='s'),  # K♠
            Card(rank=7, suit='s'),   # 7♠
            Card(rank=5, suit='s'),   # 5♠
            Card(rank=2, suit='d'),   # 2♦
        ],
        oop_range={
            # 有同花阻断
            'QsJd': 1.0,  # 有Q♠阻断
            'TsJd': 1.0,  # 有T♠阻断
            # 无阻断的强牌
            'AdAh': 1.0,  # AA无阻断
            'KdKh': 1.0,  # KK无阻断
            # 成同花
            '9s8d': 1.0,  # 同花
            '6s4d': 1.0,  # 同花
        },
        ip_range={
            'AdAh': 1.0,
            'KdKh': 1.0,
            'QdQh': 1.0,
            'JsJd': 1.0,  # 有J♠
            '8s7d': 1.0,  # 同花
        },
        solver_config=config,
        tags=['flush_board', 'blocker_important'],
    ))
    
    # 场景3：配对牌面 - 葫芦可能
    scenarios.append(ExperimentScenario(
        name="配对牌面_AAK72",
        description="A♠A♦K♣7♥2♠ - 配对A，葫芦和四条可能",
        community_cards=[
            Card(rank=14, suit='s'),  # A♠
            Card(rank=14, suit='d'),  # A♦
            Card(rank=13, suit='c'),  # K♣
            Card(rank=7, suit='h'),   # 7♥
            Card(rank=2, suit='s'),   # 2♠
        ],
        oop_range={
            # 葫芦
            'KsKd': 1.0,  # KK full house
            '7s7d': 1.0,  # 77 full house
            # 三条A
            'AhKh': 1.0,  # trips with K kicker
            'AcQc': 1.0,  # trips with Q kicker
            # 空气
            'QhJh': 1.0,
            'ThJh': 1.0,
        },
        ip_range={
            'KsKh': 1.0,  # KK full house
            'AhQh': 1.0,  # trips
            'QsQd': 1.0,  # QQ
            'JsJd': 1.0,  # JJ
            '9h8h': 1.0,  # 空气
        },
        solver_config=config,
        tags=['paired_board', 'full_house_possible'],
    ))
    
    # 场景4：极化vs线性 - 范围结构差异
    scenarios.append(ExperimentScenario(
        name="极化vs线性_K82r",
        description="K♠8♦2♣5♥9♠ - 干燥牌面，测试范围结构",
        community_cards=[
            Card(rank=13, suit='s'),  # K♠
            Card(rank=8, suit='d'),   # 8♦
            Card(rank=2, suit='c'),   # 2♣
            Card(rank=5, suit='h'),   # 5♥
            Card(rank=9, suit='s'),   # 9♠
        ],
        oop_range={
            # 极化范围：坚果和空气
            'AsAd': 1.0,  # 坚果
            'KhKd': 1.0,  # 顶set
            '3h4h': 1.0,  # 空气
            '6h7h': 1.0,  # 空气
        },
        ip_range={
            # 线性范围：中等强度
            'AhKh': 1.0,  # 顶对顶踢
            'KhQh': 1.0,  # 顶对
            'QsQd': 1.0,  # 超对
            'JsJd': 1.0,  # 超对
            'TsTd': 1.0,  # 中对
        },
        solver_config=config,
        tags=['dry_board', 'polarized_vs_linear'],
    ))
    
    # 场景5：边缘牌决策 - 胜率接近阈值
    scenarios.append(ExperimentScenario(
        name="边缘决策_QJ973",
        description="Q♠J♦9♣7♥3♠ - 中等连接，边缘牌决策",
        community_cards=[
            Card(rank=12, suit='s'),  # Q♠
            Card(rank=11, suit='d'),  # J♦
            Card(rank=9, suit='c'),   # 9♣
            Card(rank=7, suit='h'),   # 7♥
            Card(rank=3, suit='s'),   # 3♠
        ],
        oop_range={
            # 强牌
            'AsAd': 1.0,
            'KhKd': 1.0,
            # 边缘牌
            'QhTh': 1.0,  # 顶对弱踢
            'JhTh': 1.0,  # 第二对
            # 弱牌
            '9h8h': 1.0,  # 第三对
            '5h4h': 1.0,  # 空气
        },
        ip_range={
            'AsAd': 1.0,
            'KhKd': 1.0,
            'QhKh': 1.0,  # 顶对强踢
            'JhKh': 1.0,  # 第二对
            'Th8h': 1.0,  # 顺子听牌完成
        },
        solver_config=config,
        tags=['marginal_decisions', 'medium_board'],
    ))
    
    return scenarios


def run_detailed_experiment(scenario: ExperimentScenario) -> DetailedExperimentResult:
    """运行单个详细实验。"""
    print(f"\n{'='*60}")
    print(f"场景: {scenario.name}")
    print(f"描述: {scenario.description}")
    print(f"公共牌: {cards_to_str(scenario.community_cards)}")
    print(f"{'='*60}")
    
    # 初始化组件
    calculator = RangeVsRangeCalculator()
    solver = SimpleCFRSolver(scenario.solver_config)
    
    pot_size = scenario.solver_config.pot_size
    bet_size = pot_size * scenario.solver_config.oop_bet_sizes[0]
    
    comparator = StrategyComparator(
        pot_size=pot_size,
        bet_size=bet_size,
    )
    
    # 计算OOP范围对IP范围的胜率
    print("\n--- 计算胜率 ---")
    oop_equity = calculator.calculate_range_vs_range_equity(
        scenario.oop_range,
        scenario.ip_range,
        scenario.community_cards
    )
    
    print(f"OOP范围中有效手牌数: {len(oop_equity)}")
    for hand, eq in sorted(oop_equity.items(), key=lambda x: -x[1]):
        print(f"  {hand}: 胜率 = {eq:.4f}")
    
    # 基于胜率生成策略
    print("\n--- 基于胜率的策略 ---")
    equity_strategy = comparator.equity_to_strategy(oop_equity, 'oop_root')
    
    for hand, strat in sorted(equity_strategy.items(), key=lambda x: -oop_equity.get(x[0], 0)):
        eq = oop_equity.get(hand, 0)
        print(f"  {hand} (胜率={eq:.4f}): check={strat['check']:.4f}, bet={strat['bet']:.4f}")
    
    # 运行Solver
    print("\n--- Solver策略 ---")
    solver_result = solver.solve(
        scenario.community_cards,
        scenario.oop_range,
        scenario.ip_range,
        iterations=200
    )
    
    solver_strategy = solver_result.strategies.get('root', {})
    
    for hand, strat in sorted(solver_strategy.items(), key=lambda x: -oop_equity.get(x[0], 0)):
        eq = oop_equity.get(hand, 0)
        print(f"  {hand} (胜率={eq:.4f}): check={strat.get('check', 0):.4f}, bet={strat.get('bet', 0):.4f}")
    
    # 对比策略
    print("\n--- 策略对比 ---")
    comparison = comparator.compare_strategies(equity_strategy, solver_strategy)
    
    print(f"总变差距离 (TVD): {comparison.metrics.total_variation_distance:.4f}")
    print(f"动作一致率: {comparison.metrics.action_agreement_rate:.4f}")
    
    # 找出显著差异的手牌
    significant_hands = []
    print("\n--- 显著差异的手牌 (TVD > 0.3) ---")
    
    for hand, diff in sorted(comparison.per_hand_diff.items(), key=lambda x: -x[1]):
        if diff > 0.3:
            eq = oop_equity.get(hand, 0)
            eq_strat = equity_strategy.get(hand, {})
            sol_strat = solver_strategy.get(hand, {})
            
            print(f"\n  手牌: {hand}")
            print(f"    胜率: {eq:.4f}")
            print(f"    胜率策略: check={eq_strat.get('check', 0):.4f}, bet={eq_strat.get('bet', 0):.4f}")
            print(f"    Solver策略: check={sol_strat.get('check', 0):.4f}, bet={sol_strat.get('bet', 0):.4f}")
            print(f"    差异 (TVD): {diff:.4f}")
            
            # 分析差异原因
            eq_best = max(eq_strat.items(), key=lambda x: x[1])[0] if eq_strat else 'N/A'
            sol_best = max(sol_strat.items(), key=lambda x: x[1])[0] if sol_strat else 'N/A'
            
            if eq_best != sol_best:
                print(f"    ⚠️ 最优动作不同: 胜率方法={eq_best}, Solver={sol_best}")
            
            significant_hands.append({
                'hand': hand,
                'equity': eq,
                'equity_strategy': eq_strat,
                'solver_strategy': sol_strat,
                'tvd': diff,
                'equity_best_action': eq_best,
                'solver_best_action': sol_best,
            })
    
    return DetailedExperimentResult(
        scenario_name=scenario.name,
        description=scenario.description,
        community_cards=cards_to_str(scenario.community_cards),
        oop_range=scenario.oop_range,
        ip_range=scenario.ip_range,
        pot_size=pot_size,
        bet_size=bet_size,
        oop_equity_vs_ip=oop_equity,
        equity_based_strategy=equity_strategy,
        solver_strategy=solver_strategy,
        total_variation_distance=comparison.metrics.total_variation_distance,
        action_agreement_rate=comparison.metrics.action_agreement_rate,
        per_hand_diff=comparison.per_hand_diff,
        significant_diff_hands=significant_hands,
    )


def main():
    """主函数。"""
    print("=" * 80)
    print("详细胜率-Solver验证实验")
    print("=" * 80)
    print("\n本实验验证：简单胜率标量是否能替代Solver所需的完整信息")
    print("输入信息：手牌、公共牌、对手范围、我的范围")
    print("输出：策略分布（check/bet概率）")
    
    scenarios = create_detailed_scenarios()
    results = []
    
    for scenario in scenarios:
        result = run_detailed_experiment(scenario)
        results.append(result)
    
    # 汇总报告
    print("\n" + "=" * 80)
    print("汇总报告")
    print("=" * 80)
    
    print("\n场景对比:")
    print(f"{'场景名称':<25} {'TVD':<10} {'一致率':<10} {'显著差异手牌数':<15}")
    print("-" * 60)
    
    for r in results:
        print(f"{r.scenario_name:<25} {r.total_variation_distance:<10.4f} "
              f"{r.action_agreement_rate:<10.4f} {len(r.significant_diff_hands):<15}")
    
    # 找出差异最大的场景
    worst = max(results, key=lambda x: x.total_variation_distance)
    print(f"\n差异最大的场景: {worst.scenario_name}")
    print(f"  TVD: {worst.total_variation_distance:.4f}")
    print(f"  显著差异手牌: {len(worst.significant_diff_hands)}")
    
    # 保存详细结果到JSON
    output_data = []
    for r in results:
        output_data.append({
            'scenario_name': r.scenario_name,
            'description': r.description,
            'community_cards': r.community_cards,
            'oop_range': r.oop_range,
            'ip_range': r.ip_range,
            'pot_size': r.pot_size,
            'bet_size': r.bet_size,
            'oop_equity_vs_ip': r.oop_equity_vs_ip,
            'equity_based_strategy': r.equity_based_strategy,
            'solver_strategy': r.solver_strategy,
            'total_variation_distance': r.total_variation_distance,
            'action_agreement_rate': r.action_agreement_rate,
            'per_hand_diff': r.per_hand_diff,
            'significant_diff_hands': r.significant_diff_hands,
        })
    
    output_path = 'experiments/results/detailed_validation_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_path}")
    
    # 生成可读报告
    report_path = 'experiments/results/detailed_validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("详细胜率-Solver验证实验报告\n")
        f.write("=" * 80 + "\n\n")
        
        for r in results:
            f.write(f"\n{'='*60}\n")
            f.write(f"场景: {r.scenario_name}\n")
            f.write(f"描述: {r.description}\n")
            f.write(f"公共牌: {r.community_cards}\n")
            f.write(f"底池: {r.pot_size}, 下注: {r.bet_size}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write("OOP范围:\n")
            for hand, weight in r.oop_range.items():
                f.write(f"  {hand}: {weight}\n")
            
            f.write("\nIP范围:\n")
            for hand, weight in r.ip_range.items():
                f.write(f"  {hand}: {weight}\n")
            
            f.write("\n--- 胜率计算结果 ---\n")
            for hand, eq in sorted(r.oop_equity_vs_ip.items(), key=lambda x: -x[1]):
                f.write(f"  {hand}: {eq:.4f}\n")
            
            f.write("\n--- 策略对比 ---\n")
            f.write(f"总变差距离: {r.total_variation_distance:.4f}\n")
            f.write(f"动作一致率: {r.action_agreement_rate:.4f}\n")
            
            f.write("\n--- 逐手牌对比 ---\n")
            f.write(f"{'手牌':<10} {'胜率':<8} {'胜率check':<12} {'胜率bet':<12} "
                   f"{'Solver check':<14} {'Solver bet':<12} {'TVD':<8}\n")
            f.write("-" * 80 + "\n")
            
            for hand in sorted(r.oop_equity_vs_ip.keys(), 
                             key=lambda x: -r.oop_equity_vs_ip.get(x, 0)):
                eq = r.oop_equity_vs_ip.get(hand, 0)
                eq_strat = r.equity_based_strategy.get(hand, {})
                sol_strat = r.solver_strategy.get(hand, {})
                diff = r.per_hand_diff.get(hand, 0)
                
                f.write(f"{hand:<10} {eq:<8.4f} {eq_strat.get('check', 0):<12.4f} "
                       f"{eq_strat.get('bet', 0):<12.4f} {sol_strat.get('check', 0):<14.4f} "
                       f"{sol_strat.get('bet', 0):<12.4f} {diff:<8.4f}\n")
            
            if r.significant_diff_hands:
                f.write("\n--- 显著差异分析 ---\n")
                for sh in r.significant_diff_hands:
                    f.write(f"\n手牌: {sh['hand']}\n")
                    f.write(f"  胜率: {sh['equity']:.4f}\n")
                    f.write(f"  胜率方法最优动作: {sh['equity_best_action']}\n")
                    f.write(f"  Solver最优动作: {sh['solver_best_action']}\n")
                    f.write(f"  TVD: {sh['tvd']:.4f}\n")
            
            f.write("\n")
    
    print(f"可读报告已保存到: {report_path}")
    
    print("\n" + "=" * 80)
    print("实验结论")
    print("=" * 80)
    avg_tvd = sum(r.total_variation_distance for r in results) / len(results)
    avg_agreement = sum(r.action_agreement_rate for r in results) / len(results)
    
    print(f"\n平均总变差距离: {avg_tvd:.4f}")
    print(f"平均动作一致率: {avg_agreement:.4f}")
    
    if avg_tvd > 0.4:
        print("\n⚠️ 结论: 简单胜率标量与Solver策略存在显著差异")
        print("   胜率方法无法完全替代Solver所需的完整信息")
        print("\n主要差异来源:")
        print("   1. 胜率方法忽略了范围结构（极化vs线性）")
        print("   2. 胜率方法忽略了阻断效应")
        print("   3. 胜率方法使用固定阈值，无法适应不同场景")
        print("   4. Solver考虑了博弈论均衡，胜率方法只考虑期望值")
    else:
        print("\n✓ 结论: 胜率方法在某些场景下可以近似Solver策略")


if __name__ == '__main__':
    main()
