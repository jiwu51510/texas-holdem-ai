#!/usr/bin/env python3
"""综合胜率验证实验 - 完整输入输出数据。

生成所有显著差异的实验案例，包含：
1. 完整的输入数据（公共牌、范围、底池）
2. 详细的胜率计算过程
3. 手牌强度评估
4. 策略对比分析
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from models.core import Card
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    DeadCardRemover,
    RangeVsRangeCalculator,
)
from environment.hand_evaluator import HandEvaluator, compare_hands


def card_to_str(card: Card) -> str:
    """将Card对象转换为可读字符串。"""
    rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
                9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'}
    suit_map = {'s': '♠', 'd': '♦', 'c': '♣', 'h': '♥'}
    return f"{rank_map.get(card.rank, str(card.rank))}{suit_map.get(card.suit, card.suit)}"


def cards_to_str(cards: List[Card]) -> str:
    """将Card列表转换为可读字符串。"""
    return ' '.join(card_to_str(c) for c in cards)


def parse_hand_string(hand_str: str) -> Optional[Tuple[Card, Card]]:
    """解析手牌字符串。"""
    rank_map = {
        'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
        '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2
    }
    if len(hand_str) == 4:
        r1 = rank_map.get(hand_str[0].upper())
        s1 = hand_str[1].lower()
        r2 = rank_map.get(hand_str[2].upper())
        s2 = hand_str[3].lower()
        if r1 and r2 and s1 in 'hdcs' and s2 in 'hdcs':
            return (Card(rank=r1, suit=s1), Card(rank=r2, suit=s2))
    return None


def evaluate_hand_strength(hole_cards: Tuple[Card, Card], community_cards: List[Card]) -> Tuple[str, int]:
    """评估手牌强度。"""
    from models.core import HandRank
    
    all_cards = list(hole_cards) + community_cards
    
    # 获取最佳5张牌组合
    hand_rank, kickers = HandEvaluator.evaluate_hand(all_cards)
    
    hand_names = {
        HandRank.HIGH_CARD: "高牌",
        HandRank.PAIR: "一对",
        HandRank.TWO_PAIR: "两对",
        HandRank.THREE_OF_A_KIND: "三条",
        HandRank.STRAIGHT: "顺子",
        HandRank.FLUSH: "同花",
        HandRank.FULL_HOUSE: "葫芦",
        HandRank.FOUR_OF_A_KIND: "四条",
        HandRank.STRAIGHT_FLUSH: "同花顺",
    }
    
    return hand_names.get(hand_rank, f"类型{hand_rank}"), hand_rank.value


@dataclass
class HandAnalysis:
    """单手牌分析结果。"""
    hand_str: str
    hand_display: str
    hand_type: str
    hand_rank: int
    equity_vs_range: float
    matchup_details: List[Dict]


@dataclass
class ScenarioResult:
    """场景结果。"""
    name: str
    description: str
    community_cards_str: str
    community_cards_display: str
    pot_size: float
    bet_size: float
    oop_range: Dict[str, float]
    ip_range: Dict[str, float]
    oop_hand_analyses: List[HandAnalysis]
    summary: Dict


def run_comprehensive_analysis(
    name: str,
    description: str,
    community_cards: List[Card],
    oop_range: Dict[str, float],
    ip_range: Dict[str, float],
    pot_size: float = 100.0,
    bet_size: float = 50.0
) -> ScenarioResult:
    """运行综合分析。"""
    
    print(f"\n{'#'*80}")
    print(f"# 场景: {name}")
    print(f"# {description}")
    print(f"{'#'*80}")
    
    # 显示输入
    print(f"\n【公共牌】")
    print(f"  {cards_to_str(community_cards)}")
    
    print(f"\n【底池信息】")
    print(f"  底池大小: {pot_size}")
    print(f"  下注大小: {bet_size}")
    print(f"  底池赔率: {bet_size / (pot_size + 2 * bet_size):.4f}")
    
    print(f"\n【OOP范围】")
    for hand, weight in oop_range.items():
        print(f"  {hand}: 权重={weight}")
    
    print(f"\n【IP范围】")
    for hand, weight in ip_range.items():
        print(f"  {hand}: 权重={weight}")
    
    # 初始化计算器
    calculator = RangeVsRangeCalculator()
    remover = DeadCardRemover()
    
    # 展开范围
    oop_expanded = {}
    for hand, weight in oop_range.items():
        concrete = remover.expand_abstract_hand(hand)
        for c in concrete:
            cards = parse_hand_string(c)
            if cards:
                # 检查与公共牌冲突
                board_set = set((card.rank, card.suit) for card in community_cards)
                if not any((card.rank, card.suit) in board_set for card in cards):
                    oop_expanded[c] = weight / len(concrete)
    
    ip_expanded = {}
    for hand, weight in ip_range.items():
        concrete = remover.expand_abstract_hand(hand)
        for c in concrete:
            cards = parse_hand_string(c)
            if cards:
                board_set = set((card.rank, card.suit) for card in community_cards)
                if not any((card.rank, card.suit) in board_set for card in cards):
                    ip_expanded[c] = weight / len(concrete)
    
    print(f"\n【展开后的有效手牌】")
    print(f"  OOP有效手牌数: {len(oop_expanded)}")
    print(f"  IP有效手牌数: {len(ip_expanded)}")
    
    # 详细分析每个OOP手牌
    print(f"\n{'='*80}")
    print("详细手牌分析")
    print(f"{'='*80}")
    
    hand_analyses = []
    
    for oop_hand_str, oop_weight in sorted(oop_expanded.items()):
        oop_cards = parse_hand_string(oop_hand_str)
        if not oop_cards:
            continue
        
        # 评估手牌强度
        hand_type, hand_rank = evaluate_hand_strength(oop_cards, community_cards)
        
        # 计算对每个IP手牌的胜率
        matchups = []
        total_weight = 0
        weighted_wins = 0
        
        for ip_hand_str, ip_weight in ip_expanded.items():
            ip_cards = parse_hand_string(ip_hand_str)
            if not ip_cards:
                continue
            
            # 检查手牌冲突
            oop_set = set((c.rank, c.suit) for c in oop_cards)
            ip_set = set((c.rank, c.suit) for c in ip_cards)
            if oop_set & ip_set:
                continue
            
            # 比较手牌
            result = compare_hands(list(oop_cards), list(ip_cards), community_cards)
            
            if result == 0:  # OOP胜
                outcome = "胜"
                equity = 1.0
            elif result == -1:  # 平局
                outcome = "平"
                equity = 0.5
            else:  # IP胜
                outcome = "负"
                equity = 0.0
            
            ip_type, ip_rank = evaluate_hand_strength(ip_cards, community_cards)
            
            matchups.append({
                'ip_hand': ip_hand_str,
                'ip_type': ip_type,
                'outcome': outcome,
                'equity': equity,
                'weight': ip_weight,
            })
            
            weighted_wins += equity * ip_weight
            total_weight += ip_weight
        
        overall_equity = weighted_wins / total_weight if total_weight > 0 else 0.5
        
        analysis = HandAnalysis(
            hand_str=oop_hand_str,
            hand_display=f"{card_to_str(oop_cards[0])}{card_to_str(oop_cards[1])}",
            hand_type=hand_type,
            hand_rank=hand_rank,
            equity_vs_range=overall_equity,
            matchup_details=matchups,
        )
        hand_analyses.append(analysis)
        
        # 打印详细信息
        print(f"\n--- {oop_hand_str} ({analysis.hand_display}) ---")
        print(f"  牌型: {hand_type}")
        print(f"  对IP范围胜率: {overall_equity:.4f}")
        print(f"  对战详情:")
        
        for m in matchups:
            print(f"    vs {m['ip_hand']} ({m['ip_type']}): {m['outcome']} (权重={m['weight']:.4f})")
    
    # 汇总
    print(f"\n{'='*80}")
    print("汇总表格")
    print(f"{'='*80}")
    
    print(f"\n{'手牌':<12} {'牌型':<10} {'胜率':<10} {'建议动作':<10}")
    print("-" * 50)
    
    # 按胜率排序
    sorted_analyses = sorted(hand_analyses, key=lambda x: -x.equity_vs_range)
    
    for a in sorted_analyses:
        # 简单策略建议
        if a.equity_vs_range >= 0.7:
            action = "价值下注"
        elif a.equity_vs_range <= 0.3:
            action = "过牌/弃牌"
        else:
            action = "边缘决策"
        
        print(f"{a.hand_str:<12} {a.hand_type:<10} {a.equity_vs_range:<10.4f} {action:<10}")
    
    # 计算范围平均胜率
    avg_equity = sum(a.equity_vs_range for a in hand_analyses) / len(hand_analyses) if hand_analyses else 0
    
    summary = {
        'avg_equity': avg_equity,
        'num_hands': len(hand_analyses),
        'strong_hands': sum(1 for a in hand_analyses if a.equity_vs_range >= 0.7),
        'weak_hands': sum(1 for a in hand_analyses if a.equity_vs_range <= 0.3),
        'marginal_hands': sum(1 for a in hand_analyses if 0.3 < a.equity_vs_range < 0.7),
    }
    
    print(f"\n【范围汇总】")
    print(f"  平均胜率: {summary['avg_equity']:.4f}")
    print(f"  强牌数量 (胜率>=0.7): {summary['strong_hands']}")
    print(f"  弱牌数量 (胜率<=0.3): {summary['weak_hands']}")
    print(f"  边缘牌数量: {summary['marginal_hands']}")
    
    return ScenarioResult(
        name=name,
        description=description,
        community_cards_str=' '.join(f"{c.rank}{c.suit}" for c in community_cards),
        community_cards_display=cards_to_str(community_cards),
        pot_size=pot_size,
        bet_size=bet_size,
        oop_range=oop_range,
        ip_range=ip_range,
        oop_hand_analyses=hand_analyses,
        summary=summary,
    )


def main():
    """主函数。"""
    print("=" * 80)
    print("综合胜率验证实验 - 完整输入输出数据")
    print("=" * 80)
    
    results = []
    
    # 场景1：极化范围场景
    results.append(run_comprehensive_analysis(
        name="场景1_极化范围",
        description="干燥牌面K♠8♦2♣5♥9♠，OOP持有极化范围（坚果+空气）",
        community_cards=[
            Card(rank=13, suit='s'),
            Card(rank=8, suit='d'),
            Card(rank=2, suit='c'),
            Card(rank=5, suit='h'),
            Card(rank=9, suit='s'),
        ],
        oop_range={
            'AsAd': 1.0,  # AA - 坚果
            'KhKd': 1.0,  # KK - 顶set
            '3h4h': 1.0,  # 空气
            '6h7c': 1.0,  # 空气（但有顺子）
        },
        ip_range={
            'AhKh': 1.0,
            'KhQh': 1.0,
            'QsQd': 1.0,
            'JsJd': 1.0,
            'TsTd': 1.0,
        },
    ))
    
    # 场景2：湿润牌面
    results.append(run_comprehensive_analysis(
        name="场景2_湿润牌面",
        description="顺子牌面J♠T♠9♦8♣7♥，多种成牌可能",
        community_cards=[
            Card(rank=11, suit='s'),
            Card(rank=10, suit='s'),
            Card(rank=9, suit='d'),
            Card(rank=8, suit='c'),
            Card(rank=7, suit='h'),
        ],
        oop_range={
            'QsKs': 1.0,  # Q高顺子
            '6h5h': 1.0,  # 6高顺子
            'AsAd': 1.0,  # 超对
            'KhKd': 1.0,  # 超对
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
    
    # 场景3：同花牌面
    results.append(run_comprehensive_analysis(
        name="场景3_同花牌面",
        description="四张同花A♠K♠7♠5♠2♦，阻断效应重要",
        community_cards=[
            Card(rank=14, suit='s'),
            Card(rank=13, suit='s'),
            Card(rank=7, suit='s'),
            Card(rank=5, suit='s'),
            Card(rank=2, suit='d'),
        ],
        oop_range={
            'QsJd': 1.0,  # Q♠阻断
            'TsJd': 1.0,  # T♠阻断
            '9s8d': 1.0,  # 同花
            '6s4d': 1.0,  # 同花
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
    
    # 场景4：配对牌面
    results.append(run_comprehensive_analysis(
        name="场景4_配对牌面",
        description="配对A牌面A♠A♦K♣7♥2♠，葫芦可能",
        community_cards=[
            Card(rank=14, suit='s'),
            Card(rank=14, suit='d'),
            Card(rank=13, suit='c'),
            Card(rank=7, suit='h'),
            Card(rank=2, suit='s'),
        ],
        oop_range={
            'KsKd': 1.0,  # KK葫芦
            '7s7d': 1.0,  # 77葫芦
            'AhKh': 1.0,  # 三条A
            'AcQc': 1.0,  # 三条A
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
    
    # 场景5：边缘决策
    results.append(run_comprehensive_analysis(
        name="场景5_边缘决策",
        description="中等牌面Q♠J♦9♣7♥3♠，多个边缘牌",
        community_cards=[
            Card(rank=12, suit='s'),
            Card(rank=11, suit='d'),
            Card(rank=9, suit='c'),
            Card(rank=7, suit='h'),
            Card(rank=3, suit='s'),
        ],
        oop_range={
            'AsAd': 1.0,  # 超对
            'KhKd': 1.0,  # 超对
            'QhTh': 1.0,  # 顶对
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
    
    # 保存完整结果
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)
    
    output_data = []
    for r in results:
        scenario_data = {
            'name': r.name,
            'description': r.description,
            'community_cards': r.community_cards_display,
            'pot_size': r.pot_size,
            'bet_size': r.bet_size,
            'oop_range': r.oop_range,
            'ip_range': r.ip_range,
            'hand_analyses': [
                {
                    'hand': a.hand_str,
                    'display': a.hand_display,
                    'type': a.hand_type,
                    'equity': a.equity_vs_range,
                    'matchups': a.matchup_details,
                }
                for a in r.oop_hand_analyses
            ],
            'summary': r.summary,
        }
        output_data.append(scenario_data)
    
    output_path = 'experiments/results/comprehensive_validation.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"JSON结果已保存到: {output_path}")
    
    # 生成可读报告
    report_path = 'experiments/results/comprehensive_validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("综合胜率验证实验 - 完整输入输出报告\n")
        f.write("=" * 80 + "\n\n")
        
        for r in results:
            f.write(f"\n{'#'*80}\n")
            f.write(f"# {r.name}\n")
            f.write(f"# {r.description}\n")
            f.write(f"{'#'*80}\n\n")
            
            f.write(f"【输入】\n")
            f.write(f"  公共牌: {r.community_cards_display}\n")
            f.write(f"  底池: {r.pot_size}, 下注: {r.bet_size}\n\n")
            
            f.write(f"  OOP范围:\n")
            for hand, weight in r.oop_range.items():
                f.write(f"    {hand}: {weight}\n")
            
            f.write(f"\n  IP范围:\n")
            for hand, weight in r.ip_range.items():
                f.write(f"    {hand}: {weight}\n")
            
            f.write(f"\n【输出】\n")
            f.write(f"  {'手牌':<12} {'牌型':<10} {'胜率':<10}\n")
            f.write(f"  {'-'*40}\n")
            
            for a in sorted(r.oop_hand_analyses, key=lambda x: -x.equity_vs_range):
                f.write(f"  {a.hand_str:<12} {a.hand_type:<10} {a.equity_vs_range:<10.4f}\n")
            
            f.write(f"\n【汇总】\n")
            f.write(f"  平均胜率: {r.summary['avg_equity']:.4f}\n")
            f.write(f"  强牌: {r.summary['strong_hands']}, 弱牌: {r.summary['weak_hands']}, 边缘: {r.summary['marginal_hands']}\n")
            
            f.write(f"\n【详细对战】\n")
            for a in r.oop_hand_analyses:
                f.write(f"\n  {a.hand_str} ({a.hand_type}, 胜率={a.equity_vs_range:.4f}):\n")
                for m in a.matchup_details:
                    f.write(f"    vs {m['ip_hand']} ({m['ip_type']}): {m['outcome']}\n")
            
            f.write("\n")
    
    print(f"文本报告已保存到: {report_path}")
    
    # 最终汇总
    print("\n" + "=" * 80)
    print("最终汇总")
    print("=" * 80)
    
    print(f"\n{'场景':<25} {'平均胜率':<12} {'强牌':<8} {'弱牌':<8} {'边缘':<8}")
    print("-" * 65)
    
    for r in results:
        print(f"{r.name:<25} {r.summary['avg_equity']:<12.4f} "
              f"{r.summary['strong_hands']:<8} {r.summary['weak_hands']:<8} "
              f"{r.summary['marginal_hands']:<8}")


if __name__ == '__main__':
    main()
