#!/usr/bin/env python3
"""双维度胜率-策略相关性验证实验。

验证核心问题：当两个手牌同时满足以下条件时，它们的最优策略是否相同？
1. 手牌vs对手范围的胜率相同或接近
2. 自己范围vs对手范围的胜率相同或接近

这是更严格的实验设计，同时控制两个胜率维度。
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from models.core import Card
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    DeadCardRemover,
    RangeVsRangeCalculator,
)
from environment.hand_evaluator import HandEvaluator, compare_hands
from models.core import HandRank


def card_to_str(card: Card) -> str:
    """将Card对象转换为可读字符串。"""
    rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
                9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'}
    suit_map = {'s': '♠', 'd': '♦', 'c': '♣', 'h': '♥'}
    return f"{rank_map.get(card.rank, str(card.rank))}{suit_map.get(card.suit, card.suit)}"


def cards_to_str(cards: List[Card]) -> str:
    """将Card列表转换为可读字符串。"""
    return ' '.join(card_to_str(c) for c in cards)


def parse_hand(hand_str: str) -> Optional[Tuple[Card, Card]]:
    """解析手牌字符串为Card元组。"""
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


class DualEquityRiverGame:
    """河牌阶段博弈，同时计算双维度胜率。"""
    
    def __init__(
        self,
        community_cards: List[Card],
        oop_hands: List[str],
        ip_hands: List[str],
        pot_size: float = 100.0,
        bet_size: float = 50.0
    ):
        self.community_cards = community_cards
        self.oop_hands = oop_hands
        self.ip_hands = ip_hands
        self.pot_size = pot_size
        self.bet_size = bet_size
        
        # 计算胜率矩阵
        self.equity_matrix = self._compute_equity_matrix()
        
        # 计算每个手牌的胜率
        self.hand_equities = self._compute_hand_equities()
        
        # 计算范围vs范围胜率
        self.range_equity = self._compute_range_equity()
    
    def _compute_equity_matrix(self) -> np.ndarray:
        """计算OOP vs IP的胜率矩阵。"""
        n_oop = len(self.oop_hands)
        n_ip = len(self.ip_hands)
        matrix = np.zeros((n_oop, n_ip))
        
        for i, oop_h in enumerate(self.oop_hands):
            oop_cards = parse_hand(oop_h)
            if not oop_cards:
                continue
            
            for j, ip_h in enumerate(self.ip_hands):
                ip_cards = parse_hand(ip_h)
                if not ip_cards:
                    continue
                
                # 检查手牌冲突
                oop_set = set((c.rank, c.suit) for c in oop_cards)
                ip_set = set((c.rank, c.suit) for c in ip_cards)
                if oop_set & ip_set:
                    matrix[i, j] = np.nan  # 冲突时标记为nan
                    continue
                
                # 比较手牌
                result = compare_hands(list(oop_cards), list(ip_cards), self.community_cards)
                
                if result == 0:  # OOP胜
                    matrix[i, j] = 1.0
                elif result == -1:  # 平局
                    matrix[i, j] = 0.5
                else:  # IP胜
                    matrix[i, j] = 0.0
        
        return matrix

    def _compute_hand_equities(self) -> Dict[str, float]:
        """计算每个OOP手牌vs对手范围的胜率。"""
        equities = {}
        n_ip = len(self.ip_hands)
        
        for i, hand in enumerate(self.oop_hands):
            valid_matchups = ~np.isnan(self.equity_matrix[i])
            if valid_matchups.sum() > 0:
                equities[hand] = np.nanmean(self.equity_matrix[i])
            else:
                equities[hand] = 0.5
        
        return equities
    
    def _compute_range_equity(self) -> float:
        """计算OOP范围vs IP范围的整体胜率。"""
        valid = ~np.isnan(self.equity_matrix)
        if valid.sum() > 0:
            return np.nanmean(self.equity_matrix)
        return 0.5
    
    def solve_with_cfr(self, iterations: int = 10000) -> Dict[str, Dict[str, float]]:
        """使用CFR求解简化的河牌博弈。"""
        n_oop = len(self.oop_hands)
        n_ip = len(self.ip_hands)
        
        # 初始化遗憾值和策略累积
        oop_regrets = np.zeros((n_oop, 2))  # [check, bet]
        oop_strategy_sum = np.zeros((n_oop, 2))
        
        ip_regrets_vs_check = np.zeros((n_ip, 2))  # [check, bet]
        ip_regrets_vs_bet = np.zeros((n_ip, 2))  # [fold, call]
        ip_strategy_sum_vs_check = np.zeros((n_ip, 2))
        ip_strategy_sum_vs_bet = np.zeros((n_ip, 2))
        
        oop_regrets_vs_bet = np.zeros((n_oop, 2))  # [fold, call]
        oop_strategy_sum_vs_bet = np.zeros((n_oop, 2))
        
        pot = self.pot_size
        bet = self.bet_size
        
        # 处理nan值
        equity_matrix = np.nan_to_num(self.equity_matrix, nan=0.5)
        
        for t in range(iterations):
            oop_strategy = self._regret_matching(oop_regrets)
            ip_strategy_check = self._regret_matching(ip_regrets_vs_check)
            ip_strategy_bet = self._regret_matching(ip_regrets_vs_bet)
            oop_strategy_vs_bet = self._regret_matching(oop_regrets_vs_bet)
            
            oop_strategy_sum += oop_strategy
            ip_strategy_sum_vs_check += ip_strategy_check
            ip_strategy_sum_vs_bet += ip_strategy_bet
            oop_strategy_sum_vs_bet += oop_strategy_vs_bet

            # 计算OOP遗憾值
            for i in range(n_oop):
                ev_check = 0.0
                ev_bet = 0.0
                
                for j in range(n_ip):
                    equity = equity_matrix[i, j]
                    
                    ip_check_prob = ip_strategy_check[j, 0]
                    ip_bet_prob = ip_strategy_check[j, 1]
                    
                    ev_check_check = equity * pot
                    
                    oop_fold_prob = oop_strategy_vs_bet[i, 0]
                    oop_call_prob = oop_strategy_vs_bet[i, 1]
                    
                    ev_check_bet_fold = 0
                    ev_check_bet_call = equity * (pot + 2 * bet) - (1 - equity) * bet
                    ev_check_bet = oop_fold_prob * ev_check_bet_fold + oop_call_prob * ev_check_bet_call
                    
                    ev_check += (ip_check_prob * ev_check_check + ip_bet_prob * ev_check_bet) / n_ip
                    
                    ip_fold_prob = ip_strategy_bet[j, 0]
                    ip_call_prob = ip_strategy_bet[j, 1]
                    
                    ev_bet_fold = pot
                    ev_bet_call = equity * (pot + 2 * bet) - (1 - equity) * bet
                    
                    ev_bet += (ip_fold_prob * ev_bet_fold + ip_call_prob * ev_bet_call) / n_ip
                
                current_ev = oop_strategy[i, 0] * ev_check + oop_strategy[i, 1] * ev_bet
                oop_regrets[i, 0] += ev_check - current_ev
                oop_regrets[i, 1] += ev_bet - current_ev
            
            # 更新IP遗憾值
            for j in range(n_ip):
                ev_ip_check = 0.0
                ev_ip_bet = 0.0
                
                for i in range(n_oop):
                    equity = 1 - equity_matrix[i, j]
                    oop_check_prob = oop_strategy[i, 0]
                    
                    ev_ip_check += oop_check_prob * (equity * pot) / n_oop
                    
                    oop_fold = oop_strategy_vs_bet[i, 0]
                    oop_call = oop_strategy_vs_bet[i, 1]
                    
                    ev_ip_bet_fold = pot
                    ev_ip_bet_call = equity * (pot + 2 * bet) - (1 - equity) * bet
                    ev_ip_bet += oop_check_prob * (oop_fold * ev_ip_bet_fold + oop_call * ev_ip_bet_call) / n_oop
                
                current_ev_ip = ip_strategy_check[j, 0] * ev_ip_check + ip_strategy_check[j, 1] * ev_ip_bet
                ip_regrets_vs_check[j, 0] += ev_ip_check - current_ev_ip
                ip_regrets_vs_check[j, 1] += ev_ip_bet - current_ev_ip

                # 面对OOP bet
                ev_ip_fold = 0.0
                ev_ip_call = 0.0
                
                for i in range(n_oop):
                    equity = 1 - equity_matrix[i, j]
                    oop_bet_prob = oop_strategy[i, 1]
                    
                    ev_ip_fold += oop_bet_prob * 0 / n_oop
                    ev_ip_call += oop_bet_prob * (equity * (pot + 2 * bet) - (1 - equity) * bet) / n_oop
                
                current_ev_ip_bet = ip_strategy_bet[j, 0] * ev_ip_fold + ip_strategy_bet[j, 1] * ev_ip_call
                ip_regrets_vs_bet[j, 0] += ev_ip_fold - current_ev_ip_bet
                ip_regrets_vs_bet[j, 1] += ev_ip_call - current_ev_ip_bet
        
        # 计算平均策略
        oop_avg_strategy = {}
        for i, hand in enumerate(self.oop_hands):
            total = oop_strategy_sum[i].sum()
            if total > 0:
                probs = oop_strategy_sum[i] / total
            else:
                probs = np.array([0.5, 0.5])
            oop_avg_strategy[hand] = {'check': float(probs[0]), 'bet': float(probs[1])}
        
        return oop_avg_strategy
    
    def _regret_matching(self, regrets: np.ndarray) -> np.ndarray:
        """遗憾匹配算法。"""
        positive = np.maximum(regrets, 0)
        sums = positive.sum(axis=1, keepdims=True)
        sums = np.where(sums > 0, sums, 1)
        strategy = positive / sums
        uniform = np.ones_like(strategy) / strategy.shape[1]
        strategy = np.where(positive.sum(axis=1, keepdims=True) > 0, strategy, uniform)
        return strategy


def expand_range(range_dict: Dict[str, float], community_cards: List[Card]) -> List[str]:
    """展开范围为具体手牌列表。"""
    remover = DeadCardRemover()
    expanded = []
    board_set = set((card.rank, card.suit) for card in community_cards)
    
    for hand, weight in range_dict.items():
        concrete = remover.expand_abstract_hand(hand)
        for c in concrete:
            cards = parse_hand(c)
            if cards:
                if not any((card.rank, card.suit) in board_set for card in cards):
                    expanded.append(c)
    
    return list(set(expanded))


def run_dual_equity_experiment(
    name: str,
    description: str,
    community_cards: List[Card],
    oop_range: Dict[str, float],
    ip_range: Dict[str, float],
    pot_size: float = 100.0,
    bet_size: float = 50.0,
    cfr_iterations: int = 5000,
    equity_threshold: float = 0.03  # 胜率差异阈值
):
    """运行双维度胜率验证实验。
    
    核心：找出同时满足以下条件的手牌对：
    1. 手牌vs对手范围的胜率相近（差异<threshold）
    2. 它们所在的范围vs对手范围的胜率相同（因为是同一个范围）
    
    然后比较这些手牌对的策略是否相同。
    """
    
    print(f"\n{'='*80}")
    print(f"实验: {name}")
    print(f"描述: {description}")
    print(f"{'='*80}")
    
    print(f"\n【输入参数】")
    print(f"公共牌: {cards_to_str(community_cards)}")
    print(f"底池: {pot_size}, 下注: {bet_size}")
    print(f"胜率差异阈值: {equity_threshold*100:.1f}%")
    
    # 展开范围
    oop_expanded = expand_range(oop_range, community_cards)
    ip_expanded = expand_range(ip_range, community_cards)
    
    print(f"\nOOP有效手牌数: {len(oop_expanded)}")
    print(f"IP有效手牌数: {len(ip_expanded)}")
    
    # 创建博弈并求解
    print(f"\n【运行CFR求解器】(迭代次数: {cfr_iterations})")
    game = DualEquityRiverGame(
        community_cards=community_cards,
        oop_hands=oop_expanded,
        ip_hands=ip_expanded,
        pot_size=pot_size,
        bet_size=bet_size
    )
    
    solver_strategies = game.solve_with_cfr(iterations=cfr_iterations)
    
    # 范围vs范围胜率（所有手牌共享这个值）
    range_equity = game.range_equity
    print(f"\n【范围vs范围胜率】: {range_equity:.4f}")
    
    # 收集所有手牌的数据
    results = []
    for hand in oop_expanded:
        cards = parse_hand(hand)
        if not cards:
            continue
        
        hand_type = get_hand_type(cards, community_cards)
        hand_equity = game.hand_equities.get(hand, 0.5)
        strategy = solver_strategies.get(hand, {'check': 0.5, 'bet': 0.5})
        
        results.append({
            'hand': hand,
            'type': hand_type,
            'hand_equity': hand_equity,  # 手牌vs对手范围
            'range_equity': range_equity,  # 范围vs范围（所有手牌相同）
            'check': strategy['check'],
            'bet': strategy['bet'],
        })
    
    # 按手牌胜率排序
    results.sort(key=lambda x: x['hand_equity'])

    # 打印所有手牌数据
    print(f"\n【所有手牌的双维度胜率和策略】")
    print(f"{'手牌':<10} {'牌型':<8} {'手牌胜率':<10} {'范围胜率':<10} {'check':<10} {'bet':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['hand']:<10} {r['type']:<8} {r['hand_equity']:<10.4f} "
              f"{r['range_equity']:<10.4f} {r['check']:<10.4f} {r['bet']:<10.4f}")
    
    # 找出手牌胜率相近的手牌对
    # 注意：范围胜率对所有手牌都相同，所以只需要比较手牌胜率
    print(f"\n{'='*80}")
    print("【关键分析】找出双维度胜率都相近的手牌对")
    print(f"条件: 手牌胜率差异 < {equity_threshold*100:.1f}%")
    print(f"注意: 范围胜率对所有手牌相同 = {range_equity:.4f}")
    print(f"{'='*80}")
    
    pairs = []
    for i, r1 in enumerate(results):
        for r2 in results[i+1:]:
            hand_equity_diff = abs(r1['hand_equity'] - r2['hand_equity'])
            
            # 只有手牌胜率相近的才比较
            if hand_equity_diff < equity_threshold:
                strategy_diff = 0.5 * (abs(r1['check'] - r2['check']) + abs(r1['bet'] - r2['bet']))
                
                r1_best = 'bet' if r1['bet'] > 0.5 else 'check'
                r2_best = 'bet' if r2['bet'] > 0.5 else 'check'
                
                pairs.append({
                    'hand1': r1['hand'],
                    'hand2': r2['hand'],
                    'type1': r1['type'],
                    'type2': r2['type'],
                    'hand_equity1': r1['hand_equity'],
                    'hand_equity2': r2['hand_equity'],
                    'hand_equity_diff': hand_equity_diff,
                    'range_equity': range_equity,
                    'strategy_diff': strategy_diff,
                    'same_action': r1_best == r2_best,
                    'strategy1': {'check': r1['check'], 'bet': r1['bet']},
                    'strategy2': {'check': r2['check'], 'bet': r2['bet']},
                })
    
    # 按策略差异排序
    pairs.sort(key=lambda x: -x['strategy_diff'])
    
    if pairs:
        print(f"\n找到 {len(pairs)} 对双维度胜率相近的手牌")
        
        # 显示策略差异显著的
        significant = [p for p in pairs if p['strategy_diff'] > 0.1]
        
        if significant:
            print(f"\n【策略差异显著的手牌对】(策略差异>10%)")
            print(f"这些手牌对满足: 手牌胜率相近 + 范围胜率相同，但策略不同！")
            print("-" * 80)

            for p in significant[:15]:
                print(f"\n  手牌1: {p['hand1']} ({p['type1']})")
                print(f"  手牌2: {p['hand2']} ({p['type2']})")
                print(f"  手牌胜率: {p['hand_equity1']:.4f} vs {p['hand_equity2']:.4f} (差异: {p['hand_equity_diff']:.4f})")
                print(f"  范围胜率: {p['range_equity']:.4f} (相同)")
                print(f"  策略1: check={p['strategy1']['check']:.4f}, bet={p['strategy1']['bet']:.4f}")
                print(f"  策略2: check={p['strategy2']['check']:.4f}, bet={p['strategy2']['bet']:.4f}")
                print(f"  策略差异: {p['strategy_diff']:.4f}")
                if not p['same_action']:
                    print(f"  ⚠️ 最优动作完全不同!")
        else:
            print("\n✓ 没有发现策略差异显著的手牌对")
        
        # 统计
        avg_diff = np.mean([p['strategy_diff'] for p in pairs])
        max_diff = max(p['strategy_diff'] for p in pairs)
        same_rate = sum(1 for p in pairs if p['same_action']) / len(pairs) if pairs else 1.0
        
        print(f"\n【统计汇总】")
        print(f"  双维度胜率相近的手牌对数: {len(pairs)}")
        print(f"  策略差异显著的对数: {len(significant)}")
        print(f"  平均策略差异: {avg_diff:.4f}")
        print(f"  最大策略差异: {max_diff:.4f}")
        print(f"  最优动作一致率: {same_rate:.2%}")
    else:
        print("\n没有找到双维度胜率相近的手牌对")
    
    return {
        'name': name,
        'range_equity': range_equity,
        'results': results,
        'pairs': pairs,
    }


def main():
    print("=" * 80)
    print("双维度胜率-策略相关性验证实验")
    print("=" * 80)
    print("\n核心问题: 当两个手牌同时满足以下条件时，它们的最优策略是否相同？")
    print("  1. 手牌vs对手范围的胜率相同或接近")
    print("  2. 自己范围vs对手范围的胜率相同（因为是同一个范围）")
    
    all_results = []

    # 场景1：干燥牌面 - 大范围
    all_results.append(run_dual_equity_experiment(
        name="场景1_干燥牌面_大范围",
        description="K♠T♦7♣4♥2♠ - 干燥牌面，包含多种牌力",
        community_cards=[
            Card(rank=13, suit='s'),
            Card(rank=10, suit='d'),
            Card(rank=7, suit='c'),
            Card(rank=4, suit='h'),
            Card(rank=2, suit='s'),
        ],
        oop_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0,
            '99': 1.0, '88': 1.0, '77': 1.0, '66': 1.0, '55': 1.0,
            'AKs': 1.0, 'AKo': 1.0, 'KQs': 1.0, 'KQo': 1.0,
            'AQs': 1.0, 'AJs': 1.0, 'KJs': 1.0, 'QJs': 1.0,
        },
        ip_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0,
            '99': 1.0, '88': 1.0,
            'AKs': 1.0, 'AKo': 1.0, 'KQs': 1.0,
            'AQs': 1.0, 'AJs': 1.0, 'QJs': 1.0,
        },
        equity_threshold=0.03,
        cfr_iterations=2000,
    ))
    
    # 场景2：湿润牌面 - 顺子可能
    all_results.append(run_dual_equity_experiment(
        name="场景2_湿润牌面_顺子可能",
        description="J♠T♥9♦5♣2♠ - 连接牌面，顺子和两对可能",
        community_cards=[
            Card(rank=11, suit='s'),
            Card(rank=10, suit='h'),
            Card(rank=9, suit='d'),
            Card(rank=5, suit='c'),
            Card(rank=2, suit='s'),
        ],
        oop_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0,
            '99': 1.0, '88': 1.0, '77': 1.0,
            'KQs': 1.0, 'KQo': 1.0, 'QJs': 1.0, 'JTs': 1.0,
            '87s': 1.0, '76s': 1.0, 'Q8s': 1.0,
        },
        ip_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0,
            '99': 1.0,
            'KQs': 1.0, 'QJs': 1.0, 'JTs': 1.0,
            '87s': 1.0, 'Q8s': 1.0,
        },
        equity_threshold=0.03,
    ))

    # 场景3：同花牌面 - 阻断效应
    all_results.append(run_dual_equity_experiment(
        name="场景3_同花牌面_阻断效应",
        description="A♥J♥8♥5♥2♦ - 四张同花，阻断效应明显",
        community_cards=[
            Card(rank=14, suit='h'),
            Card(rank=11, suit='h'),
            Card(rank=8, suit='h'),
            Card(rank=5, suit='h'),
            Card(rank=2, suit='d'),
        ],
        oop_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0,
            'KhQh': 1.0, 'KhTh': 1.0, 'QhTh': 1.0, 'Th9h': 1.0,
            'KsQs': 1.0, 'KsTs': 1.0, 'QsTs': 1.0,
            'KdQd': 1.0, 'KdTd': 1.0,
        },
        ip_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0,
            'KhQh': 1.0, 'QhTh': 1.0,
            'KsQs': 1.0, 'KdQd': 1.0,
        },
        equity_threshold=0.03,
    ))
    
    # 场景4：配对牌面 - 葫芦可能
    all_results.append(run_dual_equity_experiment(
        name="场景4_配对牌面_葫芦可能",
        description="K♠K♦9♣5♥2♠ - 配对牌面，葫芦和三条可能",
        community_cards=[
            Card(rank=13, suit='s'),
            Card(rank=13, suit='d'),
            Card(rank=9, suit='c'),
            Card(rank=5, suit='h'),
            Card(rank=2, suit='s'),
        ],
        oop_range={
            'AA': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0, '99': 1.0,
            '88': 1.0, '77': 1.0, '66': 1.0, '55': 1.0,
            'AKs': 1.0, 'AKo': 1.0, 'KQs': 1.0, 'KJs': 1.0,
            'AQs': 1.0, 'AJs': 1.0, 'A9s': 1.0, 'A5s': 1.0,
        },
        ip_range={
            'AA': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0, '99': 1.0,
            'AKs': 1.0, 'AKo': 1.0, 'KQs': 1.0,
            'AQs': 1.0, 'AJs': 1.0, 'A9s': 1.0,
        },
        equity_threshold=0.03,
    ))

    # 保存结果
    output_path = 'experiments/results/dual_equity_validation.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")
    
    # 生成详细报告
    generate_report(all_results)


def generate_report(all_results: List[Dict]):
    """生成详细的实验报告。"""
    
    report = []
    report.append("# 双维度胜率-策略相关性验证实验报告\n")
    report.append("## 实验目的\n")
    report.append("验证：**当两个手牌同时满足以下条件时，它们的最优策略是否相同？**\n")
    report.append("1. 手牌vs对手范围的胜率相同或接近\n")
    report.append("2. 自己范围vs对手范围的胜率相同（因为是同一个范围）\n\n")
    
    report.append("## 实验方法\n")
    report.append("1. 使用CFR算法计算河牌阶段的纳什均衡策略\n")
    report.append("2. 计算每个手牌的两个胜率维度：\n")
    report.append("   - 手牌vs对手范围的胜率\n")
    report.append("   - 自己范围vs对手范围的胜率（所有手牌共享）\n")
    report.append("3. 找出双维度胜率都相近的手牌对\n")
    report.append("4. 比较这些手牌对的策略差异\n\n")
    
    report.append("## 关键发现\n\n")
    
    total_pairs = 0
    total_significant = 0
    
    for result in all_results:
        name = result['name']
        range_equity = result['range_equity']
        pairs = result['pairs']
        
        report.append(f"### {name}\n\n")
        report.append(f"**范围vs范围胜率**: {range_equity:.4f}\n\n")
        
        significant = [p for p in pairs if p['strategy_diff'] > 0.1]
        total_pairs += len(pairs)
        total_significant += len(significant)
        
        if significant:
            report.append("**发现双维度胜率相近但策略不同的手牌对：**\n\n")
            report.append("| 手牌1 | 手牌2 | 手牌胜率1 | 手牌胜率2 | 策略1 | 策略2 | 策略差异 |\n")
            report.append("|-------|-------|-----------|-----------|-------|-------|----------|\n")
            
            for p in significant[:10]:
                s1 = f"bet {p['strategy1']['bet']*100:.0f}%" if p['strategy1']['bet'] > 0.5 else f"check {p['strategy1']['check']*100:.0f}%"
                s2 = f"bet {p['strategy2']['bet']*100:.0f}%" if p['strategy2']['bet'] > 0.5 else f"check {p['strategy2']['check']*100:.0f}%"
                report.append(f"| {p['hand1']} ({p['type1']}) | {p['hand2']} ({p['type2']}) | "
                            f"{p['hand_equity1']:.2%} | {p['hand_equity2']:.2%} | "
                            f"{s1} | {s2} | {p['strategy_diff']:.2%} |\n")
            report.append("\n")
        else:
            report.append("✓ 没有发现策略差异显著的手牌对\n\n")

    report.append("## 统计汇总\n\n")
    report.append("| 场景 | 范围胜率 | 双维度相近对数 | 策略差异显著数 |\n")
    report.append("|------|----------|----------------|----------------|\n")
    
    for result in all_results:
        pairs = result['pairs']
        significant = [p for p in pairs if p['strategy_diff'] > 0.1]
        report.append(f"| {result['name']} | {result['range_equity']:.2%} | "
                     f"{len(pairs)} | {len(significant)} |\n")
    
    report.append(f"\n**总计**: {total_pairs} 对双维度胜率相近的手牌，其中 {total_significant} 对策略差异显著\n\n")
    
    report.append("## 核心结论\n\n")
    
    if total_significant > 0:
        report.append("### ⚠️ 双维度胜率标量不足以决定最优策略\n\n")
        report.append("实验发现大量满足以下条件的手牌对：\n")
        report.append("- 手牌vs对手范围的胜率相近\n")
        report.append("- 范围vs范围的胜率相同\n")
        report.append("- **但策略完全不同！**\n\n")
        report.append("这证明了仅靠两个胜率标量无法替代传统solver所需的完整信息。\n\n")
        report.append("### 需要的额外信息\n\n")
        report.append("除了胜率标量，最优策略还需要考虑：\n")
        report.append("1. **绝对牌力**：手牌的牌型（三条 > 两对 > 一对）\n")
        report.append("2. **相对牌力**：在当前牌面的相对强度\n")
        report.append("3. **阻断效应**：是否阻断对手的强牌\n")
        report.append("4. **范围结构**：自己范围是极化还是线性\n")
    else:
        report.append("### ✓ 双维度胜率可能足以决定策略\n\n")
        report.append("在测试的场景中，双维度胜率相近的手牌策略也相近。\n")
    
    report.append("\n## 实验数据文件\n\n")
    report.append("- 完整结果：`experiments/results/dual_equity_validation.json`\n")
    report.append("- 本报告：`experiments/results/dual_equity_validation_report.md`\n")
    
    # 保存报告
    report_path = 'experiments/results/dual_equity_validation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    print(f"\n报告已保存到: {report_path}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)
    print(f"\n总共找到 {total_pairs} 对双维度胜率相近的手牌")
    print(f"其中策略差异显著的: {total_significant} 对")
    
    if total_significant > 0:
        print("\n⚠️ 结论: 发现双维度胜率相近但策略不同的手牌对")
        print("   这表明仅靠两个胜率标量不足以完全决定最优策略")
    else:
        print("\n✓ 结论: 双维度胜率相近的手牌策略也相近")


if __name__ == '__main__':
    main()
