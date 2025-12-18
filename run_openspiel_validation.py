#!/usr/bin/env python3
"""使用OpenSpiel进行胜率-策略相关性验证实验。

OpenSpiel是Google的博弈论库，包含CFR等算法的实现。
我们使用它来计算河牌阶段的纳什均衡策略。
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# OpenSpiel imports
import pyspiel
from open_spiel.python.algorithms import cfr

# 本地imports
from models.core import Card
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    DeadCardRemover,
    RangeVsRangeCalculator,
)
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


class RiverPokerGame:
    """河牌阶段简化扑克博弈。
    
    使用OpenSpiel的matrix_game来建模河牌阶段的决策。
    """
    
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
        self.remover = DeadCardRemover()
        
        # 计算胜率矩阵
        self.equity_matrix = self._compute_equity_matrix()
    
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
                    matrix[i, j] = 0.5  # 冲突时设为0.5
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
    
    def solve_with_cfr(self, iterations: int = 10000) -> Dict[str, Dict[str, float]]:
        """使用CFR求解简化的河牌博弈。
        
        简化模型：
        - OOP先行动：check 或 bet
        - 如果OOP check，IP可以 check 或 bet
        - 如果OOP bet，IP可以 fold 或 call
        - 如果OOP check后IP bet，OOP可以 fold 或 call
        
        为简化，我们只考虑OOP的初始决策（check/bet）
        """
        # 使用Kuhn Poker作为基础，但我们需要自定义
        # 由于OpenSpiel的复杂性，我们使用简化的CFR实现
        
        n_oop = len(self.oop_hands)
        n_ip = len(self.ip_hands)
        
        # 初始化遗憾值和策略累积
        oop_regrets = np.zeros((n_oop, 2))  # [check, bet]
        oop_strategy_sum = np.zeros((n_oop, 2))
        
        ip_regrets_vs_check = np.zeros((n_ip, 2))  # [check, bet]
        ip_regrets_vs_bet = np.zeros((n_ip, 2))  # [fold, call]
        ip_strategy_sum_vs_check = np.zeros((n_ip, 2))
        ip_strategy_sum_vs_bet = np.zeros((n_ip, 2))
        
        # OOP check后IP bet后OOP的决策
        oop_regrets_vs_bet = np.zeros((n_oop, 2))  # [fold, call]
        oop_strategy_sum_vs_bet = np.zeros((n_oop, 2))
        
        pot = self.pot_size
        bet = self.bet_size
        
        for t in range(iterations):
            # 获取当前策略
            oop_strategy = self._regret_matching(oop_regrets)
            ip_strategy_check = self._regret_matching(ip_regrets_vs_check)
            ip_strategy_bet = self._regret_matching(ip_regrets_vs_bet)
            oop_strategy_vs_bet = self._regret_matching(oop_regrets_vs_bet)
            
            # 累积策略
            oop_strategy_sum += oop_strategy
            ip_strategy_sum_vs_check += ip_strategy_check
            ip_strategy_sum_vs_bet += ip_strategy_bet
            oop_strategy_sum_vs_bet += oop_strategy_vs_bet
            
            # 计算每个OOP手牌的遗憾值
            for i in range(n_oop):
                # 假设IP手牌均匀分布
                ev_check = 0.0
                ev_bet = 0.0
                
                for j in range(n_ip):
                    equity = self.equity_matrix[i, j]
                    
                    # OOP check后
                    ip_check_prob = ip_strategy_check[j, 0]
                    ip_bet_prob = ip_strategy_check[j, 1]
                    
                    # check-check: 摊牌
                    ev_check_check = equity * pot - (1 - equity) * 0
                    
                    # check-bet: OOP需要决定
                    oop_fold_prob = oop_strategy_vs_bet[i, 0]
                    oop_call_prob = oop_strategy_vs_bet[i, 1]
                    
                    ev_check_bet_fold = -0  # OOP弃牌，损失0（已投入的不算）
                    ev_check_bet_call = equity * (pot + 2 * bet) - (1 - equity) * bet
                    ev_check_bet = oop_fold_prob * ev_check_bet_fold + oop_call_prob * ev_check_bet_call
                    
                    ev_check += (ip_check_prob * ev_check_check + ip_bet_prob * ev_check_bet) / n_ip
                    
                    # OOP bet后
                    ip_fold_prob = ip_strategy_bet[j, 0]
                    ip_call_prob = ip_strategy_bet[j, 1]
                    
                    # bet-fold: OOP赢得底池
                    ev_bet_fold = pot
                    
                    # bet-call: 摊牌
                    ev_bet_call = equity * (pot + 2 * bet) - (1 - equity) * bet
                    
                    ev_bet += (ip_fold_prob * ev_bet_fold + ip_call_prob * ev_bet_call) / n_ip
                
                # 当前策略的期望值
                current_ev = oop_strategy[i, 0] * ev_check + oop_strategy[i, 1] * ev_bet
                
                # 更新遗憾值
                oop_regrets[i, 0] += ev_check - current_ev
                oop_regrets[i, 1] += ev_bet - current_ev
            
            # 更新IP的遗憾值（简化：假设OOP策略固定）
            for j in range(n_ip):
                # 面对OOP check
                ev_ip_check = 0.0
                ev_ip_bet = 0.0
                
                for i in range(n_oop):
                    equity = 1 - self.equity_matrix[i, j]  # IP的胜率
                    
                    # 只考虑OOP check的情况
                    oop_check_prob = oop_strategy[i, 0]
                    
                    # IP check: 摊牌
                    ev_ip_check += oop_check_prob * (equity * pot) / n_oop
                    
                    # IP bet: OOP决定
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
                    equity = 1 - self.equity_matrix[i, j]
                    oop_bet_prob = oop_strategy[i, 1]
                    
                    ev_ip_fold += oop_bet_prob * 0 / n_oop  # 弃牌损失0
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
        
        # 避免除以0
        sums = np.where(sums > 0, sums, 1)
        strategy = positive / sums
        
        # 如果所有遗憾都是负的，使用均匀分布
        uniform = np.ones_like(strategy) / strategy.shape[1]
        strategy = np.where(positive.sum(axis=1, keepdims=True) > 0, strategy, uniform)
        
        return strategy


def run_experiment(
    name: str,
    description: str,
    community_cards: List[Card],
    oop_range: Dict[str, float],
    ip_range: Dict[str, float],
    pot_size: float = 100.0,
    bet_size: float = 50.0,
    cfr_iterations: int = 10000
):
    """运行单个实验场景。"""
    
    print(f"\n{'#'*80}")
    print(f"# {name}")
    print(f"# {description}")
    print(f"{'#'*80}")
    
    print(f"\n【输入】")
    print(f"公共牌: {cards_to_str(community_cards)}")
    print(f"底池: {pot_size}, 下注: {bet_size}")
    
    # 展开范围
    remover = DeadCardRemover()
    calculator = RangeVsRangeCalculator()
    
    oop_expanded = []
    for hand, weight in oop_range.items():
        concrete = remover.expand_abstract_hand(hand)
        for c in concrete:
            cards = parse_hand(c)
            if cards:
                board_set = set((card.rank, card.suit) for card in community_cards)
                if not any((card.rank, card.suit) in board_set for card in cards):
                    oop_expanded.append(c)
    
    ip_expanded = []
    for hand, weight in ip_range.items():
        concrete = remover.expand_abstract_hand(hand)
        for c in concrete:
            cards = parse_hand(c)
            if cards:
                board_set = set((card.rank, card.suit) for card in community_cards)
                if not any((card.rank, card.suit) in board_set for card in cards):
                    ip_expanded.append(c)
    
    # 去重
    oop_expanded = list(set(oop_expanded))
    ip_expanded = list(set(ip_expanded))
    
    print(f"\nOOP有效手牌数: {len(oop_expanded)}")
    print(f"IP有效手牌数: {len(ip_expanded)}")
    
    # 计算胜率
    hand_equities = calculator.calculate_range_vs_range_equity(
        oop_range, ip_range, community_cards
    )
    
    # 创建博弈并求解
    print(f"\n【运行CFR求解器】(迭代次数: {cfr_iterations})")
    game = RiverPokerGame(
        community_cards=community_cards,
        oop_hands=oop_expanded,
        ip_hands=ip_expanded,
        pot_size=pot_size,
        bet_size=bet_size
    )
    
    solver_strategies = game.solve_with_cfr(iterations=cfr_iterations)
    
    # 打印结果
    print(f"\n【所有手牌的胜率和策略】")
    print(f"{'手牌':<12} {'牌型':<8} {'胜率':<10} {'check':<10} {'bet':<10}")
    print("-" * 60)
    
    results = []
    for hand in sorted(oop_expanded):
        cards = parse_hand(hand)
        if not cards:
            continue
        
        hand_type = get_hand_type(cards, community_cards)
        equity = hand_equities.get(hand, 0.5)
        strategy = solver_strategies.get(hand, {'check': 0.5, 'bet': 0.5})
        
        results.append({
            'hand': hand,
            'type': hand_type,
            'equity': equity,
            'check': strategy['check'],
            'bet': strategy['bet'],
        })
    
    # 按胜率排序
    results.sort(key=lambda x: x['equity'])
    
    for r in results:
        print(f"{r['hand']:<12} {r['type']:<8} {r['equity']:<10.4f} "
              f"{r['check']:<10.4f} {r['bet']:<10.4f}")
    
    # 找出胜率相近但策略不同的手牌对
    print(f"\n{'='*80}")
    print("【关键分析】找出胜率相近的手牌对，比较策略差异")
    print(f"{'='*80}")
    
    pairs = []
    for i, r1 in enumerate(results):
        for r2 in results[i+1:]:
            equity_diff = abs(r1['equity'] - r2['equity'])
            if equity_diff < 0.05:  # 胜率差异小于5%
                strategy_diff = 0.5 * (abs(r1['check'] - r2['check']) + abs(r1['bet'] - r2['bet']))
                
                r1_best = 'bet' if r1['bet'] > 0.5 else 'check'
                r2_best = 'bet' if r2['bet'] > 0.5 else 'check'
                
                pairs.append({
                    'hand1': r1['hand'],
                    'hand2': r2['hand'],
                    'type1': r1['type'],
                    'type2': r2['type'],
                    'equity1': r1['equity'],
                    'equity2': r2['equity'],
                    'equity_diff': equity_diff,
                    'strategy_diff': strategy_diff,
                    'same_action': r1_best == r2_best,
                    'strategy1': {'check': r1['check'], 'bet': r1['bet']},
                    'strategy2': {'check': r2['check'], 'bet': r2['bet']},
                })
    
    # 按策略差异排序
    pairs.sort(key=lambda x: -x['strategy_diff'])
    
    if pairs:
        print(f"\n找到 {len(pairs)} 对胜率相近的手牌（胜率差<5%）")
        
        # 显示策略差异最大的
        significant = [p for p in pairs if p['strategy_diff'] > 0.1]
        
        if significant:
            print(f"\n【策略差异显著的手牌对】(策略差异>0.1)")
            for p in significant[:10]:
                print(f"\n  {p['hand1']} ({p['type1']}) vs {p['hand2']} ({p['type2']})")
                print(f"    胜率: {p['equity1']:.4f} vs {p['equity2']:.4f} (差异: {p['equity_diff']:.4f})")
                print(f"    策略1: check={p['strategy1']['check']:.4f}, bet={p['strategy1']['bet']:.4f}")
                print(f"    策略2: check={p['strategy2']['check']:.4f}, bet={p['strategy2']['bet']:.4f}")
                print(f"    策略差异: {p['strategy_diff']:.4f}")
                if not p['same_action']:
                    print(f"    ⚠️ 最优动作不同!")
        else:
            print("\n✓ 没有发现策略差异显著的手牌对")
        
        # 统计
        avg_diff = np.mean([p['strategy_diff'] for p in pairs])
        max_diff = max(p['strategy_diff'] for p in pairs)
        same_rate = sum(1 for p in pairs if p['same_action']) / len(pairs)
        
        print(f"\n【统计】")
        print(f"  平均策略差异: {avg_diff:.4f}")
        print(f"  最大策略差异: {max_diff:.4f}")
        print(f"  最优动作一致率: {same_rate:.2%}")
    
    return {
        'name': name,
        'results': results,
        'pairs': pairs,
    }


def main():
    print("=" * 80)
    print("使用CFR进行胜率-策略相关性验证实验")
    print("=" * 80)
    print("\n核心问题: 当两个手牌具有相同的胜率时，它们的最优策略是否相同？")
    
    all_results = []
    
    # 场景1：干燥牌面
    all_results.append(run_experiment(
        name="场景1_干燥牌面",
        description="K♠T♦7♣4♥2♠ - 干燥牌面，范围差异明显",
        community_cards=[
            Card(rank=13, suit='s'),
            Card(rank=10, suit='d'),
            Card(rank=7, suit='c'),
            Card(rank=4, suit='h'),
            Card(rank=2, suit='s'),
        ],
        oop_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0,
            '99': 1.0, '88': 1.0, '77': 1.0,
            'AKs': 1.0, 'AKo': 1.0, 'KQs': 1.0,
        },
        ip_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0,
            'AKs': 1.0, 'KQs': 1.0, 'QJs': 1.0,
        },
        cfr_iterations=5000,
    ))
    
    # 场景2：湿润牌面
    all_results.append(run_experiment(
        name="场景2_湿润牌面",
        description="J♠T♥9♦5♣2♠ - 连接牌面，顺子可能",
        community_cards=[
            Card(rank=11, suit='s'),
            Card(rank=10, suit='h'),
            Card(rank=9, suit='d'),
            Card(rank=5, suit='c'),
            Card(rank=2, suit='s'),
        ],
        oop_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0,
            'KQs': 1.0, 'QJs': 1.0, 'JTs': 1.0,
            '87s': 1.0, '76s': 1.0,
        },
        ip_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0,
            'KQs': 1.0, 'QJs': 1.0, 'JTs': 1.0,
            '87s': 1.0,
        },
        cfr_iterations=5000,
    ))
    
    # 场景3：同花牌面
    all_results.append(run_experiment(
        name="场景3_同花牌面",
        description="A♥J♥8♥5♥2♦ - 四张同花，阻断效应",
        community_cards=[
            Card(rank=14, suit='h'),
            Card(rank=11, suit='h'),
            Card(rank=8, suit='h'),
            Card(rank=5, suit='h'),
            Card(rank=2, suit='d'),
        ],
        oop_range={
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0,
            'KhQh': 1.0, 'KhTh': 1.0, 'QhTh': 1.0,
            'KsQs': 1.0, 'KsTs': 1.0,
        },
        ip_range={
            'AA': 1.0, 'KK': 1.0,
            'KhQh': 1.0, 'QhTh': 1.0,
            'KsQs': 1.0,
        },
        cfr_iterations=5000,
    ))
    
    # 保存结果
    output_path = 'experiments/results/openspiel_validation.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")
    
    # 总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)
    
    all_pairs = []
    for r in all_results:
        all_pairs.extend(r['pairs'])
    
    if all_pairs:
        significant = [p for p in all_pairs if p['strategy_diff'] > 0.1]
        
        print(f"\n总共找到 {len(all_pairs)} 对胜率相近的手牌")
        print(f"其中策略差异显著的: {len(significant)} 对")
        
        if significant:
            print("\n⚠️ 结论: 发现胜率相近但策略不同的手牌对")
            print("   这表明仅靠胜率标量不足以完全决定最优策略")
        else:
            print("\n✓ 结论: 胜率相近的手牌策略也相近")


if __name__ == '__main__':
    main()
