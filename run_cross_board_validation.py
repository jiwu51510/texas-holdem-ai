#!/usr/bin/env python3
"""跨公共牌的双维度胜率-策略验证实验。

核心问题：固定一个手牌，在不同的公共牌情况下：
1. 计算手牌vs对手范围的胜率（去除死牌）
2. 计算自己范围vs对手范围的胜率（去除死牌）
3. 找出这两个胜率都相近的不同公共牌情况
4. 比较这些情况下的策略是否相同

如果两个胜率标量能完全决定策略，那么在不同公共牌下，
只要这两个胜率相同，策略就应该相同。
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations

from models.core import Card
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    DeadCardRemover,
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


@dataclass
class BoardScenario:
    """一个公共牌场景的数据。"""
    board: List[Card]
    board_str: str
    hero_hand: str
    hero_type: str
    hero_equity: float  # 手牌vs对手范围胜率
    range_equity: float  # 范围vs范围胜率
    strategy: Dict[str, float]  # CFR求解的策略


class CrossBoardSolver:
    """跨公共牌求解器。"""
    
    def __init__(
        self,
        hero_hand: str,  # 固定的手牌，如 "AhKh"
        oop_range: Dict[str, float],  # OOP范围
        ip_range: Dict[str, float],  # IP范围
        pot_size: float = 100.0,
        bet_size: float = 50.0,
    ):
        self.hero_hand = hero_hand
        self.hero_cards = parse_hand(hero_hand)
        self.oop_range = oop_range
        self.ip_range = ip_range
        self.pot_size = pot_size
        self.bet_size = bet_size
        self.remover = DeadCardRemover()
        
        # 去除与hero手牌冲突的组合
        self.hero_set = set((c.rank, c.suit) for c in self.hero_cards)
    
    def expand_range_with_dead_cards(
        self, 
        range_dict: Dict[str, float], 
        dead_cards: List[Card]
    ) -> List[str]:
        """展开范围，去除死牌。"""
        dead_set = set((c.rank, c.suit) for c in dead_cards)
        expanded = []
        
        for hand, weight in range_dict.items():
            concrete = self.remover.expand_abstract_hand(hand)
            for c in concrete:
                cards = parse_hand(c)
                if cards:
                    card_set = set((card.rank, card.suit) for card in cards)
                    # 去除与死牌冲突的组合
                    if not (card_set & dead_set):
                        expanded.append(c)
        
        return list(set(expanded))

    def compute_hero_equity(
        self, 
        board: List[Card], 
        opponent_hands: List[str]
    ) -> float:
        """计算hero手牌vs对手范围的胜率。"""
        if not opponent_hands:
            return 0.5
        
        wins = 0
        ties = 0
        total = 0
        
        for opp_hand in opponent_hands:
            opp_cards = parse_hand(opp_hand)
            if not opp_cards:
                continue
            
            # 检查冲突
            opp_set = set((c.rank, c.suit) for c in opp_cards)
            if opp_set & self.hero_set:
                continue
            
            result = compare_hands(
                list(self.hero_cards), 
                list(opp_cards), 
                board
            )
            
            if result == 0:  # hero胜
                wins += 1
            elif result == -1:  # 平局
                ties += 1
            total += 1
        
        if total == 0:
            return 0.5
        
        return (wins + 0.5 * ties) / total
    
    def compute_range_equity(
        self, 
        board: List[Card],
        oop_hands: List[str],
        ip_hands: List[str]
    ) -> float:
        """计算OOP范围vs IP范围的胜率。"""
        if not oop_hands or not ip_hands:
            return 0.5
        
        total_equity = 0.0
        total_matchups = 0
        
        for oop_hand in oop_hands:
            oop_cards = parse_hand(oop_hand)
            if not oop_cards:
                continue
            
            oop_set = set((c.rank, c.suit) for c in oop_cards)
            
            for ip_hand in ip_hands:
                ip_cards = parse_hand(ip_hand)
                if not ip_cards:
                    continue
                
                ip_set = set((c.rank, c.suit) for c in ip_cards)
                
                # 检查冲突
                if oop_set & ip_set:
                    continue
                
                result = compare_hands(list(oop_cards), list(ip_cards), board)
                
                if result == 0:  # OOP胜
                    total_equity += 1.0
                elif result == -1:  # 平局
                    total_equity += 0.5
                
                total_matchups += 1
        
        if total_matchups == 0:
            return 0.5
        
        return total_equity / total_matchups

    def solve_cfr(
        self, 
        board: List[Card],
        oop_hands: List[str],
        ip_hands: List[str],
        iterations: int = 2000
    ) -> Dict[str, Dict[str, float]]:
        """使用CFR求解策略。"""
        n_oop = len(oop_hands)
        n_ip = len(ip_hands)
        
        if n_oop == 0 or n_ip == 0:
            return {}
        
        # 计算胜率矩阵
        equity_matrix = np.zeros((n_oop, n_ip))
        
        for i, oop_h in enumerate(oop_hands):
            oop_cards = parse_hand(oop_h)
            if not oop_cards:
                continue
            
            oop_set = set((c.rank, c.suit) for c in oop_cards)
            
            for j, ip_h in enumerate(ip_hands):
                ip_cards = parse_hand(ip_h)
                if not ip_cards:
                    continue
                
                ip_set = set((c.rank, c.suit) for c in ip_cards)
                
                if oop_set & ip_set:
                    equity_matrix[i, j] = 0.5
                    continue
                
                result = compare_hands(list(oop_cards), list(ip_cards), board)
                
                if result == 0:
                    equity_matrix[i, j] = 1.0
                elif result == -1:
                    equity_matrix[i, j] = 0.5
                else:
                    equity_matrix[i, j] = 0.0
        
        # CFR求解
        oop_regrets = np.zeros((n_oop, 2))
        oop_strategy_sum = np.zeros((n_oop, 2))
        ip_regrets_vs_bet = np.zeros((n_ip, 2))
        
        pot = self.pot_size
        bet = self.bet_size
        
        for t in range(iterations):
            oop_strategy = self._regret_matching(oop_regrets)
            ip_strategy_bet = self._regret_matching(ip_regrets_vs_bet)
            
            oop_strategy_sum += oop_strategy
            
            for i in range(n_oop):
                ev_check = 0.0
                ev_bet = 0.0
                
                for j in range(n_ip):
                    equity = equity_matrix[i, j]
                    
                    ev_check += equity * pot / n_ip
                    
                    ip_fold = ip_strategy_bet[j, 0]
                    ip_call = ip_strategy_bet[j, 1]
                    
                    ev_bet_fold = pot
                    ev_bet_call = equity * (pot + 2 * bet) - (1 - equity) * bet
                    ev_bet += (ip_fold * ev_bet_fold + ip_call * ev_bet_call) / n_ip
                
                current_ev = oop_strategy[i, 0] * ev_check + oop_strategy[i, 1] * ev_bet
                oop_regrets[i, 0] += ev_check - current_ev
                oop_regrets[i, 1] += ev_bet - current_ev
            
            for j in range(n_ip):
                ev_fold = 0.0
                ev_call = 0.0
                
                for i in range(n_oop):
                    equity = 1 - equity_matrix[i, j]
                    oop_bet_prob = oop_strategy[i, 1]
                    
                    ev_call += oop_bet_prob * (equity * (pot + 2*bet) - (1-equity) * bet) / n_oop
                
                current_ev = ip_strategy_bet[j, 0] * ev_fold + ip_strategy_bet[j, 1] * ev_call
                ip_regrets_vs_bet[j, 0] += ev_fold - current_ev
                ip_regrets_vs_bet[j, 1] += ev_call - current_ev
        
        # 计算平均策略
        strategies = {}
        for i, hand in enumerate(oop_hands):
            total = oop_strategy_sum[i].sum()
            if total > 0:
                probs = oop_strategy_sum[i] / total
            else:
                probs = np.array([0.5, 0.5])
            strategies[hand] = {'check': float(probs[0]), 'bet': float(probs[1])}
        
        return strategies
    
    def _regret_matching(self, regrets: np.ndarray) -> np.ndarray:
        positive = np.maximum(regrets, 0)
        sums = positive.sum(axis=1, keepdims=True)
        sums = np.where(sums > 0, sums, 1)
        strategy = positive / sums
        uniform = np.ones_like(strategy) / strategy.shape[1]
        strategy = np.where(positive.sum(axis=1, keepdims=True) > 0, strategy, uniform)
        return strategy

    def analyze_board(self, board: List[Card]) -> Optional[BoardScenario]:
        """分析一个公共牌场景。"""
        # 检查hero手牌是否与公共牌冲突
        board_set = set((c.rank, c.suit) for c in board)
        if self.hero_set & board_set:
            return None
        
        # 死牌 = hero手牌 + 公共牌
        dead_cards = list(self.hero_cards) + board
        
        # 展开范围，去除死牌
        oop_hands = self.expand_range_with_dead_cards(self.oop_range, dead_cards)
        ip_hands = self.expand_range_with_dead_cards(self.ip_range, dead_cards)
        
        if not oop_hands or not ip_hands:
            return None
        
        # 计算hero胜率（假设hero是OOP）
        hero_equity = self.compute_hero_equity(board, ip_hands)
        
        # 计算范围胜率
        range_equity = self.compute_range_equity(board, oop_hands, ip_hands)
        
        # CFR求解
        strategies = self.solve_cfr(board, oop_hands, ip_hands)
        
        hero_strategy = strategies.get(self.hero_hand, {'check': 0.5, 'bet': 0.5})
        hero_type = get_hand_type(self.hero_cards, board)
        
        return BoardScenario(
            board=board,
            board_str=cards_to_str(board),
            hero_hand=self.hero_hand,
            hero_type=hero_type,
            hero_equity=hero_equity,
            range_equity=range_equity,
            strategy=hero_strategy,
        )


def generate_boards(exclude_cards: List[Card], num_boards: int = 50) -> List[List[Card]]:
    """生成多个不同的公共牌。"""
    all_cards = []
    for rank in range(2, 15):
        for suit in ['s', 'd', 'c', 'h']:
            card = Card(rank=rank, suit=suit)
            if (card.rank, card.suit) not in [(c.rank, c.suit) for c in exclude_cards]:
                all_cards.append(card)
    
    boards = []
    # 生成一些有代表性的公共牌
    
    # 干燥牌面
    dry_boards = [
        [Card(13,'s'), Card(10,'d'), Card(7,'c'), Card(4,'h'), Card(2,'s')],  # K-T-7-4-2
        [Card(14,'s'), Card(9,'d'), Card(5,'c'), Card(3,'h'), Card(2,'d')],   # A-9-5-3-2
        [Card(12,'s'), Card(8,'d'), Card(4,'c'), Card(3,'h'), Card(2,'s')],   # Q-8-4-3-2
    ]
    
    # 湿润牌面（连接）
    wet_boards = [
        [Card(11,'s'), Card(10,'h'), Card(9,'d'), Card(5,'c'), Card(2,'s')],  # J-T-9-5-2
        [Card(10,'s'), Card(9,'h'), Card(8,'d'), Card(4,'c'), Card(2,'s')],   # T-9-8-4-2
        [Card(9,'s'), Card(8,'h'), Card(7,'d'), Card(3,'c'), Card(2,'s')],    # 9-8-7-3-2
    ]
    
    # 同花牌面
    flush_boards = [
        [Card(14,'h'), Card(11,'h'), Card(8,'h'), Card(5,'h'), Card(2,'d')],  # A-J-8-5-2 四红桃
        [Card(13,'s'), Card(10,'s'), Card(7,'s'), Card(4,'s'), Card(2,'d')],  # K-T-7-4-2 四黑桃
    ]
    
    # 配对牌面
    paired_boards = [
        [Card(13,'s'), Card(13,'d'), Card(9,'c'), Card(5,'h'), Card(2,'s')],  # K-K-9-5-2
        [Card(10,'s'), Card(10,'d'), Card(7,'c'), Card(4,'h'), Card(2,'s')],  # T-T-7-4-2
    ]
    
    boards.extend(dry_boards)
    boards.extend(wet_boards)
    boards.extend(flush_boards)
    boards.extend(paired_boards)
    
    return boards


def run_cross_board_experiment(
    hero_hand: str,
    oop_range: Dict[str, float],
    ip_range: Dict[str, float],
    equity_threshold: float = 0.05,
):
    """运行跨公共牌验证实验。"""
    
    print(f"\n{'='*80}")
    print(f"跨公共牌双维度胜率验证实验")
    print(f"{'='*80}")
    print(f"\n固定手牌: {hero_hand}")
    print(f"胜率差异阈值: {equity_threshold*100:.1f}%")
    
    hero_cards = parse_hand(hero_hand)
    if not hero_cards:
        print("无效的手牌")
        return
    
    solver = CrossBoardSolver(
        hero_hand=hero_hand,
        oop_range=oop_range,
        ip_range=ip_range,
    )
    
    # 生成公共牌
    boards = generate_boards(list(hero_cards))
    
    print(f"\n分析 {len(boards)} 个不同的公共牌场景...")
    
    scenarios = []
    for board in boards:
        scenario = solver.analyze_board(board)
        if scenario:
            scenarios.append(scenario)
    
    print(f"有效场景数: {len(scenarios)}")
    
    # 打印所有场景
    print(f"\n【所有场景的双维度胜率和策略】")
    print(f"{'公共牌':<25} {'牌型':<8} {'手牌胜率':<10} {'范围胜率':<10} {'check':<10} {'bet':<10}")
    print("-" * 85)
    
    for s in scenarios:
        print(f"{s.board_str:<25} {s.hero_type:<8} {s.hero_equity:<10.4f} "
              f"{s.range_equity:<10.4f} {s.strategy['check']:<10.4f} {s.strategy['bet']:<10.4f}")
    
    # 找出双维度胜率相近的场景对
    print(f"\n{'='*80}")
    print("【关键分析】找出双维度胜率都相近的不同公共牌场景")
    print(f"条件: 手牌胜率差异 < {equity_threshold*100:.1f}% AND 范围胜率差异 < {equity_threshold*100:.1f}%")
    print(f"{'='*80}")
    
    pairs = []
    for i, s1 in enumerate(scenarios):
        for s2 in scenarios[i+1:]:
            hero_eq_diff = abs(s1.hero_equity - s2.hero_equity)
            range_eq_diff = abs(s1.range_equity - s2.range_equity)
            
            # 双维度都相近
            if hero_eq_diff < equity_threshold and range_eq_diff < equity_threshold:
                strategy_diff = 0.5 * (
                    abs(s1.strategy['check'] - s2.strategy['check']) + 
                    abs(s1.strategy['bet'] - s2.strategy['bet'])
                )
                
                s1_best = 'bet' if s1.strategy['bet'] > 0.5 else 'check'
                s2_best = 'bet' if s2.strategy['bet'] > 0.5 else 'check'
                
                pairs.append({
                    'board1': s1.board_str,
                    'board2': s2.board_str,
                    'type1': s1.hero_type,
                    'type2': s2.hero_type,
                    'hero_eq1': s1.hero_equity,
                    'hero_eq2': s2.hero_equity,
                    'hero_eq_diff': hero_eq_diff,
                    'range_eq1': s1.range_equity,
                    'range_eq2': s2.range_equity,
                    'range_eq_diff': range_eq_diff,
                    'strategy1': s1.strategy,
                    'strategy2': s2.strategy,
                    'strategy_diff': strategy_diff,
                    'same_action': s1_best == s2_best,
                })
    
    pairs.sort(key=lambda x: -x['strategy_diff'])
    
    if pairs:
        print(f"\n找到 {len(pairs)} 对双维度胜率相近的场景")
        
        significant = [p for p in pairs if p['strategy_diff'] > 0.1]
        
        if significant:
            print(f"\n【策略差异显著的场景对】(策略差异>10%)")
            print("-" * 80)
            
            for p in significant[:10]:
                print(f"\n  场景1: {p['board1']} ({p['type1']})")
                print(f"  场景2: {p['board2']} ({p['type2']})")
                print(f"  手牌胜率: {p['hero_eq1']:.4f} vs {p['hero_eq2']:.4f} (差异: {p['hero_eq_diff']:.4f})")
                print(f"  范围胜率: {p['range_eq1']:.4f} vs {p['range_eq2']:.4f} (差异: {p['range_eq_diff']:.4f})")
                print(f"  策略1: check={p['strategy1']['check']:.4f}, bet={p['strategy1']['bet']:.4f}")
                print(f"  策略2: check={p['strategy2']['check']:.4f}, bet={p['strategy2']['bet']:.4f}")
                print(f"  策略差异: {p['strategy_diff']:.4f}")
                if not p['same_action']:
                    print(f"  ⚠️ 最优动作完全不同!")
        else:
            print("\n✓ 没有发现策略差异显著的场景对")
        
        # 统计
        avg_diff = np.mean([p['strategy_diff'] for p in pairs])
        max_diff = max(p['strategy_diff'] for p in pairs) if pairs else 0
        same_rate = sum(1 for p in pairs if p['same_action']) / len(pairs) if pairs else 1.0
        
        print(f"\n【统计汇总】")
        print(f"  双维度胜率相近的场景对数: {len(pairs)}")
        print(f"  策略差异显著的对数: {len(significant)}")
        print(f"  平均策略差异: {avg_diff:.4f}")
        print(f"  最大策略差异: {max_diff:.4f}")
        print(f"  最优动作一致率: {same_rate:.2%}")
    else:
        print("\n没有找到双维度胜率相近的场景对")
    
    return {
        'hero_hand': hero_hand,
        'scenarios': [
            {
                'board': s.board_str,
                'type': s.hero_type,
                'hero_equity': s.hero_equity,
                'range_equity': s.range_equity,
                'strategy': s.strategy,
            }
            for s in scenarios
        ],
        'pairs': pairs,
    }


def main():
    print("=" * 80)
    print("跨公共牌双维度胜率-策略验证实验")
    print("=" * 80)
    print("\n核心问题: 固定一个手牌，在不同公共牌下：")
    print("  当手牌胜率和范围胜率都相近时，策略是否相同？")
    
    # 定义范围
    oop_range = {
        'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0,
        '99': 1.0, '88': 1.0, '77': 1.0, '66': 1.0, '55': 1.0,
        'AKs': 1.0, 'AKo': 1.0, 'AQs': 1.0, 'AQo': 1.0,
        'AJs': 1.0, 'ATs': 1.0, 'KQs': 1.0, 'KQo': 1.0,
        'KJs': 1.0, 'QJs': 1.0, 'JTs': 1.0, 'T9s': 1.0,
    }
    
    ip_range = {
        'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0,
        '99': 1.0, '88': 1.0, '77': 1.0,
        'AKs': 1.0, 'AKo': 1.0, 'AQs': 1.0,
        'AJs': 1.0, 'KQs': 1.0, 'QJs': 1.0, 'JTs': 1.0,
    }
    
    all_results = []
    
    # 测试多个不同的手牌
    test_hands = [
        'AhKh',  # 同花AK
        'QsQd',  # 口袋Q
        'JhTh',  # 同花JT
        '9s9d',  # 口袋9
    ]
    
    for hero_hand in test_hands:
        result = run_cross_board_experiment(
            hero_hand=hero_hand,
            oop_range=oop_range,
            ip_range=ip_range,
            equity_threshold=0.05,
        )
        if result:
            all_results.append(result)
    
    # 保存结果
    output_path = 'experiments/results/cross_board_validation.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")
    
    # 生成报告
    generate_report(all_results)


def generate_report(all_results: List[Dict]):
    """生成实验报告。"""
    
    report = []
    report.append("# 跨公共牌双维度胜率-策略验证实验报告\n\n")
    report.append("## 实验目的\n\n")
    report.append("验证：**固定一个手牌，在不同公共牌情况下：**\n")
    report.append("当以下两个条件同时满足时，策略是否相同？\n")
    report.append("1. 手牌vs对手范围的胜率相近（差异<5%）\n")
    report.append("2. 自己范围vs对手范围的胜率相近（差异<5%）\n\n")
    
    report.append("## 实验方法\n\n")
    report.append("1. 固定一个手牌（如AhKh）\n")
    report.append("2. 生成多个不同的公共牌场景\n")
    report.append("3. 对每个场景：\n")
    report.append("   - 从对手范围中去除与我手牌冲突的组合\n")
    report.append("   - 计算手牌vs对手范围的胜率\n")
    report.append("   - 计算范围vs范围的胜率\n")
    report.append("   - 使用CFR求解最优策略\n")
    report.append("4. 找出双维度胜率都相近的不同公共牌场景\n")
    report.append("5. 比较这些场景下的策略是否相同\n\n")
    
    report.append("## 关键发现\n\n")
    
    total_pairs = 0
    total_significant = 0
    
    for result in all_results:
        hero_hand = result['hero_hand']
        pairs = result['pairs']
        scenarios = result['scenarios']
        
        report.append(f"### 手牌: {hero_hand}\n\n")
        report.append(f"分析了 {len(scenarios)} 个公共牌场景\n\n")
        
        significant = [p for p in pairs if p['strategy_diff'] > 0.1]
        total_pairs += len(pairs)
        total_significant += len(significant)
        
        if significant:
            report.append("**发现双维度胜率相近但策略不同的场景对：**\n\n")
            report.append("| 公共牌1 | 公共牌2 | 手牌胜率 | 范围胜率 | 策略1 | 策略2 | 差异 |\n")
            report.append("|---------|---------|----------|----------|-------|-------|------|\n")
            
            for p in significant[:5]:
                s1 = f"bet {p['strategy1']['bet']*100:.0f}%" if p['strategy1']['bet'] > 0.5 else f"check {p['strategy1']['check']*100:.0f}%"
                s2 = f"bet {p['strategy2']['bet']*100:.0f}%" if p['strategy2']['bet'] > 0.5 else f"check {p['strategy2']['check']*100:.0f}%"
                report.append(f"| {p['board1']} | {p['board2']} | "
                            f"{p['hero_eq1']:.0%} vs {p['hero_eq2']:.0%} | "
                            f"{p['range_eq1']:.0%} vs {p['range_eq2']:.0%} | "
                            f"{s1} | {s2} | {p['strategy_diff']:.0%} |\n")
            report.append("\n")
        else:
            report.append("✓ 没有发现策略差异显著的场景对\n\n")
    
    report.append("## 统计汇总\n\n")
    report.append(f"- 总共找到 {total_pairs} 对双维度胜率相近的场景\n")
    report.append(f"- 其中策略差异显著的: {total_significant} 对\n\n")
    
    report.append("## 结论\n\n")
    if total_significant > 0:
        report.append("### ⚠️ 双维度胜率标量不足以决定最优策略\n\n")
        report.append("实验发现：即使在不同公共牌下，手牌胜率和范围胜率都相近，\n")
        report.append("策略仍然可能完全不同。这证明了仅靠两个胜率标量无法替代solver。\n")
    else:
        report.append("### ✓ 在测试的场景中，双维度胜率相近时策略也相近\n")
    
    # 保存报告
    report_path = 'experiments/results/cross_board_validation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    print(f"\n报告已保存到: {report_path}")


if __name__ == '__main__':
    main()
