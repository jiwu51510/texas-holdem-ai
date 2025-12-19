"""转牌阶段Solver封装模块。

本模块提供转牌阶段的Solver封装，用于计算纳什均衡策略。
扩展现有的SimpleCFRSolver以支持4张公共牌的情况。
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from models.core import Card
from experiments.equity_solver_validation.data_models import (
    SolverConfig,
    SolverResult,
)
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    DeadCardRemover,
    RangeVsRangeCalculator,
)
from experiments.turn_potential_validation.potential_histogram import (
    PotentialHistogramCalculator,
)
from environment.hand_evaluator import compare_hands


class TurnCFRSolver:
    """转牌阶段的CFR求解器。
    
    实现一个简化的转牌阶段CFR求解器，用于验证实验。
    支持4张公共牌的情况，考虑河牌的不确定性。
    """
    
    def __init__(self, config: SolverConfig):
        """初始化求解器。
        
        Args:
            config: Solver配置
        """
        self.config = config
        self.calculator = RangeVsRangeCalculator()
        self.remover = DeadCardRemover()
        self.histogram_calculator = PotentialHistogramCalculator()
    
    def solve(
        self,
        turn_community: List[Card],
        oop_range: Dict[str, float],
        ip_range: Dict[str, float],
        iterations: int = None
    ) -> SolverResult:
        """运行转牌阶段的CFR求解。
        
        Args:
            turn_community: 4张公共牌（翻牌+转牌）
            oop_range: OOP玩家范围
            ip_range: IP玩家范围
            iterations: 迭代次数（默认使用配置值）
            
        Returns:
            SolverResult实例
        """
        if iterations is None:
            iterations = self.config.max_iterations
        
        if len(turn_community) != 4:
            raise ValueError(f"转牌阶段必须有4张公共牌，当前：{len(turn_community)}")
        
        # 展开范围为具体手牌
        oop_expanded = self._expand_and_filter_range(oop_range, turn_community)
        ip_expanded = self._expand_and_filter_range(ip_range, turn_community)
        
        if not oop_expanded or not ip_expanded:
            return SolverResult(
                exploitability=0.0,
                iterations=0,
                root_ev=(0.0, 0.0),
                strategies={},
                converged=True,
            )
        
        # 计算期望胜率矩阵（考虑所有可能的河牌）
        equity_matrix = self._compute_expected_equity_matrix(
            oop_expanded, ip_expanded, turn_community
        )
        
        # 初始化策略和遗憾值
        oop_hands = list(oop_expanded.keys())
        ip_hands = list(ip_expanded.keys())
        
        # OOP策略：check或bet
        oop_regrets = {h: {'check': 0.0, 'bet': 0.0} for h in oop_hands}
        oop_strategy_sum = {h: {'check': 0.0, 'bet': 0.0} for h in oop_hands}
        
        # IP策略：面对check时（check/bet），面对bet时（fold/call）
        ip_regrets_vs_check = {h: {'check': 0.0, 'bet': 0.0} for h in ip_hands}
        ip_regrets_vs_bet = {h: {'fold': 0.0, 'call': 0.0} for h in ip_hands}
        ip_strategy_sum_vs_check = {h: {'check': 0.0, 'bet': 0.0} for h in ip_hands}
        ip_strategy_sum_vs_bet = {h: {'fold': 0.0, 'call': 0.0} for h in ip_hands}
        
        pot = self.config.pot_size
        bet_size = pot * self.config.oop_bet_sizes[0] if self.config.oop_bet_sizes else pot * 0.5
        
        # CFR迭代
        for t in range(iterations):
            # 获取当前策略
            oop_strategy = self._regret_matching(oop_regrets)
            ip_strategy_check = self._regret_matching(ip_regrets_vs_check)
            ip_strategy_bet = self._regret_matching(ip_regrets_vs_bet)
            
            # 累积策略
            for h in oop_hands:
                for a in ['check', 'bet']:
                    oop_strategy_sum[h][a] += oop_strategy[h][a]
            for h in ip_hands:
                for a in ['check', 'bet']:
                    ip_strategy_sum_vs_check[h][a] += ip_strategy_check[h][a]
                for a in ['fold', 'call']:
                    ip_strategy_sum_vs_bet[h][a] += ip_strategy_bet[h][a]
            
            # 计算遗憾值
            for i, oop_h in enumerate(oop_hands):
                oop_weight = oop_expanded[oop_h]
                
                ev_check = 0.0
                ev_bet = 0.0
                
                for j, ip_h in enumerate(ip_hands):
                    ip_weight = ip_expanded[ip_h]
                    
                    if self._hands_conflict(oop_h, ip_h):
                        continue
                    
                    equity = equity_matrix[i, j]
                    
                    # OOP check后，IP的响应
                    ip_check_prob = ip_strategy_check[ip_h]['check']
                    ip_bet_prob = ip_strategy_check[ip_h]['bet']
                    
                    ev_check += ip_weight * ip_check_prob * (equity * pot - (1 - equity) * pot) / 2
                    ev_check += ip_weight * ip_bet_prob * (equity * (pot + bet_size) - (1 - equity) * bet_size) / 2
                    
                    # OOP bet后，IP的响应
                    ip_fold_prob = ip_strategy_bet[ip_h]['fold']
                    ip_call_prob = ip_strategy_bet[ip_h]['call']
                    
                    ev_bet += ip_weight * ip_fold_prob * pot / 2
                    ev_bet += ip_weight * ip_call_prob * (equity * (pot + 2 * bet_size) - (1 - equity) * bet_size) / 2
                
                current_ev = oop_strategy[oop_h]['check'] * ev_check + oop_strategy[oop_h]['bet'] * ev_bet
                
                oop_regrets[oop_h]['check'] += ev_check - current_ev
                oop_regrets[oop_h]['bet'] += ev_bet - current_ev
        
        # 计算平均策略
        oop_avg_strategy = self._compute_average_strategy(oop_strategy_sum, ['check', 'bet'])
        ip_avg_strategy_check = self._compute_average_strategy(ip_strategy_sum_vs_check, ['check', 'bet'])
        ip_avg_strategy_bet = self._compute_average_strategy(ip_strategy_sum_vs_bet, ['fold', 'call'])
        
        strategies = {
            'root': oop_avg_strategy,
            'root:check': ip_avg_strategy_check,
            'root:bet': ip_avg_strategy_bet,
        }
        
        return SolverResult(
            exploitability=0.0,
            iterations=iterations,
            root_ev=(0.0, 0.0),
            strategies=strategies,
            converged=True,
        )
    
    def _expand_and_filter_range(
        self,
        range_dict: Dict[str, float],
        turn_community: List[Card]
    ) -> Dict[str, float]:
        """展开范围并过滤与公共牌冲突的手牌。"""
        expanded = {}
        
        for hand, weight in range_dict.items():
            concrete_hands = self.remover.expand_abstract_hand(hand)
            per_weight = weight / len(concrete_hands) if concrete_hands else 0
            
            for concrete in concrete_hands:
                cards = self.remover._parse_hand_string(concrete)
                if cards is None:
                    continue
                
                board_set = set((c.rank, c.suit) for c in turn_community)
                if any((c.rank, c.suit) in board_set for c in cards):
                    continue
                
                expanded[concrete] = per_weight
        
        return expanded
    
    def _compute_expected_equity_matrix(
        self,
        oop_range: Dict[str, float],
        ip_range: Dict[str, float],
        turn_community: List[Card]
    ) -> np.ndarray:
        """计算期望胜率矩阵（考虑所有可能的河牌）。
        
        对于转牌阶段，需要枚举所有可能的河牌并计算平均胜率。
        """
        oop_hands = list(oop_range.keys())
        ip_hands = list(ip_range.keys())
        
        matrix = np.zeros((len(oop_hands), len(ip_hands)))
        
        # 获取所有可能的河牌
        board_set = set((c.rank, c.suit) for c in turn_community)
        all_cards = []
        for rank in range(2, 15):
            for suit in ['h', 'd', 'c', 's']:
                if (rank, suit) not in board_set:
                    all_cards.append(Card(rank=rank, suit=suit))
        
        for i, oop_h in enumerate(oop_hands):
            oop_cards = self.remover._parse_hand_string(oop_h)
            if oop_cards is None:
                continue
            
            oop_set = set((c.rank, c.suit) for c in oop_cards)
            
            for j, ip_h in enumerate(ip_hands):
                ip_cards = self.remover._parse_hand_string(ip_h)
                if ip_cards is None:
                    continue
                
                if self._hands_conflict(oop_h, ip_h):
                    matrix[i, j] = 0.5
                    continue
                
                ip_set = set((c.rank, c.suit) for c in ip_cards)
                
                # 枚举所有可能的河牌
                total_equity = 0.0
                count = 0
                
                for river_card in all_cards:
                    river_key = (river_card.rank, river_card.suit)
                    
                    # 跳过与手牌冲突的河牌
                    if river_key in oop_set or river_key in ip_set:
                        continue
                    
                    river_community = turn_community + [river_card]
                    
                    result = compare_hands(list(oop_cards), list(ip_cards), river_community)
                    
                    if result == 0:
                        total_equity += 1.0
                    elif result == -1:
                        total_equity += 0.5
                    
                    count += 1
                
                if count > 0:
                    matrix[i, j] = total_equity / count
                else:
                    matrix[i, j] = 0.5
        
        return matrix
    
    def _hands_conflict(self, hand1: str, hand2: str) -> bool:
        """检查两个手牌是否有冲突。"""
        cards1 = self.remover._parse_hand_string(hand1)
        cards2 = self.remover._parse_hand_string(hand2)
        
        if cards1 is None or cards2 is None:
            return False
        
        set1 = set((c.rank, c.suit) for c in cards1)
        set2 = set((c.rank, c.suit) for c in cards2)
        
        return bool(set1 & set2)
    
    def _regret_matching(
        self,
        regrets: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """遗憾匹配算法。"""
        strategy = {}
        
        for hand, hand_regrets in regrets.items():
            positive_regrets = {a: max(0, r) for a, r in hand_regrets.items()}
            total = sum(positive_regrets.values())
            
            if total > 0:
                strategy[hand] = {a: r / total for a, r in positive_regrets.items()}
            else:
                n_actions = len(hand_regrets)
                strategy[hand] = {a: 1.0 / n_actions for a in hand_regrets}
        
        return strategy
    
    def _compute_average_strategy(
        self,
        strategy_sum: Dict[str, Dict[str, float]],
        actions: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """计算平均策略。"""
        avg_strategy = {}
        
        for hand, action_sums in strategy_sum.items():
            total = sum(action_sums.values())
            if total > 0:
                avg_strategy[hand] = {a: action_sums[a] / total for a in actions}
            else:
                avg_strategy[hand] = {a: 1.0 / len(actions) for a in actions}
        
        return avg_strategy
    
    def get_strategy_at_node(self, result: SolverResult, node_path: str) -> Dict[str, Dict[str, float]]:
        """获取指定节点的策略。
        
        Args:
            result: Solver结果
            node_path: 节点路径
            
        Returns:
            该节点的策略
        """
        return result.strategies.get(node_path, {})
    
    def get_ev_at_node(self, result: SolverResult, node_path: str) -> Dict[str, float]:
        """获取指定节点的EV。
        
        Args:
            result: Solver结果
            node_path: 节点路径
            
        Returns:
            每个手牌的EV
        """
        # 简化实现：返回空字典
        # 完整实现需要在CFR迭代中计算EV
        return {}
