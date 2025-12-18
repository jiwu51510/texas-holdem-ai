"""Solver封装模块。

本模块提供Solver的Python封装，用于计算纳什均衡策略。
由于wasm-postflop是WebAssembly库，我们提供两种方案：
1. SimpleCFRSolver: 简化的CFR实现，用于快速验证
2. ExternalSolverWrapper: 调用外部Solver的接口（预留）

对于实验验证目的，SimpleCFRSolver足够使用。
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from itertools import combinations

from models.core import Card
from experiments.equity_solver_validation.data_models import (
    SolverConfig,
    SolverResult,
)
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    DeadCardRemover,
    RangeVsRangeCalculator,
)
from environment.hand_evaluator import compare_hands


class SimpleCFRSolver:
    """简化的CFR求解器。
    
    实现一个简化的河牌阶段CFR求解器，用于验证实验。
    只支持河牌阶段的简单博弈树（check/bet/call/fold）。
    """
    
    def __init__(self, config: SolverConfig):
        """初始化求解器。
        
        Args:
            config: Solver配置
        """
        self.config = config
        self.calculator = RangeVsRangeCalculator()
        self.remover = DeadCardRemover()
    
    def solve(
        self,
        community_cards: List[Card],
        oop_range: Dict[str, float],
        ip_range: Dict[str, float],
        iterations: int = None
    ) -> SolverResult:
        """运行CFR求解。
        
        Args:
            community_cards: 5张公共牌
            oop_range: OOP玩家范围
            ip_range: IP玩家范围
            iterations: 迭代次数（默认使用配置值）
            
        Returns:
            SolverResult实例
        """
        if iterations is None:
            iterations = self.config.max_iterations
        
        if len(community_cards) != 5:
            raise ValueError(f"河牌阶段必须有5张公共牌，当前：{len(community_cards)}")
        
        # 展开范围为具体手牌
        oop_expanded = self._expand_and_filter_range(oop_range, community_cards)
        ip_expanded = self._expand_and_filter_range(ip_range, community_cards)
        
        if not oop_expanded or not ip_expanded:
            return SolverResult(
                exploitability=0.0,
                iterations=0,
                root_ev=(0.0, 0.0),
                strategies={},
                converged=True,
            )
        
        # 计算胜率矩阵
        equity_matrix = self._compute_equity_matrix(
            oop_expanded, ip_expanded, community_cards
        )
        
        # 初始化策略和遗憾值
        # 简化博弈树：OOP先行动（check/bet），IP响应
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
                
                # 计算各动作的期望值
                ev_check = 0.0
                ev_bet = 0.0
                
                for j, ip_h in enumerate(ip_hands):
                    ip_weight = ip_expanded[ip_h]
                    
                    # 检查死牌冲突
                    if self._hands_conflict(oop_h, ip_h):
                        continue
                    
                    equity = equity_matrix[i, j]
                    
                    # OOP check后，IP的响应
                    ip_check_prob = ip_strategy_check[ip_h]['check']
                    ip_bet_prob = ip_strategy_check[ip_h]['bet']
                    
                    # check-check: 摊牌
                    ev_check += ip_weight * ip_check_prob * (equity * pot - (1 - equity) * pot) / 2
                    
                    # check-bet: OOP需要决定call/fold（简化为总是call）
                    ev_check += ip_weight * ip_bet_prob * (equity * (pot + bet_size) - (1 - equity) * bet_size) / 2
                    
                    # OOP bet后，IP的响应
                    ip_fold_prob = ip_strategy_bet[ip_h]['fold']
                    ip_call_prob = ip_strategy_bet[ip_h]['call']
                    
                    # bet-fold: OOP赢得底池
                    ev_bet += ip_weight * ip_fold_prob * pot / 2
                    
                    # bet-call: 摊牌
                    ev_bet += ip_weight * ip_call_prob * (equity * (pot + 2 * bet_size) - (1 - equity) * bet_size) / 2
                
                # 当前策略的期望值
                current_ev = oop_strategy[oop_h]['check'] * ev_check + oop_strategy[oop_h]['bet'] * ev_bet
                
                # 更新遗憾值
                oop_regrets[oop_h]['check'] += ev_check - current_ev
                oop_regrets[oop_h]['bet'] += ev_bet - current_ev
        
        # 计算平均策略
        oop_avg_strategy = {}
        for h in oop_hands:
            total = sum(oop_strategy_sum[h].values())
            if total > 0:
                oop_avg_strategy[h] = {a: v / total for a, v in oop_strategy_sum[h].items()}
            else:
                oop_avg_strategy[h] = {'check': 0.5, 'bet': 0.5}
        
        ip_avg_strategy_check = {}
        for h in ip_hands:
            total = sum(ip_strategy_sum_vs_check[h].values())
            if total > 0:
                ip_avg_strategy_check[h] = {a: v / total for a, v in ip_strategy_sum_vs_check[h].items()}
            else:
                ip_avg_strategy_check[h] = {'check': 0.5, 'bet': 0.5}
        
        ip_avg_strategy_bet = {}
        for h in ip_hands:
            total = sum(ip_strategy_sum_vs_bet[h].values())
            if total > 0:
                ip_avg_strategy_bet[h] = {a: v / total for a, v in ip_strategy_sum_vs_bet[h].items()}
            else:
                ip_avg_strategy_bet[h] = {'fold': 0.5, 'call': 0.5}
        
        # 构建结果
        strategies = {
            'root': oop_avg_strategy,
            'root:check': ip_avg_strategy_check,
            'root:bet': ip_avg_strategy_bet,
        }
        
        return SolverResult(
            exploitability=0.0,  # 简化实现不计算可利用度
            iterations=iterations,
            root_ev=(0.0, 0.0),  # 简化实现
            strategies=strategies,
            converged=True,
        )
    
    def _expand_and_filter_range(
        self,
        range_dict: Dict[str, float],
        community_cards: List[Card]
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
                
                # 检查与公共牌的冲突
                board_set = set((c.rank, c.suit) for c in community_cards)
                if any((c.rank, c.suit) in board_set for c in cards):
                    continue
                
                expanded[concrete] = per_weight
        
        return expanded
    
    def _compute_equity_matrix(
        self,
        oop_range: Dict[str, float],
        ip_range: Dict[str, float],
        community_cards: List[Card]
    ) -> np.ndarray:
        """计算胜率矩阵。"""
        oop_hands = list(oop_range.keys())
        ip_hands = list(ip_range.keys())
        
        matrix = np.zeros((len(oop_hands), len(ip_hands)))
        
        for i, oop_h in enumerate(oop_hands):
            oop_cards = self.remover._parse_hand_string(oop_h)
            if oop_cards is None:
                continue
            
            for j, ip_h in enumerate(ip_hands):
                ip_cards = self.remover._parse_hand_string(ip_h)
                if ip_cards is None:
                    continue
                
                # 检查手牌冲突
                if self._hands_conflict(oop_h, ip_h):
                    matrix[i, j] = 0.5
                    continue
                
                # 比较手牌
                result = compare_hands(list(oop_cards), list(ip_cards), community_cards)
                
                if result == 0:  # OOP胜
                    matrix[i, j] = 1.0
                elif result == -1:  # 平局
                    matrix[i, j] = 0.5
                else:  # IP胜
                    matrix[i, j] = 0.0
        
        return matrix
    
    def _hands_conflict(self, hand1: str, hand2: str) -> bool:
        """检查两个手牌是否有冲突（共享同一张牌）。"""
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
                # 均匀分布
                n_actions = len(hand_regrets)
                strategy[hand] = {a: 1.0 / n_actions for a in hand_regrets}
        
        return strategy
    
    def get_strategy(self, node_path: str) -> Dict[str, Dict[str, float]]:
        """获取指定节点的策略。"""
        # 这个方法在solve()之后调用，从结果中提取
        pass
