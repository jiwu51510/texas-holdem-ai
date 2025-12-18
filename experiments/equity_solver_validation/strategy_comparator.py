"""策略对比器模块。

本模块实现胜率方法与Solver方法的策略对比功能。
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from experiments.equity_solver_validation.data_models import (
    ValidationMetrics,
    ComparisonResult,
)


class StrategyComparator:
    """策略对比器。
    
    对比基于胜率的策略与Solver计算的策略。
    """
    
    def __init__(
        self,
        pot_size: float,
        bet_size: float,
        fold_threshold: float = 0.3,
        raise_threshold: float = 0.7
    ):
        """初始化对比器。
        
        Args:
            pot_size: 底池大小
            bet_size: 下注大小
            fold_threshold: 弃牌阈值（胜率低于此值倾向弃牌）
            raise_threshold: 加注阈值（胜率高于此值倾向加注）
        """
        self.pot_size = pot_size
        self.bet_size = bet_size
        self.fold_threshold = fold_threshold
        self.raise_threshold = raise_threshold
    
    def equity_to_strategy(
        self,
        equity_vector: Dict[str, float],
        action_type: str = 'oop_root'
    ) -> Dict[str, Dict[str, float]]:
        """将胜率向量转换为简化策略。
        
        基于胜率阈值决定动作概率分布。
        
        Args:
            equity_vector: 胜率向量 {hand: equity}
            action_type: 动作类型
                - 'oop_root': OOP根节点（check/bet）
                - 'ip_vs_check': IP面对check（check/bet）
                - 'ip_vs_bet': IP面对bet（fold/call）
            
        Returns:
            策略 {hand: {action: prob}}
        """
        strategy = {}
        
        for hand, equity in equity_vector.items():
            if action_type == 'oop_root':
                # OOP根节点：check或bet
                # 高胜率倾向bet，低胜率倾向check
                bet_prob = self._sigmoid(equity, center=0.5, steepness=5)
                strategy[hand] = {
                    'check': 1 - bet_prob,
                    'bet': bet_prob,
                }
            
            elif action_type == 'ip_vs_check':
                # IP面对check：check或bet
                bet_prob = self._sigmoid(equity, center=0.5, steepness=5)
                strategy[hand] = {
                    'check': 1 - bet_prob,
                    'bet': bet_prob,
                }
            
            elif action_type == 'ip_vs_bet':
                # IP面对bet：fold或call
                # 需要考虑底池赔率
                pot_odds = self.bet_size / (self.pot_size + 2 * self.bet_size)
                
                # 如果胜率高于底池赔率，倾向call
                if equity >= pot_odds:
                    call_prob = self._sigmoid(equity - pot_odds, center=0, steepness=10)
                else:
                    call_prob = self._sigmoid(equity - pot_odds, center=0, steepness=10)
                
                call_prob = max(0, min(1, call_prob))
                strategy[hand] = {
                    'fold': 1 - call_prob,
                    'call': call_prob,
                }
        
        return strategy
    
    def _sigmoid(self, x: float, center: float = 0, steepness: float = 1) -> float:
        """Sigmoid函数。"""
        return 1 / (1 + np.exp(-steepness * (x - center)))
    
    def compare_strategies(
        self,
        equity_strategy: Dict[str, Dict[str, float]],
        solver_strategy: Dict[str, Dict[str, float]]
    ) -> ComparisonResult:
        """对比两种策略。
        
        Args:
            equity_strategy: 基于胜率的策略
            solver_strategy: Solver策略
            
        Returns:
            ComparisonResult实例
        """
        # 找到共同的手牌
        common_hands = set(equity_strategy.keys()) & set(solver_strategy.keys())
        
        if not common_hands:
            return ComparisonResult(
                metrics=ValidationMetrics(),
                per_hand_diff={},
                action_distribution={},
            )
        
        # 计算各种指标
        tvd_values = []
        agreement_count = 0
        per_hand_diff = {}
        
        equity_action_counts = {}
        solver_action_counts = {}
        
        for hand in common_hands:
            eq_strat = equity_strategy[hand]
            sol_strat = solver_strategy[hand]
            
            # 确保动作集合相同
            actions = set(eq_strat.keys()) & set(sol_strat.keys())
            if not actions:
                continue
            
            # 计算总变差距离
            tvd = 0.5 * sum(abs(eq_strat.get(a, 0) - sol_strat.get(a, 0)) for a in actions)
            tvd_values.append(tvd)
            per_hand_diff[hand] = tvd
            
            # 检查最高概率动作是否一致
            eq_best = max(eq_strat.items(), key=lambda x: x[1])[0]
            sol_best = max(sol_strat.items(), key=lambda x: x[1])[0]
            if eq_best == sol_best:
                agreement_count += 1
            
            # 统计动作分布
            for action in actions:
                if action not in equity_action_counts:
                    equity_action_counts[action] = 0
                    solver_action_counts[action] = 0
                equity_action_counts[action] += eq_strat.get(action, 0)
                solver_action_counts[action] += sol_strat.get(action, 0)
        
        # 计算汇总指标
        avg_tvd = np.mean(tvd_values) if tvd_values else 0.0
        agreement_rate = agreement_count / len(common_hands) if common_hands else 0.0
        
        # 归一化动作分布
        total_eq = sum(equity_action_counts.values()) or 1
        total_sol = sum(solver_action_counts.values()) or 1
        
        action_distribution = {}
        all_actions = set(equity_action_counts.keys()) | set(solver_action_counts.keys())
        for action in all_actions:
            eq_freq = equity_action_counts.get(action, 0) / total_eq
            sol_freq = solver_action_counts.get(action, 0) / total_sol
            action_distribution[action] = (eq_freq, sol_freq)
        
        metrics = ValidationMetrics(
            total_variation_distance=avg_tvd,
            action_agreement_rate=agreement_rate,
        )
        
        return ComparisonResult(
            metrics=metrics,
            per_hand_diff=per_hand_diff,
            action_distribution=action_distribution,
            equity_strategy=equity_strategy,
            solver_strategy=solver_strategy,
        )
    
    def compute_total_variation_distance(
        self,
        dist1: Dict[str, float],
        dist2: Dict[str, float]
    ) -> float:
        """计算两个分布的总变差距离。
        
        TVD = 0.5 * sum(|p(x) - q(x)|)
        
        Args:
            dist1: 第一个分布
            dist2: 第二个分布
            
        Returns:
            总变差距离 [0, 1]
        """
        all_keys = set(dist1.keys()) | set(dist2.keys())
        tvd = 0.5 * sum(abs(dist1.get(k, 0) - dist2.get(k, 0)) for k in all_keys)
        return min(1.0, max(0.0, tvd))
    
    def compute_kl_divergence(
        self,
        p: Dict[str, float],
        q: Dict[str, float],
        epsilon: float = 1e-10
    ) -> float:
        """计算KL散度 D_KL(P || Q)。
        
        Args:
            p: 真实分布
            q: 近似分布
            epsilon: 平滑参数
            
        Returns:
            KL散度（非负）
        """
        all_keys = set(p.keys()) | set(q.keys())
        kl = 0.0
        
        for k in all_keys:
            p_k = p.get(k, 0) + epsilon
            q_k = q.get(k, 0) + epsilon
            
            # 归一化
            p_k = p_k / (sum(p.values()) + len(all_keys) * epsilon)
            q_k = q_k / (sum(q.values()) + len(all_keys) * epsilon)
            
            if p_k > 0:
                kl += p_k * np.log(p_k / q_k)
        
        return max(0.0, kl)
