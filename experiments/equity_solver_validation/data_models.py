"""胜率-Solver验证实验的数据模型定义。

本模块定义了实验所需的核心数据类：
- SolverConfig: Solver配置
- SolverResult: Solver计算结果
- ValidationMetrics: 验证指标
- ComparisonResult: 策略对比结果
- ExperimentScenario: 实验场景配置
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import json
import numpy as np

from models.core import Card


@dataclass
class SolverConfig:
    """Solver配置数据类。
    
    定义了博弈论求解器的所有配置参数，包括底池大小、筹码深度、
    下注选项等。
    
    Attributes:
        pot_size: 底池大小
        effective_stack: 有效筹码（双方中较小的筹码量）
        oop_bet_sizes: OOP（Out of Position）玩家的下注尺寸（占底池比例）
        ip_bet_sizes: IP（In Position）玩家的下注尺寸（占底池比例）
        oop_raise_sizes: OOP玩家的加注尺寸（占底池比例）
        ip_raise_sizes: IP玩家的加注尺寸（占底池比例）
        all_in_threshold: 全押阈值（当剩余筹码/底池 < 阈值时自动全押）
        max_iterations: 最大迭代次数
        target_exploitability: 目标可利用度（收敛条件）
    """
    pot_size: float
    effective_stack: float
    oop_bet_sizes: List[float] = field(default_factory=lambda: [0.5, 1.0])
    ip_bet_sizes: List[float] = field(default_factory=lambda: [0.5, 1.0])
    oop_raise_sizes: List[float] = field(default_factory=lambda: [0.5, 1.0])
    ip_raise_sizes: List[float] = field(default_factory=lambda: [0.5, 1.0])
    all_in_threshold: float = 0.67
    max_iterations: int = 1000
    target_exploitability: float = 0.5
    
    def __post_init__(self):
        """验证配置参数的有效性。"""
        if self.pot_size <= 0:
            raise ValueError(f"底池大小必须为正数，当前值：{self.pot_size}")
        if self.effective_stack < 0:
            raise ValueError(f"有效筹码不能为负数，当前值：{self.effective_stack}")
        if not 0 < self.all_in_threshold <= 1:
            raise ValueError(f"全押阈值必须在(0, 1]范围内，当前值：{self.all_in_threshold}")
        if self.max_iterations <= 0:
            raise ValueError(f"最大迭代次数必须为正数，当前值：{self.max_iterations}")
        if self.target_exploitability <= 0:
            raise ValueError(f"目标可利用度必须为正数，当前值：{self.target_exploitability}")
        
        # 验证下注尺寸
        for sizes, name in [
            (self.oop_bet_sizes, "OOP下注尺寸"),
            (self.ip_bet_sizes, "IP下注尺寸"),
            (self.oop_raise_sizes, "OOP加注尺寸"),
            (self.ip_raise_sizes, "IP加注尺寸"),
        ]:
            for size in sizes:
                if size <= 0:
                    raise ValueError(f"{name}必须为正数，当前值：{size}")
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式。"""
        return {
            'pot_size': self.pot_size,
            'effective_stack': self.effective_stack,
            'oop_bet_sizes': self.oop_bet_sizes,
            'ip_bet_sizes': self.ip_bet_sizes,
            'oop_raise_sizes': self.oop_raise_sizes,
            'ip_raise_sizes': self.ip_raise_sizes,
            'all_in_threshold': self.all_in_threshold,
            'max_iterations': self.max_iterations,
            'target_exploitability': self.target_exploitability,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SolverConfig':
        """从字典创建配置对象。"""
        return cls(
            pot_size=data['pot_size'],
            effective_stack=data['effective_stack'],
            oop_bet_sizes=data.get('oop_bet_sizes', [0.5, 1.0]),
            ip_bet_sizes=data.get('ip_bet_sizes', [0.5, 1.0]),
            oop_raise_sizes=data.get('oop_raise_sizes', [0.5, 1.0]),
            ip_raise_sizes=data.get('ip_raise_sizes', [0.5, 1.0]),
            all_in_threshold=data.get('all_in_threshold', 0.67),
            max_iterations=data.get('max_iterations', 1000),
            target_exploitability=data.get('target_exploitability', 0.5),
        )


@dataclass
class SolverResult:
    """Solver计算结果数据类。
    
    存储博弈论求解器的计算结果，包括可利用度、迭代次数、
    根节点EV和各节点策略。
    
    Attributes:
        exploitability: 可利用度（衡量策略接近纳什均衡的程度）
        iterations: 实际迭代次数
        root_ev: 根节点EV，(OOP_EV, IP_EV)元组
        strategies: 各节点的策略，格式为 {node_path: {hand: {action: prob}}}
        converged: 是否收敛
    """
    exploitability: float
    iterations: int
    root_ev: Tuple[float, float]
    strategies: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    converged: bool = True
    
    def __post_init__(self):
        """验证结果数据的有效性。"""
        if self.exploitability < 0:
            raise ValueError(f"可利用度不能为负数，当前值：{self.exploitability}")
        if self.iterations < 0:
            raise ValueError(f"迭代次数不能为负数，当前值：{self.iterations}")
    
    def get_strategy_at_node(self, node_path: str) -> Dict[str, Dict[str, float]]:
        """获取指定节点的策略。
        
        Args:
            node_path: 节点路径（如 "root", "root:check", "root:bet50"）
            
        Returns:
            该节点的策略，格式为 {hand: {action: prob}}
        """
        return self.strategies.get(node_path, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典格式。"""
        return {
            'exploitability': self.exploitability,
            'iterations': self.iterations,
            'root_ev': list(self.root_ev),
            'strategies': self.strategies,
            'converged': self.converged,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SolverResult':
        """从字典创建结果对象。"""
        return cls(
            exploitability=data['exploitability'],
            iterations=data['iterations'],
            root_ev=tuple(data['root_ev']),
            strategies=data.get('strategies', {}),
            converged=data.get('converged', True),
        )


@dataclass
class ValidationMetrics:
    """验证指标数据类。
    
    存储胜率方法与Solver方法对比的各种指标。
    
    Attributes:
        total_variation_distance: 总变差距离（策略分布差异）
        kl_divergence: KL散度（信息论距离）
        action_agreement_rate: 动作一致率（最高概率动作相同的比例）
        ev_correlation: EV相关系数
        ev_rmse: EV均方根误差
        equity_strategy_correlation: 胜率与策略的相关性
    """
    total_variation_distance: float = 0.0
    kl_divergence: float = 0.0
    action_agreement_rate: float = 0.0
    ev_correlation: float = 0.0
    ev_rmse: float = 0.0
    equity_strategy_correlation: float = 0.0
    
    def __post_init__(self):
        """验证指标数据的有效性。"""
        if not 0 <= self.total_variation_distance <= 1:
            raise ValueError(
                f"总变差距离必须在[0, 1]范围内，当前值：{self.total_variation_distance}"
            )
        if self.kl_divergence < 0:
            raise ValueError(f"KL散度不能为负数，当前值：{self.kl_divergence}")
        if not 0 <= self.action_agreement_rate <= 1:
            raise ValueError(
                f"动作一致率必须在[0, 1]范围内，当前值：{self.action_agreement_rate}"
            )
        if not -1 <= self.ev_correlation <= 1:
            raise ValueError(
                f"EV相关系数必须在[-1, 1]范围内，当前值：{self.ev_correlation}"
            )
        if self.ev_rmse < 0:
            raise ValueError(f"EV RMSE不能为负数，当前值：{self.ev_rmse}")
        if not -1 <= self.equity_strategy_correlation <= 1:
            raise ValueError(
                f"胜率-策略相关系数必须在[-1, 1]范围内，当前值：{self.equity_strategy_correlation}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """将指标转换为字典格式。"""
        return {
            'total_variation_distance': self.total_variation_distance,
            'kl_divergence': self.kl_divergence,
            'action_agreement_rate': self.action_agreement_rate,
            'ev_correlation': self.ev_correlation,
            'ev_rmse': self.ev_rmse,
            'equity_strategy_correlation': self.equity_strategy_correlation,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationMetrics':
        """从字典创建指标对象。"""
        return cls(
            total_variation_distance=data.get('total_variation_distance', 0.0),
            kl_divergence=data.get('kl_divergence', 0.0),
            action_agreement_rate=data.get('action_agreement_rate', 0.0),
            ev_correlation=data.get('ev_correlation', 0.0),
            ev_rmse=data.get('ev_rmse', 0.0),
            equity_strategy_correlation=data.get('equity_strategy_correlation', 0.0),
        )


@dataclass
class ComparisonResult:
    """策略对比结果数据类。
    
    存储胜率方法与Solver方法的详细对比结果。
    
    Attributes:
        metrics: 验证指标
        per_hand_diff: 每手牌的策略差异，格式为 {hand: diff_value}
        action_distribution: 动作分布对比，格式为 {action: (equity_freq, solver_freq)}
        equity_vector: 胜率向量，格式为 {hand: equity}
        solver_strategy: Solver策略，格式为 {hand: {action: prob}}
        equity_strategy: 基于胜率的策略，格式为 {hand: {action: prob}}
    """
    metrics: ValidationMetrics
    per_hand_diff: Dict[str, float] = field(default_factory=dict)
    action_distribution: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    equity_vector: Dict[str, float] = field(default_factory=dict)
    solver_strategy: Dict[str, Dict[str, float]] = field(default_factory=dict)
    equity_strategy: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def get_worst_hands(self, n: int = 10) -> List[Tuple[str, float]]:
        """获取策略差异最大的手牌。
        
        Args:
            n: 返回的手牌数量
            
        Returns:
            (hand, diff) 元组列表，按差异降序排列
        """
        sorted_hands = sorted(
            self.per_hand_diff.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_hands[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典格式。"""
        return {
            'metrics': self.metrics.to_dict(),
            'per_hand_diff': self.per_hand_diff,
            'action_distribution': {
                k: list(v) for k, v in self.action_distribution.items()
            },
            'equity_vector': self.equity_vector,
            'solver_strategy': self.solver_strategy,
            'equity_strategy': self.equity_strategy,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComparisonResult':
        """从字典创建结果对象。"""
        return cls(
            metrics=ValidationMetrics.from_dict(data['metrics']),
            per_hand_diff=data.get('per_hand_diff', {}),
            action_distribution={
                k: tuple(v) for k, v in data.get('action_distribution', {}).items()
            },
            equity_vector=data.get('equity_vector', {}),
            solver_strategy=data.get('solver_strategy', {}),
            equity_strategy=data.get('equity_strategy', {}),
        )


@dataclass
class ExperimentScenario:
    """实验场景配置数据类。
    
    定义了一个完整的实验场景，包括公共牌、双方范围和Solver配置。
    
    Attributes:
        name: 场景名称
        description: 场景描述
        community_cards: 5张公共牌（河牌阶段）
        oop_range: OOP玩家的范围，格式为 {hand: weight}
        ip_range: IP玩家的范围，格式为 {hand: weight}
        solver_config: Solver配置
        tags: 场景标签（如 "dry_board", "wet_board" 等）
    """
    name: str
    description: str
    community_cards: List[Card]
    oop_range: Dict[str, float]
    ip_range: Dict[str, float]
    solver_config: SolverConfig
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """验证场景配置的有效性。"""
        if not self.name:
            raise ValueError("场景名称不能为空")
        if len(self.community_cards) != 5:
            raise ValueError(
                f"河牌阶段必须有5张公共牌，当前：{len(self.community_cards)}"
            )
        if not self.oop_range:
            raise ValueError("OOP范围不能为空")
        if not self.ip_range:
            raise ValueError("IP范围不能为空")
        
        # 验证范围权重
        for range_name, range_dict in [("OOP", self.oop_range), ("IP", self.ip_range)]:
            for hand, weight in range_dict.items():
                if weight < 0:
                    raise ValueError(f"{range_name}范围中手牌{hand}的权重不能为负数")
        
        # 验证公共牌无重复
        card_set = set()
        for card in self.community_cards:
            card_key = (card.rank, card.suit)
            if card_key in card_set:
                raise ValueError(f"公共牌中有重复的牌：{card}")
            card_set.add(card_key)
    
    def get_community_cards_str(self) -> str:
        """获取公共牌的字符串表示。"""
        return ' '.join(str(card) for card in self.community_cards)
    
    def to_dict(self) -> Dict[str, Any]:
        """将场景转换为字典格式。"""
        return {
            'name': self.name,
            'description': self.description,
            'community_cards': [
                {'rank': c.rank, 'suit': c.suit} for c in self.community_cards
            ],
            'oop_range': self.oop_range,
            'ip_range': self.ip_range,
            'solver_config': self.solver_config.to_dict(),
            'tags': self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentScenario':
        """从字典创建场景对象。"""
        community_cards = [
            Card(rank=c['rank'], suit=c['suit'])
            for c in data['community_cards']
        ]
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            community_cards=community_cards,
            oop_range=data['oop_range'],
            ip_range=data['ip_range'],
            solver_config=SolverConfig.from_dict(data['solver_config']),
            tags=data.get('tags', []),
        )
    
    def save_to_file(self, filepath: str) -> None:
        """保存场景到JSON文件。"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ExperimentScenario':
        """从JSON文件加载场景。"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ExperimentResult:
    """实验结果数据类。
    
    存储单个实验场景的完整结果。
    
    Attributes:
        scenario: 实验场景
        comparison: 对比结果
        solver_result: Solver原始结果
        execution_time: 执行时间（秒）
        error: 错误信息（如果有）
    """
    scenario: ExperimentScenario
    comparison: Optional[ComparisonResult] = None
    solver_result: Optional[SolverResult] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """检查实验是否成功。"""
        return self.error is None and self.comparison is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典格式。"""
        return {
            'scenario': self.scenario.to_dict(),
            'comparison': self.comparison.to_dict() if self.comparison else None,
            'solver_result': self.solver_result.to_dict() if self.solver_result else None,
            'execution_time': self.execution_time,
            'error': self.error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        """从字典创建结果对象。"""
        return cls(
            scenario=ExperimentScenario.from_dict(data['scenario']),
            comparison=ComparisonResult.from_dict(data['comparison']) if data.get('comparison') else None,
            solver_result=SolverResult.from_dict(data['solver_result']) if data.get('solver_result') else None,
            execution_time=data.get('execution_time', 0.0),
            error=data.get('error'),
        )


@dataclass
class BatchExperimentResult:
    """批量实验结果数据类。
    
    存储多个实验场景的汇总结果。
    
    Attributes:
        results: 各场景的实验结果列表
        summary_metrics: 汇总指标
        total_execution_time: 总执行时间（秒）
    """
    results: List[ExperimentResult] = field(default_factory=list)
    summary_metrics: Dict[str, float] = field(default_factory=dict)
    total_execution_time: float = 0.0
    
    @property
    def success_count(self) -> int:
        """成功的实验数量。"""
        return sum(1 for r in self.results if r.success)
    
    @property
    def failure_count(self) -> int:
        """失败的实验数量。"""
        return sum(1 for r in self.results if not r.success)
    
    def compute_summary(self) -> None:
        """计算汇总指标。"""
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return
        
        # 收集各指标
        tvd_values = []
        agreement_values = []
        ev_corr_values = []
        
        for result in successful_results:
            if result.comparison:
                tvd_values.append(result.comparison.metrics.total_variation_distance)
                agreement_values.append(result.comparison.metrics.action_agreement_rate)
                ev_corr_values.append(result.comparison.metrics.ev_correlation)
        
        # 计算汇总
        if tvd_values:
            self.summary_metrics['avg_total_variation_distance'] = np.mean(tvd_values)
            self.summary_metrics['max_total_variation_distance'] = np.max(tvd_values)
            self.summary_metrics['min_total_variation_distance'] = np.min(tvd_values)
        
        if agreement_values:
            self.summary_metrics['avg_action_agreement_rate'] = np.mean(agreement_values)
            self.summary_metrics['min_action_agreement_rate'] = np.min(agreement_values)
        
        if ev_corr_values:
            self.summary_metrics['avg_ev_correlation'] = np.mean(ev_corr_values)
        
        self.summary_metrics['success_rate'] = len(successful_results) / len(self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典格式。"""
        return {
            'results': [r.to_dict() for r in self.results],
            'summary_metrics': self.summary_metrics,
            'total_execution_time': self.total_execution_time,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchExperimentResult':
        """从字典创建结果对象。"""
        return cls(
            results=[ExperimentResult.from_dict(r) for r in data.get('results', [])],
            summary_metrics=data.get('summary_metrics', {}),
            total_execution_time=data.get('total_execution_time', 0.0),
        )
