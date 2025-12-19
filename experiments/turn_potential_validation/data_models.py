"""转牌阶段Potential直方图验证实验的数据模型定义。

本模块定义了转牌验证实验所需的核心数据类：
- TurnScenario: 转牌实验场景配置
- CorrelationResult: Potential直方图与策略的相关性结果
- ClusteringComparisonResult: 聚类与策略比较结果
- TurnValidationMetrics: 转牌验证指标
- TurnExperimentResult: 单个实验结果
- TurnBatchExperimentResult: 批量实验结果
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import json
import numpy as np

from models.core import Card
from experiments.equity_solver_validation.data_models import SolverConfig, SolverResult


@dataclass
class TurnScenario:
    """转牌实验场景配置数据类。
    
    定义了一个完整的转牌实验场景，包括4张公共牌、双方范围和Solver配置。
    
    Attributes:
        name: 场景名称
        description: 场景描述
        turn_community: 4张公共牌（翻牌+转牌）
        oop_range: OOP玩家的范围，格式为 {hand: weight}
        ip_range: IP玩家的范围，格式为 {hand: weight}
        solver_config: Solver配置
        tags: 场景标签（如 "dry_board", "wet_board", "draw_board" 等）
        num_histogram_bins: Potential直方图的区间数量
    """
    name: str
    description: str
    turn_community: List[Card]
    oop_range: Dict[str, float]
    ip_range: Dict[str, float]
    solver_config: SolverConfig
    tags: List[str] = field(default_factory=list)
    num_histogram_bins: int = 50
    
    def __post_init__(self):
        """验证场景配置的有效性。"""
        if not self.name:
            raise ValueError("场景名称不能为空")
        if len(self.turn_community) != 4:
            raise ValueError(
                f"转牌阶段必须有4张公共牌，当前：{len(self.turn_community)}"
            )
        if not self.oop_range:
            raise ValueError("OOP范围不能为空")
        if not self.ip_range:
            raise ValueError("IP范围不能为空")
        if self.num_histogram_bins <= 0:
            raise ValueError(f"直方图区间数必须为正数，当前：{self.num_histogram_bins}")
        
        # 验证范围权重
        for range_name, range_dict in [("OOP", self.oop_range), ("IP", self.ip_range)]:
            for hand, weight in range_dict.items():
                if weight < 0:
                    raise ValueError(f"{range_name}范围中手牌{hand}的权重不能为负数")
        
        # 验证公共牌无重复
        card_set = set()
        for card in self.turn_community:
            card_key = (card.rank, card.suit)
            if card_key in card_set:
                raise ValueError(f"公共牌中有重复的牌：{card}")
            card_set.add(card_key)
    
    def get_turn_community_str(self) -> str:
        """获取转牌公共牌的字符串表示。"""
        return ' '.join(str(card) for card in self.turn_community)
    
    def to_dict(self) -> Dict[str, Any]:
        """将场景转换为字典格式。"""
        return {
            'name': self.name,
            'description': self.description,
            'turn_community': [
                {'rank': c.rank, 'suit': c.suit} for c in self.turn_community
            ],
            'oop_range': self.oop_range,
            'ip_range': self.ip_range,
            'solver_config': self.solver_config.to_dict(),
            'tags': self.tags,
            'num_histogram_bins': self.num_histogram_bins,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TurnScenario':
        """从字典创建场景对象。"""
        turn_community = [
            Card(rank=c['rank'], suit=c['suit'])
            for c in data['turn_community']
        ]
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            turn_community=turn_community,
            oop_range=data['oop_range'],
            ip_range=data['ip_range'],
            solver_config=SolverConfig.from_dict(data['solver_config']),
            tags=data.get('tags', []),
            num_histogram_bins=data.get('num_histogram_bins', 50),
        )
    
    def save_to_file(self, filepath: str) -> None:
        """保存场景到JSON文件。"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TurnScenario':
        """从JSON文件加载场景。"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class CorrelationResult:
    """Potential直方图与策略的相关性结果数据类。
    
    存储Potential直方图特征与Solver策略之间的相关性分析结果。
    
    Attributes:
        mean_equity_correlation: 平均Equity与策略的相关性
        variance_correlation: Equity方差与策略的相关性
        intra_action_emd: 同一动作内手牌的平均EMD距离
        inter_action_emd: 不同动作间手牌的平均EMD距离
        clustering_purity: 聚类纯度
        histogram_entropy_correlation: 直方图熵与策略的相关性
    """
    mean_equity_correlation: float = 0.0
    variance_correlation: float = 0.0
    intra_action_emd: Dict[str, float] = field(default_factory=dict)
    inter_action_emd: Dict[str, float] = field(default_factory=dict)
    clustering_purity: float = 0.0
    histogram_entropy_correlation: float = 0.0
    
    def __post_init__(self):
        """验证结果数据的有效性。"""
        if not -1 <= self.mean_equity_correlation <= 1:
            raise ValueError(
                f"平均Equity相关系数必须在[-1, 1]范围内，当前值：{self.mean_equity_correlation}"
            )
        if not -1 <= self.variance_correlation <= 1:
            raise ValueError(
                f"方差相关系数必须在[-1, 1]范围内，当前值：{self.variance_correlation}"
            )
        if not 0 <= self.clustering_purity <= 1:
            raise ValueError(
                f"聚类纯度必须在[0, 1]范围内，当前值：{self.clustering_purity}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典格式。"""
        return {
            'mean_equity_correlation': self.mean_equity_correlation,
            'variance_correlation': self.variance_correlation,
            'intra_action_emd': self.intra_action_emd,
            'inter_action_emd': self.inter_action_emd,
            'clustering_purity': self.clustering_purity,
            'histogram_entropy_correlation': self.histogram_entropy_correlation,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorrelationResult':
        """从字典创建结果对象。"""
        return cls(
            mean_equity_correlation=data.get('mean_equity_correlation', 0.0),
            variance_correlation=data.get('variance_correlation', 0.0),
            intra_action_emd=data.get('intra_action_emd', {}),
            inter_action_emd=data.get('inter_action_emd', {}),
            clustering_purity=data.get('clustering_purity', 0.0),
            histogram_entropy_correlation=data.get('histogram_entropy_correlation', 0.0),
        )


@dataclass
class ClusteringComparisonResult:
    """聚类与策略比较结果数据类。
    
    存储基于Potential直方图的聚类与Solver策略的比较结果。
    
    Attributes:
        num_clusters: 聚类数量
        purity: 聚类纯度（同一聚类内主要动作的比例）
        normalized_mutual_info: 归一化互信息
        action_distribution_per_cluster: 每个聚类的动作分布
        cluster_sizes: 每个聚类的大小
        silhouette_score: 轮廓系数
    """
    num_clusters: int = 0
    purity: float = 0.0
    normalized_mutual_info: float = 0.0
    action_distribution_per_cluster: Dict[int, Dict[str, float]] = field(default_factory=dict)
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    silhouette_score: float = 0.0
    
    def __post_init__(self):
        """验证结果数据的有效性。"""
        if self.num_clusters < 0:
            raise ValueError(f"聚类数量不能为负数，当前值：{self.num_clusters}")
        if not 0 <= self.purity <= 1:
            raise ValueError(f"聚类纯度必须在[0, 1]范围内，当前值：{self.purity}")
        if not 0 <= self.normalized_mutual_info <= 1:
            raise ValueError(
                f"归一化互信息必须在[0, 1]范围内，当前值：{self.normalized_mutual_info}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典格式。"""
        return {
            'num_clusters': self.num_clusters,
            'purity': self.purity,
            'normalized_mutual_info': self.normalized_mutual_info,
            'action_distribution_per_cluster': {
                str(k): v for k, v in self.action_distribution_per_cluster.items()
            },
            'cluster_sizes': {str(k): v for k, v in self.cluster_sizes.items()},
            'silhouette_score': self.silhouette_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusteringComparisonResult':
        """从字典创建结果对象。"""
        return cls(
            num_clusters=data.get('num_clusters', 0),
            purity=data.get('purity', 0.0),
            normalized_mutual_info=data.get('normalized_mutual_info', 0.0),
            action_distribution_per_cluster={
                int(k): v for k, v in data.get('action_distribution_per_cluster', {}).items()
            },
            cluster_sizes={
                int(k): v for k, v in data.get('cluster_sizes', {}).items()
            },
            silhouette_score=data.get('silhouette_score', 0.0),
        )


@dataclass
class TurnValidationMetrics:
    """转牌验证指标数据类。
    
    存储转牌阶段Potential直方图验证的各种指标。
    
    Attributes:
        avg_histogram_entropy: 平均直方图熵
        histogram_sparsity: 直方图稀疏度（非零区间比例）
        strategy_correlation: 策略相关性
        ev_correlation: EV相关性
        clustering_purity: 聚类纯度
        silhouette_score: 轮廓系数
        avg_intra_action_emd: 同一动作内的平均EMD
        avg_inter_action_emd: 不同动作间的平均EMD
        action_agreement_rate: 动作一致率
    """
    avg_histogram_entropy: float = 0.0
    histogram_sparsity: float = 0.0
    strategy_correlation: float = 0.0
    ev_correlation: float = 0.0
    clustering_purity: float = 0.0
    silhouette_score: float = 0.0
    avg_intra_action_emd: float = 0.0
    avg_inter_action_emd: float = 0.0
    action_agreement_rate: float = 0.0
    
    def __post_init__(self):
        """验证指标数据的有效性。"""
        if self.avg_histogram_entropy < 0:
            raise ValueError(f"平均直方图熵不能为负数，当前值：{self.avg_histogram_entropy}")
        if not 0 <= self.histogram_sparsity <= 1:
            raise ValueError(
                f"直方图稀疏度必须在[0, 1]范围内，当前值：{self.histogram_sparsity}"
            )
        if not -1 <= self.strategy_correlation <= 1:
            raise ValueError(
                f"策略相关性必须在[-1, 1]范围内，当前值：{self.strategy_correlation}"
            )
        if not -1 <= self.ev_correlation <= 1:
            raise ValueError(
                f"EV相关性必须在[-1, 1]范围内，当前值：{self.ev_correlation}"
            )
        if not 0 <= self.clustering_purity <= 1:
            raise ValueError(
                f"聚类纯度必须在[0, 1]范围内，当前值：{self.clustering_purity}"
            )
        if not 0 <= self.action_agreement_rate <= 1:
            raise ValueError(
                f"动作一致率必须在[0, 1]范围内，当前值：{self.action_agreement_rate}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """将指标转换为字典格式。"""
        return {
            'avg_histogram_entropy': self.avg_histogram_entropy,
            'histogram_sparsity': self.histogram_sparsity,
            'strategy_correlation': self.strategy_correlation,
            'ev_correlation': self.ev_correlation,
            'clustering_purity': self.clustering_purity,
            'silhouette_score': self.silhouette_score,
            'avg_intra_action_emd': self.avg_intra_action_emd,
            'avg_inter_action_emd': self.avg_inter_action_emd,
            'action_agreement_rate': self.action_agreement_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TurnValidationMetrics':
        """从字典创建指标对象。"""
        return cls(
            avg_histogram_entropy=data.get('avg_histogram_entropy', 0.0),
            histogram_sparsity=data.get('histogram_sparsity', 0.0),
            strategy_correlation=data.get('strategy_correlation', 0.0),
            ev_correlation=data.get('ev_correlation', 0.0),
            clustering_purity=data.get('clustering_purity', 0.0),
            silhouette_score=data.get('silhouette_score', 0.0),
            avg_intra_action_emd=data.get('avg_intra_action_emd', 0.0),
            avg_inter_action_emd=data.get('avg_inter_action_emd', 0.0),
            action_agreement_rate=data.get('action_agreement_rate', 0.0),
        )


@dataclass
class TurnExperimentResult:
    """转牌实验结果数据类。
    
    存储单个转牌实验场景的完整结果。
    
    Attributes:
        scenario: 实验场景
        potential_histograms: 每个手牌的Potential直方图
        correlation_result: 相关性分析结果
        clustering_result: 聚类比较结果
        metrics: 验证指标
        solver_result: Solver原始结果
        execution_time: 执行时间（秒）
        error: 错误信息（如果有）
    """
    scenario: TurnScenario
    potential_histograms: Dict[str, np.ndarray] = field(default_factory=dict)
    correlation_result: Optional[CorrelationResult] = None
    clustering_result: Optional[ClusteringComparisonResult] = None
    metrics: Optional[TurnValidationMetrics] = None
    solver_result: Optional[SolverResult] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """检查实验是否成功。"""
        return self.error is None and self.metrics is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典格式。"""
        # 将numpy数组转换为列表
        histograms_dict = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self.potential_histograms.items()
        }
        
        return {
            'scenario': self.scenario.to_dict(),
            'potential_histograms': histograms_dict,
            'correlation_result': self.correlation_result.to_dict() if self.correlation_result else None,
            'clustering_result': self.clustering_result.to_dict() if self.clustering_result else None,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'solver_result': self.solver_result.to_dict() if self.solver_result else None,
            'execution_time': self.execution_time,
            'error': self.error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TurnExperimentResult':
        """从字典创建结果对象。"""
        # 将列表转换回numpy数组
        histograms = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in data.get('potential_histograms', {}).items()
        }
        
        return cls(
            scenario=TurnScenario.from_dict(data['scenario']),
            potential_histograms=histograms,
            correlation_result=CorrelationResult.from_dict(data['correlation_result']) 
                if data.get('correlation_result') else None,
            clustering_result=ClusteringComparisonResult.from_dict(data['clustering_result'])
                if data.get('clustering_result') else None,
            metrics=TurnValidationMetrics.from_dict(data['metrics']) if data.get('metrics') else None,
            solver_result=SolverResult.from_dict(data['solver_result']) if data.get('solver_result') else None,
            execution_time=data.get('execution_time', 0.0),
            error=data.get('error'),
        )


@dataclass
class TurnBatchExperimentResult:
    """转牌批量实验结果数据类。
    
    存储多个转牌实验场景的汇总结果。
    
    Attributes:
        results: 各场景的实验结果列表
        summary_metrics: 汇总指标
        total_execution_time: 总执行时间（秒）
    """
    results: List[TurnExperimentResult] = field(default_factory=list)
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
        entropy_values = []
        sparsity_values = []
        strategy_corr_values = []
        ev_corr_values = []
        purity_values = []
        agreement_values = []
        
        for result in successful_results:
            if result.metrics:
                entropy_values.append(result.metrics.avg_histogram_entropy)
                sparsity_values.append(result.metrics.histogram_sparsity)
                strategy_corr_values.append(result.metrics.strategy_correlation)
                ev_corr_values.append(result.metrics.ev_correlation)
                purity_values.append(result.metrics.clustering_purity)
                agreement_values.append(result.metrics.action_agreement_rate)
        
        # 计算汇总
        if entropy_values:
            self.summary_metrics['avg_histogram_entropy'] = float(np.mean(entropy_values))
        
        if sparsity_values:
            self.summary_metrics['avg_histogram_sparsity'] = float(np.mean(sparsity_values))
        
        if strategy_corr_values:
            self.summary_metrics['avg_strategy_correlation'] = float(np.mean(strategy_corr_values))
        
        if ev_corr_values:
            self.summary_metrics['avg_ev_correlation'] = float(np.mean(ev_corr_values))
        
        if purity_values:
            self.summary_metrics['avg_clustering_purity'] = float(np.mean(purity_values))
        
        if agreement_values:
            self.summary_metrics['avg_action_agreement_rate'] = float(np.mean(agreement_values))
        
        self.summary_metrics['success_rate'] = len(successful_results) / len(self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典格式。"""
        return {
            'results': [r.to_dict() for r in self.results],
            'summary_metrics': self.summary_metrics,
            'total_execution_time': self.total_execution_time,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TurnBatchExperimentResult':
        """从字典创建结果对象。"""
        return cls(
            results=[TurnExperimentResult.from_dict(r) for r in data.get('results', [])],
            summary_metrics=data.get('summary_metrics', {}),
            total_execution_time=data.get('total_execution_time', 0.0),
        )
    
    def save_to_file(self, filepath: str) -> None:
        """保存结果到JSON文件。"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TurnBatchExperimentResult':
        """从JSON文件加载结果。"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
