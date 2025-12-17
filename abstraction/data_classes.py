"""Potential-Aware卡牌抽象模块的数据类定义。

本模块定义了卡牌抽象所需的核心数据类：
- AbstractionConfig: 抽象配置
- EquityHistogram: Equity分布直方图
- AbstractionResult: 抽象结果
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


@dataclass
class AbstractionConfig:
    """抽象配置数据类。
    
    定义了卡牌抽象的所有配置参数，包括每个游戏阶段的桶数量、
    Equity直方图的区间数、k-means聚类参数等。
    
    Attributes:
        preflop_buckets: 翻牌前桶数（169=无抽象，对应169种起手牌类型）
        flop_buckets: 翻牌阶段桶数
        turn_buckets: 转牌阶段桶数
        river_buckets: 河牌阶段桶数
        equity_bins: Equity直方图的区间数量（默认50，每个区间宽度0.02）
        kmeans_restarts: k-means算法的重启次数（选择最佳结果）
        kmeans_max_iters: k-means算法的最大迭代次数
        use_potential_aware: 是否使用Potential-Aware抽象（否则使用Distribution-Aware）
        random_seed: 随机种子（用于可重复性）
        num_workers: 并行计算的工作进程数（0=使用所有CPU核心）
    """
    preflop_buckets: int = 169  # 翻牌前桶数（169=无抽象）
    flop_buckets: int = 5000    # 翻牌桶数
    turn_buckets: int = 5000    # 转牌桶数
    river_buckets: int = 5000   # 河牌桶数
    equity_bins: int = 50       # Equity直方图区间数
    kmeans_restarts: int = 25   # k-means重启次数
    kmeans_max_iters: int = 100 # k-means最大迭代次数
    use_potential_aware: bool = True  # 是否使用Potential-Aware抽象
    random_seed: int = 42       # 随机种子
    num_workers: int = 0        # 并行工作进程数（0=使用所有CPU）
    
    def __post_init__(self):
        """验证配置参数的有效性。"""
        if self.preflop_buckets <= 0:
            raise ValueError(f"翻牌前桶数必须为正数，当前值：{self.preflop_buckets}")
        if self.preflop_buckets > 169:
            raise ValueError(f"翻牌前桶数不能超过169（起手牌类型数），当前值：{self.preflop_buckets}")
        if self.flop_buckets <= 0:
            raise ValueError(f"翻牌桶数必须为正数，当前值：{self.flop_buckets}")
        if self.turn_buckets <= 0:
            raise ValueError(f"转牌桶数必须为正数，当前值：{self.turn_buckets}")
        if self.river_buckets <= 0:
            raise ValueError(f"河牌桶数必须为正数，当前值：{self.river_buckets}")
        if self.equity_bins <= 0:
            raise ValueError(f"Equity区间数必须为正数，当前值：{self.equity_bins}")
        if self.kmeans_restarts <= 0:
            raise ValueError(f"k-means重启次数必须为正数，当前值：{self.kmeans_restarts}")
        if self.kmeans_max_iters <= 0:
            raise ValueError(f"k-means最大迭代次数必须为正数，当前值：{self.kmeans_max_iters}")
        if self.num_workers < 0:
            raise ValueError(f"工作进程数不能为负数，当前值：{self.num_workers}")
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式。
        
        Returns:
            包含所有配置参数的字典
        """
        return {
            'preflop_buckets': self.preflop_buckets,
            'flop_buckets': self.flop_buckets,
            'turn_buckets': self.turn_buckets,
            'river_buckets': self.river_buckets,
            'equity_bins': self.equity_bins,
            'kmeans_restarts': self.kmeans_restarts,
            'kmeans_max_iters': self.kmeans_max_iters,
            'use_potential_aware': self.use_potential_aware,
            'random_seed': self.random_seed,
            'num_workers': self.num_workers,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AbstractionConfig':
        """从字典创建配置对象。
        
        Args:
            data: 包含配置参数的字典
            
        Returns:
            AbstractionConfig实例
        """
        return cls(
            preflop_buckets=data.get('preflop_buckets', 169),
            flop_buckets=data.get('flop_buckets', 5000),
            turn_buckets=data.get('turn_buckets', 5000),
            river_buckets=data.get('river_buckets', 5000),
            equity_bins=data.get('equity_bins', 50),
            kmeans_restarts=data.get('kmeans_restarts', 25),
            kmeans_max_iters=data.get('kmeans_max_iters', 100),
            use_potential_aware=data.get('use_potential_aware', True),
            random_seed=data.get('random_seed', 42),
            num_workers=data.get('num_workers', 0),
        )


@dataclass
class EquityHistogram:
    """Equity分布直方图数据类。
    
    表示手牌在不同公共牌面下的胜率分布。直方图将[0, 1]的Equity范围
    划分为多个区间，记录落入每个区间的概率。
    
    Attributes:
        bins: 直方图区间边界数组，长度为num_bins+1
        counts: 每个区间的计数/概率数组，长度为num_bins
    """
    bins: np.ndarray  # 直方图区间边界
    counts: np.ndarray  # 每个区间的计数/概率
    
    def __post_init__(self):
        """验证直方图数据的有效性。"""
        if len(self.bins) != len(self.counts) + 1:
            raise ValueError(
                f"区间边界数量({len(self.bins)})必须比计数数量({len(self.counts)})多1"
            )
        if len(self.counts) == 0:
            raise ValueError("直方图不能为空")
        if np.any(self.counts < 0):
            raise ValueError("直方图计数不能为负数")
    
    @property
    def num_bins(self) -> int:
        """返回直方图的区间数量。"""
        return len(self.counts)
    
    def normalize(self) -> 'EquityHistogram':
        """归一化直方图，使概率和为1。
        
        Returns:
            归一化后的新EquityHistogram实例
        """
        total = np.sum(self.counts)
        if total == 0:
            # 如果总和为0，返回均匀分布
            normalized_counts = np.ones_like(self.counts) / len(self.counts)
        else:
            normalized_counts = self.counts / total
        return EquityHistogram(bins=self.bins.copy(), counts=normalized_counts)
    
    def is_normalized(self, tolerance: float = 1e-6) -> bool:
        """检查直方图是否已归一化。
        
        Args:
            tolerance: 允许的误差范围
            
        Returns:
            如果概率和在1±tolerance范围内，返回True
        """
        return abs(np.sum(self.counts) - 1.0) < tolerance
    
    def to_sparse(self) -> Tuple[np.ndarray, np.ndarray]:
        """转换为稀疏表示（非零索引和值）。
        
        Returns:
            (indices, values) 元组，其中indices是非零元素的索引，
            values是对应的值
        """
        nonzero_mask = self.counts != 0
        indices = np.where(nonzero_mask)[0]
        values = self.counts[nonzero_mask]
        return indices, values
    
    @classmethod
    def from_sparse(cls, indices: np.ndarray, values: np.ndarray, 
                    num_bins: int) -> 'EquityHistogram':
        """从稀疏表示创建直方图。
        
        Args:
            indices: 非零元素的索引
            values: 非零元素的值
            num_bins: 总区间数量
            
        Returns:
            EquityHistogram实例
        """
        counts = np.zeros(num_bins)
        counts[indices] = values
        bins = np.linspace(0, 1, num_bins + 1)
        return cls(bins=bins, counts=counts)
    
    @classmethod
    def create_empty(cls, num_bins: int = 50) -> 'EquityHistogram':
        """创建空的直方图。
        
        Args:
            num_bins: 区间数量
            
        Returns:
            空的EquityHistogram实例
        """
        bins = np.linspace(0, 1, num_bins + 1)
        counts = np.zeros(num_bins)
        return cls(bins=bins, counts=counts)
    
    def __eq__(self, other: object) -> bool:
        """检查两个直方图是否相等。"""
        if not isinstance(other, EquityHistogram):
            return False
        return (np.allclose(self.bins, other.bins) and 
                np.allclose(self.counts, other.counts))


@dataclass
class AbstractionResult:
    """抽象结果数据类。
    
    存储完整的卡牌抽象结果，包括配置、每个阶段的映射表、
    聚类中心和质量指标。
    
    Attributes:
        config: 抽象配置
        preflop_mapping: 翻牌前映射（起手牌索引 -> 桶ID）
        flop_mapping: 翻牌映射（翻牌手牌索引 -> 桶ID）
        turn_mapping: 转牌映射（转牌手牌索引 -> 桶ID）
        river_mapping: 河牌映射（河牌手牌索引 -> 桶ID）
        flop_centers: 翻牌聚类中心（直方图表示）
        turn_centers: 转牌聚类中心（直方图表示）
        river_centers: 河牌聚类中心（直方图表示）
        wcss: 每个阶段的Within-Cluster Sum of Squares
        generation_time: 生成抽象所花费的时间（秒）
    """
    config: AbstractionConfig
    preflop_mapping: Optional[np.ndarray] = None  # 翻牌前映射
    flop_mapping: Optional[np.ndarray] = None     # 翻牌映射
    turn_mapping: Optional[np.ndarray] = None     # 转牌映射
    river_mapping: Optional[np.ndarray] = None    # 河牌映射
    flop_centers: Optional[np.ndarray] = None     # 翻牌聚类中心
    turn_centers: Optional[np.ndarray] = None     # 转牌聚类中心
    river_centers: Optional[np.ndarray] = None    # 河牌聚类中心
    wcss: Dict[str, float] = field(default_factory=dict)  # 每个阶段的WCSS
    generation_time: float = 0.0  # 生成耗时（秒）
    
    def get_bucket_count(self, stage: str) -> int:
        """获取指定阶段的实际桶数量。
        
        Args:
            stage: 游戏阶段（'preflop', 'flop', 'turn', 'river'）
            
        Returns:
            该阶段的桶数量
        """
        mapping = self._get_mapping(stage)
        if mapping is None:
            return 0
        return len(np.unique(mapping))
    
    def get_bucket_sizes(self, stage: str) -> Dict[str, Any]:
        """获取指定阶段的桶大小统计。
        
        Args:
            stage: 游戏阶段
            
        Returns:
            包含桶数量、平均大小、最大大小、最小大小的字典
        """
        mapping = self._get_mapping(stage)
        if mapping is None:
            return {'count': 0, 'avg_size': 0, 'max_size': 0, 'min_size': 0}
        
        unique, counts = np.unique(mapping, return_counts=True)
        return {
            'count': len(unique),
            'avg_size': float(np.mean(counts)),
            'max_size': int(np.max(counts)),
            'min_size': int(np.min(counts)),
        }
    
    def _get_mapping(self, stage: str) -> Optional[np.ndarray]:
        """获取指定阶段的映射数组。"""
        stage_lower = stage.lower()
        if stage_lower == 'preflop':
            return self.preflop_mapping
        elif stage_lower == 'flop':
            return self.flop_mapping
        elif stage_lower == 'turn':
            return self.turn_mapping
        elif stage_lower == 'river':
            return self.river_mapping
        else:
            raise ValueError(f"未知的游戏阶段：{stage}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取抽象结果的完整统计信息。
        
        Returns:
            包含所有阶段统计信息的字典
        """
        stats = {
            'config': self.config.to_dict(),
            'generation_time': self.generation_time,
            'wcss': self.wcss,
            'stages': {}
        }
        
        for stage in ['preflop', 'flop', 'turn', 'river']:
            stats['stages'][stage] = self.get_bucket_sizes(stage)
        
        return stats
    
    def is_complete(self) -> bool:
        """检查抽象结果是否完整。
        
        Returns:
            如果所有阶段的映射都已生成，返回True
        """
        return (self.preflop_mapping is not None and
                self.flop_mapping is not None and
                self.turn_mapping is not None and
                self.river_mapping is not None)
    
    def to_dict(self) -> Dict[str, Any]:
        """将抽象结果转换为可序列化的字典格式。
        
        注意：numpy数组会被转换为列表
        
        Returns:
            可序列化的字典
        """
        return {
            'config': self.config.to_dict(),
            'preflop_mapping': self.preflop_mapping.tolist() if self.preflop_mapping is not None else None,
            'flop_mapping': self.flop_mapping.tolist() if self.flop_mapping is not None else None,
            'turn_mapping': self.turn_mapping.tolist() if self.turn_mapping is not None else None,
            'river_mapping': self.river_mapping.tolist() if self.river_mapping is not None else None,
            'flop_centers': self.flop_centers.tolist() if self.flop_centers is not None else None,
            'turn_centers': self.turn_centers.tolist() if self.turn_centers is not None else None,
            'river_centers': self.river_centers.tolist() if self.river_centers is not None else None,
            'wcss': self.wcss,
            'generation_time': self.generation_time,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AbstractionResult':
        """从字典创建抽象结果对象。
        
        Args:
            data: 包含抽象结果数据的字典
            
        Returns:
            AbstractionResult实例
        """
        config = AbstractionConfig.from_dict(data['config'])
        
        def to_array(value):
            if value is None:
                return None
            return np.array(value)
        
        return cls(
            config=config,
            preflop_mapping=to_array(data.get('preflop_mapping')),
            flop_mapping=to_array(data.get('flop_mapping')),
            turn_mapping=to_array(data.get('turn_mapping')),
            river_mapping=to_array(data.get('river_mapping')),
            flop_centers=to_array(data.get('flop_centers')),
            turn_centers=to_array(data.get('turn_centers')),
            river_centers=to_array(data.get('river_centers')),
            wcss=data.get('wcss', {}),
            generation_time=data.get('generation_time', 0.0),
        )
