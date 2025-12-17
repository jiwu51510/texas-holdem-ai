"""Potential-Aware卡牌抽象模块。

本模块实现了Potential-Aware抽象方法，用于减少德州扑克翻后阶段的状态空间。
通过考虑手牌在所有未来轮次的强度分布轨迹，将相似的信息集聚类在一起，
从而大幅减少需要学习的状态空间，加速训练收敛。

主要组件：
- AbstractionConfig: 抽象配置数据类
- EquityHistogram: Equity分布直方图数据类
- AbstractionResult: 抽象结果数据类
- EquityCalculator: Equity计算器
- EMDCalculator: Earth Mover's Distance计算器
- PotentialAwareAbstractor: Potential-Aware抽象器
- CardAbstraction: 卡牌抽象管理器
- AbstractionCache: 抽象缓存
"""

from abstraction.data_classes import (
    AbstractionConfig,
    EquityHistogram,
    AbstractionResult,
)

from abstraction.equity_calculator import (
    EquityCalculator,
    get_canonical_hand_index,
    index_to_canonical_hand,
)

from abstraction.emd_calculator import EMDCalculator
from abstraction.potential_aware_abstractor import PotentialAwareAbstractor
from abstraction.card_abstraction import CardAbstraction
from abstraction.abstraction_cache import AbstractionCache
from abstraction.abstraction_evaluator import (
    AbstractionEvaluator,
    BucketSizeStats,
    AbstractionQualityReport,
)

__all__ = [
    'AbstractionConfig',
    'EquityHistogram',
    'AbstractionResult',
    'EquityCalculator',
    'get_canonical_hand_index',
    'index_to_canonical_hand',
    'EMDCalculator',
    'PotentialAwareAbstractor',
    'CardAbstraction',
    'AbstractionCache',
    'AbstractionEvaluator',
    'BucketSizeStats',
    'AbstractionQualityReport',
]
