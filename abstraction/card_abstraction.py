"""卡牌抽象管理器模块。

本模块实现了CardAbstraction类，用于管理完整的卡牌抽象生成、
查询、保存和加载功能。

主要功能：
- generate_abstraction: 生成完整的卡牌抽象
- get_bucket_id: 查询手牌组合对应的桶ID
- save: 保存抽象结果到文件
- load: 从文件加载抽象结果
- get_abstraction_stats: 获取抽象统计信息
"""

import json
import os
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from models.core import Card, GameStage
from abstraction.data_classes import AbstractionConfig, AbstractionResult
from abstraction.potential_aware_abstractor import PotentialAwareAbstractor
from abstraction.equity_calculator import get_canonical_hand_index


class CardAbstraction:
    """卡牌抽象管理器。
    
    管理完整的卡牌抽象生命周期，包括生成、查询、保存和加载。
    支持Potential-Aware和Distribution-Aware两种抽象方法。
    
    Attributes:
        config: 抽象配置
        result: 抽象结果（生成或加载后可用）
        _abstractor: 内部使用的抽象器实例
    """
    
    def __init__(self, config: Optional[AbstractionConfig] = None):
        """初始化卡牌抽象管理器。
        
        Args:
            config: 抽象配置，如果为None则使用默认配置
        """
        self.config = config or AbstractionConfig()
        self.result: Optional[AbstractionResult] = None
        self._abstractor: Optional[PotentialAwareAbstractor] = None
    
    def generate_abstraction(self) -> AbstractionResult:
        """生成完整的卡牌抽象。
        
        按照从后向前的顺序计算：河牌 -> 转牌 -> 翻牌 -> 翻牌前
        
        Returns:
            AbstractionResult实例，包含所有阶段的抽象映射
        """
        self._abstractor = PotentialAwareAbstractor(self.config)
        self.result = self._abstractor.generate_full_abstraction()
        return self.result
    
    def get_bucket_id(self, hole_cards: Tuple[Card, Card],
                      community_cards: List[Card]) -> int:
        """获取手牌组合对应的桶ID。
        
        根据当前游戏阶段（由公共牌数量决定）查询对应的桶ID。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 当前的公共牌（0-5张）
            
        Returns:
            桶ID（整数）
            
        Raises:
            ValueError: 如果抽象结果未生成或加载
            ValueError: 如果公共牌数量无效
        """
        if self.result is None:
            raise ValueError("抽象结果未生成或加载，请先调用generate_abstraction()或load()")
        
        num_community = len(community_cards)
        
        if num_community == 0:
            # 翻牌前：使用规范化手牌索引
            return self._get_preflop_bucket(hole_cards)
        elif num_community == 3:
            # 翻牌阶段
            return self._get_flop_bucket(hole_cards, community_cards)
        elif num_community == 4:
            # 转牌阶段
            return self._get_turn_bucket(hole_cards, community_cards)
        elif num_community == 5:
            # 河牌阶段
            return self._get_river_bucket(hole_cards, community_cards)
        else:
            raise ValueError(f"无效的公共牌数量：{num_community}，必须是0、3、4或5")
    
    def _get_preflop_bucket(self, hole_cards: Tuple[Card, Card]) -> int:
        """获取翻牌前的桶ID。
        
        翻牌前使用规范化手牌索引作为桶ID。
        
        Args:
            hole_cards: 玩家的两张手牌
            
        Returns:
            桶ID
        """
        canonical_index = get_canonical_hand_index(hole_cards)
        
        if self.result.preflop_mapping is not None:
            if canonical_index < len(self.result.preflop_mapping):
                return int(self.result.preflop_mapping[canonical_index])
        
        # 如果没有映射，直接返回规范化索引
        return canonical_index
    
    def _get_flop_bucket(self, hole_cards: Tuple[Card, Card],
                         community_cards: List[Card]) -> int:
        """获取翻牌阶段的桶ID。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 翻牌的3张公共牌
            
        Returns:
            桶ID
        """
        if self.result.flop_mapping is None:
            return 0
        
        # 计算翻牌手牌的索引
        index = self._compute_hand_index(hole_cards, community_cards)
        
        if index < len(self.result.flop_mapping):
            return int(self.result.flop_mapping[index])
        
        # 索引超出范围，使用模运算
        return int(self.result.flop_mapping[index % len(self.result.flop_mapping)])
    
    def _get_turn_bucket(self, hole_cards: Tuple[Card, Card],
                         community_cards: List[Card]) -> int:
        """获取转牌阶段的桶ID。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 转牌的4张公共牌
            
        Returns:
            桶ID
        """
        if self.result.turn_mapping is None:
            return 0
        
        # 计算转牌手牌的索引
        index = self._compute_hand_index(hole_cards, community_cards)
        
        if index < len(self.result.turn_mapping):
            return int(self.result.turn_mapping[index])
        
        return int(self.result.turn_mapping[index % len(self.result.turn_mapping)])
    
    def _get_river_bucket(self, hole_cards: Tuple[Card, Card],
                          community_cards: List[Card]) -> int:
        """获取河牌阶段的桶ID。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 河牌的5张公共牌
            
        Returns:
            桶ID
        """
        if self.result.river_mapping is None:
            return 0
        
        # 计算河牌手牌的索引
        index = self._compute_hand_index(hole_cards, community_cards)
        
        if index < len(self.result.river_mapping):
            return int(self.result.river_mapping[index])
        
        return int(self.result.river_mapping[index % len(self.result.river_mapping)])
    
    def _compute_hand_index(self, hole_cards: Tuple[Card, Card],
                            community_cards: List[Card]) -> int:
        """计算手牌组合的唯一索引。
        
        将手牌和公共牌组合编码为唯一索引。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 公共牌
            
        Returns:
            手牌组合的唯一索引
        """
        def card_to_index(card: Card) -> int:
            """将牌转换为0-51的索引。"""
            suit_order = {'h': 0, 'd': 1, 'c': 2, 's': 3}
            return (card.rank - 2) * 4 + suit_order.get(card.suit, 0)
        
        # 对手牌排序以确保一致性
        hole_indices = sorted([card_to_index(c) for c in hole_cards])
        community_indices = sorted([card_to_index(c) for c in community_cards])
        
        # 组合成唯一索引
        # 使用多项式哈希
        index = 0
        base = 52
        
        for idx in hole_indices:
            index = index * base + idx
        
        for idx in community_indices:
            index = index * base + idx
        
        # 限制索引范围以避免溢出
        return index % (10 ** 9)
    
    def save(self, path: str) -> None:
        """保存抽象结果到文件。
        
        保存内容包括：
        - 抽象配置（JSON格式）
        - 每个阶段的桶映射表（numpy数组）
        - 聚类中心（numpy数组）
        - 元数据（生成时间、WCSS等）
        
        Args:
            path: 保存路径（目录路径）
            
        Raises:
            ValueError: 如果抽象结果未生成
        """
        if self.result is None:
            raise ValueError("抽象结果未生成，请先调用generate_abstraction()")
        
        # 创建目录
        os.makedirs(path, exist_ok=True)
        
        # 保存元数据和配置
        metadata = {
            'config': self.result.config.to_dict(),
            'wcss': self.result.wcss,
            'generation_time': self.result.generation_time,
            'mappings': {
                'preflop': 'preflop_mapping.npy',
                'flop': 'flop_mapping.npy',
                'turn': 'turn_mapping.npy',
                'river': 'river_mapping.npy',
            },
            'centers': {
                'flop': 'flop_centers.npy',
                'turn': 'turn_centers.npy',
                'river': 'river_centers.npy',
            }
        }
        
        metadata_path = os.path.join(path, 'abstraction.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 保存映射数组
        if self.result.preflop_mapping is not None:
            np.save(os.path.join(path, 'preflop_mapping.npy'), 
                    self.result.preflop_mapping)
        
        if self.result.flop_mapping is not None:
            np.save(os.path.join(path, 'flop_mapping.npy'), 
                    self.result.flop_mapping)
        
        if self.result.turn_mapping is not None:
            np.save(os.path.join(path, 'turn_mapping.npy'), 
                    self.result.turn_mapping)
        
        if self.result.river_mapping is not None:
            np.save(os.path.join(path, 'river_mapping.npy'), 
                    self.result.river_mapping)
        
        # 保存聚类中心
        if self.result.flop_centers is not None:
            np.save(os.path.join(path, 'flop_centers.npy'), 
                    self.result.flop_centers)
        
        if self.result.turn_centers is not None:
            np.save(os.path.join(path, 'turn_centers.npy'), 
                    self.result.turn_centers)
        
        if self.result.river_centers is not None:
            np.save(os.path.join(path, 'river_centers.npy'), 
                    self.result.river_centers)
    
    def load(self, path: str) -> AbstractionResult:
        """从文件加载抽象结果。
        
        Args:
            path: 抽象文件路径（目录路径）
            
        Returns:
            加载的AbstractionResult实例
            
        Raises:
            FileNotFoundError: 如果文件不存在
            ValueError: 如果文件格式无效
        """
        metadata_path = os.path.join(path, 'abstraction.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"抽象元数据文件不存在：{metadata_path}")
        
        # 加载元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 解析配置
        config = AbstractionConfig.from_dict(metadata['config'])
        self.config = config
        
        # 加载映射数组
        def load_array(filename: str) -> Optional[np.ndarray]:
            filepath = os.path.join(path, filename)
            if os.path.exists(filepath):
                return np.load(filepath)
            return None
        
        preflop_mapping = load_array('preflop_mapping.npy')
        flop_mapping = load_array('flop_mapping.npy')
        turn_mapping = load_array('turn_mapping.npy')
        river_mapping = load_array('river_mapping.npy')
        
        # 加载聚类中心
        flop_centers = load_array('flop_centers.npy')
        turn_centers = load_array('turn_centers.npy')
        river_centers = load_array('river_centers.npy')
        
        # 创建结果对象
        self.result = AbstractionResult(
            config=config,
            preflop_mapping=preflop_mapping,
            flop_mapping=flop_mapping,
            turn_mapping=turn_mapping,
            river_mapping=river_mapping,
            flop_centers=flop_centers,
            turn_centers=turn_centers,
            river_centers=river_centers,
            wcss=metadata.get('wcss', {}),
            generation_time=metadata.get('generation_time', 0.0),
        )
        
        return self.result
    
    def get_abstraction_stats(self) -> Dict[str, Any]:
        """获取抽象统计信息。
        
        Returns:
            包含以下信息的字典：
            - config: 抽象配置
            - generation_time: 生成耗时
            - wcss: 每个阶段的WCSS
            - stages: 每个阶段的桶统计（数量、平均大小、最大大小）
            
        Raises:
            ValueError: 如果抽象结果未生成或加载
        """
        if self.result is None:
            raise ValueError("抽象结果未生成或加载")
        
        return self.result.get_stats()
    
    def is_loaded(self) -> bool:
        """检查抽象结果是否已加载。
        
        Returns:
            如果抽象结果已生成或加载，返回True
        """
        return self.result is not None
    
    def is_complete(self) -> bool:
        """检查抽象结果是否完整。
        
        Returns:
            如果所有阶段的映射都已生成，返回True
        """
        if self.result is None:
            return False
        return self.result.is_complete()
    
    def config_matches(self, other_config: AbstractionConfig) -> bool:
        """检查配置是否匹配。
        
        用于检测抽象配置是否发生变化。
        
        Args:
            other_config: 要比较的配置
            
        Returns:
            如果配置匹配，返回True
        """
        if self.result is None:
            return False
        
        current = self.result.config
        return (current.preflop_buckets == other_config.preflop_buckets and
                current.flop_buckets == other_config.flop_buckets and
                current.turn_buckets == other_config.turn_buckets and
                current.river_buckets == other_config.river_buckets and
                current.equity_bins == other_config.equity_bins and
                current.use_potential_aware == other_config.use_potential_aware and
                current.random_seed == other_config.random_seed)
    
    def get_stage_from_community_cards(self, num_community: int) -> str:
        """根据公共牌数量获取游戏阶段名称。
        
        Args:
            num_community: 公共牌数量
            
        Returns:
            游戏阶段名称
        """
        if num_community == 0:
            return 'preflop'
        elif num_community == 3:
            return 'flop'
        elif num_community == 4:
            return 'turn'
        elif num_community == 5:
            return 'river'
        else:
            raise ValueError(f"无效的公共牌数量：{num_community}")
