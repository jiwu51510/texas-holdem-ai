"""抽象缓存模块。

本模块实现了AbstractionCache类，用于管理抽象结果的缓存和快速查询。
支持O(1)时间复杂度的桶ID查询，以及手牌规范化处理。

主要功能：
- O(1)时间复杂度的桶ID查询
- 手牌规范化（处理花色同构性）
- 支持内存映射（memmap）处理大型映射表
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import os

from models.core import Card
from abstraction.data_classes import AbstractionConfig, AbstractionResult
from abstraction.equity_calculator import get_canonical_hand_index


class AbstractionCache:
    """抽象缓存类。
    
    管理抽象结果的缓存，提供O(1)时间复杂度的桶ID查询。
    使用numpy数组作为查找表，支持内存映射处理大型映射表。
    
    Attributes:
        config: 抽象配置
        _preflop_cache: 翻牌前桶ID缓存（169个条目）
        _flop_cache: 翻牌桶ID缓存
        _turn_cache: 转牌桶ID缓存
        _river_cache: 河牌桶ID缓存
        _use_memmap: 是否使用内存映射
    """
    
    # 花色映射表，用于规范化
    SUIT_ORDER = {'h': 0, 'd': 1, 'c': 2, 's': 3}
    SUITS = ['h', 'd', 'c', 's']
    
    def __init__(self, result: Optional[AbstractionResult] = None,
                 use_memmap: bool = False):
        """初始化抽象缓存。
        
        Args:
            result: 抽象结果，如果提供则立即构建缓存
            use_memmap: 是否使用内存映射处理大型映射表
        """
        self.config: Optional[AbstractionConfig] = None
        self._preflop_cache: Optional[np.ndarray] = None
        self._flop_cache: Optional[np.ndarray] = None
        self._turn_cache: Optional[np.ndarray] = None
        self._river_cache: Optional[np.ndarray] = None
        self._use_memmap = use_memmap
        
        if result is not None:
            self.build_cache(result)
    
    def build_cache(self, result: AbstractionResult) -> None:
        """从抽象结果构建缓存。
        
        Args:
            result: 抽象结果
        """
        self.config = result.config
        
        # 构建翻牌前缓存
        if result.preflop_mapping is not None:
            self._preflop_cache = result.preflop_mapping.astype(np.int32)
        else:
            # 默认：每个规范化手牌一个桶
            self._preflop_cache = np.arange(169, dtype=np.int32)
        
        # 构建翻牌缓存
        if result.flop_mapping is not None:
            self._flop_cache = result.flop_mapping.astype(np.int32)
        
        # 构建转牌缓存
        if result.turn_mapping is not None:
            self._turn_cache = result.turn_mapping.astype(np.int32)
        
        # 构建河牌缓存
        if result.river_mapping is not None:
            self._river_cache = result.river_mapping.astype(np.int32)
    
    def get_bucket(self, hole_cards: Tuple[Card, Card],
                   community_cards: List[Card]) -> int:
        """快速查询桶ID。
        
        使用预计算的查找表，时间复杂度O(1)。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 当前的公共牌（0-5张）
            
        Returns:
            桶ID
            
        Raises:
            ValueError: 如果缓存未构建
        """
        num_community = len(community_cards)
        
        if num_community == 0:
            return self._get_preflop_bucket(hole_cards)
        elif num_community == 3:
            return self._get_flop_bucket(hole_cards, community_cards)
        elif num_community == 4:
            return self._get_turn_bucket(hole_cards, community_cards)
        elif num_community == 5:
            return self._get_river_bucket(hole_cards, community_cards)
        else:
            raise ValueError(f"无效的公共牌数量：{num_community}")
    
    def _get_preflop_bucket(self, hole_cards: Tuple[Card, Card]) -> int:
        """获取翻牌前的桶ID（O(1)）。
        
        Args:
            hole_cards: 玩家的两张手牌
            
        Returns:
            桶ID
        """
        if self._preflop_cache is None:
            raise ValueError("翻牌前缓存未构建")
        
        canonical_index = get_canonical_hand_index(hole_cards)
        return int(self._preflop_cache[canonical_index])
    
    def _get_flop_bucket(self, hole_cards: Tuple[Card, Card],
                         community_cards: List[Card]) -> int:
        """获取翻牌阶段的桶ID（O(1)）。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 翻牌的3张公共牌
            
        Returns:
            桶ID
        """
        if self._flop_cache is None:
            return 0
        
        index = self._compute_hand_index(hole_cards, community_cards)
        return int(self._flop_cache[index % len(self._flop_cache)])
    
    def _get_turn_bucket(self, hole_cards: Tuple[Card, Card],
                         community_cards: List[Card]) -> int:
        """获取转牌阶段的桶ID（O(1)）。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 转牌的4张公共牌
            
        Returns:
            桶ID
        """
        if self._turn_cache is None:
            return 0
        
        index = self._compute_hand_index(hole_cards, community_cards)
        return int(self._turn_cache[index % len(self._turn_cache)])
    
    def _get_river_bucket(self, hole_cards: Tuple[Card, Card],
                          community_cards: List[Card]) -> int:
        """获取河牌阶段的桶ID（O(1)）。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 河牌的5张公共牌
            
        Returns:
            桶ID
        """
        if self._river_cache is None:
            return 0
        
        index = self._compute_hand_index(hole_cards, community_cards)
        return int(self._river_cache[index % len(self._river_cache)])
    
    def _compute_hand_index(self, hole_cards: Tuple[Card, Card],
                            community_cards: List[Card]) -> int:
        """计算手牌组合的唯一索引（O(1)）。
        
        使用位运算和哈希计算唯一索引。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 公共牌
            
        Returns:
            手牌组合的唯一索引
        """
        # 将所有牌转换为索引
        all_cards = list(hole_cards) + community_cards
        indices = [self._card_to_index(c) for c in all_cards]
        
        # 使用多项式哈希计算唯一索引
        # 这是O(1)操作，因为牌的数量是固定的（最多7张）
        hash_value = 0
        for idx in sorted(indices):
            hash_value = hash_value * 53 + idx + 1
        
        return hash_value % (10 ** 9)
    
    @staticmethod
    def _card_to_index(card: Card) -> int:
        """将牌转换为0-51的索引（O(1)）。
        
        Args:
            card: 扑克牌
            
        Returns:
            0-51的索引
        """
        suit_idx = AbstractionCache.SUIT_ORDER.get(card.suit, 0)
        return (card.rank - 2) * 4 + suit_idx
    
    @staticmethod
    def get_canonical_hand(hole_cards: Tuple[Card, Card]) -> int:
        """获取手牌的规范化表示。
        
        将手牌转换为规范化索引，用于查找表。
        考虑花色同构性（如AhKh和AsKs是等价的）。
        
        Args:
            hole_cards: 玩家的两张手牌
            
        Returns:
            规范化的手牌索引（0-168）
        """
        return get_canonical_hand_index(hole_cards)
    
    def load_from_files(self, path: str) -> None:
        """从文件加载缓存。
        
        支持内存映射模式，用于处理大型映射表。
        
        Args:
            path: 抽象文件目录路径
        """
        mmap_mode = 'r' if self._use_memmap else None
        
        # 加载翻牌前缓存
        preflop_path = os.path.join(path, 'preflop_mapping.npy')
        if os.path.exists(preflop_path):
            self._preflop_cache = np.load(preflop_path, mmap_mode=mmap_mode)
        
        # 加载翻牌缓存
        flop_path = os.path.join(path, 'flop_mapping.npy')
        if os.path.exists(flop_path):
            self._flop_cache = np.load(flop_path, mmap_mode=mmap_mode)
        
        # 加载转牌缓存
        turn_path = os.path.join(path, 'turn_mapping.npy')
        if os.path.exists(turn_path):
            self._turn_cache = np.load(turn_path, mmap_mode=mmap_mode)
        
        # 加载河牌缓存
        river_path = os.path.join(path, 'river_mapping.npy')
        if os.path.exists(river_path):
            self._river_cache = np.load(river_path, mmap_mode=mmap_mode)
    
    def is_loaded(self) -> bool:
        """检查缓存是否已加载。
        
        Returns:
            如果至少有一个缓存已加载，返回True
        """
        return (self._preflop_cache is not None or
                self._flop_cache is not None or
                self._turn_cache is not None or
                self._river_cache is not None)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息。
        
        Returns:
            包含每个阶段缓存大小的字典
        """
        stats = {}
        
        if self._preflop_cache is not None:
            stats['preflop_size'] = len(self._preflop_cache)
        
        if self._flop_cache is not None:
            stats['flop_size'] = len(self._flop_cache)
        
        if self._turn_cache is not None:
            stats['turn_size'] = len(self._turn_cache)
        
        if self._river_cache is not None:
            stats['river_size'] = len(self._river_cache)
        
        return stats
    
    def clear(self) -> None:
        """清除所有缓存。"""
        self._preflop_cache = None
        self._flop_cache = None
        self._turn_cache = None
        self._river_cache = None
        self.config = None
