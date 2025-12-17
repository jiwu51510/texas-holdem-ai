"""翻牌纹理分类器模块。

将翻牌分为多个 bucket，用于 CFR 分层采样。
分类维度包括：同花、顺子潜力、牌面高低、对子、A-high 等。
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
import random
from itertools import combinations

from models.core import Card


@dataclass
class FlopFeatures:
    """翻牌特征数据类。
    
    Attributes:
        flush_count: 同花牌数量（2或3）
        straight_potential: 顺子潜力（0=无, 1=后门顺, 2=两头顺/卡顺, 3=已成顺）
        high_card_count: 高牌数量（T+，即10以上的牌）
        is_paired: 是否有对子
        has_ace: 是否有A
        is_monotone: 是否三张同花
        is_rainbow: 是否三张不同花色
        max_rank: 最大牌面值
        min_rank: 最小牌面值
        connectedness: 连接度（相邻牌的数量）
    """
    flush_count: int
    straight_potential: int
    high_card_count: int
    is_paired: bool
    has_ace: bool
    is_monotone: bool
    is_rainbow: bool
    max_rank: int
    min_rank: int
    connectedness: int


class FlopBucketClassifier:
    """翻牌纹理分类器。
    
    将翻牌分为约 30 个 bucket，用于分层采样。
    """
    
    # 高牌阈值（T=10 及以上为高牌）
    HIGH_CARD_THRESHOLD = 10
    
    def __init__(self, num_buckets: int = 30):
        """初始化分类器。
        
        Args:
            num_buckets: bucket 数量（默认30，范围20-80）
        """
        self.num_buckets = num_buckets
        self._bucket_to_flops: Dict[int, List[Tuple[Card, Card, Card]]] = {}
        self._build_bucket_mapping()

    def _build_bucket_mapping(self) -> None:
        """构建所有可能翻牌到 bucket 的映射。"""
        # 生成所有可能的翻牌组合
        all_cards = []
        for suit in ['h', 'd', 'c', 's']:
            for rank in range(2, 15):  # 2-14 (A=14)
                all_cards.append(Card(rank, suit))
        
        # 遍历所有翻牌组合并分类
        for flop in combinations(all_cards, 3):
            bucket = self.classify(list(flop))
            if bucket not in self._bucket_to_flops:
                self._bucket_to_flops[bucket] = []
            self._bucket_to_flops[bucket].append(flop)
    
    def classify(self, flop: List[Card]) -> int:
        """将翻牌分类到 bucket。
        
        Args:
            flop: 三张翻牌
            
        Returns:
            bucket 索引（0 到 num_buckets-1）
        """
        features = self._extract_features(flop)
        return self._features_to_bucket(features)
    
    def _extract_features(self, flop: List[Card]) -> FlopFeatures:
        """提取翻牌特征。
        
        Args:
            flop: 三张翻牌
            
        Returns:
            翻牌特征对象
        """
        ranks = sorted([card.rank for card in flop], reverse=True)
        suits = [card.suit for card in flop]
        
        # 同花特征
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        max_suit_count = max(suit_counts.values())
        is_monotone = max_suit_count == 3
        is_rainbow = max_suit_count == 1
        flush_count = max_suit_count
        
        # 高牌特征
        high_card_count = sum(1 for r in ranks if r >= self.HIGH_CARD_THRESHOLD)
        has_ace = 14 in ranks
        
        # 对子特征
        is_paired = len(set(ranks)) < 3
        
        # 顺子潜力
        straight_potential = self._calculate_straight_potential(ranks)
        
        # 连接度
        connectedness = self._calculate_connectedness(ranks)
        
        return FlopFeatures(
            flush_count=flush_count,
            straight_potential=straight_potential,
            high_card_count=high_card_count,
            is_paired=is_paired,
            has_ace=has_ace,
            is_monotone=is_monotone,
            is_rainbow=is_rainbow,
            max_rank=max(ranks),
            min_rank=min(ranks),
            connectedness=connectedness
        )
    
    def _calculate_straight_potential(self, ranks: List[int]) -> int:
        """计算顺子潜力。
        
        Args:
            ranks: 排序后的牌面值列表
            
        Returns:
            顺子潜力（0=无, 1=后门顺, 2=两头顺/卡顺, 3=已成顺）
        """
        unique_ranks = sorted(set(ranks))
        
        # 处理 A 可以作为 1 的情况
        if 14 in unique_ranks:
            unique_ranks_with_low_ace = sorted(set(unique_ranks + [1]))
        else:
            unique_ranks_with_low_ace = unique_ranks
        
        # 检查是否已成顺（三张连续）
        for i in range(len(unique_ranks_with_low_ace) - 2):
            if (unique_ranks_with_low_ace[i+1] - unique_ranks_with_low_ace[i] == 1 and
                unique_ranks_with_low_ace[i+2] - unique_ranks_with_low_ace[i+1] == 1):
                return 3  # 已成顺
        
        # 计算牌面跨度
        span = max(unique_ranks) - min(unique_ranks)
        if 14 in unique_ranks:
            # 考虑 A 作为 1 的情况
            span_with_low_ace = max(unique_ranks_with_low_ace) - min(unique_ranks_with_low_ace)
            span = min(span, span_with_low_ace)
        
        # 根据跨度判断顺子潜力
        if span <= 4:
            return 2  # 两头顺或卡顺潜力
        elif span <= 6:
            return 1  # 后门顺潜力
        else:
            return 0  # 无顺子潜力
    
    def _calculate_connectedness(self, ranks: List[int]) -> int:
        """计算连接度（相邻牌的数量）。
        
        Args:
            ranks: 排序后的牌面值列表
            
        Returns:
            连接度（0-2）
        """
        unique_ranks = sorted(set(ranks))
        connectedness = 0
        
        for i in range(len(unique_ranks) - 1):
            if unique_ranks[i+1] - unique_ranks[i] == 1:
                connectedness += 1
        
        # 处理 A-2 连接
        if 14 in unique_ranks and 2 in unique_ranks:
            connectedness += 1
        
        return connectedness

    def _features_to_bucket(self, features: FlopFeatures) -> int:
        """将特征映射到 bucket 索引。
        
        分类逻辑：
        1. 首先按同花/彩虹分大类
        2. 然后按对子/非对子分
        3. 再按牌面高低分
        4. 最后按顺子潜力分
        
        Args:
            features: 翻牌特征
            
        Returns:
            bucket 索引
        """
        bucket = 0
        
        # 第一维：同花类型（0-2）
        # 0: 彩虹（三张不同花色）
        # 1: 两张同花
        # 2: 三张同花（单色）
        if features.is_monotone:
            bucket += 20  # 单色翻牌
        elif features.is_rainbow:
            bucket += 0   # 彩虹翻牌
        else:
            bucket += 10  # 两张同花
        
        # 第二维：对子（0 或 5）
        if features.is_paired:
            bucket += 5
        
        # 第三维：牌面高低（0-2）
        # 根据高牌数量分类
        if features.high_card_count >= 2:
            bucket += 2  # 高牌翻牌
        elif features.high_card_count == 1:
            bucket += 1  # 中等翻牌
        else:
            bucket += 0  # 低牌翻牌
        
        # 第四维：顺子潜力（0-2）
        # 简化为三档
        if features.straight_potential >= 2:
            bucket += 0  # 高顺子潜力
        elif features.straight_potential == 1:
            bucket += 0  # 中等顺子潜力（合并）
        else:
            bucket += 0  # 低顺子潜力（合并）
        
        # 确保 bucket 在有效范围内
        return bucket % self.num_buckets
    
    def sample_flop_from_bucket(
        self, 
        bucket: int, 
        used_cards: Optional[Set[Tuple[int, str]]] = None
    ) -> Optional[List[Card]]:
        """从指定 bucket 中采样一个翻牌。
        
        Args:
            bucket: bucket 索引
            used_cards: 已使用的牌（私牌），格式为 {(rank, suit), ...}
            
        Returns:
            三张翻牌，如果无法采样则返回 None
        """
        if bucket not in self._bucket_to_flops:
            return None
        
        available_flops = self._bucket_to_flops[bucket]
        
        if used_cards:
            # 过滤掉包含已使用牌的翻牌
            available_flops = [
                flop for flop in available_flops
                if not any((card.rank, card.suit) in used_cards for card in flop)
            ]
        
        if not available_flops:
            return None
        
        # 随机选择一个翻牌
        selected = random.choice(available_flops)
        return list(selected)
    
    def get_bucket_for_sampling(self) -> int:
        """随机选择一个 bucket 用于采样。
        
        按 bucket 中翻牌数量加权采样，确保覆盖均匀。
        
        Returns:
            bucket 索引
        """
        # 计算每个 bucket 的权重（翻牌数量）
        buckets = list(self._bucket_to_flops.keys())
        weights = [len(self._bucket_to_flops[b]) for b in buckets]
        
        # 按权重采样
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(buckets) if buckets else 0
        
        r = random.random() * total_weight
        cumulative = 0
        for bucket, weight in zip(buckets, weights):
            cumulative += weight
            if r <= cumulative:
                return bucket
        
        return buckets[-1] if buckets else 0
    
    def sample_flop(
        self, 
        used_cards: Optional[Set[Tuple[int, str]]] = None
    ) -> Optional[List[Card]]:
        """分层采样一个翻牌。
        
        先随机选择 bucket，再从 bucket 中采样翻牌。
        
        Args:
            used_cards: 已使用的牌（私牌）
            
        Returns:
            三张翻牌
        """
        # 尝试多次采样（避免所有翻牌都被过滤的情况）
        for _ in range(10):
            bucket = self.get_bucket_for_sampling()
            flop = self.sample_flop_from_bucket(bucket, used_cards)
            if flop is not None:
                return flop
        
        # 如果分层采样失败，回退到均匀采样
        return self._uniform_sample_flop(used_cards)
    
    def _uniform_sample_flop(
        self, 
        used_cards: Optional[Set[Tuple[int, str]]] = None
    ) -> Optional[List[Card]]:
        """均匀采样一个翻牌（回退方法）。
        
        Args:
            used_cards: 已使用的牌
            
        Returns:
            三张翻牌
        """
        # 生成所有可用的牌
        available_cards = []
        for suit in ['h', 'd', 'c', 's']:
            for rank in range(2, 15):
                if used_cards is None or (rank, suit) not in used_cards:
                    available_cards.append(Card(rank, suit))
        
        if len(available_cards) < 3:
            return None
        
        # 随机选择三张
        return random.sample(available_cards, 3)
    
    def get_bucket_count(self) -> int:
        """获取实际使用的 bucket 数量。"""
        return len(self._bucket_to_flops)
    
    def get_bucket_sizes(self) -> Dict[int, int]:
        """获取每个 bucket 的翻牌数量。"""
        return {b: len(flops) for b, flops in self._bucket_to_flops.items()}
