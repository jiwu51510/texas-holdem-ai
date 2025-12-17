"""Equity计算器模块。

本模块实现了手牌Equity（胜率）的计算功能，包括：
- 计算手牌对抗随机对手的Equity
- 生成Equity分布直方图
- 计算翻牌手牌在转牌桶上的分布（用于Potential-Aware抽象）

Equity计算是卡牌抽象的基础，用于衡量手牌在不同公共牌面下的强度。
"""

from typing import List, Tuple, Optional, Dict, Any
from itertools import combinations
from multiprocessing import Pool, cpu_count
import numpy as np

from models.core import Card
from environment.hand_evaluator import compare_hands
from abstraction.data_classes import EquityHistogram, AbstractionConfig


def _create_deck() -> List[Card]:
    """创建一副完整的52张扑克牌。
    
    Returns:
        包含52张牌的列表
    """
    deck = []
    for rank in range(2, 15):  # 2-14 (A=14)
        for suit in ['h', 'd', 'c', 's']:
            deck.append(Card(rank=rank, suit=suit))
    return deck


def _get_remaining_deck(used_cards: List[Card]) -> List[Card]:
    """获取剩余的牌（排除已使用的牌）。
    
    Args:
        used_cards: 已使用的牌列表
        
    Returns:
        剩余牌的列表
    """
    deck = _create_deck()
    used_set = set((c.rank, c.suit) for c in used_cards)
    return [c for c in deck if (c.rank, c.suit) not in used_set]


def _calculate_equity_single(args: Tuple) -> float:
    """计算单个手牌组合的Equity（用于并行计算）。
    
    Args:
        args: (hole_cards, community_cards) 元组
        
    Returns:
        Equity值（0-1之间）
    """
    hole_cards, community_cards = args
    return EquityCalculator._calculate_equity_impl(hole_cards, community_cards)


class EquityCalculator:
    """Equity计算器类。
    
    提供手牌Equity计算的各种方法，支持多进程并行计算以提高效率。
    """
    
    def __init__(self, num_workers: int = 0):
        """初始化Equity计算器。
        
        Args:
            num_workers: 并行工作进程数（0=使用所有CPU核心）
        """
        self.num_workers = num_workers if num_workers > 0 else cpu_count()
    
    @staticmethod
    def _calculate_equity_impl(hole_cards: Tuple[Card, Card],
                               community_cards: List[Card]) -> float:
        """计算手牌对抗随机对手的Equity（内部实现）。
        
        通过枚举所有可能的对手手牌和剩余公共牌，计算胜率。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 当前的公共牌（0-5张）
            
        Returns:
            Equity值（0-1之间的浮点数）
        """
        # 获取已使用的牌
        used_cards = list(hole_cards) + community_cards
        remaining_deck = _get_remaining_deck(used_cards)
        
        # 需要发的公共牌数量
        cards_to_deal = 5 - len(community_cards)
        
        wins = 0
        ties = 0
        total = 0
        
        # 枚举所有可能的对手手牌
        for opp_cards in combinations(remaining_deck, 2):
            opp_hand = list(opp_cards)
            
            # 获取排除对手手牌后的剩余牌
            remaining_after_opp = [c for c in remaining_deck 
                                   if c not in opp_hand]
            
            if cards_to_deal == 0:
                # 河牌阶段，直接比较
                result = compare_hands(list(hole_cards), opp_hand, community_cards)
                if result == 0:  # 我方胜
                    wins += 1
                elif result == -1:  # 平局
                    ties += 1
                total += 1
            else:
                # 需要枚举剩余公共牌
                for board_cards in combinations(remaining_after_opp, cards_to_deal):
                    full_community = community_cards + list(board_cards)
                    result = compare_hands(list(hole_cards), opp_hand, full_community)
                    if result == 0:  # 我方胜
                        wins += 1
                    elif result == -1:  # 平局
                        ties += 1
                    total += 1
        
        if total == 0:
            return 0.5
        
        # Equity = 胜率 + 0.5 * 平局率
        return (wins + 0.5 * ties) / total
    
    def calculate_equity(self, hole_cards: Tuple[Card, Card],
                        community_cards: List[Card]) -> float:
        """计算手牌对抗随机对手的Equity。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 当前的公共牌（0-5张）
            
        Returns:
            Equity值（0-1之间的浮点数）
        """
        return self._calculate_equity_impl(hole_cards, community_cards)
    
    def calculate_equity_distribution(self, hole_cards: Tuple[Card, Card],
                                      community_cards: List[Card],
                                      num_bins: int = 50) -> EquityHistogram:
        """计算手牌在所有可能公共牌面下的Equity分布直方图。
        
        对于给定的手牌和当前公共牌，枚举所有可能的后续公共牌，
        计算每种情况下的Equity，并生成分布直方图。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 当前的公共牌
            num_bins: 直方图的区间数量（默认50，每个区间宽度0.02）
            
        Returns:
            EquityHistogram实例，表示Equity分布
        """
        # 获取已使用的牌
        used_cards = list(hole_cards) + community_cards
        remaining_deck = _get_remaining_deck(used_cards)
        
        # 需要发的公共牌数量
        cards_to_deal = 5 - len(community_cards)
        
        if cards_to_deal == 0:
            # 河牌阶段，只有一个Equity值
            equity = self.calculate_equity(hole_cards, community_cards)
            bins = np.linspace(0, 1, num_bins + 1)
            counts = np.zeros(num_bins)
            bin_idx = min(int(equity * num_bins), num_bins - 1)
            counts[bin_idx] = 1.0
            return EquityHistogram(bins=bins, counts=counts)
        
        # 收集所有可能公共牌面下的Equity值
        equities = []
        for board_cards in combinations(remaining_deck, cards_to_deal):
            full_community = community_cards + list(board_cards)
            equity = self.calculate_equity(hole_cards, full_community)
            equities.append(equity)
        
        # 生成直方图
        bins = np.linspace(0, 1, num_bins + 1)
        counts, _ = np.histogram(equities, bins=bins)
        
        # 归一化
        total = np.sum(counts)
        if total > 0:
            counts = counts.astype(float) / total
        
        return EquityHistogram(bins=bins, counts=counts)
    
    def calculate_equity_distribution_fast(self, hole_cards: Tuple[Card, Card],
                                           community_cards: List[Card],
                                           num_bins: int = 50,
                                           num_samples: int = 1000) -> EquityHistogram:
        """使用蒙特卡洛采样快速计算Equity分布直方图。
        
        当精确计算太慢时，使用随机采样来近似Equity分布。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 当前的公共牌
            num_bins: 直方图的区间数量
            num_samples: 采样数量
            
        Returns:
            EquityHistogram实例
        """
        # 获取已使用的牌
        used_cards = list(hole_cards) + community_cards
        remaining_deck = _get_remaining_deck(used_cards)
        
        # 需要发的公共牌数量
        cards_to_deal = 5 - len(community_cards)
        
        if cards_to_deal == 0:
            # 河牌阶段，只有一个Equity值
            equity = self.calculate_equity(hole_cards, community_cards)
            bins = np.linspace(0, 1, num_bins + 1)
            counts = np.zeros(num_bins)
            bin_idx = min(int(equity * num_bins), num_bins - 1)
            counts[bin_idx] = 1.0
            return EquityHistogram(bins=bins, counts=counts)
        
        # 蒙特卡洛采样
        equities = []
        rng = np.random.default_rng()
        
        for _ in range(num_samples):
            # 随机选择公共牌
            indices = rng.choice(len(remaining_deck), size=cards_to_deal, replace=False)
            board_cards = [remaining_deck[i] for i in indices]
            full_community = community_cards + board_cards
            
            # 计算Equity（使用简化版本）
            equity = self._calculate_equity_sampled(hole_cards, full_community, 
                                                    remaining_deck, board_cards)
            equities.append(equity)
        
        # 生成直方图
        bins = np.linspace(0, 1, num_bins + 1)
        counts, _ = np.histogram(equities, bins=bins)
        
        # 归一化
        total = np.sum(counts)
        if total > 0:
            counts = counts.astype(float) / total
        
        return EquityHistogram(bins=bins, counts=counts)
    
    def _calculate_equity_sampled(self, hole_cards: Tuple[Card, Card],
                                  community_cards: List[Card],
                                  remaining_deck: List[Card],
                                  excluded_cards: List[Card],
                                  num_opponent_samples: int = 100) -> float:
        """使用采样计算Equity（用于快速近似）。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 完整的公共牌（5张）
            remaining_deck: 剩余牌堆
            excluded_cards: 需要排除的牌（已用于公共牌）
            num_opponent_samples: 对手手牌采样数量
            
        Returns:
            近似的Equity值
        """
        # 排除已用于公共牌的牌
        available = [c for c in remaining_deck if c not in excluded_cards]
        
        if len(available) < 2:
            return 0.5
        
        wins = 0
        ties = 0
        total = 0
        
        rng = np.random.default_rng()
        
        for _ in range(num_opponent_samples):
            # 随机选择对手手牌
            indices = rng.choice(len(available), size=2, replace=False)
            opp_hand = [available[indices[0]], available[indices[1]]]
            
            result = compare_hands(list(hole_cards), opp_hand, community_cards)
            if result == 0:
                wins += 1
            elif result == -1:
                ties += 1
            total += 1
        
        if total == 0:
            return 0.5
        
        return (wins + 0.5 * ties) / total
    
    def calculate_turn_bucket_distribution(self, hole_cards: Tuple[Card, Card],
                                           flop_cards: List[Card],
                                           turn_bucket_mapping: np.ndarray,
                                           num_turn_buckets: int) -> np.ndarray:
        """计算翻牌手牌在转牌阶段各个桶的分布。
        
        用于Potential-Aware抽象：计算给定翻牌手牌在所有可能转牌下
        落入各个转牌桶的概率分布。
        
        Args:
            hole_cards: 玩家的两张手牌
            flop_cards: 翻牌的3张公共牌
            turn_bucket_mapping: 转牌阶段的桶映射（手牌索引 -> 桶ID）
            num_turn_buckets: 转牌阶段的桶数量
            
        Returns:
            直方图数组，表示落入每个转牌桶的概率
        """
        if len(flop_cards) != 3:
            raise ValueError(f"翻牌必须是3张牌，当前：{len(flop_cards)}")
        
        # 获取剩余牌
        used_cards = list(hole_cards) + flop_cards
        remaining_deck = _get_remaining_deck(used_cards)
        
        # 统计落入每个桶的次数
        bucket_counts = np.zeros(num_turn_buckets)
        
        for turn_card in remaining_deck:
            # 计算转牌手牌的索引
            turn_community = flop_cards + [turn_card]
            turn_index = self._get_turn_hand_index(hole_cards, turn_community)
            
            # 查找对应的桶ID
            if turn_index < len(turn_bucket_mapping):
                bucket_id = turn_bucket_mapping[turn_index]
                if 0 <= bucket_id < num_turn_buckets:
                    bucket_counts[bucket_id] += 1
        
        # 归一化
        total = np.sum(bucket_counts)
        if total > 0:
            bucket_counts = bucket_counts / total
        
        return bucket_counts
    
    def _get_turn_hand_index(self, hole_cards: Tuple[Card, Card],
                             community_cards: List[Card]) -> int:
        """计算转牌手牌的索引。
        
        将手牌和公共牌组合编码为唯一索引。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 公共牌（4张，翻牌+转牌）
            
        Returns:
            手牌组合的唯一索引
        """
        # 简化实现：使用牌的编码计算索引
        # 实际实现中可能需要更复杂的索引方案
        def card_to_index(card: Card) -> int:
            return (card.rank - 2) * 4 + ['h', 'd', 'c', 's'].index(card.suit)
        
        # 对手牌和公共牌排序以确保一致性
        hole_indices = sorted([card_to_index(c) for c in hole_cards])
        community_indices = sorted([card_to_index(c) for c in community_cards])
        
        # 组合成唯一索引（简化版本）
        index = 0
        for i, idx in enumerate(hole_indices + community_indices):
            index = index * 52 + idx
        
        return index % (10 ** 9)  # 限制索引范围
    
    def calculate_equity_batch(self, hands: List[Tuple[Tuple[Card, Card], List[Card]]],
                               use_parallel: bool = True) -> List[float]:
        """批量计算多个手牌的Equity。
        
        Args:
            hands: 手牌列表，每个元素是 (hole_cards, community_cards) 元组
            use_parallel: 是否使用并行计算
            
        Returns:
            Equity值列表
        """
        if not use_parallel or len(hands) < 10:
            return [self.calculate_equity(h[0], h[1]) for h in hands]
        
        with Pool(self.num_workers) as pool:
            results = pool.map(_calculate_equity_single, hands)
        
        return results
    
    def calculate_river_equity(self, hole_cards: Tuple[Card, Card],
                               community_cards: List[Card]) -> float:
        """计算河牌阶段的最终Equity。
        
        河牌阶段公共牌已完整（5张），直接计算对抗随机对手的胜率。
        
        Args:
            hole_cards: 玩家的两张手牌
            community_cards: 完整的公共牌（5张）
            
        Returns:
            Equity值
        """
        if len(community_cards) != 5:
            raise ValueError(f"河牌阶段必须有5张公共牌，当前：{len(community_cards)}")
        
        return self.calculate_equity(hole_cards, community_cards)


def get_canonical_hand_index(hole_cards: Tuple[Card, Card]) -> int:
    """获取手牌的规范化索引。
    
    将手牌转换为规范化索引，考虑花色同构性。
    例如：AhKh 和 AsKs 是等价的（同花），应该映射到同一索引。
    
    索引布局：
    - 0-12: 对子 (AA=0, KK=1, ..., 22=12)
    - 13-90: 同花非对子 (AKs=13, AQs=14, ..., 32s=90)
    - 91-168: 异花非对子 (AKo=91, AQo=92, ..., 32o=168)
    
    Args:
        hole_cards: 玩家的两张手牌
        
    Returns:
        规范化的手牌索引（0-168）
    """
    card1, card2 = hole_cards
    
    # 确保 rank1 >= rank2
    if card1.rank < card2.rank:
        card1, card2 = card2, card1
    
    rank1, rank2 = card1.rank, card2.rank
    same_suit = card1.suit == card2.suit
    
    if rank1 == rank2:
        # 对子：AA=0, KK=1, ..., 22=12
        return 14 - rank1
    
    # 非对子：计算在非对子组合中的位置
    # 高牌索引：A=0, K=1, ..., 3=11
    high_idx = 14 - rank1
    # 低牌相对于高牌的偏移：比高牌小1的牌偏移为0
    low_offset = rank1 - rank2 - 1
    
    # 计算在非对子组合中的位置
    # 对于高牌索引 h，之前有 h*(h-1)/2 + h = h*(h+1)/2 个组合
    # 但我们需要的是：之前所有高牌的组合数
    # AK, AQ, AJ, ..., A2 = 12个
    # KQ, KJ, ..., K2 = 11个
    # ...
    # 高牌索引为 h 时，之前有 sum(12-i for i in range(h)) = 12*h - h*(h-1)/2 个组合
    
    # 简化：使用累积计数
    # 高牌为 A(14) 时，有 12 个组合 (AK, AQ, ..., A2)
    # 高牌为 K(13) 时，有 11 个组合 (KQ, KJ, ..., K2)
    # ...
    # 高牌为 3(3) 时，有 1 个组合 (32)
    
    # 之前所有高牌的组合数
    combos_before = 0
    for r in range(14, rank1, -1):
        combos_before += r - 2  # 高牌为 r 时有 r-2 个组合
    
    # 当前高牌内的偏移
    position = combos_before + low_offset
    
    if same_suit:
        return 13 + position  # 同花从索引13开始
    else:
        return 13 + 78 + position  # 异花从索引91开始（13 + 78 = 91）


def index_to_canonical_hand(index: int) -> Tuple[int, int, bool]:
    """将规范化索引转换回手牌表示。
    
    Args:
        index: 规范化的手牌索引（0-168）
        
    Returns:
        (rank1, rank2, same_suit) 元组，其中 rank1 >= rank2
    """
    if index < 0 or index > 168:
        raise ValueError(f"索引必须在0-168范围内，当前：{index}")
    
    if index < 13:
        # 对子：AA=0, KK=1, ..., 22=12
        rank = 14 - index
        return (rank, rank, False)
    
    # 非对子
    if index < 91:
        # 同花
        position = index - 13
        suited = True
    else:
        # 异花
        position = index - 91
        suited = False
    
    # 根据位置反推高牌和低牌
    # 遍历找到对应的高牌
    cumulative = 0
    for high_rank in range(14, 2, -1):
        num_combos = high_rank - 2  # 高牌为 high_rank 时的组合数
        if cumulative + num_combos > position:
            # 找到了高牌
            low_offset = position - cumulative
            low_rank = high_rank - 1 - low_offset
            return (high_rank, low_rank, suited)
        cumulative += num_combos
    
    # 不应该到达这里
    raise ValueError(f"无法解析索引：{index}")
