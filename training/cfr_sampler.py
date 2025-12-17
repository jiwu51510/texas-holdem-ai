"""CFR 采样器模块。

实现"翻前/翻牌分层随机 + 转牌随机 + 河牌枚举"的采样策略。
支持多次采样平均以降低方差。
"""

from typing import List, Dict, Set, Tuple, Optional, Generator, Callable
from dataclasses import dataclass, field
import random
import numpy as np
from itertools import combinations

from models.core import Card, GameState
from training.flop_bucket_classifier import FlopBucketClassifier


@dataclass
class SamplerConfig:
    """采样器配置。
    
    Attributes:
        num_samples: 多次采样的采样次数（用于降低方差）
        use_averaging: 是否使用多次采样平均
    """
    num_samples: int = 1
    use_averaging: bool = False
    
    def __post_init__(self):
        """验证配置参数。"""
        if self.num_samples < 1:
            raise ValueError("采样次数必须至少为1")


@dataclass
class SampledCards:
    """采样的牌数据类。
    
    Attributes:
        player_hands: 两个玩家的私牌
        flop: 翻牌（3张）
        turn: 转牌（1张）
        river: 河牌（1张，如果是枚举则为 None）
    """
    player_hands: List[List[Card]]
    flop: List[Card]
    turn: Optional[Card]
    river: Optional[Card]


class CFRSampler:
    """CFR 采样器。
    
    实现高效的 CFR 采样策略：
    - Preflop: 均匀采样私牌组合
    - Flop: 分层采样（按翻牌纹理分 bucket）
    - Turn: 随机采样转牌
    - River: 枚举全部河牌
    """
    
    def __init__(self, num_flop_buckets: int = 30):
        """初始化采样器。
        
        Args:
            num_flop_buckets: 翻牌 bucket 数量
        """
        self.flop_classifier = FlopBucketClassifier(num_flop_buckets)
        self._all_cards = self._generate_all_cards()
    
    def _generate_all_cards(self) -> List[Card]:
        """生成一副完整的牌。"""
        cards = []
        for suit in ['h', 'd', 'c', 's']:
            for rank in range(2, 15):
                cards.append(Card(rank, suit))
        return cards
    
    def _card_to_tuple(self, card: Card) -> Tuple[int, str]:
        """将 Card 转换为元组（用于集合操作）。"""
        return (card.rank, card.suit)
    
    def _cards_to_set(self, cards: List[Card]) -> Set[Tuple[int, str]]:
        """将牌列表转换为集合。"""
        return {self._card_to_tuple(c) for c in cards}

    def sample_preflop(self) -> List[List[Card]]:
        """均匀采样私牌组合。
        
        为两个玩家各发两张私牌。
        
        Returns:
            两个玩家的私牌列表 [[p0_card1, p0_card2], [p1_card1, p1_card2]]
        """
        # 随机打乱牌堆
        deck = self._all_cards.copy()
        random.shuffle(deck)
        
        # 发私牌
        player0_hand = [deck.pop(), deck.pop()]
        player1_hand = [deck.pop(), deck.pop()]
        
        return [player0_hand, player1_hand]
    
    def sample_flop(
        self, 
        used_cards: Set[Tuple[int, str]]
    ) -> List[Card]:
        """分层采样翻牌。
        
        使用 FlopBucketClassifier 进行分层采样。
        
        Args:
            used_cards: 已使用的牌（私牌）
            
        Returns:
            三张翻牌
        """
        flop = self.flop_classifier.sample_flop(used_cards)
        if flop is None:
            # 回退到均匀采样
            flop = self._uniform_sample_cards(3, used_cards)
        return flop
    
    def sample_turn(
        self, 
        used_cards: Set[Tuple[int, str]]
    ) -> Card:
        """随机采样转牌。
        
        从剩余牌堆中均匀随机选择一张。
        
        Args:
            used_cards: 已使用的牌（私牌 + 翻牌）
            
        Returns:
            一张转牌
        """
        cards = self._uniform_sample_cards(1, used_cards)
        return cards[0] if cards else None
    
    def enumerate_river(
        self, 
        used_cards: Set[Tuple[int, str]]
    ) -> Generator[Card, None, None]:
        """枚举全部河牌。
        
        生成所有可能的河牌（通常 46 张）。
        
        Args:
            used_cards: 已使用的牌（私牌 + 翻牌 + 转牌）
            
        Yields:
            每一张可能的河牌
        """
        for card in self._all_cards:
            card_tuple = self._card_to_tuple(card)
            if card_tuple not in used_cards:
                yield card
    
    def get_remaining_river_count(
        self, 
        used_cards: Set[Tuple[int, str]]
    ) -> int:
        """获取剩余河牌数量。
        
        Args:
            used_cards: 已使用的牌
            
        Returns:
            剩余河牌数量
        """
        return 52 - len(used_cards)
    
    def _uniform_sample_cards(
        self, 
        count: int, 
        used_cards: Set[Tuple[int, str]]
    ) -> List[Card]:
        """均匀采样指定数量的牌。
        
        Args:
            count: 需要采样的牌数量
            used_cards: 已使用的牌
            
        Returns:
            采样的牌列表
        """
        available = [
            card for card in self._all_cards
            if self._card_to_tuple(card) not in used_cards
        ]
        
        if len(available) < count:
            return available
        
        return random.sample(available, count)
    
    def sample_full_board(
        self, 
        player_hands: List[List[Card]]
    ) -> SampledCards:
        """采样完整的公共牌（翻牌 + 转牌 + 河牌）。
        
        注意：河牌使用随机采样而非枚举（用于快速测试）。
        
        Args:
            player_hands: 两个玩家的私牌
            
        Returns:
            采样的牌数据
        """
        # 收集已使用的牌
        used_cards = set()
        for hand in player_hands:
            for card in hand:
                used_cards.add(self._card_to_tuple(card))
        
        # 采样翻牌
        flop = self.sample_flop(used_cards)
        for card in flop:
            used_cards.add(self._card_to_tuple(card))
        
        # 采样转牌
        turn = self.sample_turn(used_cards)
        if turn:
            used_cards.add(self._card_to_tuple(turn))
        
        # 采样河牌（非枚举模式）
        river = self.sample_turn(used_cards)  # 复用 sample_turn
        
        return SampledCards(
            player_hands=player_hands,
            flop=flop,
            turn=turn,
            river=river
        )

    def sample_game_state(self) -> Tuple[List[List[Card]], List[Card], Card]:
        """采样一个完整的游戏状态（到转牌阶段）。
        
        用于 CFR 迭代，河牌阶段需要枚举。
        
        Returns:
            (player_hands, flop, turn) 元组
        """
        # 采样私牌
        player_hands = self.sample_preflop()
        
        # 收集已使用的牌
        used_cards = set()
        for hand in player_hands:
            for card in hand:
                used_cards.add(self._card_to_tuple(card))
        
        # 采样翻牌
        flop = self.sample_flop(used_cards)
        for card in flop:
            used_cards.add(self._card_to_tuple(card))
        
        # 采样转牌
        turn = self.sample_turn(used_cards)
        
        return player_hands, flop, turn
    
    def create_game_state_with_cards(
        self,
        player_hands: List[List[Card]],
        community_cards: List[Card]
    ) -> GameState:
        """使用指定的牌创建游戏状态。
        
        Args:
            player_hands: 两个玩家的私牌
            community_cards: 公共牌
            
        Returns:
            游戏状态对象
        """
        from models.core import GameStage
        
        # 根据公共牌数量确定游戏阶段
        num_community = len(community_cards)
        if num_community == 0:
            stage = GameStage.PREFLOP
        elif num_community == 3:
            stage = GameStage.FLOP
        elif num_community == 4:
            stage = GameStage.TURN
        else:
            stage = GameStage.RIVER
        
        return GameState(
            player_hands=player_hands,
            community_cards=community_cards,
            current_player=0,
            pot=0,
            player_stacks=[1000, 1000],  # 默认筹码
            current_bets=[0, 0],
            stage=stage,
            action_history=[],
            is_terminal=False,
            min_raise=0,
            last_raise_amount=0
        )


class MultiSampleAverager:
    """多次采样平均器。
    
    通过多次采样取平均来降低方差。
    根据统计学原理，n次独立采样的平均值方差为单次采样方差的1/n。
    """
    
    def __init__(self, config: Optional[SamplerConfig] = None):
        """初始化多次采样平均器。
        
        Args:
            config: 采样器配置
        """
        self.config = config or SamplerConfig()
    
    def sample_with_averaging(
        self,
        sample_func: Callable[[], float],
        num_samples: Optional[int] = None
    ) -> float:
        """执行多次采样并返回平均值。
        
        通过多次独立采样取平均来降低方差。
        
        Args:
            sample_func: 单次采样函数，返回一个浮点数值
            num_samples: 采样次数，如果为None则使用配置中的默认值
            
        Returns:
            多次采样的平均值
        """
        n = num_samples if num_samples is not None else self.config.num_samples
        
        if n <= 0:
            raise ValueError("采样次数必须为正整数")
        
        if n == 1:
            return sample_func()
        
        # 执行多次采样
        samples = [sample_func() for _ in range(n)]
        return np.mean(samples)
    
    def sample_array_with_averaging(
        self,
        sample_func: Callable[[], np.ndarray],
        num_samples: Optional[int] = None
    ) -> np.ndarray:
        """执行多次采样并返回数组的平均值。
        
        用于采样函数返回数组的情况（如遗憾值数组）。
        
        Args:
            sample_func: 单次采样函数，返回一个numpy数组
            num_samples: 采样次数，如果为None则使用配置中的默认值
            
        Returns:
            多次采样的平均数组
        """
        n = num_samples if num_samples is not None else self.config.num_samples
        
        if n <= 0:
            raise ValueError("采样次数必须为正整数")
        
        if n == 1:
            return sample_func()
        
        # 执行多次采样
        samples = [sample_func() for _ in range(n)]
        return np.mean(samples, axis=0)
    
    def estimate_variance_reduction(
        self,
        sample_func: Callable[[], float],
        num_samples: int,
        num_trials: int = 100
    ) -> Tuple[float, float, float]:
        """估计多次采样平均的方差降低效果。
        
        通过实验估计单次采样方差和多次采样平均后的方差。
        
        Args:
            sample_func: 单次采样函数
            num_samples: 每次平均使用的采样次数
            num_trials: 实验次数
            
        Returns:
            (单次采样方差, 多次采样平均方差, 方差比率)
        """
        # 收集单次采样的样本
        single_samples = [sample_func() for _ in range(num_trials)]
        single_variance = np.var(single_samples)
        
        # 收集多次采样平均的样本
        averaged_samples = [
            self.sample_with_averaging(sample_func, num_samples)
            for _ in range(num_trials)
        ]
        averaged_variance = np.var(averaged_samples)
        
        # 计算方差比率
        if single_variance > 0:
            variance_ratio = averaged_variance / single_variance
        else:
            variance_ratio = 0.0
        
        return single_variance, averaged_variance, variance_ratio


class RiverEnumerator:
    """河牌枚举器。
    
    用于在河牌阶段枚举所有可能的河牌并计算平均收益。
    """
    
    def __init__(self, sampler: CFRSampler):
        """初始化枚举器。
        
        Args:
            sampler: CFR 采样器
        """
        self.sampler = sampler
    
    def enumerate_and_average(
        self,
        used_cards: Set[Tuple[int, str]],
        value_func: callable
    ) -> float:
        """枚举所有河牌并计算平均收益。
        
        Args:
            used_cards: 已使用的牌（私牌 + 翻牌 + 转牌）
            value_func: 计算单个河牌收益的函数，签名为 (river_card) -> float
            
        Returns:
            所有河牌的平均收益
        """
        total_value = 0.0
        count = 0
        
        for river_card in self.sampler.enumerate_river(used_cards):
            value = value_func(river_card)
            total_value += value
            count += 1
        
        if count == 0:
            return 0.0
        
        return total_value / count
    
    def enumerate_with_weights(
        self,
        used_cards: Set[Tuple[int, str]],
        value_func: callable,
        weight_func: Optional[callable] = None
    ) -> float:
        """枚举所有河牌并计算加权平均收益。
        
        Args:
            used_cards: 已使用的牌
            value_func: 计算单个河牌收益的函数
            weight_func: 计算权重的函数（可选，默认均匀权重）
            
        Returns:
            加权平均收益
        """
        total_weighted_value = 0.0
        total_weight = 0.0
        
        for river_card in self.sampler.enumerate_river(used_cards):
            value = value_func(river_card)
            weight = weight_func(river_card) if weight_func else 1.0
            total_weighted_value += value * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_weighted_value / total_weight
