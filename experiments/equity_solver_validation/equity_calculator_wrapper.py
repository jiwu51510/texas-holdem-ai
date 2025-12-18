"""胜率计算器封装模块。

本模块封装现有的EquityCalculator，提供范围VS范围的胜率计算功能。
由于poker-odds-calc是TypeScript库，不适合直接在Python中使用，
因此我们扩展现有的EquityCalculator来满足实验需求。
"""

from typing import Dict, List, Tuple, Optional
from itertools import combinations
import numpy as np

from models.core import Card
from abstraction.equity_calculator import EquityCalculator
from environment.hand_evaluator import compare_hands


class DeadCardRemover:
    """死牌移除器。
    
    从手牌范围中移除与已知牌（死牌）冲突的手牌组合。
    """
    
    @staticmethod
    def remove_dead_cards(
        range_weights: Dict[str, float],
        dead_cards: List[Card]
    ) -> Dict[str, float]:
        """从范围中移除包含死牌的组合。
        
        Args:
            range_weights: 手牌组合到权重的映射（如 {"AhKh": 1.0, "AsKs": 0.5}）
            dead_cards: 死牌列表（我的手牌 + 公共牌）
            
        Returns:
            移除死牌后的范围权重
        """
        dead_set = set((c.rank, c.suit) for c in dead_cards)
        result = {}
        
        for hand_str, weight in range_weights.items():
            # 解析手牌字符串
            cards = DeadCardRemover._parse_hand_string(hand_str)
            if cards is None:
                continue
            
            # 检查是否与死牌冲突
            has_conflict = False
            for card in cards:
                if (card.rank, card.suit) in dead_set:
                    has_conflict = True
                    break
            
            if not has_conflict:
                result[hand_str] = weight
        
        return result
    
    @staticmethod
    def _parse_hand_string(hand_str: str) -> Optional[Tuple[Card, Card]]:
        """解析手牌字符串为Card对象。
        
        支持格式：
        - "AhKh" - 具体牌面
        - "AA" - 对子（返回None，需要展开）
        - "AKs" - 同花（返回None，需要展开）
        - "AKo" - 异花（返回None，需要展开）
        
        Args:
            hand_str: 手牌字符串
            
        Returns:
            (Card, Card) 元组，如果是抽象表示则返回None
        """
        if len(hand_str) == 4:
            # 具体牌面格式：AhKh
            rank1 = DeadCardRemover._parse_rank(hand_str[0])
            suit1 = hand_str[1].lower()
            rank2 = DeadCardRemover._parse_rank(hand_str[2])
            suit2 = hand_str[3].lower()
            
            if rank1 is None or rank2 is None:
                return None
            if suit1 not in 'hdcs' or suit2 not in 'hdcs':
                return None
            
            return (Card(rank=rank1, suit=suit1), Card(rank=rank2, suit=suit2))
        
        # 抽象表示（AA, AKs, AKo）
        return None
    
    @staticmethod
    def _parse_rank(rank_char: str) -> Optional[int]:
        """解析牌面字符为数值。"""
        rank_map = {
            'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
            '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2
        }
        return rank_map.get(rank_char.upper())
    
    @staticmethod
    def expand_abstract_hand(hand_str: str) -> List[str]:
        """将抽象手牌表示展开为所有具体组合。
        
        Args:
            hand_str: 抽象手牌字符串（如 "AA", "AKs", "AKo"）
            
        Returns:
            具体手牌字符串列表
        """
        suits = ['h', 'd', 'c', 's']
        
        if len(hand_str) == 2:
            # 对子：AA, KK, etc.
            rank = hand_str[0]
            combos = []
            for i, s1 in enumerate(suits):
                for s2 in suits[i+1:]:
                    combos.append(f"{rank}{s1}{rank}{s2}")
            return combos
        
        elif len(hand_str) == 3:
            rank1, rank2, suit_type = hand_str[0], hand_str[1], hand_str[2].lower()
            
            if suit_type == 's':
                # 同花
                return [f"{rank1}{s}{rank2}{s}" for s in suits]
            elif suit_type == 'o':
                # 异花
                combos = []
                for s1 in suits:
                    for s2 in suits:
                        if s1 != s2:
                            combos.append(f"{rank1}{s1}{rank2}{s2}")
                return combos
        
        # 已经是具体格式
        return [hand_str]


class RangeVsRangeCalculator:
    """范围VS范围胜率计算器。
    
    计算手牌对抗对手范围的胜率，以及范围VS范围的胜率矩阵。
    """
    
    def __init__(self, num_workers: int = 0):
        """初始化计算器。
        
        Args:
            num_workers: 并行工作进程数（0=使用所有CPU核心）
        """
        self.equity_calculator = EquityCalculator(num_workers=num_workers)
        self.dead_card_remover = DeadCardRemover()
    
    def calculate_hand_vs_range_equity(
        self,
        hole_cards: Tuple[Card, Card],
        community_cards: List[Card],
        opponent_range: Dict[str, float]
    ) -> float:
        """计算单个手牌对抗对手范围的胜率。
        
        Args:
            hole_cards: 我的手牌
            community_cards: 5张公共牌
            opponent_range: 对手范围（已移除死牌或未移除）
            
        Returns:
            胜率标量 [0, 1]
        """
        if len(community_cards) != 5:
            raise ValueError(f"河牌阶段必须有5张公共牌，当前：{len(community_cards)}")
        
        # 移除死牌
        dead_cards = list(hole_cards) + community_cards
        clean_range = self.dead_card_remover.remove_dead_cards(opponent_range, dead_cards)
        
        # 展开抽象手牌
        expanded_range = self._expand_range(clean_range)
        
        if not expanded_range:
            return 0.5  # 对手范围为空，返回中性值
        
        # 计算加权胜率
        total_weight = 0.0
        weighted_equity = 0.0
        
        for opp_hand_str, weight in expanded_range.items():
            opp_cards = self.dead_card_remover._parse_hand_string(opp_hand_str)
            if opp_cards is None:
                continue
            
            # 再次检查死牌（展开后可能有新的冲突）
            opp_dead = set((c.rank, c.suit) for c in dead_cards)
            if any((c.rank, c.suit) in opp_dead for c in opp_cards):
                continue
            
            # 比较手牌
            result = compare_hands(list(hole_cards), list(opp_cards), community_cards)
            
            if result == 0:  # 我方胜
                equity = 1.0
            elif result == -1:  # 平局
                equity = 0.5
            else:  # 对方胜
                equity = 0.0
            
            weighted_equity += equity * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_equity / total_weight
    
    def calculate_range_vs_range_equity(
        self,
        my_range: Dict[str, float],
        opponent_range: Dict[str, float],
        community_cards: List[Card]
    ) -> Dict[str, float]:
        """计算我的范围中每个手牌对抗对手范围的胜率。
        
        Args:
            my_range: 我的范围
            opponent_range: 对手范围
            community_cards: 5张公共牌
            
        Returns:
            每个手牌组合的胜率映射
        """
        if len(community_cards) != 5:
            raise ValueError(f"河牌阶段必须有5张公共牌，当前：{len(community_cards)}")
        
        # 展开我的范围
        expanded_my_range = self._expand_range(my_range)
        
        # 移除与公共牌冲突的手牌
        board_dead = community_cards
        clean_my_range = self.dead_card_remover.remove_dead_cards(expanded_my_range, board_dead)
        
        result = {}
        
        for hand_str, weight in clean_my_range.items():
            cards = self.dead_card_remover._parse_hand_string(hand_str)
            if cards is None:
                continue
            
            equity = self.calculate_hand_vs_range_equity(cards, community_cards, opponent_range)
            result[hand_str] = equity
        
        return result
    
    def _expand_range(self, range_dict: Dict[str, float]) -> Dict[str, float]:
        """展开范围中的抽象手牌表示。
        
        Args:
            range_dict: 可能包含抽象表示的范围
            
        Returns:
            只包含具体手牌的范围
        """
        expanded = {}
        
        for hand_str, weight in range_dict.items():
            concrete_hands = self.dead_card_remover.expand_abstract_hand(hand_str)
            
            # 分配权重
            per_hand_weight = weight / len(concrete_hands) if concrete_hands else 0
            
            for concrete in concrete_hands:
                if concrete in expanded:
                    expanded[concrete] += per_hand_weight
                else:
                    expanded[concrete] = per_hand_weight
        
        return expanded
    
    def get_range_equity_vector(
        self,
        my_range: Dict[str, float],
        opponent_range: Dict[str, float],
        community_cards: List[Card]
    ) -> Tuple[List[str], np.ndarray]:
        """获取范围胜率向量（用于与Solver策略对比）。
        
        Args:
            my_range: 我的范围
            opponent_range: 对手范围
            community_cards: 5张公共牌
            
        Returns:
            (手牌列表, 胜率数组) 元组
        """
        equity_dict = self.calculate_range_vs_range_equity(
            my_range, opponent_range, community_cards
        )
        
        hands = list(equity_dict.keys())
        equities = np.array([equity_dict[h] for h in hands])
        
        return hands, equities


def create_full_range() -> Dict[str, float]:
    """创建完整的手牌范围（所有169种起手牌类型）。
    
    Returns:
        完整范围，所有手牌权重为1.0
    """
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    range_dict = {}
    
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            if i == j:
                # 对子
                range_dict[f"{r1}{r2}"] = 1.0
            elif i < j:
                # 同花
                range_dict[f"{r1}{r2}s"] = 1.0
                # 异花
                range_dict[f"{r1}{r2}o"] = 1.0
    
    return range_dict


def create_top_range(percent: float) -> Dict[str, float]:
    """创建前N%的手牌范围。
    
    Args:
        percent: 范围百分比（0-100）
        
    Returns:
        前N%的手牌范围
    """
    # 按强度排序的手牌（简化版本）
    top_hands = [
        'AA', 'KK', 'QQ', 'JJ', 'AKs', 'TT', 'AKo', 'AQs', '99', 'AJs',
        'AQo', 'KQs', '88', 'ATs', 'AJo', 'KJs', 'KQo', '77', 'A9s', 'KTs',
        'ATo', 'QJs', 'KJo', 'A8s', '66', 'QTs', 'K9s', 'A7s', 'A9o', 'QJo',
        'A5s', 'JTs', 'A6s', 'KTo', 'A8o', '55', 'K8s', 'A4s', 'Q9s', 'A3s',
        'K7s', 'QTo', 'A7o', 'J9s', 'A2s', 'K9o', 'JTo', 'K6s', 'A6o', '44',
    ]
    
    # 计算需要的手牌数量
    num_hands = int(len(top_hands) * percent / 100)
    num_hands = max(1, min(num_hands, len(top_hands)))
    
    return {hand: 1.0 for hand in top_hands[:num_hands]}
