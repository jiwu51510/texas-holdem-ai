"""河牌枚举器模块。

本模块实现了河牌枚举功能，用于枚举转牌阶段所有可能的河牌。
"""

from typing import List, Tuple, Set
from models.core import Card


class RiverCardEnumerator:
    """河牌枚举器类。
    
    枚举转牌阶段所有可能的河牌（52-2手牌-4公共牌=46张）。
    """
    
    @staticmethod
    def create_full_deck() -> List[Card]:
        """创建一副完整的52张扑克牌。
        
        Returns:
            包含52张牌的列表
        """
        deck = []
        for rank in range(2, 15):  # 2-14 (A=14)
            for suit in ['h', 'd', 'c', 's']:
                deck.append(Card(rank=rank, suit=suit))
        return deck
    
    @staticmethod
    def get_used_cards_set(
        hole_cards: Tuple[Card, Card],
        turn_community: List[Card]
    ) -> Set[Tuple[int, str]]:
        """获取已使用牌的集合。
        
        Args:
            hole_cards: 玩家的两张手牌
            turn_community: 4张公共牌（翻牌+转牌）
            
        Returns:
            已使用牌的集合，每张牌用(rank, suit)元组表示
        """
        used = set()
        
        # 添加手牌
        for card in hole_cards:
            used.add((card.rank, card.suit))
        
        # 添加公共牌
        for card in turn_community:
            used.add((card.rank, card.suit))
        
        return used
    
    def enumerate_river_cards(
        self,
        hole_cards: Tuple[Card, Card],
        turn_community: List[Card]
    ) -> List[Card]:
        """枚举所有可能的河牌。
        
        Args:
            hole_cards: 玩家的两张手牌
            turn_community: 4张公共牌（翻牌+转牌）
            
        Returns:
            可能的河牌列表（46张）
            
        Raises:
            ValueError: 如果公共牌数量不是4张
        """
        if len(turn_community) != 4:
            raise ValueError(f"转牌阶段必须有4张公共牌，当前：{len(turn_community)}")
        
        # 获取已使用的牌
        used_cards = self.get_used_cards_set(hole_cards, turn_community)
        
        # 验证已使用牌的数量
        if len(used_cards) != 6:
            raise ValueError(
                f"手牌和公共牌中有重复，期望6张不同的牌，实际：{len(used_cards)}"
            )
        
        # 枚举剩余的牌
        river_cards = []
        full_deck = self.create_full_deck()
        
        for card in full_deck:
            card_key = (card.rank, card.suit)
            if card_key not in used_cards:
                river_cards.append(card)
        
        return river_cards
    
    def enumerate_river_cards_for_range(
        self,
        turn_community: List[Card]
    ) -> List[Card]:
        """枚举所有可能的河牌（仅考虑公共牌）。
        
        用于范围VS范围计算时，先枚举不与公共牌冲突的河牌。
        
        Args:
            turn_community: 4张公共牌（翻牌+转牌）
            
        Returns:
            可能的河牌列表（48张）
        """
        if len(turn_community) != 4:
            raise ValueError(f"转牌阶段必须有4张公共牌，当前：{len(turn_community)}")
        
        # 获取公共牌集合
        community_set = set((card.rank, card.suit) for card in turn_community)
        
        # 枚举剩余的牌
        river_cards = []
        full_deck = self.create_full_deck()
        
        for card in full_deck:
            card_key = (card.rank, card.suit)
            if card_key not in community_set:
                river_cards.append(card)
        
        return river_cards
    
    def is_valid_river_card(
        self,
        river_card: Card,
        hole_cards: Tuple[Card, Card],
        turn_community: List[Card]
    ) -> bool:
        """检查河牌是否有效（不与手牌和公共牌冲突）。
        
        Args:
            river_card: 要检查的河牌
            hole_cards: 玩家的两张手牌
            turn_community: 4张公共牌
            
        Returns:
            如果河牌有效返回True
        """
        used_cards = self.get_used_cards_set(hole_cards, turn_community)
        river_key = (river_card.rank, river_card.suit)
        return river_key not in used_cards
