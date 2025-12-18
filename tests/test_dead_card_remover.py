"""死牌移除器的属性测试。

**Feature: equity-solver-validation, Property 2: 死牌移除正确性**
**Validates: Requirements 1.2, 2.2**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import List, Set, Tuple

from models.core import Card
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    DeadCardRemover,
    RangeVsRangeCalculator,
    create_full_range,
)


# ============================================================================
# 策略生成器
# ============================================================================

@st.composite
def valid_card(draw):
    """生成有效的Card。"""
    rank = draw(st.integers(min_value=2, max_value=14))
    suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
    return Card(rank=rank, suit=suit)


@st.composite
def unique_cards(draw, count: int):
    """生成指定数量的不重复牌。"""
    cards = []
    used = set()
    
    while len(cards) < count:
        rank = draw(st.integers(min_value=2, max_value=14))
        suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
        key = (rank, suit)
        
        if key not in used:
            used.add(key)
            cards.append(Card(rank=rank, suit=suit))
    
    return cards


@st.composite
def dead_cards_scenario(draw):
    """生成死牌场景：手牌(2张) + 公共牌(5张)。"""
    cards = draw(unique_cards(7))
    hole_cards = cards[:2]
    community_cards = cards[2:]
    return hole_cards, community_cards


@st.composite
def concrete_hand_range(draw):
    """生成具体手牌范围（非抽象表示）。"""
    suits = ['h', 'd', 'c', 's']
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    
    # 生成一些随机的具体手牌
    num_hands = draw(st.integers(min_value=1, max_value=20))
    range_dict = {}
    
    for _ in range(num_hands):
        r1 = draw(st.sampled_from(ranks))
        s1 = draw(st.sampled_from(suits))
        r2 = draw(st.sampled_from(ranks))
        s2 = draw(st.sampled_from(suits))
        
        # 确保不是同一张牌
        if r1 == r2 and s1 == s2:
            continue
        
        hand_str = f"{r1}{s1}{r2}{s2}"
        weight = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        range_dict[hand_str] = weight
    
    assume(len(range_dict) > 0)
    return range_dict


# ============================================================================
# Property 2: 死牌移除正确性
# ============================================================================

class TestDeadCardRemoverProperty:
    """死牌移除正确性属性测试。
    
    **Feature: equity-solver-validation, Property 2: 死牌移除正确性**
    **Validates: Requirements 1.2, 2.2**
    """
    
    @given(scenario=dead_cards_scenario(), range_dict=concrete_hand_range())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_no_conflict_after_removal(self, scenario, range_dict):
        """验证移除后的范围不包含任何与死牌冲突的组合。"""
        hole_cards, community_cards = scenario
        dead_cards = hole_cards + community_cards
        
        # 移除死牌
        clean_range = DeadCardRemover.remove_dead_cards(range_dict, dead_cards)
        
        # 构建死牌集合
        dead_set = set((c.rank, c.suit) for c in dead_cards)
        
        # 验证：移除后的范围中没有任何手牌包含死牌
        for hand_str in clean_range.keys():
            cards = DeadCardRemover._parse_hand_string(hand_str)
            if cards is not None:
                for card in cards:
                    assert (card.rank, card.suit) not in dead_set, \
                        f"手牌 {hand_str} 包含死牌 {card}"
    
    @given(scenario=dead_cards_scenario(), range_dict=concrete_hand_range())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    def test_valid_hands_preserved(self, scenario, range_dict):
        """验证所有不冲突的组合都被保留。"""
        hole_cards, community_cards = scenario
        dead_cards = hole_cards + community_cards
        
        # 移除死牌
        clean_range = DeadCardRemover.remove_dead_cards(range_dict, dead_cards)
        
        # 构建死牌集合
        dead_set = set((c.rank, c.suit) for c in dead_cards)
        
        # 验证：所有不冲突的手牌都被保留
        for hand_str, weight in range_dict.items():
            cards = DeadCardRemover._parse_hand_string(hand_str)
            if cards is None:
                continue
            
            has_conflict = any((c.rank, c.suit) in dead_set for c in cards)
            
            if not has_conflict:
                assert hand_str in clean_range, \
                    f"有效手牌 {hand_str} 被错误移除"
                assert clean_range[hand_str] == weight, \
                    f"手牌 {hand_str} 的权重被修改"
    
    @given(scenario=dead_cards_scenario())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.large_base_example])
    def test_empty_range_returns_empty(self, scenario):
        """验证空范围返回空结果。"""
        hole_cards, community_cards = scenario
        dead_cards = hole_cards + community_cards
        
        clean_range = DeadCardRemover.remove_dead_cards({}, dead_cards)
        
        assert clean_range == {}
    
    @given(range_dict=concrete_hand_range())
    @settings(max_examples=50)
    def test_no_dead_cards_preserves_all(self, range_dict):
        """验证没有死牌时保留所有手牌。"""
        clean_range = DeadCardRemover.remove_dead_cards(range_dict, [])
        
        assert clean_range == range_dict


# ============================================================================
# 手牌展开测试
# ============================================================================

class TestHandExpansion:
    """手牌展开功能测试。"""
    
    def test_expand_pair(self):
        """测试对子展开。"""
        combos = DeadCardRemover.expand_abstract_hand("AA")
        
        # 对子有6种组合
        assert len(combos) == 6
        
        # 验证所有组合都是AA
        for combo in combos:
            assert combo[0] == 'A' and combo[2] == 'A'
    
    def test_expand_suited(self):
        """测试同花展开。"""
        combos = DeadCardRemover.expand_abstract_hand("AKs")
        
        # 同花有4种组合
        assert len(combos) == 4
        
        # 验证所有组合都是同花
        for combo in combos:
            assert combo[1] == combo[3]  # 花色相同
    
    def test_expand_offsuit(self):
        """测试异花展开。"""
        combos = DeadCardRemover.expand_abstract_hand("AKo")
        
        # 异花有12种组合
        assert len(combos) == 12
        
        # 验证所有组合都是异花
        for combo in combos:
            assert combo[1] != combo[3]  # 花色不同
    
    def test_expand_concrete_hand(self):
        """测试具体手牌不展开。"""
        combos = DeadCardRemover.expand_abstract_hand("AhKh")
        
        assert len(combos) == 1
        assert combos[0] == "AhKh"


# ============================================================================
# 手牌解析测试
# ============================================================================

class TestHandParsing:
    """手牌解析功能测试。"""
    
    def test_parse_concrete_hand(self):
        """测试解析具体手牌。"""
        cards = DeadCardRemover._parse_hand_string("AhKs")
        
        assert cards is not None
        assert cards[0].rank == 14
        assert cards[0].suit == 'h'
        assert cards[1].rank == 13
        assert cards[1].suit == 's'
    
    def test_parse_abstract_hand_returns_none(self):
        """测试抽象手牌返回None。"""
        assert DeadCardRemover._parse_hand_string("AA") is None
        assert DeadCardRemover._parse_hand_string("AKs") is None
        assert DeadCardRemover._parse_hand_string("AKo") is None
    
    def test_parse_invalid_hand(self):
        """测试无效手牌。"""
        assert DeadCardRemover._parse_hand_string("XhYs") is None
        assert DeadCardRemover._parse_hand_string("") is None


# ============================================================================
# 集成测试
# ============================================================================

class TestDeadCardRemoverIntegration:
    """死牌移除器集成测试。"""
    
    def test_remove_from_full_range(self):
        """测试从完整范围中移除死牌。"""
        full_range = create_full_range()
        
        # 假设我们持有 AhKh，公共牌是 QhJhTh9h8h
        dead_cards = [
            Card(rank=14, suit='h'),  # Ah
            Card(rank=13, suit='h'),  # Kh
            Card(rank=12, suit='h'),  # Qh
            Card(rank=11, suit='h'),  # Jh
            Card(rank=10, suit='h'),  # Th
            Card(rank=9, suit='h'),   # 9h
            Card(rank=8, suit='h'),   # 8h
        ]
        
        # 展开范围
        remover = DeadCardRemover()
        expanded = {}
        for hand, weight in full_range.items():
            for concrete in remover.expand_abstract_hand(hand):
                expanded[concrete] = weight / len(remover.expand_abstract_hand(hand))
        
        # 移除死牌
        clean_range = remover.remove_dead_cards(expanded, dead_cards)
        
        # 验证：所有包含红桃A、K、Q、J、T、9、8的手牌都被移除
        for hand_str in clean_range.keys():
            cards = remover._parse_hand_string(hand_str)
            if cards:
                for card in cards:
                    assert not (card.suit == 'h' and card.rank in [14, 13, 12, 11, 10, 9, 8])
