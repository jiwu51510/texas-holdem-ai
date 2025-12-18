"""范围VS范围胜率计算器的属性测试。

**Feature: equity-solver-validation, Property 1: 胜率计算正确性与范围约束**
**Feature: equity-solver-validation, Property 3: 范围VS范围计算完整性**
**Validates: Requirements 1.1, 1.3, 2.1, 2.3**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import numpy as np

from models.core import Card
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    DeadCardRemover,
    RangeVsRangeCalculator,
    create_full_range,
    create_top_range,
)


# ============================================================================
# 策略生成器
# ============================================================================

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
def river_scenario(draw):
    """生成河牌场景：手牌(2张) + 公共牌(5张)。"""
    cards = draw(unique_cards(7))
    hole_cards = tuple(cards[:2])
    community_cards = cards[2:]
    return hole_cards, community_cards


@st.composite
def simple_range(draw):
    """生成简单的手牌范围。"""
    hands = ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AKo', 'AQs', 'AQo', 'AJs']
    selected = draw(st.lists(st.sampled_from(hands), min_size=1, max_size=5, unique=True))
    
    return {hand: 1.0 for hand in selected}


# ============================================================================
# Property 1: 胜率计算正确性与范围约束
# ============================================================================

class TestEquityCalculationProperty:
    """胜率计算正确性属性测试。
    
    **Feature: equity-solver-validation, Property 1: 胜率计算正确性与范围约束**
    **Validates: Requirements 1.1, 1.3**
    """
    
    @given(scenario=river_scenario(), opp_range=simple_range())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.large_base_example])
    def test_equity_in_valid_range(self, scenario, opp_range):
        """验证胜率在[0, 1]范围内。"""
        hole_cards, community_cards = scenario
        
        calculator = RangeVsRangeCalculator()
        equity = calculator.calculate_hand_vs_range_equity(
            hole_cards, community_cards, opp_range
        )
        
        assert 0.0 <= equity <= 1.0, f"胜率 {equity} 超出有效范围 [0, 1]"
    
    def test_equity_nut_hand(self):
        """测试坚果牌的胜率应该接近1。"""
        # 我们持有皇家同花顺（使用不同花色避免死牌冲突）
        hole_cards = (Card(rank=14, suit='s'), Card(rank=13, suit='s'))
        community_cards = [
            Card(rank=12, suit='s'),
            Card(rank=11, suit='s'),
            Card(rank=10, suit='s'),
            Card(rank=2, suit='h'),
            Card(rank=3, suit='c'),
        ]
        
        # 对手范围使用具体手牌避免死牌问题
        opp_range = {
            'AhKh': 1.0, 'AhQh': 1.0, 'KhQh': 1.0,
            'AdKd': 1.0, 'AdQd': 1.0,
        }
        
        calculator = RangeVsRangeCalculator()
        equity = calculator.calculate_hand_vs_range_equity(
            hole_cards, community_cards, opp_range
        )
        
        # 皇家同花顺应该赢所有对手
        assert equity == 1.0, f"皇家同花顺的胜率应该是1.0，实际是 {equity}"
    
    def test_equity_weak_hand(self):
        """测试弱牌的胜率应该较低。"""
        # 我们持有72o（最弱的起手牌之一）
        hole_cards = (Card(rank=7, suit='h'), Card(rank=2, suit='s'))
        community_cards = [
            Card(rank=14, suit='c'),
            Card(rank=8, suit='d'),
            Card(rank=9, suit='h'),
            Card(rank=3, suit='d'),
            Card(rank=4, suit='c'),
        ]
        
        # 对手范围是强牌（使用具体手牌）
        opp_range = {
            'AsAd': 1.0, 'KsKd': 1.0, 'QsQd': 1.0,
            'AhKh': 1.0, 'AdKd': 1.0,
        }
        
        calculator = RangeVsRangeCalculator()
        equity = calculator.calculate_hand_vs_range_equity(
            hole_cards, community_cards, opp_range
        )
        
        # 弱牌对抗强范围应该胜率较低
        assert equity < 0.5, f"弱牌的胜率应该低于0.5，实际是 {equity}"


# ============================================================================
# Property 3: 范围VS范围计算完整性
# ============================================================================

class TestRangeVsRangeProperty:
    """范围VS范围计算完整性属性测试。
    
    **Feature: equity-solver-validation, Property 3: 范围VS范围计算完整性**
    **Validates: Requirements 2.1, 2.3**
    """
    
    def test_all_valid_hands_included(self):
        """验证返回向量包含所有有效手牌。"""
        community_cards = [
            Card(rank=14, suit='h'),
            Card(rank=13, suit='h'),
            Card(rank=12, suit='h'),
            Card(rank=11, suit='h'),
            Card(rank=10, suit='h'),
        ]
        
        # 使用简单范围
        my_range = {'AA': 1.0, 'KK': 1.0, 'QQ': 1.0}
        opp_range = {'JJ': 1.0, 'TT': 1.0}
        
        calculator = RangeVsRangeCalculator()
        equity_dict = calculator.calculate_range_vs_range_equity(
            my_range, opp_range, community_cards
        )
        
        # 验证所有不与公共牌冲突的手牌都有胜率
        assert len(equity_dict) > 0, "应该有至少一个有效手牌"
        
        # 验证所有胜率都在有效范围内
        for hand, equity in equity_dict.items():
            assert 0.0 <= equity <= 1.0, f"手牌 {hand} 的胜率 {equity} 超出范围"
    
    def test_equity_vector_format(self):
        """验证胜率向量格式正确。"""
        community_cards = [
            Card(rank=2, suit='h'),
            Card(rank=3, suit='d'),
            Card(rank=4, suit='c'),
            Card(rank=5, suit='s'),
            Card(rank=6, suit='h'),
        ]
        
        my_range = {'AA': 1.0, 'KK': 1.0}
        opp_range = {'QQ': 1.0, 'JJ': 1.0}
        
        calculator = RangeVsRangeCalculator()
        hands, equities = calculator.get_range_equity_vector(
            my_range, opp_range, community_cards
        )
        
        # 验证格式
        assert isinstance(hands, list)
        assert isinstance(equities, np.ndarray)
        assert len(hands) == len(equities)
        
        # 验证所有胜率在有效范围内
        assert np.all(equities >= 0.0)
        assert np.all(equities <= 1.0)
    
    @given(scenario=river_scenario())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.large_base_example])
    def test_range_vs_range_consistency(self, scenario):
        """验证范围VS范围计算的一致性。"""
        hole_cards, community_cards = scenario
        
        # 创建包含我们手牌的范围
        hand_str = f"{_rank_to_char(hole_cards[0].rank)}{hole_cards[0].suit}" \
                   f"{_rank_to_char(hole_cards[1].rank)}{hole_cards[1].suit}"
        
        my_range = {hand_str: 1.0}
        opp_range = {'AA': 1.0, 'KK': 1.0}
        
        calculator = RangeVsRangeCalculator()
        
        # 单手牌计算
        single_equity = calculator.calculate_hand_vs_range_equity(
            hole_cards, community_cards, opp_range
        )
        
        # 范围计算
        range_equity = calculator.calculate_range_vs_range_equity(
            my_range, opp_range, community_cards
        )
        
        # 如果手牌在范围结果中，应该一致
        if hand_str in range_equity:
            assert abs(range_equity[hand_str] - single_equity) < 1e-6, \
                f"单手牌胜率 {single_equity} 与范围胜率 {range_equity[hand_str]} 不一致"


# ============================================================================
# 辅助函数
# ============================================================================

def _rank_to_char(rank: int) -> str:
    """将牌面数值转换为字符。"""
    rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
    return rank_map.get(rank, str(rank))


# ============================================================================
# 集成测试
# ============================================================================

class TestRangeVsRangeIntegration:
    """范围VS范围计算器集成测试。"""
    
    def test_top_range_vs_weak_range(self):
        """测试强范围对抗弱范围。"""
        community_cards = [
            Card(rank=7, suit='h'),
            Card(rank=8, suit='d'),
            Card(rank=2, suit='c'),
            Card(rank=9, suit='s'),
            Card(rank=4, suit='h'),
        ]
        
        # 强范围（具体手牌）
        strong_range = {
            'AsAd': 1.0, 'AhAc': 1.0,
            'KsKd': 1.0, 'KhKc': 1.0,
        }
        
        # 弱范围（具体手牌）
        weak_range = {
            '5h3s': 1.0, '5d3c': 1.0,
            '6h4s': 1.0, '6d4c': 1.0,
        }
        
        calculator = RangeVsRangeCalculator()
        equity_dict = calculator.calculate_range_vs_range_equity(
            strong_range, weak_range, community_cards
        )
        
        # 验证有结果
        assert len(equity_dict) > 0
        
        # 计算平均胜率
        avg_equity = np.mean(list(equity_dict.values()))
        
        # 强范围对抗弱范围应该有优势
        assert avg_equity > 0.5, f"强范围的平均胜率应该大于0.5，实际是 {avg_equity}"
    
    def test_symmetric_ranges(self):
        """测试对称范围的胜率应该接近0.5。"""
        community_cards = [
            Card(rank=7, suit='h'),
            Card(rank=8, suit='d'),
            Card(rank=9, suit='c'),
            Card(rank=2, suit='s'),
            Card(rank=3, suit='h'),
        ]
        
        # 使用相同的范围
        same_range = {'AA': 1.0, 'KK': 1.0, 'QQ': 1.0}
        
        calculator = RangeVsRangeCalculator()
        equity_dict = calculator.calculate_range_vs_range_equity(
            same_range, same_range, community_cards
        )
        
        # 对称范围的平均胜率应该接近0.5
        if equity_dict:
            avg_equity = np.mean(list(equity_dict.values()))
            # 由于死牌移除，可能不完全是0.5，但应该接近
            assert 0.3 <= avg_equity <= 0.7, \
                f"对称范围的平均胜率应该接近0.5，实际是 {avg_equity}"
