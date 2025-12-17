"""EquityCalculator的属性测试。

本模块包含EquityCalculator的属性测试，验证：
- 属性46：Equity分布直方图归一化
- 属性47：Equity直方图区间覆盖完整性
- 属性48：Potential-Aware特征维度正确性
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from models.core import Card
from abstraction import EquityCalculator, EquityHistogram
from abstraction.equity_calculator import get_canonical_hand_index, index_to_canonical_hand


# ============================================================================
# 测试策略（生成器）
# ============================================================================

@st.composite
def card_strategy(draw):
    """生成随机扑克牌。"""
    rank = draw(st.integers(min_value=2, max_value=14))
    suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
    return Card(rank=rank, suit=suit)


@st.composite
def unique_cards_strategy(draw, count: int):
    """生成指定数量的不重复扑克牌。"""
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
def hole_cards_strategy(draw):
    """生成两张不重复的手牌。"""
    cards = draw(unique_cards_strategy(2))
    return (cards[0], cards[1])


@st.composite
def hole_and_community_strategy(draw, community_count: int):
    """生成手牌和公共牌组合。"""
    total_cards = 2 + community_count
    cards = draw(unique_cards_strategy(total_cards))
    hole_cards = (cards[0], cards[1])
    community_cards = cards[2:]
    return hole_cards, community_cards


@st.composite
def num_bins_strategy(draw):
    """生成直方图区间数量。"""
    return draw(st.integers(min_value=5, max_value=100))


# ============================================================================
# 属性测试
# ============================================================================

class TestEquityDistributionNormalization:
    """属性46：Equity分布直方图归一化测试。
    
    Feature: texas-holdem-ai-training, Property 46: Equity分布直方图归一化
    验证需求：12.1, 12.2
    """
    
    @given(
        hole_and_community=hole_and_community_strategy(community_count=5),
        num_bins=num_bins_strategy()
    )
    @settings(max_examples=100, deadline=None, 
              suppress_health_check=[HealthCheck.large_base_example])
    def test_river_equity_distribution_normalized(self, hole_and_community, num_bins):
        """河牌阶段Equity分布直方图应该归一化。
        
        Feature: texas-holdem-ai-training, Property 46: Equity分布直方图归一化
        验证需求：12.1, 12.2
        """
        hole_cards, community_cards = hole_and_community
        
        calc = EquityCalculator(num_workers=1)
        hist = calc.calculate_equity_distribution(hole_cards, community_cards, num_bins=num_bins)
        
        # 验证直方图归一化
        assert hist.is_normalized(), f"直方图未归一化，概率和为 {np.sum(hist.counts)}"
    
    @given(
        hole_and_community=hole_and_community_strategy(community_count=3),
        num_bins=num_bins_strategy()
    )
    @settings(max_examples=50, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_flop_equity_distribution_normalized(self, hole_and_community, num_bins):
        """翻牌阶段Equity分布直方图应该归一化。
        
        Feature: texas-holdem-ai-training, Property 46: Equity分布直方图归一化
        验证需求：12.1, 12.2
        """
        hole_cards, community_cards = hole_and_community
        
        calc = EquityCalculator(num_workers=1)
        # 使用快速采样版本，因为精确计算太慢
        hist = calc.calculate_equity_distribution_fast(
            hole_cards, community_cards, 
            num_bins=num_bins, 
            num_samples=100
        )
        
        # 验证直方图归一化
        assert hist.is_normalized(), f"直方图未归一化，概率和为 {np.sum(hist.counts)}"


class TestEquityHistogramCoverage:
    """属性47：Equity直方图区间覆盖完整性测试。
    
    Feature: texas-holdem-ai-training, Property 47: Equity直方图区间覆盖完整性
    验证需求：12.2
    """
    
    @given(num_bins=num_bins_strategy())
    @settings(max_examples=100, deadline=None)
    def test_histogram_bins_count(self, num_bins):
        """直方图区间数量应该正确。
        
        Feature: texas-holdem-ai-training, Property 47: Equity直方图区间覆盖完整性
        验证需求：12.2
        """
        bins = np.linspace(0, 1, num_bins + 1)
        counts = np.zeros(num_bins)
        hist = EquityHistogram(bins=bins, counts=counts)
        
        assert hist.num_bins == num_bins, f"区间数量不正确：期望 {num_bins}，实际 {hist.num_bins}"
    
    @given(num_bins=num_bins_strategy())
    @settings(max_examples=100, deadline=None)
    def test_histogram_bins_coverage(self, num_bins):
        """直方图区间应该覆盖[0, 1]范围。
        
        Feature: texas-holdem-ai-training, Property 47: Equity直方图区间覆盖完整性
        验证需求：12.2
        """
        bins = np.linspace(0, 1, num_bins + 1)
        counts = np.zeros(num_bins)
        hist = EquityHistogram(bins=bins, counts=counts)
        
        # 验证区间边界
        assert np.isclose(hist.bins[0], 0.0), f"区间起始值不为0：{hist.bins[0]}"
        assert np.isclose(hist.bins[-1], 1.0), f"区间结束值不为1：{hist.bins[-1]}"
        
        # 验证区间数量
        assert len(hist.bins) == num_bins + 1, f"区间边界数量不正确"
    
    @given(
        hole_and_community=hole_and_community_strategy(community_count=5),
        num_bins=num_bins_strategy()
    )
    @settings(max_examples=100, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_calculated_histogram_coverage(self, hole_and_community, num_bins):
        """计算得到的直方图应该覆盖[0, 1]范围。
        
        Feature: texas-holdem-ai-training, Property 47: Equity直方图区间覆盖完整性
        验证需求：12.2
        """
        hole_cards, community_cards = hole_and_community
        
        calc = EquityCalculator(num_workers=1)
        hist = calc.calculate_equity_distribution(hole_cards, community_cards, num_bins=num_bins)
        
        # 验证区间覆盖
        assert np.isclose(hist.bins[0], 0.0), f"区间起始值不为0"
        assert np.isclose(hist.bins[-1], 1.0), f"区间结束值不为1"
        assert hist.num_bins == num_bins, f"区间数量不正确"


class TestPotentialAwareFeatureDimension:
    """属性48：Potential-Aware特征维度正确性测试。
    
    Feature: texas-holdem-ai-training, Property 48: Potential-Aware特征维度正确性
    验证需求：12.3
    """
    
    @given(
        hole_and_community=hole_and_community_strategy(community_count=3),
        num_turn_buckets=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=50, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_turn_bucket_distribution_dimension(self, hole_and_community, num_turn_buckets):
        """翻牌手牌的转牌桶分布维度应该等于转牌桶数量。
        
        Feature: texas-holdem-ai-training, Property 48: Potential-Aware特征维度正确性
        验证需求：12.3
        """
        hole_cards, flop_cards = hole_and_community
        
        # 创建模拟的转牌桶映射
        # 实际使用中，这个映射会由PotentialAwareAbstractor生成
        turn_bucket_mapping = np.random.randint(0, num_turn_buckets, size=10**6)
        
        calc = EquityCalculator(num_workers=1)
        distribution = calc.calculate_turn_bucket_distribution(
            hole_cards, flop_cards, 
            turn_bucket_mapping, 
            num_turn_buckets
        )
        
        # 验证维度
        assert len(distribution) == num_turn_buckets, \
            f"特征维度不正确：期望 {num_turn_buckets}，实际 {len(distribution)}"
        
        # 验证归一化（如果有非零元素）
        total = np.sum(distribution)
        if total > 0:
            assert np.isclose(total, 1.0, atol=1e-6), \
                f"分布未归一化，概率和为 {total}"


# ============================================================================
# 单元测试
# ============================================================================

class TestEquityCalculatorBasic:
    """EquityCalculator基本功能测试。"""
    
    def test_equity_range(self):
        """Equity值应该在[0, 1]范围内。"""
        calc = EquityCalculator(num_workers=1)
        
        # 测试AA（强牌）
        hole_cards = (Card(14, 'h'), Card(14, 's'))
        community = [Card(2, 'h'), Card(3, 'd'), Card(4, 'c'), Card(5, 's'), Card(7, 'h')]
        equity = calc.calculate_equity(hole_cards, community)
        
        assert 0 <= equity <= 1, f"Equity超出范围：{equity}"
    
    def test_strong_hand_high_equity(self):
        """强牌应该有较高的Equity。"""
        calc = EquityCalculator(num_workers=1)
        
        # AA在河牌阶段应该有较高的Equity
        hole_cards = (Card(14, 'h'), Card(14, 's'))
        community = [Card(2, 'h'), Card(3, 'd'), Card(4, 'c'), Card(5, 's'), Card(7, 'h')]
        equity = calc.calculate_equity(hole_cards, community)
        
        assert equity > 0.5, f"AA的Equity应该大于0.5，实际为 {equity}"
    
    def test_histogram_creation(self):
        """直方图创建应该正确。"""
        bins = np.linspace(0, 1, 11)
        counts = np.array([0.1] * 10)
        hist = EquityHistogram(bins=bins, counts=counts)
        
        assert hist.num_bins == 10
        assert hist.is_normalized()
    
    def test_histogram_normalize(self):
        """直方图归一化应该正确。"""
        bins = np.linspace(0, 1, 11)
        counts = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1])
        hist = EquityHistogram(bins=bins, counts=counts)
        
        normalized = hist.normalize()
        assert normalized.is_normalized()
        assert np.isclose(np.sum(normalized.counts), 1.0)


class TestCanonicalHandIndex:
    """规范化手牌索引测试。"""
    
    def test_pairs_index_range(self):
        """对子索引应该在0-12范围内。"""
        for rank in range(2, 15):
            hand = (Card(rank, 'h'), Card(rank, 's'))
            idx = get_canonical_hand_index(hand)
            assert 0 <= idx <= 12, f"对子索引超出范围：{idx}"
    
    def test_suited_index_range(self):
        """同花索引应该在13-90范围内。"""
        for rank1 in range(3, 15):
            for rank2 in range(2, rank1):
                hand = (Card(rank1, 'h'), Card(rank2, 'h'))
                idx = get_canonical_hand_index(hand)
                assert 13 <= idx <= 90, f"同花索引超出范围：{idx}"
    
    def test_offsuit_index_range(self):
        """异花索引应该在91-168范围内。"""
        for rank1 in range(3, 15):
            for rank2 in range(2, rank1):
                hand = (Card(rank1, 'h'), Card(rank2, 's'))
                idx = get_canonical_hand_index(hand)
                assert 91 <= idx <= 168, f"异花索引超出范围：{idx}"
    
    def test_index_roundtrip(self):
        """索引往返应该保持一致。"""
        for idx in range(169):
            r1, r2, suited = index_to_canonical_hand(idx)
            
            # 创建对应的手牌
            if r1 == r2:
                hand = (Card(r1, 'h'), Card(r2, 's'))
            elif suited:
                hand = (Card(r1, 'h'), Card(r2, 'h'))
            else:
                hand = (Card(r1, 'h'), Card(r2, 's'))
            
            new_idx = get_canonical_hand_index(hand)
            assert new_idx == idx, f"索引往返不一致：{idx} -> {new_idx}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
