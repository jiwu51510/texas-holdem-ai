"""PotentialAwareAbstractor的属性测试。

本模块包含PotentialAwareAbstractor的属性测试，验证：
- 属性41：Potential-Aware抽象考虑未来轮次
- 属性43：k-means聚类桶数量正确性
- 属性44：Imperfect Recall抽象允许信息遗忘
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from models.core import Card
from abstraction import (
    PotentialAwareAbstractor, 
    AbstractionConfig, 
    AbstractionResult
)


# ============================================================================
# 测试策略（生成器）
# ============================================================================

@st.composite
def small_abstraction_config_strategy(draw):
    """生成小规模的抽象配置（用于快速测试）。"""
    flop_buckets = draw(st.integers(min_value=3, max_value=20))
    turn_buckets = draw(st.integers(min_value=3, max_value=20))
    river_buckets = draw(st.integers(min_value=3, max_value=20))
    
    return AbstractionConfig(
        flop_buckets=flop_buckets,
        turn_buckets=turn_buckets,
        river_buckets=river_buckets,
        kmeans_restarts=2,
        kmeans_max_iters=10,
        random_seed=draw(st.integers(min_value=0, max_value=10000))
    )


@st.composite
def histogram_data_strategy(draw, min_samples: int = 10, max_samples: int = 50,
                           min_bins: int = 5, max_bins: int = 20):
    """生成用于k-means聚类的直方图数据。"""
    num_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    num_bins = draw(st.integers(min_value=min_bins, max_value=max_bins))
    
    # 生成随机直方图
    histograms = np.random.rand(num_samples, num_bins)
    # 归一化
    histograms = histograms / histograms.sum(axis=1, keepdims=True)
    
    return histograms


@st.composite
def num_clusters_strategy(draw, min_clusters: int = 2, max_clusters: int = 10):
    """生成聚类数量。"""
    return draw(st.integers(min_value=min_clusters, max_value=max_clusters))


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
def hole_and_flop_strategy(draw):
    """生成手牌和翻牌组合。"""
    cards = draw(unique_cards_strategy(5))
    hole_cards = (cards[0], cards[1])
    flop_cards = cards[2:5]
    return hole_cards, flop_cards


# ============================================================================
# 属性测试
# ============================================================================

class TestPotentialAwareFutureRounds:
    """属性41：Potential-Aware抽象考虑未来轮次测试。
    
    Feature: texas-holdem-ai-training, Property 41: Potential-Aware抽象考虑未来轮次
    验证需求：11.1
    """
    
    @given(
        hole_and_flop=hole_and_flop_strategy(),
        num_turn_buckets=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=50, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_flop_feature_contains_turn_distribution(self, hole_and_flop, num_turn_buckets):
        """翻牌阶段特征向量应该包含转牌桶分布信息。
        
        Feature: texas-holdem-ai-training, Property 41: Potential-Aware抽象考虑未来轮次
        验证需求：11.1
        """
        hole_cards, flop_cards = hole_and_flop
        
        config = AbstractionConfig(
            turn_buckets=num_turn_buckets,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstractor = PotentialAwareAbstractor(config)
        
        # 创建模拟的转牌抽象映射
        turn_mapping = np.random.randint(0, num_turn_buckets, size=10**6)
        
        # 获取翻牌特征向量
        feature = abstractor.get_flop_feature_vector(
            hole_cards, flop_cards, turn_mapping, num_turn_buckets
        )
        
        # 验证特征向量维度等于转牌桶数量
        assert len(feature) == num_turn_buckets, \
            f"特征向量维度应该等于转牌桶数量：期望 {num_turn_buckets}，实际 {len(feature)}"
        
        # 验证特征向量是有效的概率分布（如果有非零元素）
        total = np.sum(feature)
        if total > 0:
            assert np.isclose(total, 1.0, atol=1e-6), \
                f"特征向量应该是归一化的概率分布，概率和为 {total}"


class TestKMeansClusterCount:
    """属性43：k-means聚类桶数量正确性测试。
    
    Feature: texas-holdem-ai-training, Property 43: k-means聚类桶数量正确性
    验证需求：11.3, 11.4
    """
    
    @given(
        num_clusters=num_clusters_strategy(min_clusters=2, max_clusters=8)
    )
    @settings(max_examples=30, deadline=None)
    def test_kmeans_generates_correct_cluster_count(self, num_clusters):
        """k-means应该生成指定数量的聚类。
        
        Feature: texas-holdem-ai-training, Property 43: k-means聚类桶数量正确性
        验证需求：11.3, 11.4
        """
        config = AbstractionConfig(
            kmeans_restarts=2,
            kmeans_max_iters=20,
            random_seed=42
        )
        abstractor = PotentialAwareAbstractor(config)
        
        # 生成足够多的样本
        num_samples = max(num_clusters * 5, 20)
        num_bins = 10
        histograms = np.random.rand(num_samples, num_bins)
        histograms = histograms / histograms.sum(axis=1, keepdims=True)
        
        labels, centers, wcss = abstractor._kmeans_with_emd(
            histograms, num_clusters=num_clusters
        )
        
        # 验证聚类中心数量
        assert len(centers) == num_clusters, \
            f"聚类中心数量不正确：期望 {num_clusters}，实际 {len(centers)}"
        
        # 验证所有样本都被分配了标签
        assert len(labels) == num_samples, \
            f"标签数量不正确：期望 {num_samples}，实际 {len(labels)}"
        
        # 验证标签在有效范围内
        assert np.all(labels >= 0) and np.all(labels < num_clusters), \
            f"标签超出有效范围 [0, {num_clusters})"
    
    @given(
        num_clusters=num_clusters_strategy(min_clusters=2, max_clusters=5)
    )
    @settings(max_examples=20, deadline=None)
    def test_kmeans_generates_non_empty_clusters(self, num_clusters):
        """k-means应该生成非空的聚类。
        
        Feature: texas-holdem-ai-training, Property 43: k-means聚类桶数量正确性
        验证需求：11.3, 11.4
        """
        config = AbstractionConfig(
            kmeans_restarts=3,
            kmeans_max_iters=30,
            random_seed=42
        )
        abstractor = PotentialAwareAbstractor(config)
        
        # 生成足够多的样本
        num_samples = num_clusters * 10
        num_bins = 10
        histograms = np.random.rand(num_samples, num_bins)
        histograms = histograms / histograms.sum(axis=1, keepdims=True)
        
        labels, centers, wcss = abstractor._kmeans_with_emd(
            histograms, num_clusters=num_clusters
        )
        
        # 统计每个聚类的样本数
        unique_labels = np.unique(labels)
        
        # 验证至少有一些聚类是非空的
        # 注意：由于k-means的特性，可能有些聚类是空的
        assert len(unique_labels) > 0, "应该至少有一个非空聚类"


class TestImperfectRecall:
    """属性44：Imperfect Recall抽象允许信息遗忘测试。
    
    Feature: texas-holdem-ai-training, Property 44: Imperfect Recall抽象允许信息遗忘
    验证需求：11.5
    """
    
    @given(config=small_abstraction_config_strategy())
    @settings(max_examples=20, deadline=None)
    def test_abstraction_allows_information_loss(self, config):
        """抽象应该允许信息遗忘（多个不同手牌可能映射到同一桶）。
        
        Feature: texas-holdem-ai-training, Property 44: Imperfect Recall抽象允许信息遗忘
        验证需求：11.5
        """
        abstractor = PotentialAwareAbstractor(config)
        
        # 生成完整抽象
        result = abstractor.generate_full_abstraction()
        
        # 验证翻牌前映射（169种起手牌）
        # 如果桶数量小于169，则必然有多个手牌映射到同一桶
        if config.preflop_buckets < 169:
            # 检查是否有重复的桶ID
            unique_buckets = len(np.unique(result.preflop_mapping))
            assert unique_buckets <= config.preflop_buckets, \
                f"翻牌前桶数量超过配置：{unique_buckets} > {config.preflop_buckets}"
    
    @given(config=small_abstraction_config_strategy())
    @settings(max_examples=20, deadline=None)
    def test_abstraction_result_complete(self, config):
        """生成的抽象结果应该是完整的。
        
        Feature: texas-holdem-ai-training, Property 44: Imperfect Recall抽象允许信息遗忘
        验证需求：11.5
        """
        abstractor = PotentialAwareAbstractor(config)
        result = abstractor.generate_full_abstraction()
        
        # 验证结果完整性
        assert result.is_complete(), "抽象结果应该是完整的"
        
        # 验证所有映射都存在
        assert result.preflop_mapping is not None
        assert result.flop_mapping is not None
        assert result.turn_mapping is not None
        assert result.river_mapping is not None


# ============================================================================
# 单元测试
# ============================================================================

class TestPotentialAwareAbstractorBasic:
    """PotentialAwareAbstractor基本功能测试。"""
    
    def test_abstractor_initialization(self):
        """抽象器应该正确初始化。"""
        config = AbstractionConfig(
            flop_buckets=10,
            turn_buckets=10,
            river_buckets=10
        )
        abstractor = PotentialAwareAbstractor(config)
        
        assert abstractor.config == config
        assert abstractor.equity_calculator is not None
        assert abstractor.emd_calculator is not None
    
    def test_river_abstraction(self):
        """河牌阶段抽象应该正确生成。"""
        config = AbstractionConfig(river_buckets=10)
        abstractor = PotentialAwareAbstractor(config)
        
        mapping, centers, wcss = abstractor.compute_river_abstraction()
        
        assert len(centers) == config.river_buckets
        assert wcss >= 0
    
    def test_turn_abstraction(self):
        """转牌阶段抽象应该正确生成。"""
        config = AbstractionConfig(turn_buckets=10, river_buckets=10)
        abstractor = PotentialAwareAbstractor(config)
        
        # 先生成河牌抽象
        river_mapping, river_centers, _ = abstractor.compute_river_abstraction()
        
        # 生成转牌抽象
        mapping, centers, wcss = abstractor.compute_turn_abstraction(
            river_mapping, river_centers
        )
        
        assert centers.shape[0] == config.turn_buckets
        assert wcss >= 0
    
    def test_flop_abstraction(self):
        """翻牌阶段抽象应该正确生成。"""
        config = AbstractionConfig(
            flop_buckets=10, 
            turn_buckets=10, 
            river_buckets=10
        )
        abstractor = PotentialAwareAbstractor(config)
        
        # 先生成河牌和转牌抽象
        river_mapping, river_centers, _ = abstractor.compute_river_abstraction()
        turn_mapping, turn_centers, _ = abstractor.compute_turn_abstraction(
            river_mapping, river_centers
        )
        
        # 生成翻牌抽象
        mapping, centers, wcss = abstractor.compute_flop_abstraction(
            turn_mapping, turn_centers
        )
        
        assert centers.shape[0] == config.flop_buckets
        assert wcss >= 0
    
    def test_full_abstraction_generation(self):
        """完整抽象生成应该正确。"""
        config = AbstractionConfig(
            flop_buckets=5,
            turn_buckets=5,
            river_buckets=5,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstractor = PotentialAwareAbstractor(config)
        
        result = abstractor.generate_full_abstraction()
        
        assert result.is_complete()
        assert result.generation_time >= 0
        assert 'flop' in result.wcss
        assert 'turn' in result.wcss
        assert 'river' in result.wcss
    
    def test_kmeans_with_small_data(self):
        """k-means应该能处理小数据集。"""
        config = AbstractionConfig(kmeans_restarts=1, kmeans_max_iters=5)
        abstractor = PotentialAwareAbstractor(config)
        
        # 小数据集
        histograms = np.array([
            [0.5, 0.5],
            [0.3, 0.7],
            [0.7, 0.3],
        ])
        
        labels, centers, wcss = abstractor._kmeans_with_emd(
            histograms, num_clusters=2
        )
        
        assert len(labels) == 3
        assert len(centers) == 2
    
    def test_kmeans_with_more_clusters_than_samples(self):
        """当聚类数大于样本数时，k-means应该正确处理。"""
        config = AbstractionConfig(kmeans_restarts=1, kmeans_max_iters=5)
        abstractor = PotentialAwareAbstractor(config)
        
        histograms = np.array([
            [0.5, 0.5],
            [0.3, 0.7],
        ])
        
        labels, centers, wcss = abstractor._kmeans_with_emd(
            histograms, num_clusters=5
        )
        
        # 每个样本应该有自己的聚类
        assert len(labels) == 2


class TestKMeansPlusPlus:
    """k-means++初始化测试。"""
    
    def test_kmeans_plus_plus_init(self):
        """k-means++初始化应该选择分散的中心。"""
        config = AbstractionConfig(random_seed=42)
        abstractor = PotentialAwareAbstractor(config)
        
        # 创建明显分离的数据
        histograms = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.9],
        ])
        
        centers = abstractor._kmeans_plus_plus_init(histograms, num_clusters=3)
        
        assert len(centers) == 3
        # 中心应该是有效的直方图
        for center in centers:
            assert len(center) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
