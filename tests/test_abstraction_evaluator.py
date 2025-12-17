"""AbstractionEvaluator的属性测试。

本模块包含AbstractionEvaluator的属性测试，验证：
- 属性56：WCSS质量指标计算正确性
- 属性57：桶大小分布统计完整性
- 属性58：抽象可重复性
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from abstraction import (
    CardAbstraction,
    AbstractionConfig,
    AbstractionResult,
)
from abstraction.abstraction_evaluator import (
    AbstractionEvaluator,
    BucketSizeStats,
    AbstractionQualityReport,
)
from abstraction.emd_calculator import EMDCalculator


# ============================================================================
# 测试策略（生成器）
# ============================================================================

@st.composite
def small_abstraction_config_strategy(draw):
    """生成小规模的抽象配置（用于快速测试）。"""
    flop_buckets = draw(st.integers(min_value=3, max_value=20))
    turn_buckets = draw(st.integers(min_value=3, max_value=20))
    river_buckets = draw(st.integers(min_value=3, max_value=20))
    random_seed = draw(st.integers(min_value=0, max_value=10000))
    
    return AbstractionConfig(
        flop_buckets=flop_buckets,
        turn_buckets=turn_buckets,
        river_buckets=river_buckets,
        kmeans_restarts=2,
        kmeans_max_iters=10,
        random_seed=random_seed
    )


@st.composite
def normalized_histogram_strategy(draw, num_bins: int = 10):
    """生成归一化的直方图。"""
    # 生成非负值
    values = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=num_bins,
        max_size=num_bins
    ))
    values = np.array(values)
    
    # 归一化
    total = np.sum(values)
    if total > 0:
        values = values / total
    else:
        values = np.ones(num_bins) / num_bins
    
    return values


@st.composite
def clustering_data_strategy(draw, n_samples: int = 50, n_bins: int = 10, n_clusters: int = 5):
    """生成聚类数据（数据点、标签、中心）。"""
    # 生成数据点（归一化直方图）
    data_points = []
    for _ in range(n_samples):
        hist = draw(normalized_histogram_strategy(n_bins))
        data_points.append(hist)
    data_points = np.array(data_points)
    
    # 生成标签
    labels = draw(st.lists(
        st.integers(min_value=0, max_value=n_clusters - 1),
        min_size=n_samples,
        max_size=n_samples
    ))
    labels = np.array(labels)
    
    # 生成聚类中心（归一化直方图）
    centers = []
    for _ in range(n_clusters):
        center = draw(normalized_histogram_strategy(n_bins))
        centers.append(center)
    centers = np.array(centers)
    
    return data_points, labels, centers


@st.composite
def bucket_mapping_strategy(draw, size: int = 1000, num_buckets: int = 10):
    """生成桶映射数组。"""
    mapping = draw(st.lists(
        st.integers(min_value=0, max_value=num_buckets - 1),
        min_size=size,
        max_size=size
    ))
    return np.array(mapping, dtype=np.int32)


# ============================================================================
# 属性测试
# ============================================================================

class TestWCSSCalculation:
    """属性56：WCSS质量指标计算正确性测试。
    
    Feature: texas-holdem-ai-training, Property 56: WCSS质量指标计算正确性
    验证需求：15.1
    """
    
    @given(data=clustering_data_strategy(n_samples=30, n_bins=10, n_clusters=5))
    @settings(max_examples=50, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_wcss_equals_sum_of_squared_distances(self, data):
        """WCSS应该等于所有数据点到聚类中心距离平方和。
        
        Feature: texas-holdem-ai-training, Property 56: WCSS质量指标计算正确性
        验证需求：15.1
        """
        data_points, labels, centers = data
        
        evaluator = AbstractionEvaluator()
        emd_calculator = EMDCalculator()
        
        # 使用评估器计算WCSS
        wcss_evaluator = evaluator.calculate_wcss(data_points, labels, centers)
        
        # 手动计算WCSS
        wcss_manual = 0.0
        for i in range(len(data_points)):
            cluster_idx = labels[i]
            dist = emd_calculator.calculate_emd_1d(data_points[i], centers[cluster_idx])
            wcss_manual += dist ** 2
        
        # 验证两者相等
        assert np.isclose(wcss_evaluator, wcss_manual, rtol=1e-6), \
            f"WCSS计算不一致：评估器={wcss_evaluator}, 手动={wcss_manual}"
    
    @given(data=clustering_data_strategy(n_samples=20, n_bins=8, n_clusters=4))
    @settings(max_examples=50, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_wcss_is_non_negative(self, data):
        """WCSS应该是非负的。
        
        Feature: texas-holdem-ai-training, Property 56: WCSS质量指标计算正确性
        验证需求：15.1
        """
        data_points, labels, centers = data
        
        evaluator = AbstractionEvaluator()
        wcss = evaluator.calculate_wcss(data_points, labels, centers)
        
        assert wcss >= 0, f"WCSS应该非负，但得到 {wcss}"
    
    @given(n_bins=st.integers(min_value=5, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_wcss_is_zero_when_points_equal_centers(self, n_bins):
        """当数据点等于聚类中心时，WCSS应该为0。
        
        Feature: texas-holdem-ai-training, Property 56: WCSS质量指标计算正确性
        验证需求：15.1
        """
        # 创建数据点等于聚类中心的情况
        n_clusters = 3
        centers = np.random.rand(n_clusters, n_bins)
        # 归一化
        centers = centers / centers.sum(axis=1, keepdims=True)
        
        # 数据点就是聚类中心
        data_points = centers.copy()
        labels = np.arange(n_clusters)
        
        evaluator = AbstractionEvaluator()
        wcss = evaluator.calculate_wcss(data_points, labels, centers)
        
        assert np.isclose(wcss, 0.0, atol=1e-10), \
            f"当数据点等于聚类中心时，WCSS应该为0，但得到 {wcss}"
    
    def test_wcss_empty_data(self):
        """空数据的WCSS应该为0。
        
        Feature: texas-holdem-ai-training, Property 56: WCSS质量指标计算正确性
        验证需求：15.1
        """
        evaluator = AbstractionEvaluator()
        
        data_points = np.array([]).reshape(0, 10)
        labels = np.array([])
        centers = np.random.rand(3, 10)
        
        wcss = evaluator.calculate_wcss(data_points, labels, centers)
        
        assert wcss == 0.0, f"空数据的WCSS应该为0，但得到 {wcss}"


class TestBucketSizeDistribution:
    """属性57：桶大小分布统计完整性测试。
    
    Feature: texas-holdem-ai-training, Property 57: 桶大小分布统计完整性
    验证需求：15.2, 15.4
    """
    
    @given(mapping=bucket_mapping_strategy(size=500, num_buckets=10))
    @settings(max_examples=50, deadline=None)
    def test_stats_contain_required_fields(self, mapping):
        """统计报告应该包含桶数量、平均桶大小、最大桶大小。
        
        Feature: texas-holdem-ai-training, Property 57: 桶大小分布统计完整性
        验证需求：15.2, 15.4
        """
        evaluator = AbstractionEvaluator()
        stats = evaluator.get_bucket_size_distribution(mapping)
        
        # 验证必需字段存在
        assert hasattr(stats, 'count'), "统计应该包含count字段"
        assert hasattr(stats, 'avg_size'), "统计应该包含avg_size字段"
        assert hasattr(stats, 'max_size'), "统计应该包含max_size字段"
        assert hasattr(stats, 'min_size'), "统计应该包含min_size字段"
        assert hasattr(stats, 'std_size'), "统计应该包含std_size字段"
        assert hasattr(stats, 'size_distribution'), "统计应该包含size_distribution字段"
    
    @given(mapping=bucket_mapping_strategy(size=500, num_buckets=10))
    @settings(max_examples=50, deadline=None)
    def test_bucket_count_is_correct(self, mapping):
        """桶数量应该等于唯一桶ID的数量。
        
        Feature: texas-holdem-ai-training, Property 57: 桶大小分布统计完整性
        验证需求：15.2, 15.4
        """
        evaluator = AbstractionEvaluator()
        stats = evaluator.get_bucket_size_distribution(mapping)
        
        expected_count = len(np.unique(mapping))
        assert stats.count == expected_count, \
            f"桶数量不正确：期望 {expected_count}，得到 {stats.count}"
    
    @given(mapping=bucket_mapping_strategy(size=500, num_buckets=10))
    @settings(max_examples=50, deadline=None)
    def test_avg_size_is_correct(self, mapping):
        """平均桶大小应该等于总元素数除以桶数量。
        
        Feature: texas-holdem-ai-training, Property 57: 桶大小分布统计完整性
        验证需求：15.2, 15.4
        """
        evaluator = AbstractionEvaluator()
        stats = evaluator.get_bucket_size_distribution(mapping)
        
        # 手动计算平均桶大小
        unique, counts = np.unique(mapping, return_counts=True)
        expected_avg = float(np.mean(counts))
        
        assert np.isclose(stats.avg_size, expected_avg, rtol=1e-6), \
            f"平均桶大小不正确：期望 {expected_avg}，得到 {stats.avg_size}"
    
    @given(mapping=bucket_mapping_strategy(size=500, num_buckets=10))
    @settings(max_examples=50, deadline=None)
    def test_max_min_size_are_correct(self, mapping):
        """最大和最小桶大小应该正确。
        
        Feature: texas-holdem-ai-training, Property 57: 桶大小分布统计完整性
        验证需求：15.2, 15.4
        """
        evaluator = AbstractionEvaluator()
        stats = evaluator.get_bucket_size_distribution(mapping)
        
        # 手动计算
        unique, counts = np.unique(mapping, return_counts=True)
        expected_max = int(np.max(counts))
        expected_min = int(np.min(counts))
        
        assert stats.max_size == expected_max, \
            f"最大桶大小不正确：期望 {expected_max}，得到 {stats.max_size}"
        assert stats.min_size == expected_min, \
            f"最小桶大小不正确：期望 {expected_min}，得到 {stats.min_size}"
    
    @given(mapping=bucket_mapping_strategy(size=500, num_buckets=10))
    @settings(max_examples=50, deadline=None)
    def test_size_distribution_sums_to_total(self, mapping):
        """桶大小分布的总和应该等于映射数组的长度。
        
        Feature: texas-holdem-ai-training, Property 57: 桶大小分布统计完整性
        验证需求：15.2, 15.4
        """
        evaluator = AbstractionEvaluator()
        stats = evaluator.get_bucket_size_distribution(mapping)
        
        total_from_distribution = sum(stats.size_distribution.values())
        assert total_from_distribution == len(mapping), \
            f"分布总和不正确：期望 {len(mapping)}，得到 {total_from_distribution}"
    
    def test_empty_mapping_stats(self):
        """空映射的统计应该正确处理。
        
        Feature: texas-holdem-ai-training, Property 57: 桶大小分布统计完整性
        验证需求：15.2, 15.4
        """
        evaluator = AbstractionEvaluator()
        stats = evaluator.get_bucket_size_distribution(np.array([]))
        
        assert stats.count == 0
        assert stats.avg_size == 0.0
        assert stats.max_size == 0
        assert stats.min_size == 0
        assert len(stats.size_distribution) == 0


class TestAbstractionReproducibility:
    """属性58：抽象可重复性测试。
    
    Feature: texas-holdem-ai-training, Property 58: 抽象可重复性
    验证需求：15.3
    """
    
    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=20, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_same_seed_produces_same_result(self, seed):
        """相同配置（包括随机种子）应该生成相同结果。
        
        Feature: texas-holdem-ai-training, Property 58: 抽象可重复性
        验证需求：15.3
        """
        config = AbstractionConfig(
            flop_buckets=10,
            turn_buckets=10,
            river_buckets=10,
            kmeans_restarts=2,
            kmeans_max_iters=10,
            random_seed=seed
        )
        
        # 第一次生成
        abstraction1 = CardAbstraction(config)
        result1 = abstraction1.generate_abstraction()
        
        # 第二次生成（相同配置）
        abstraction2 = CardAbstraction(config)
        result2 = abstraction2.generate_abstraction()
        
        # 验证映射相同
        assert np.array_equal(result1.preflop_mapping, result2.preflop_mapping), \
            "翻牌前映射应该相同"
        assert np.array_equal(result1.flop_mapping, result2.flop_mapping), \
            "翻牌映射应该相同"
        assert np.array_equal(result1.turn_mapping, result2.turn_mapping), \
            "转牌映射应该相同"
        assert np.array_equal(result1.river_mapping, result2.river_mapping), \
            "河牌映射应该相同"
    
    @given(
        seed1=st.integers(min_value=0, max_value=5000),
        seed2=st.integers(min_value=5001, max_value=10000)
    )
    @settings(max_examples=15, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_different_seeds_may_produce_different_results(self, seed1, seed2):
        """不同的随机种子可能产生不同的结果。
        
        Feature: texas-holdem-ai-training, Property 58: 抽象可重复性
        验证需求：15.3
        """
        assume(seed1 != seed2)
        
        config1 = AbstractionConfig(
            flop_buckets=10,
            turn_buckets=10,
            river_buckets=10,
            kmeans_restarts=2,
            kmeans_max_iters=10,
            random_seed=seed1
        )
        
        config2 = AbstractionConfig(
            flop_buckets=10,
            turn_buckets=10,
            river_buckets=10,
            kmeans_restarts=2,
            kmeans_max_iters=10,
            random_seed=seed2
        )
        
        abstraction1 = CardAbstraction(config1)
        result1 = abstraction1.generate_abstraction()
        
        abstraction2 = CardAbstraction(config2)
        result2 = abstraction2.generate_abstraction()
        
        # 注意：不同种子不一定产生不同结果，但通常会
        # 这里只验证代码能正常运行
        assert result1 is not None
        assert result2 is not None
    
    def test_verify_reproducibility_method(self):
        """验证可重复性方法应该正确工作。
        
        Feature: texas-holdem-ai-training, Property 58: 抽象可重复性
        验证需求：15.3
        """
        evaluator = AbstractionEvaluator()
        
        config = AbstractionConfig(
            flop_buckets=5,
            turn_buckets=5,
            river_buckets=5,
            kmeans_restarts=1,
            kmeans_max_iters=5,
            random_seed=42
        )
        
        def generator_func(cfg):
            abstraction = CardAbstraction(cfg)
            return abstraction.generate_abstraction()
        
        is_reproducible, message = evaluator.verify_reproducibility(
            config, generator_func, num_runs=2
        )
        
        assert is_reproducible, f"抽象应该是可重复的：{message}"


class TestQualityReport:
    """质量报告生成测试。"""
    
    @given(config=small_abstraction_config_strategy())
    @settings(max_examples=15, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_report_contains_all_fields(self, config):
        """质量报告应该包含所有必需字段。
        
        Feature: texas-holdem-ai-training, Property 57: 桶大小分布统计完整性
        验证需求：15.4
        """
        abstraction = CardAbstraction(config)
        result = abstraction.generate_abstraction()
        
        evaluator = AbstractionEvaluator()
        report = evaluator.generate_report(result)
        
        # 验证报告字段
        assert report.config is not None
        assert report.wcss is not None
        assert report.bucket_stats is not None
        assert report.generation_time >= 0
        assert report.total_buckets >= 0
        assert report.compression_ratio > 0
    
    @given(config=small_abstraction_config_strategy())
    @settings(max_examples=15, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_report_dict_is_serializable(self, config):
        """报告字典应该是可序列化的。
        
        Feature: texas-holdem-ai-training, Property 57: 桶大小分布统计完整性
        验证需求：15.4
        """
        import json
        
        abstraction = CardAbstraction(config)
        result = abstraction.generate_abstraction()
        
        evaluator = AbstractionEvaluator()
        report_dict = evaluator.generate_report_dict(result)
        
        # 验证可以序列化为JSON
        try:
            json_str = json.dumps(report_dict)
            assert len(json_str) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"报告字典无法序列化为JSON：{e}")
    
    @given(config=small_abstraction_config_strategy())
    @settings(max_examples=15, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_bucket_stats_for_all_stages(self, config):
        """报告应该包含所有游戏阶段的桶统计。
        
        Feature: texas-holdem-ai-training, Property 57: 桶大小分布统计完整性
        验证需求：15.4
        """
        abstraction = CardAbstraction(config)
        result = abstraction.generate_abstraction()
        
        evaluator = AbstractionEvaluator()
        report = evaluator.generate_report(result)
        
        # 验证所有阶段都有统计
        for stage in ['preflop', 'flop', 'turn', 'river']:
            assert stage in report.bucket_stats, \
                f"报告应该包含{stage}阶段的统计"


class TestAbstractionComparison:
    """抽象比较测试。"""
    
    def test_compare_single_abstraction(self):
        """比较单个抽象应该成功。"""
        config = AbstractionConfig(
            flop_buckets=5,
            turn_buckets=5,
            river_buckets=5,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstraction = CardAbstraction(config)
        result = abstraction.generate_abstraction()
        
        evaluator = AbstractionEvaluator()
        comparison = evaluator.compare_abstractions([result], ['test'])
        
        assert 'abstractions' in comparison
        assert len(comparison['abstractions']) == 1
        assert comparison['summary']['num_abstractions'] == 1
    
    def test_compare_multiple_abstractions(self):
        """比较多个抽象应该成功。"""
        configs = [
            AbstractionConfig(
                flop_buckets=5,
                turn_buckets=5,
                river_buckets=5,
                kmeans_restarts=1,
                kmeans_max_iters=5,
                random_seed=i
            )
            for i in range(3)
        ]
        
        results = []
        for config in configs:
            abstraction = CardAbstraction(config)
            results.append(abstraction.generate_abstraction())
        
        evaluator = AbstractionEvaluator()
        comparison = evaluator.compare_abstractions(
            results, 
            ['config_0', 'config_1', 'config_2']
        )
        
        assert 'abstractions' in comparison
        assert len(comparison['abstractions']) == 3
        assert 'rankings' in comparison
        assert 'by_wcss' in comparison['rankings']
        assert 'by_time' in comparison['rankings']
        assert comparison['summary']['num_abstractions'] == 3
    
    def test_compare_empty_list(self):
        """比较空列表应该返回错误。"""
        evaluator = AbstractionEvaluator()
        comparison = evaluator.compare_abstractions([], [])
        
        assert 'error' in comparison


# ============================================================================
# 单元测试
# ============================================================================

class TestAbstractionEvaluatorBasic:
    """AbstractionEvaluator基本功能测试。"""
    
    def test_initialization(self):
        """初始化应该成功。"""
        evaluator = AbstractionEvaluator()
        assert evaluator.emd_calculator is not None
    
    def test_calculate_wcss_basic(self):
        """基本WCSS计算应该成功。"""
        evaluator = AbstractionEvaluator()
        
        # 创建简单的测试数据
        data_points = np.array([
            [0.5, 0.5],
            [0.6, 0.4],
            [0.4, 0.6],
        ])
        labels = np.array([0, 0, 0])
        centers = np.array([[0.5, 0.5]])
        
        wcss = evaluator.calculate_wcss(data_points, labels, centers)
        
        assert wcss >= 0
        assert isinstance(wcss, float)
    
    def test_calculate_wcss_with_ground_distances(self):
        """带地面距离的WCSS计算应该成功。"""
        evaluator = AbstractionEvaluator()
        
        data_points = np.array([
            [0.5, 0.5],
            [0.6, 0.4],
        ])
        labels = np.array([0, 0])
        centers = np.array([[0.5, 0.5]])
        ground_distances = np.array([[0.0, 0.1], [0.1, 0.0]])
        
        wcss = evaluator.calculate_wcss(
            data_points, labels, centers, ground_distances
        )
        
        assert wcss >= 0
    
    def test_calculate_wcss_invalid_labels(self):
        """无效标签应该抛出异常。"""
        evaluator = AbstractionEvaluator()
        
        data_points = np.array([[0.5, 0.5]])
        labels = np.array([5])  # 无效标签
        centers = np.array([[0.5, 0.5]])
        
        with pytest.raises(ValueError):
            evaluator.calculate_wcss(data_points, labels, centers)
    
    def test_calculate_wcss_mismatched_lengths(self):
        """数据点和标签长度不匹配应该抛出异常。"""
        evaluator = AbstractionEvaluator()
        
        data_points = np.array([[0.5, 0.5], [0.6, 0.4]])
        labels = np.array([0])  # 长度不匹配
        centers = np.array([[0.5, 0.5]])
        
        with pytest.raises(ValueError):
            evaluator.calculate_wcss(data_points, labels, centers)
    
    def test_get_bucket_size_distribution_basic(self):
        """基本桶大小分布计算应该成功。"""
        evaluator = AbstractionEvaluator()
        
        mapping = np.array([0, 0, 0, 1, 1, 2])
        stats = evaluator.get_bucket_size_distribution(mapping)
        
        assert stats.count == 3  # 3个不同的桶
        assert stats.max_size == 3  # 桶0有3个元素
        assert stats.min_size == 1  # 桶2有1个元素
    
    def test_get_all_bucket_stats(self):
        """获取所有阶段桶统计应该成功。"""
        config = AbstractionConfig(
            flop_buckets=5,
            turn_buckets=5,
            river_buckets=5,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstraction = CardAbstraction(config)
        result = abstraction.generate_abstraction()
        
        evaluator = AbstractionEvaluator()
        all_stats = evaluator.get_all_bucket_stats(result)
        
        assert 'preflop' in all_stats
        assert 'flop' in all_stats
        assert 'turn' in all_stats
        assert 'river' in all_stats
    
    def test_silhouette_score_basic(self):
        """基本轮廓系数计算应该成功。"""
        evaluator = AbstractionEvaluator()
        
        # 创建两个明显分离的聚类
        data_points = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ])
        labels = np.array([0, 0, 1, 1])
        
        score = evaluator.calculate_silhouette_score(data_points, labels)
        
        # 轮廓系数范围是[-1, 1]
        assert -1 <= score <= 1
    
    def test_silhouette_score_single_cluster(self):
        """单个聚类的轮廓系数应该为0。"""
        evaluator = AbstractionEvaluator()
        
        data_points = np.array([
            [0.5, 0.5],
            [0.6, 0.4],
        ])
        labels = np.array([0, 0])
        
        score = evaluator.calculate_silhouette_score(data_points, labels)
        
        assert score == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
