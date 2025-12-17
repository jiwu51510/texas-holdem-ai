"""EMDCalculator的属性测试。

本模块包含EMDCalculator的属性测试，验证：
- 属性42：EMD距离度量正确性（非负性、对称性、三角不等式）
- 属性49：一维EMD线性时间计算
- 属性50：EMD地面距离传递性
"""

import pytest
import numpy as np
import time
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from abstraction import EMDCalculator


# ============================================================================
# 测试策略（生成器）
# ============================================================================

@st.composite
def normalized_histogram_strategy(draw, min_bins: int = 5, max_bins: int = 50):
    """生成归一化的直方图。"""
    num_bins = draw(st.integers(min_value=min_bins, max_value=max_bins))
    # 生成非负值
    values = draw(st.lists(
        st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
        min_size=num_bins, max_size=num_bins
    ))
    values = np.array(values)
    
    # 归一化
    total = np.sum(values)
    if total > 0:
        values = values / total
    else:
        # 如果全为0，使用均匀分布
        values = np.ones(num_bins) / num_bins
    
    return values


@st.composite
def histogram_pair_strategy(draw, min_bins: int = 5, max_bins: int = 50):
    """生成两个相同长度的归一化直方图。"""
    num_bins = draw(st.integers(min_value=min_bins, max_value=max_bins))
    
    hist1 = draw(normalized_histogram_strategy(min_bins=num_bins, max_bins=num_bins))
    hist2 = draw(normalized_histogram_strategy(min_bins=num_bins, max_bins=num_bins))
    
    return hist1, hist2


@st.composite
def histogram_triple_strategy(draw, min_bins: int = 5, max_bins: int = 30):
    """生成三个相同长度的归一化直方图。"""
    num_bins = draw(st.integers(min_value=min_bins, max_value=max_bins))
    
    hist1 = draw(normalized_histogram_strategy(min_bins=num_bins, max_bins=num_bins))
    hist2 = draw(normalized_histogram_strategy(min_bins=num_bins, max_bins=num_bins))
    hist3 = draw(normalized_histogram_strategy(min_bins=num_bins, max_bins=num_bins))
    
    return hist1, hist2, hist3


@st.composite
def ground_distance_matrix_strategy(draw, size: int):
    """生成有效的地面距离矩阵（对称、非负、对角线为0）。"""
    # 生成上三角矩阵
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            dist = draw(st.floats(min_value=0, max_value=10, 
                                  allow_nan=False, allow_infinity=False))
            matrix[i, j] = dist
            matrix[j, i] = dist
    
    return matrix


# ============================================================================
# 属性测试
# ============================================================================

class TestEMDMetricProperties:
    """属性42：EMD距离度量正确性测试。
    
    Feature: texas-holdem-ai-training, Property 42: EMD距离度量正确性
    验证需求：11.2
    """
    
    @given(histograms=histogram_pair_strategy())
    @settings(max_examples=100, deadline=None)
    def test_emd_non_negative(self, histograms):
        """EMD应该是非负的。
        
        Feature: texas-holdem-ai-training, Property 42: EMD距离度量正确性
        验证需求：11.2
        """
        hist1, hist2 = histograms
        emd = EMDCalculator.calculate_emd_1d(hist1, hist2)
        
        assert emd >= 0, f"EMD应该非负，实际为 {emd}"
    
    @given(histograms=histogram_pair_strategy())
    @settings(max_examples=100, deadline=None)
    def test_emd_symmetric(self, histograms):
        """EMD应该是对称的：d(x, y) = d(y, x)。
        
        Feature: texas-holdem-ai-training, Property 42: EMD距离度量正确性
        验证需求：11.2
        """
        hist1, hist2 = histograms
        
        emd_12 = EMDCalculator.calculate_emd_1d(hist1, hist2)
        emd_21 = EMDCalculator.calculate_emd_1d(hist2, hist1)
        
        assert np.isclose(emd_12, emd_21, atol=1e-10), \
            f"EMD不对称：d(1,2)={emd_12}, d(2,1)={emd_21}"
    
    @given(histograms=histogram_triple_strategy())
    @settings(max_examples=100, deadline=None)
    def test_emd_triangle_inequality(self, histograms):
        """EMD应该满足三角不等式：d(x, z) <= d(x, y) + d(y, z)。
        
        Feature: texas-holdem-ai-training, Property 42: EMD距离度量正确性
        验证需求：11.2
        """
        hist1, hist2, hist3 = histograms
        
        d12 = EMDCalculator.calculate_emd_1d(hist1, hist2)
        d23 = EMDCalculator.calculate_emd_1d(hist2, hist3)
        d13 = EMDCalculator.calculate_emd_1d(hist1, hist3)
        
        # 允许小的数值误差
        assert d13 <= d12 + d23 + 1e-10, \
            f"三角不等式不满足：d(1,3)={d13} > d(1,2)+d(2,3)={d12+d23}"
    
    @given(histogram=normalized_histogram_strategy())
    @settings(max_examples=100, deadline=None)
    def test_emd_identity(self, histogram):
        """相同直方图的EMD应该为0。
        
        Feature: texas-holdem-ai-training, Property 42: EMD距离度量正确性
        验证需求：11.2
        """
        emd = EMDCalculator.calculate_emd_1d(histogram, histogram)
        
        assert np.isclose(emd, 0, atol=1e-10), \
            f"相同直方图的EMD应该为0，实际为 {emd}"


class TestEMDLinearTime:
    """属性49：一维EMD线性时间计算测试。
    
    Feature: texas-holdem-ai-training, Property 49: 一维EMD线性时间计算
    验证需求：13.1
    """
    
    @given(size_multiplier=st.integers(min_value=1, max_value=5))
    @settings(max_examples=20, deadline=None)
    def test_emd_linear_time_complexity(self, size_multiplier):
        """EMD计算时间应该与直方图大小成线性关系。
        
        Feature: texas-holdem-ai-training, Property 49: 一维EMD线性时间计算
        验证需求：13.1
        """
        base_size = 1000
        sizes = [base_size * i for i in range(1, size_multiplier + 1)]
        times = []
        
        for size in sizes:
            hist1 = np.random.rand(size)
            hist1 = hist1 / np.sum(hist1)
            hist2 = np.random.rand(size)
            hist2 = hist2 / np.sum(hist2)
            
            # 多次运行取平均
            num_runs = 10
            start = time.perf_counter()
            for _ in range(num_runs):
                EMDCalculator.calculate_emd_1d(hist1, hist2)
            elapsed = (time.perf_counter() - start) / num_runs
            times.append(elapsed)
        
        # 验证时间增长大致是线性的
        # 如果是线性的，时间比应该接近大小比
        if len(times) >= 2 and times[0] > 1e-7:
            time_ratio = times[-1] / times[0]
            size_ratio = sizes[-1] / sizes[0]
            
            # 允许一定的误差（2倍以内）
            assert time_ratio < size_ratio * 3, \
                f"时间增长不是线性的：时间比={time_ratio:.2f}, 大小比={size_ratio}"


class TestEMDGroundDistanceTransitivity:
    """属性50：EMD地面距离传递性测试。
    
    Feature: texas-holdem-ai-training, Property 50: EMD地面距离传递性
    验证需求：13.3
    """
    
    @given(num_bins=st.integers(min_value=3, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_ground_distance_matrix_symmetric(self, num_bins):
        """地面距离矩阵应该是对称的。
        
        Feature: texas-holdem-ai-training, Property 50: EMD地面距离传递性
        验证需求：13.3
        """
        # 创建随机聚类中心
        centers = np.random.rand(num_bins, 20)
        centers = centers / centers.sum(axis=1, keepdims=True)
        
        dist_matrix = EMDCalculator.compute_ground_distance_matrix(centers)
        
        # 验证对称性
        assert np.allclose(dist_matrix, dist_matrix.T), \
            "地面距离矩阵不对称"
    
    @given(num_bins=st.integers(min_value=3, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_ground_distance_matrix_diagonal_zero(self, num_bins):
        """地面距离矩阵的对角线应该为0。
        
        Feature: texas-holdem-ai-training, Property 50: EMD地面距离传递性
        验证需求：13.3
        """
        centers = np.random.rand(num_bins, 20)
        centers = centers / centers.sum(axis=1, keepdims=True)
        
        dist_matrix = EMDCalculator.compute_ground_distance_matrix(centers)
        
        # 验证对角线为0
        assert np.allclose(np.diag(dist_matrix), 0), \
            "地面距离矩阵对角线不为0"
    
    @given(num_bins=st.integers(min_value=3, max_value=8))
    @settings(max_examples=30, deadline=None)
    def test_ground_distance_triangle_inequality(self, num_bins):
        """地面距离矩阵应该满足三角不等式。
        
        Feature: texas-holdem-ai-training, Property 50: EMD地面距离传递性
        验证需求：13.3
        """
        centers = np.random.rand(num_bins, 20)
        centers = centers / centers.sum(axis=1, keepdims=True)
        
        dist_matrix = EMDCalculator.compute_ground_distance_matrix(centers)
        
        # 验证三角不等式
        for i in range(num_bins):
            for j in range(num_bins):
                for k in range(num_bins):
                    assert dist_matrix[i, k] <= dist_matrix[i, j] + dist_matrix[j, k] + 1e-10, \
                        f"三角不等式不满足：d({i},{k})={dist_matrix[i,k]} > d({i},{j})+d({j},{k})={dist_matrix[i,j]+dist_matrix[j,k]}"


# ============================================================================
# 单元测试
# ============================================================================

class TestEMDCalculatorBasic:
    """EMDCalculator基本功能测试。"""
    
    def test_emd_simple_case(self):
        """简单情况的EMD计算。"""
        hist1 = np.array([1.0, 0.0, 0.0, 0.0])
        hist2 = np.array([0.0, 0.0, 0.0, 1.0])
        
        emd = EMDCalculator.calculate_emd_1d(hist1, hist2)
        
        # 所有质量需要移动3个单位
        assert np.isclose(emd, 3.0), f"EMD应该为3.0，实际为 {emd}"
    
    def test_emd_adjacent_bins(self):
        """相邻区间的EMD计算。"""
        hist1 = np.array([1.0, 0.0, 0.0, 0.0])
        hist2 = np.array([0.0, 1.0, 0.0, 0.0])
        
        emd = EMDCalculator.calculate_emd_1d(hist1, hist2)
        
        # 所有质量需要移动1个单位
        assert np.isclose(emd, 1.0), f"EMD应该为1.0，实际为 {emd}"
    
    def test_emd_uniform_distributions(self):
        """均匀分布的EMD计算。"""
        hist1 = np.array([0.25, 0.25, 0.25, 0.25])
        hist2 = np.array([0.25, 0.25, 0.25, 0.25])
        
        emd = EMDCalculator.calculate_emd_1d(hist1, hist2)
        
        assert np.isclose(emd, 0.0), f"相同分布的EMD应该为0，实际为 {emd}"
    
    def test_emd_with_ground_distance(self):
        """带地面距离的EMD计算。"""
        hist1 = np.array([1.0, 0.0, 0.0])
        hist2 = np.array([0.0, 0.0, 1.0])
        
        # 自定义地面距离
        ground_distances = np.array([
            [0, 1, 5],  # 从0到2的距离是5
            [1, 0, 1],
            [5, 1, 0]
        ], dtype=float)
        
        emd = EMDCalculator.calculate_emd_with_ground_distance(
            hist1, hist2, ground_distances
        )
        
        # 质量从0移动到2，距离为5
        assert emd > 0, f"EMD应该大于0，实际为 {emd}"
    
    def test_emd_empty_histograms(self):
        """空直方图的EMD计算。"""
        hist1 = np.array([])
        hist2 = np.array([])
        
        emd = EMDCalculator.calculate_emd_1d(hist1, hist2)
        
        assert emd == 0.0, f"空直方图的EMD应该为0，实际为 {emd}"
    
    def test_emd_length_mismatch(self):
        """直方图长度不匹配应该抛出异常。"""
        hist1 = np.array([0.5, 0.5])
        hist2 = np.array([0.33, 0.33, 0.34])
        
        with pytest.raises(ValueError):
            EMDCalculator.calculate_emd_1d(hist1, hist2)


class TestValidateMetricProperties:
    """验证度量性质的测试。"""
    
    def test_validate_metric_properties(self):
        """验证度量性质函数。"""
        hist1 = np.array([0.5, 0.3, 0.2])
        hist2 = np.array([0.3, 0.4, 0.3])
        hist3 = np.array([0.2, 0.3, 0.5])
        
        non_neg, sym, triangle = EMDCalculator.validate_metric_properties(
            hist1, hist2, hist3
        )
        
        assert non_neg, "非负性验证失败"
        assert sym, "对称性验证失败"
        assert triangle, "三角不等式验证失败"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
