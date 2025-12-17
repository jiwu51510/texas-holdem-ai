"""CFR 采样器属性测试模块。

使用Hypothesis库进行属性测试，验证多次采样平均降低方差的正确性属性。
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from training.cfr_sampler import (
    SamplerConfig, 
    MultiSampleAverager,
    CFRSampler
)


# ============================================================================
# 测试策略（生成器）
# ============================================================================

# 生成有效的采样次数
valid_num_samples = st.integers(min_value=2, max_value=50)

# 生成随机种子（用于可重复的随机采样函数）
random_seeds = st.integers(min_value=0, max_value=2**31 - 1)

# 生成正态分布参数
normal_mean = st.floats(min_value=-100.0, max_value=100.0, 
                        allow_nan=False, allow_infinity=False)
normal_std = st.floats(min_value=0.1, max_value=50.0,
                       allow_nan=False, allow_infinity=False)


# ============================================================================
# Property 5: 多次采样降低方差
# **Feature: strategy-convergence-control, Property 5: 多次采样降低方差**
# **验证需求: 2.2**
# ============================================================================

class TestProperty5MultiSampleVarianceReduction:
    """属性测试：多次采样降低方差。
    
    对于任何采样函数和采样次数n，n次采样平均值的方差应该约等于单次采样方差除以n。
    """
    
    @given(valid_num_samples, normal_mean, normal_std, random_seeds)
    @settings(max_examples=100)
    def test_variance_reduction_ratio(self, num_samples, mean, std, seed):
        """
        **Feature: strategy-convergence-control, Property 5: 多次采样降低方差**
        **验证需求: 2.2**
        
        测试多次采样平均的方差降低比率接近理论值1/n。
        """
        # 设置随机种子以确保可重复性
        np.random.seed(seed)
        
        # 创建一个返回正态分布随机数的采样函数
        def sample_func():
            return np.random.normal(mean, std)
        
        averager = MultiSampleAverager()
        
        # 使用足够多的试验次数来获得稳定的方差估计
        num_trials = 500
        
        # 收集单次采样的样本
        single_samples = [sample_func() for _ in range(num_trials)]
        single_variance = np.var(single_samples)
        
        # 跳过方差太小的情况（可能导致数值不稳定）
        assume(single_variance > 0.01)
        
        # 收集多次采样平均的样本
        averaged_samples = [
            averager.sample_with_averaging(sample_func, num_samples)
            for _ in range(num_trials)
        ]
        averaged_variance = np.var(averaged_samples)
        
        # 理论上，n次采样平均的方差应该是单次采样方差的1/n
        expected_variance = single_variance / num_samples
        
        # 由于是统计估计，允许一定的误差范围（使用相对误差）
        # 对于500次试验，允许50%的相对误差
        relative_error = abs(averaged_variance - expected_variance) / expected_variance
        
        assert relative_error < 0.5, (
            f"方差降低比率不符合预期。"
            f"单次方差: {single_variance:.4f}, "
            f"平均后方差: {averaged_variance:.4f}, "
            f"期望方差: {expected_variance:.4f}, "
            f"相对误差: {relative_error:.2%}"
        )
    
    @given(valid_num_samples, random_seeds)
    @settings(max_examples=100)
    def test_averaged_variance_less_than_single(self, num_samples, seed):
        """
        **Feature: strategy-convergence-control, Property 5: 多次采样降低方差**
        **验证需求: 2.2**
        
        测试多次采样平均的方差小于单次采样方差。
        """
        np.random.seed(seed)
        
        # 使用标准正态分布
        def sample_func():
            return np.random.normal(0, 1)
        
        averager = MultiSampleAverager()
        
        # 使用estimate_variance_reduction方法
        single_var, avg_var, ratio = averager.estimate_variance_reduction(
            sample_func, num_samples, num_trials=200
        )
        
        # 跳过方差太小的情况
        assume(single_var > 0.01)
        
        # 多次采样平均的方差应该小于单次采样方差
        assert avg_var < single_var, (
            f"多次采样平均的方差应小于单次采样方差。"
            f"单次方差: {single_var:.4f}, 平均后方差: {avg_var:.4f}"
        )
    
    @given(valid_num_samples, random_seeds)
    @settings(max_examples=100)
    def test_variance_ratio_bounded(self, num_samples, seed):
        """
        **Feature: strategy-convergence-control, Property 5: 多次采样降低方差**
        **验证需求: 2.2**
        
        测试方差比率在合理范围内（接近1/n）。
        """
        np.random.seed(seed)
        
        def sample_func():
            return np.random.normal(0, 10)
        
        averager = MultiSampleAverager()
        
        single_var, avg_var, ratio = averager.estimate_variance_reduction(
            sample_func, num_samples, num_trials=200
        )
        
        # 跳过方差太小的情况
        assume(single_var > 0.1)
        
        # 理论比率是1/n，允许较大的统计误差范围
        expected_ratio = 1.0 / num_samples
        
        # 比率应该在理论值的0.2到2.0倍之间（考虑统计波动）
        assert 0.2 * expected_ratio < ratio < 2.0 * expected_ratio, (
            f"方差比率超出预期范围。"
            f"实际比率: {ratio:.4f}, 期望比率: {expected_ratio:.4f}"
        )
    
    @given(st.integers(min_value=1, max_value=10), random_seeds)
    @settings(max_examples=50)
    def test_more_samples_lower_variance(self, base_samples, seed):
        """
        **Feature: strategy-convergence-control, Property 5: 多次采样降低方差**
        **验证需求: 2.2**
        
        测试更多的采样次数导致更低的方差。
        """
        np.random.seed(seed)
        
        def sample_func():
            return np.random.normal(0, 5)
        
        averager = MultiSampleAverager()
        
        # 比较两个不同的采样次数
        n1 = base_samples
        n2 = base_samples * 2
        
        _, var1, _ = averager.estimate_variance_reduction(
            sample_func, n1, num_trials=300
        )
        _, var2, _ = averager.estimate_variance_reduction(
            sample_func, n2, num_trials=300
        )
        
        # 跳过方差太小的情况
        assume(var1 > 0.01)
        
        # 更多采样次数应该导致更低的方差（允许一些统计波动）
        # 理论上var2应该是var1的一半，但由于统计波动，我们只要求var2 < var1 * 1.2
        assert var2 < var1 * 1.2, (
            f"更多采样次数应导致更低方差。"
            f"n={n1}时方差: {var1:.4f}, n={n2}时方差: {var2:.4f}"
        )


# ============================================================================
# 配置验证测试
# ============================================================================

class TestSamplerConfigValidation:
    """采样器配置验证测试。"""
    
    def test_invalid_num_samples_raises_error(self):
        """测试无效采样次数抛出错误。"""
        with pytest.raises(ValueError, match="采样次数必须至少为1"):
            SamplerConfig(num_samples=0)
        
        with pytest.raises(ValueError, match="采样次数必须至少为1"):
            SamplerConfig(num_samples=-1)
    
    def test_valid_config_creation(self):
        """测试有效配置创建成功。"""
        config = SamplerConfig(num_samples=10, use_averaging=True)
        assert config.num_samples == 10
        assert config.use_averaging is True
    
    def test_default_config_values(self):
        """测试默认配置值。"""
        config = SamplerConfig()
        assert config.num_samples == 1
        assert config.use_averaging is False


# ============================================================================
# MultiSampleAverager 功能测试
# ============================================================================

class TestMultiSampleAveragerFunctionality:
    """多次采样平均器功能测试。"""
    
    def test_single_sample_returns_direct_value(self):
        """测试单次采样直接返回值。"""
        averager = MultiSampleAverager()
        
        # 使用确定性函数
        counter = [0]
        def sample_func():
            counter[0] += 1
            return 42.0
        
        result = averager.sample_with_averaging(sample_func, num_samples=1)
        
        assert result == 42.0
        assert counter[0] == 1  # 只调用一次
    
    def test_multiple_samples_returns_average(self):
        """测试多次采样返回平均值。"""
        averager = MultiSampleAverager()
        
        # 使用返回固定序列的函数
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        index = [0]
        
        def sample_func():
            val = values[index[0] % len(values)]
            index[0] += 1
            return val
        
        result = averager.sample_with_averaging(sample_func, num_samples=5)
        
        expected = np.mean(values)
        assert abs(result - expected) < 1e-10
    
    def test_invalid_num_samples_raises_error(self):
        """测试无效采样次数抛出错误。"""
        averager = MultiSampleAverager()
        
        with pytest.raises(ValueError, match="采样次数必须为正整数"):
            averager.sample_with_averaging(lambda: 1.0, num_samples=0)
        
        with pytest.raises(ValueError, match="采样次数必须为正整数"):
            averager.sample_with_averaging(lambda: 1.0, num_samples=-1)
    
    def test_uses_config_default_num_samples(self):
        """测试使用配置的默认采样次数。"""
        config = SamplerConfig(num_samples=3)
        averager = MultiSampleAverager(config)
        
        call_count = [0]
        def sample_func():
            call_count[0] += 1
            return 1.0
        
        averager.sample_with_averaging(sample_func)
        
        assert call_count[0] == 3
    
    @given(st.lists(st.floats(min_value=-100, max_value=100, 
                              allow_nan=False, allow_infinity=False),
                    min_size=2, max_size=10))
    @settings(max_examples=50)
    def test_array_averaging_correctness(self, values):
        """测试数组平均的正确性。"""
        averager = MultiSampleAverager()
        
        # 创建返回固定数组的函数
        arrays = [np.array([v, v * 2]) for v in values]
        index = [0]
        
        def sample_func():
            arr = arrays[index[0] % len(arrays)]
            index[0] += 1
            return arr
        
        result = averager.sample_array_with_averaging(
            sample_func, num_samples=len(values)
        )
        
        expected = np.mean(arrays, axis=0)
        np.testing.assert_array_almost_equal(result, expected)


# ============================================================================
# 边缘情况测试
# ============================================================================

class TestEdgeCases:
    """边缘情况测试。"""
    
    def test_constant_sample_function(self):
        """测试常数采样函数。"""
        averager = MultiSampleAverager()
        
        result = averager.sample_with_averaging(lambda: 5.0, num_samples=10)
        
        assert result == 5.0
    
    def test_zero_variance_sample_function(self):
        """测试零方差采样函数。"""
        averager = MultiSampleAverager()
        
        single_var, avg_var, ratio = averager.estimate_variance_reduction(
            lambda: 42.0, num_samples=5, num_trials=50
        )
        
        assert single_var == 0.0
        assert avg_var == 0.0
        assert ratio == 0.0
    
    def test_large_num_samples(self):
        """测试大量采样次数。"""
        averager = MultiSampleAverager()
        
        np.random.seed(42)
        
        def sample_func():
            return np.random.normal(100, 10)
        
        # 使用100次采样
        result = averager.sample_with_averaging(sample_func, num_samples=100)
        
        # 结果应该接近均值100
        assert 90 < result < 110
