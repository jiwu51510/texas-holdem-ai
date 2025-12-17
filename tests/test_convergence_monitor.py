"""收敛监控器属性测试模块。

使用Hypothesis库进行属性测试，验证收敛监控器的核心正确性属性。
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from training.convergence_monitor import (
    ConvergenceMonitor, 
    ConvergenceMonitorConfig,
    ConvergenceMetrics
)


# ============================================================================
# 测试策略（生成器）
# ============================================================================

@st.composite
def probability_distribution_strategy(draw, min_size=2, max_size=10):
    """生成随机概率分布。
    
    生成和为1的非负概率分布。
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # 生成正数
    values = draw(st.lists(
        st.floats(min_value=0.01, max_value=10.0, 
                  allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size
    ))
    # 归一化为概率分布
    arr = np.array(values, dtype=np.float64)
    arr = arr / arr.sum()
    return arr


@st.composite
def uniform_distribution_strategy(draw, min_size=2, max_size=10):
    """生成均匀分布。"""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return np.ones(size) / size


@st.composite
def deterministic_distribution_strategy(draw, min_size=2, max_size=10):
    """生成确定性分布（只有一个元素为1，其余为0）。"""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    index = draw(st.integers(min_value=0, max_value=size-1))
    arr = np.zeros(size)
    arr[index] = 1.0
    return arr


@st.composite
def regret_array_strategy(draw, min_size=1, max_size=100):
    """生成随机遗憾值数组。"""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    values = draw(st.lists(
        st.floats(min_value=-100.0, max_value=100.0, 
                  allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size
    ))
    return np.array(values, dtype=np.float64)


@st.composite
def entropy_history_strategy(draw, min_size=3, max_size=50):
    """生成熵值历史列表。"""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    values = draw(st.lists(
        st.floats(min_value=0.0, max_value=5.0, 
                  allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size
    ))
    return values


# ============================================================================
# Property 10: 策略熵计算正确性
# **Feature: strategy-convergence-control, Property 10: 策略熵计算正确性**
# **验证需求: 4.1**
# ============================================================================

class TestProperty10EntropyCalculation:
    """属性测试：策略熵计算正确性。
    
    对于任何概率分布，熵值应该非负，且均匀分布的熵值最大。
    """
    
    @given(probability_distribution_strategy())
    @settings(max_examples=100)
    def test_entropy_non_negative(self, strategy):
        """
        **Feature: strategy-convergence-control, Property 10: 策略熵计算正确性**
        **验证需求: 4.1**
        
        测试熵值非负。
        """
        monitor = ConvergenceMonitor()
        entropy = monitor.compute_entropy(strategy)
        
        assert entropy >= -1e-10, (
            f"熵值应非负，但得到: {entropy}"
        )
    
    @given(uniform_distribution_strategy())
    @settings(max_examples=100)
    def test_uniform_distribution_has_maximum_entropy(self, uniform_dist):
        """
        **Feature: strategy-convergence-control, Property 10: 策略熵计算正确性**
        **验证需求: 4.1**
        
        测试均匀分布具有最大熵。
        """
        monitor = ConvergenceMonitor()
        
        # 计算均匀分布的熵
        uniform_entropy = monitor.compute_entropy(uniform_dist)
        
        # 理论最大熵 = log(n)
        n = len(uniform_dist)
        max_entropy = np.log(n)
        
        # 均匀分布的熵应该接近理论最大值
        assert abs(uniform_entropy - max_entropy) < 1e-6, (
            f"均匀分布熵应为log({n})={max_entropy:.4f}，但得到: {uniform_entropy:.4f}"
        )
    
    @given(deterministic_distribution_strategy())
    @settings(max_examples=100)
    def test_deterministic_distribution_has_zero_entropy(self, det_dist):
        """
        **Feature: strategy-convergence-control, Property 10: 策略熵计算正确性**
        **验证需求: 4.1**
        
        测试确定性分布的熵为0。
        """
        monitor = ConvergenceMonitor()
        entropy = monitor.compute_entropy(det_dist)
        
        # 确定性分布的熵应该接近0
        assert entropy < 1e-5, (
            f"确定性分布的熵应接近0，但得到: {entropy}"
        )
    
    @given(probability_distribution_strategy())
    @settings(max_examples=100)
    def test_entropy_bounded_by_log_n(self, strategy):
        """
        **Feature: strategy-convergence-control, Property 10: 策略熵计算正确性**
        **验证需求: 4.1**
        
        测试熵值不超过log(n)。
        """
        monitor = ConvergenceMonitor()
        entropy = monitor.compute_entropy(strategy)
        
        n = len(strategy)
        max_entropy = np.log(n)
        
        assert entropy <= max_entropy + 1e-6, (
            f"熵值应不超过log({n})={max_entropy:.4f}，但得到: {entropy:.4f}"
        )
    
    @given(probability_distribution_strategy())
    @settings(max_examples=100)
    def test_entropy_invariant_to_permutation(self, strategy):
        """
        **Feature: strategy-convergence-control, Property 10: 策略熵计算正确性**
        **验证需求: 4.1**
        
        测试熵值对排列不变。
        """
        monitor = ConvergenceMonitor()
        
        # 原始熵
        original_entropy = monitor.compute_entropy(strategy)
        
        # 随机排列后的熵
        permuted = np.random.permutation(strategy)
        permuted_entropy = monitor.compute_entropy(permuted)
        
        assert abs(original_entropy - permuted_entropy) < 1e-10, (
            f"熵值应对排列不变，原始: {original_entropy}, 排列后: {permuted_entropy}"
        )
    
    def test_entropy_empty_array_returns_zero(self):
        """
        **Feature: strategy-convergence-control, Property 10: 策略熵计算正确性**
        **验证需求: 4.1**
        
        测试空数组返回0。
        """
        monitor = ConvergenceMonitor()
        entropy = monitor.compute_entropy(np.array([]))
        
        assert entropy == 0.0, f"空数组的熵应为0，但得到: {entropy}"
    
    @given(probability_distribution_strategy(min_size=2, max_size=5))
    @settings(max_examples=100)
    def test_entropy_2d_array_averages_rows(self, strategy):
        """
        **Feature: strategy-convergence-control, Property 10: 策略熵计算正确性**
        **验证需求: 4.1**
        
        测试二维数组对每行计算熵后取平均。
        """
        monitor = ConvergenceMonitor()
        
        # 创建二维数组（多行相同的策略）
        strategy_2d = np.vstack([strategy, strategy, strategy])
        
        # 一维熵
        entropy_1d = monitor.compute_entropy(strategy)
        
        # 二维熵（应该等于一维熵，因为所有行相同）
        entropy_2d = monitor.compute_entropy(strategy_2d)
        
        assert abs(entropy_1d - entropy_2d) < 1e-10, (
            f"相同行的二维数组熵应等于一维熵，"
            f"一维: {entropy_1d}, 二维: {entropy_2d}"
        )


# ============================================================================
# 遗憾值统计测试
# ============================================================================

class TestRegretStats:
    """遗憾值统计测试。"""
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_regret_stats_mean_correct(self, regrets):
        """测试遗憾值均值计算正确。"""
        monitor = ConvergenceMonitor()
        stats = monitor.compute_regret_stats(regrets)
        
        expected_mean = np.mean(regrets)
        assert abs(stats['mean'] - expected_mean) < 1e-10, (
            f"均值计算不正确，期望: {expected_mean}, 得到: {stats['mean']}"
        )
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_regret_stats_std_non_negative(self, regrets):
        """测试遗憾值标准差非负。"""
        monitor = ConvergenceMonitor()
        stats = monitor.compute_regret_stats(regrets)
        
        assert stats['std'] >= 0, f"标准差应非负，但得到: {stats['std']}"
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_regret_stats_max_min_bounds(self, regrets):
        """测试遗憾值最大最小值边界。"""
        monitor = ConvergenceMonitor()
        stats = monitor.compute_regret_stats(regrets)
        
        assert stats['max'] >= stats['min'], (
            f"最大值应大于等于最小值，max: {stats['max']}, min: {stats['min']}"
        )
        assert stats['max'] >= stats['mean'], (
            f"最大值应大于等于均值，max: {stats['max']}, mean: {stats['mean']}"
        )
        assert stats['min'] <= stats['mean'], (
            f"最小值应小于等于均值，min: {stats['min']}, mean: {stats['mean']}"
        )
    
    def test_regret_stats_empty_array(self):
        """测试空数组返回零值。"""
        monitor = ConvergenceMonitor()
        stats = monitor.compute_regret_stats(np.array([]))
        
        assert stats['mean'] == 0.0
        assert stats['std'] == 0.0
        assert stats['max'] == 0.0
        assert stats['min'] == 0.0


# ============================================================================
# 震荡检测测试
# ============================================================================

class TestOscillationDetection:
    """震荡检测测试。"""
    
    def test_oscillation_detected_for_alternating_values(self):
        """测试交替值被检测为震荡。"""
        monitor = ConvergenceMonitor(ConvergenceMonitorConfig(
            oscillation_threshold=0.05
        ))
        
        # 创建明显的震荡模式
        oscillating_history = [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5]
        
        is_oscillating = monitor.detect_oscillation(oscillating_history)
        
        assert is_oscillating, "交替值应被检测为震荡"
    
    def test_no_oscillation_for_monotonic_values(self):
        """测试单调值不被检测为震荡。"""
        monitor = ConvergenceMonitor()
        
        # 创建单调递减的历史
        monotonic_history = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        is_oscillating = monitor.detect_oscillation(monotonic_history)
        
        assert not is_oscillating, "单调值不应被检测为震荡"
    
    def test_no_oscillation_for_constant_values(self):
        """测试常数值不被检测为震荡。"""
        monitor = ConvergenceMonitor()
        
        # 创建常数历史
        constant_history = [0.5] * 10
        
        is_oscillating = monitor.detect_oscillation(constant_history)
        
        assert not is_oscillating, "常数值不应被检测为震荡"
    
    def test_oscillation_requires_minimum_history(self):
        """测试震荡检测需要最少历史数据。"""
        monitor = ConvergenceMonitor()
        
        # 历史太短
        short_history = [0.5, 1.5]
        
        is_oscillating = monitor.detect_oscillation(short_history)
        
        assert not is_oscillating, "历史太短时不应检测到震荡"
    
    @given(entropy_history_strategy(min_size=10, max_size=30))
    @settings(max_examples=100)
    def test_oscillation_detection_returns_boolean(self, history):
        """测试震荡检测返回布尔值。"""
        monitor = ConvergenceMonitor()
        result = monitor.detect_oscillation(history)
        
        assert isinstance(result, bool), f"应返回布尔值，但得到: {type(result)}"


# ============================================================================
# KL散度测试
# ============================================================================

class TestKLDivergence:
    """KL散度测试。"""
    
    @given(probability_distribution_strategy(), probability_distribution_strategy())
    @settings(max_examples=100)
    def test_kl_divergence_non_negative(self, p, q):
        """测试KL散度非负。"""
        # 确保两个分布大小相同
        min_size = min(len(p), len(q))
        p = p[:min_size]
        q = q[:min_size]
        
        # 重新归一化
        p = p / p.sum()
        q = q / q.sum()
        
        monitor = ConvergenceMonitor()
        kl = monitor.compute_kl_divergence(p, q)
        
        assert kl >= -1e-10, f"KL散度应非负，但得到: {kl}"
    
    @given(probability_distribution_strategy())
    @settings(max_examples=100)
    def test_kl_divergence_zero_for_identical(self, p):
        """测试相同分布的KL散度为0。"""
        monitor = ConvergenceMonitor()
        kl = monitor.compute_kl_divergence(p, p)
        
        assert abs(kl) < 1e-10, f"相同分布的KL散度应为0，但得到: {kl}"
    
    def test_kl_divergence_empty_arrays(self):
        """测试空数组返回0。"""
        monitor = ConvergenceMonitor()
        kl = monitor.compute_kl_divergence(np.array([]), np.array([]))
        
        assert kl == 0.0, f"空数组的KL散度应为0，但得到: {kl}"
    
    def test_kl_divergence_shape_mismatch_raises_error(self):
        """测试形状不匹配抛出错误。"""
        monitor = ConvergenceMonitor()
        p = np.array([0.5, 0.5])
        q = np.array([0.33, 0.33, 0.34])
        
        with pytest.raises(ValueError, match="分布形状不匹配"):
            monitor.compute_kl_divergence(p, q)


# ============================================================================
# 收敛报告测试
# ============================================================================

class TestConvergenceReport:
    """收敛报告测试。"""
    
    def test_report_contains_required_fields(self):
        """测试报告包含必需字段。"""
        monitor = ConvergenceMonitor()
        report = monitor.get_convergence_report()
        
        assert 'current_iteration' in report
        assert 'total_metrics_recorded' in report
        assert 'entropy_history_size' in report
        assert 'config' in report
        assert 'is_oscillating' in report
    
    def test_report_after_updates(self):
        """测试更新后的报告。"""
        config = ConvergenceMonitorConfig(monitor_interval=1)
        monitor = ConvergenceMonitor(config)
        
        # 进行一些更新
        strategy = np.array([0.5, 0.3, 0.2])
        regrets = np.array([1.0, -0.5, 0.3])
        
        monitor.update(1, strategy, regrets)
        
        report = monitor.get_convergence_report()
        
        assert report['current_iteration'] == 1
        assert report['total_metrics_recorded'] == 1
        assert 'latest_metrics' in report
        assert 'entropy_stats' in report
    
    def test_reset_clears_state(self):
        """测试重置清除状态。"""
        config = ConvergenceMonitorConfig(monitor_interval=1)
        monitor = ConvergenceMonitor(config)
        
        # 进行一些更新
        strategy = np.array([0.5, 0.3, 0.2])
        monitor.update(1, strategy)
        
        # 重置
        monitor.reset()
        
        report = monitor.get_convergence_report()
        
        assert report['current_iteration'] == 0
        assert report['total_metrics_recorded'] == 0
        assert report['entropy_history_size'] == 0


# ============================================================================
# 配置验证测试
# ============================================================================

class TestConfigValidation:
    """配置验证测试。"""
    
    def test_invalid_entropy_window_raises_error(self):
        """测试无效熵窗口抛出错误。"""
        with pytest.raises(ValueError, match="entropy_window必须为正整数"):
            ConvergenceMonitorConfig(entropy_window=0)
        
        with pytest.raises(ValueError, match="entropy_window必须为正整数"):
            ConvergenceMonitorConfig(entropy_window=-1)
    
    def test_invalid_oscillation_threshold_raises_error(self):
        """测试无效震荡阈值抛出错误。"""
        with pytest.raises(ValueError, match="oscillation_threshold必须为正数"):
            ConvergenceMonitorConfig(oscillation_threshold=0)
        
        with pytest.raises(ValueError, match="oscillation_threshold必须为正数"):
            ConvergenceMonitorConfig(oscillation_threshold=-0.1)
    
    def test_invalid_kl_warning_threshold_raises_error(self):
        """测试无效KL警告阈值抛出错误。"""
        with pytest.raises(ValueError, match="kl_warning_threshold必须为正数"):
            ConvergenceMonitorConfig(kl_warning_threshold=0)
        
        with pytest.raises(ValueError, match="kl_warning_threshold必须为正数"):
            ConvergenceMonitorConfig(kl_warning_threshold=-0.1)
    
    def test_invalid_monitor_interval_raises_error(self):
        """测试无效监控间隔抛出错误。"""
        with pytest.raises(ValueError, match="monitor_interval必须为正整数"):
            ConvergenceMonitorConfig(monitor_interval=0)
        
        with pytest.raises(ValueError, match="monitor_interval必须为正整数"):
            ConvergenceMonitorConfig(monitor_interval=-1)
    
    def test_valid_config_creation(self):
        """测试有效配置创建成功。"""
        config = ConvergenceMonitorConfig(
            entropy_window=50,
            oscillation_threshold=0.2,
            kl_warning_threshold=1.0,
            monitor_interval=500
        )
        assert config.entropy_window == 50
        assert config.oscillation_threshold == 0.2
        assert config.kl_warning_threshold == 1.0
        assert config.monitor_interval == 500
