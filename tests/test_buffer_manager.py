"""缓冲区管理器属性测试模块。

使用Hypothesis库进行属性测试，验证缓冲区管理器的核心正确性属性。
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from collections import Counter

from training.buffer_manager import BufferManager, BufferManagerConfig
from training.reservoir_buffer import ReservoirBuffer


# ============================================================================
# 测试策略（生成器）
# ============================================================================

# 生成有效的时间衰减因子
valid_decay_factors = st.floats(
    min_value=0.5, max_value=0.999,  # 使用较大的衰减因子以确保效果明显
    allow_nan=False, allow_infinity=False
)

# 生成有效的迭代次数
valid_iterations = st.integers(min_value=1, max_value=10000)

# 生成有效的批次大小
valid_batch_sizes = st.integers(min_value=1, max_value=100)


@st.composite
def buffer_with_samples_strategy(draw, min_samples=10, max_samples=100):
    """生成包含样本的缓冲区。
    
    生成一个包含不同迭代次数样本的缓冲区，用于测试时间衰减采样。
    """
    num_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    capacity = num_samples + 10  # 确保容量足够
    
    buffer = ReservoirBuffer(capacity)
    
    # 生成样本，迭代次数从1到num_samples
    state_dim = 10
    target_dim = 4
    
    for i in range(num_samples):
        state = np.random.randn(state_dim).astype(np.float32)
        target = np.random.randn(target_dim).astype(np.float32)
        iteration = i + 1  # 迭代次数从1开始
        buffer.add(state, target, iteration)
    
    return buffer


@st.composite
def buffer_with_age_groups_strategy(draw):
    """生成包含明确年龄分组的缓冲区。
    
    创建两组样本：近期样本和旧样本，用于验证时间衰减采样的偏好。
    """
    # 每组样本数量
    samples_per_group = draw(st.integers(min_value=20, max_value=50))
    
    # 当前迭代次数
    current_iteration = 1000
    
    # 旧样本的迭代次数（年龄大）
    old_iteration = 100
    
    # 近期样本的迭代次数（年龄小）
    recent_iteration = 990
    
    capacity = samples_per_group * 2 + 10
    buffer = ReservoirBuffer(capacity)
    
    state_dim = 10
    target_dim = 4
    
    # 添加旧样本
    for _ in range(samples_per_group):
        state = np.random.randn(state_dim).astype(np.float32)
        target = np.random.randn(target_dim).astype(np.float32)
        buffer.add(state, target, old_iteration)
    
    # 添加近期样本
    for _ in range(samples_per_group):
        state = np.random.randn(state_dim).astype(np.float32)
        target = np.random.randn(target_dim).astype(np.float32)
        buffer.add(state, target, recent_iteration)
    
    return buffer, current_iteration, old_iteration, recent_iteration, samples_per_group


# ============================================================================
# Property 11: 时间衰减采样偏好近期样本
# **Feature: strategy-convergence-control, Property 11: 时间衰减采样偏好近期样本**
# **验证需求: 5.1**
# ============================================================================

class TestProperty11TimeDecaySampling:
    """属性测试：时间衰减采样偏好近期样本。
    
    对于任何缓冲区和时间衰减因子，近期样本的采样概率应该高于旧样本。
    """
    
    @given(buffer_with_age_groups_strategy(), valid_decay_factors)
    @settings(max_examples=100)
    def test_recent_samples_sampled_more_frequently(
        self, 
        buffer_data, 
        decay_factor
    ):
        """
        **Feature: strategy-convergence-control, Property 11: 时间衰减采样偏好近期样本**
        **验证需求: 5.1**
        
        测试近期样本被采样的频率高于旧样本。
        """
        buffer, current_iteration, old_iteration, recent_iteration, samples_per_group = buffer_data
        
        config = BufferManagerConfig(time_decay_factor=decay_factor)
        manager = BufferManager(config)
        
        # 多次采样并统计
        num_trials = 100
        batch_size = samples_per_group  # 采样一半的样本
        
        recent_count = 0
        old_count = 0
        
        for _ in range(num_trials):
            _, _, iterations = manager.sample_with_time_decay(
                buffer, batch_size, current_iteration
            )
            
            for it in iterations:
                if it == recent_iteration:
                    recent_count += 1
                elif it == old_iteration:
                    old_count += 1
        
        # 近期样本应该被采样更多
        # 由于衰减因子的作用，近期样本的权重更高
        # 计算期望的比例
        recent_age = current_iteration - recent_iteration  # 10
        old_age = current_iteration - old_iteration  # 900
        
        expected_ratio = (decay_factor ** recent_age) / (decay_factor ** old_age)
        
        # 如果期望比例大于1，近期样本应该更多
        if expected_ratio > 1.5:  # 只在期望差异明显时检查
            assert recent_count > old_count, (
                f"近期样本应被采样更多，但近期: {recent_count}, 旧: {old_count}, "
                f"期望比例: {expected_ratio:.2f}"
            )
    
    @given(buffer_with_samples_strategy(min_samples=50, max_samples=100))
    @settings(max_examples=100)
    def test_time_decay_weights_decrease_with_age(self, buffer):
        """
        **Feature: strategy-convergence-control, Property 11: 时间衰减采样偏好近期样本**
        **验证需求: 5.1**
        
        测试时间衰减权重随年龄增加而减小。
        """
        config = BufferManagerConfig(time_decay_factor=0.99)
        manager = BufferManager(config)
        
        current_iteration = len(buffer) + 100
        
        weights = manager._compute_time_decay_weights(buffer, current_iteration)
        
        # 获取每个样本的迭代次数
        iterations = [sample[2] for sample in buffer.buffer]
        
        # 验证：迭代次数越大（越近期），权重越高
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                if iterations[i] > iterations[j]:
                    # i 更近期，权重应该更高
                    assert weights[i] >= weights[j], (
                        f"近期样本权重应更高: "
                        f"iteration[{i}]={iterations[i]}, weight={weights[i]}, "
                        f"iteration[{j}]={iterations[j]}, weight={weights[j]}"
                    )
                elif iterations[i] < iterations[j]:
                    # j 更近期，权重应该更高
                    assert weights[j] >= weights[i], (
                        f"近期样本权重应更高: "
                        f"iteration[{i}]={iterations[i]}, weight={weights[i]}, "
                        f"iteration[{j}]={iterations[j]}, weight={weights[j]}"
                    )
    
    @given(valid_decay_factors, valid_iterations)
    @settings(max_examples=100)
    def test_weight_formula_correctness(self, decay_factor, current_iteration):
        """
        **Feature: strategy-convergence-control, Property 11: 时间衰减采样偏好近期样本**
        **验证需求: 5.1**
        
        测试权重计算公式的正确性：weight = decay_factor^age
        """
        config = BufferManagerConfig(time_decay_factor=decay_factor)
        manager = BufferManager(config)
        
        # 创建一个简单的缓冲区
        buffer = ReservoirBuffer(10)
        
        # 添加几个不同迭代次数的样本
        sample_iterations = [1, current_iteration // 2, current_iteration - 1]
        
        for it in sample_iterations:
            if it > 0:
                state = np.random.randn(5).astype(np.float32)
                target = np.random.randn(3).astype(np.float32)
                buffer.add(state, target, it)
        
        if len(buffer) == 0:
            return  # 跳过空缓冲区
        
        weights = manager._compute_time_decay_weights(buffer, current_iteration)
        
        # 验证每个权重
        for i, (_, _, iteration) in enumerate(buffer.buffer):
            age = current_iteration - iteration
            expected_weight = decay_factor ** age
            
            # 考虑最小权重限制
            expected_weight = max(expected_weight, 1e-10)
            
            np.testing.assert_almost_equal(
                weights[i], expected_weight, decimal=10,
                err_msg=f"权重计算错误: age={age}, expected={expected_weight}, got={weights[i]}"
            )
    
    @given(buffer_with_samples_strategy(min_samples=20, max_samples=50), valid_batch_sizes)
    @settings(max_examples=100)
    def test_sample_returns_correct_batch_size(self, buffer, batch_size):
        """
        **Feature: strategy-convergence-control, Property 11: 时间衰减采样偏好近期样本**
        **验证需求: 5.1**
        
        测试采样返回正确的批次大小。
        """
        manager = BufferManager()
        current_iteration = len(buffer) + 100
        
        states, targets, iterations = manager.sample_with_time_decay(
            buffer, batch_size, current_iteration
        )
        
        expected_size = min(batch_size, len(buffer))
        
        assert len(states) == expected_size, (
            f"状态数量应为 {expected_size}，但得到 {len(states)}"
        )
        assert len(targets) == expected_size, (
            f"目标数量应为 {expected_size}，但得到 {len(targets)}"
        )
        assert len(iterations) == expected_size, (
            f"迭代次数数量应为 {expected_size}，但得到 {len(iterations)}"
        )
    
    def test_empty_buffer_returns_empty_arrays(self):
        """
        **Feature: strategy-convergence-control, Property 11: 时间衰减采样偏好近期样本**
        **验证需求: 5.1**
        
        测试空缓冲区返回空数组。
        """
        manager = BufferManager()
        buffer = ReservoirBuffer(100)
        
        states, targets, iterations = manager.sample_with_time_decay(
            buffer, 10, 100
        )
        
        assert len(states) == 0
        assert len(targets) == 0
        assert len(iterations) == 0


# ============================================================================
# 配置验证测试
# ============================================================================

class TestConfigValidation:
    """配置验证测试。"""
    
    def test_invalid_decay_factor_raises_error(self):
        """测试无效衰减因子抛出错误。"""
        with pytest.raises(ValueError, match="时间衰减因子必须在0-1之间"):
            BufferManagerConfig(time_decay_factor=1.5)
        
        with pytest.raises(ValueError, match="时间衰减因子必须在0-1之间"):
            BufferManagerConfig(time_decay_factor=-0.1)
    
    def test_invalid_importance_threshold_raises_error(self):
        """测试无效重要性阈值抛出错误。"""
        with pytest.raises(ValueError, match="重要性阈值必须非负"):
            BufferManagerConfig(importance_threshold=-0.1)
    
    def test_invalid_max_sample_age_raises_error(self):
        """测试无效最大样本年龄抛出错误。"""
        with pytest.raises(ValueError, match="最大样本年龄必须为正整数"):
            BufferManagerConfig(max_sample_age=0)
        
        with pytest.raises(ValueError, match="最大样本年龄必须为正整数"):
            BufferManagerConfig(max_sample_age=-100)
    
    def test_valid_config_creation(self):
        """测试有效配置创建成功。"""
        config = BufferManagerConfig(
            time_decay_factor=0.95,
            importance_threshold=0.5,
            max_sample_age=5000,
            stratified_sampling=True
        )
        assert config.time_decay_factor == 0.95
        assert config.importance_threshold == 0.5
        assert config.max_sample_age == 5000
        assert config.stratified_sampling is True


# ============================================================================
# 清理过旧样本测试
# ============================================================================

class TestCleanupOldSamples:
    """清理过旧样本测试。"""
    
    def test_cleanup_removes_old_samples(self):
        """测试清理移除过旧样本。"""
        manager = BufferManager(BufferManagerConfig(max_sample_age=100))
        buffer = ReservoirBuffer(100)
        
        # 添加不同年龄的样本
        for i in range(50):
            state = np.random.randn(5).astype(np.float32)
            target = np.random.randn(3).astype(np.float32)
            buffer.add(state, target, i * 10)  # 迭代次数: 0, 10, 20, ..., 490
        
        current_iteration = 500
        removed = manager.cleanup_old_samples(buffer, current_iteration)
        
        # 应该移除年龄超过100的样本（迭代次数 < 400）
        assert removed > 0, "应该移除一些旧样本"
        
        # 验证剩余样本的年龄都在阈值内
        for _, _, iteration in buffer.buffer:
            age = current_iteration - iteration
            assert age <= 100, f"样本年龄 {age} 超过阈值 100"
    
    def test_cleanup_preserves_recent_samples(self):
        """测试清理保留近期样本。"""
        manager = BufferManager(BufferManagerConfig(max_sample_age=100))
        buffer = ReservoirBuffer(100)
        
        current_iteration = 500
        
        # 只添加近期样本
        for i in range(20):
            state = np.random.randn(5).astype(np.float32)
            target = np.random.randn(3).astype(np.float32)
            buffer.add(state, target, current_iteration - 50 + i)
        
        original_size = len(buffer)
        removed = manager.cleanup_old_samples(buffer, current_iteration)
        
        assert removed == 0, "不应移除任何近期样本"
        assert len(buffer) == original_size, "缓冲区大小应保持不变"


# ============================================================================
# 缓冲区统计测试
# ============================================================================

class TestBufferStatistics:
    """缓冲区统计测试。"""
    
    def test_statistics_for_empty_buffer(self):
        """测试空缓冲区的统计信息。"""
        manager = BufferManager()
        buffer = ReservoirBuffer(100)
        
        stats = manager.get_buffer_statistics(buffer, 100)
        
        assert stats["size"] == 0
        assert stats["capacity"] == 100
        assert stats["fill_ratio"] == 0.0
        assert stats["avg_age"] == 0.0
    
    def test_statistics_for_filled_buffer(self):
        """测试填充缓冲区的统计信息。"""
        manager = BufferManager()
        buffer = ReservoirBuffer(100)
        
        # 添加样本
        for i in range(50):
            state = np.random.randn(5).astype(np.float32)
            target = np.random.randn(3).astype(np.float32)
            buffer.add(state, target, i + 1)
        
        current_iteration = 100
        stats = manager.get_buffer_statistics(buffer, current_iteration)
        
        assert stats["size"] == 50
        assert stats["capacity"] == 100
        assert stats["fill_ratio"] == 0.5
        assert stats["min_age"] >= 0
        assert stats["max_age"] <= current_iteration
        assert stats["avg_age"] > 0
