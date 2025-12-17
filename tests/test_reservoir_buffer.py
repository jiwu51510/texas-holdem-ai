"""蓄水池缓冲区（ReservoirBuffer）属性测试模块。

使用Hypothesis库进行属性测试，验证蓄水池缓冲区的核心正确性属性。
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from training.reservoir_buffer import ReservoirBuffer


# ============================================================================
# 测试策略（生成器）
# ============================================================================

# 有效的缓冲区容量
valid_capacity = st.integers(min_value=1, max_value=10000)

# 状态编码维度
state_dim = 370

# 目标值维度（遗憾值或策略概率）
target_dim = 4

# 生成随机状态编码
state_strategy = st.lists(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=state_dim,
    max_size=state_dim
).map(np.array)

# 生成随机目标值
target_strategy = st.lists(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=target_dim,
    max_size=target_dim
).map(np.array)

# 生成随机迭代编号
iteration_strategy = st.integers(min_value=0, max_value=1000000)


# 生成随机样本
@st.composite
def sample_strategy(draw):
    """生成一个随机样本（状态、目标、迭代编号）。"""
    state = draw(state_strategy)
    target = draw(target_strategy)
    iteration = draw(iteration_strategy)
    return state, target, iteration


# 生成多个随机样本
@st.composite
def samples_strategy(draw, min_samples=1, max_samples=100):
    """生成多个随机样本。"""
    num_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    samples = []
    for _ in range(num_samples):
        sample = draw(sample_strategy())
        samples.append(sample)
    return samples


# ============================================================================
# Property 5: 缓冲区添加行为（未满）
# **Feature: deep-cfr-refactor, Property 5: 缓冲区添加行为（未满）**
# **验证需求：3.2**
# ============================================================================

class TestProperty5BufferAddBehaviorNotFull:
    """属性测试：缓冲区添加行为（未满）。
    
    对于任何未满的缓冲区，添加一个样本后，缓冲区大小应该增加1。
    """
    
    @given(
        capacity=st.integers(min_value=2, max_value=1000),
        num_initial_samples=st.integers(min_value=0, max_value=100),
        new_sample=sample_strategy()
    )
    @settings(max_examples=100)
    def test_add_to_not_full_buffer_increases_size(self, capacity, num_initial_samples, new_sample):
        """
        **Feature: deep-cfr-refactor, Property 5: 缓冲区添加行为（未满）**
        **验证需求：3.2**
        
        测试向未满缓冲区添加样本后大小增加1。
        """
        # 确保初始样本数量小于容量
        num_initial_samples = min(num_initial_samples, capacity - 1)
        
        buffer = ReservoirBuffer(capacity)
        
        # 添加初始样本
        for i in range(num_initial_samples):
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, i)
        
        # 确保缓冲区未满
        assume(len(buffer) < capacity)
        
        # 记录添加前的大小
        size_before = len(buffer)
        
        # 添加新样本
        state, target, iteration = new_sample
        buffer.add(state, target, iteration)
        
        # 验证大小增加1
        assert len(buffer) == size_before + 1, (
            f"添加样本后缓冲区大小应增加1，"
            f"添加前: {size_before}，添加后: {len(buffer)}"
        )
    
    @given(capacity=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=100)
    def test_add_to_empty_buffer(self, capacity):
        """
        **Feature: deep-cfr-refactor, Property 5: 缓冲区添加行为（未满）**
        **验证需求：3.2**
        
        测试向空缓冲区添加样本。
        """
        buffer = ReservoirBuffer(capacity)
        
        assert len(buffer) == 0, "新创建的缓冲区应为空"
        
        # 添加一个样本
        state = np.random.randn(state_dim)
        target = np.random.randn(target_dim)
        buffer.add(state, target, 0)
        
        assert len(buffer) == 1, "添加一个样本后缓冲区大小应为1"
    
    @given(capacity=st.integers(min_value=1, max_value=100))
    @settings(max_examples=100)
    def test_fill_buffer_to_capacity(self, capacity):
        """
        **Feature: deep-cfr-refactor, Property 5: 缓冲区添加行为（未满）**
        **验证需求：3.2**
        
        测试填充缓冲区到容量。
        """
        buffer = ReservoirBuffer(capacity)
        
        # 添加样本直到达到容量
        for i in range(capacity):
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, i)
            
            # 验证每次添加后大小正确
            assert len(buffer) == i + 1, (
                f"添加第{i+1}个样本后大小应为{i+1}，实际为{len(buffer)}"
            )
        
        # 验证最终大小等于容量
        assert len(buffer) == capacity, (
            f"填充后缓冲区大小应等于容量{capacity}，实际为{len(buffer)}"
        )


# ============================================================================
# Property 6: 缓冲区大小上限
# **Feature: deep-cfr-refactor, Property 6: 缓冲区大小上限**
# **验证需求：3.3**
# ============================================================================

class TestProperty6BufferSizeLimit:
    """属性测试：缓冲区大小上限。
    
    对于任何已满的缓冲区，添加样本后，缓冲区大小应该保持不变（等于最大容量）。
    """
    
    @given(
        capacity=st.integers(min_value=1, max_value=100),
        extra_samples=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100)
    def test_full_buffer_size_unchanged(self, capacity, extra_samples):
        """
        **Feature: deep-cfr-refactor, Property 6: 缓冲区大小上限**
        **验证需求：3.3**
        
        测试已满缓冲区添加样本后大小不变。
        """
        buffer = ReservoirBuffer(capacity)
        
        # 先填满缓冲区
        for i in range(capacity):
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, i)
        
        assert len(buffer) == capacity, "缓冲区应已满"
        
        # 继续添加更多样本
        for i in range(extra_samples):
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, capacity + i)
            
            # 验证大小保持不变
            assert len(buffer) == capacity, (
                f"已满缓冲区添加样本后大小应保持{capacity}，"
                f"添加第{i+1}个额外样本后大小为{len(buffer)}"
            )
    
    @given(capacity=st.integers(min_value=1, max_value=100))
    @settings(max_examples=100)
    def test_buffer_never_exceeds_capacity(self, capacity):
        """
        **Feature: deep-cfr-refactor, Property 6: 缓冲区大小上限**
        **验证需求：3.3**
        
        测试缓冲区大小永远不超过容量。
        """
        buffer = ReservoirBuffer(capacity)
        
        # 添加大量样本
        num_samples = capacity * 3
        for i in range(num_samples):
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, i)
            
            # 验证大小不超过容量
            assert len(buffer) <= capacity, (
                f"缓冲区大小不应超过容量{capacity}，"
                f"添加第{i+1}个样本后大小为{len(buffer)}"
            )
    
    @given(capacity=st.integers(min_value=1, max_value=50))
    @settings(max_examples=100)
    def test_is_full_property(self, capacity):
        """
        **Feature: deep-cfr-refactor, Property 6: 缓冲区大小上限**
        **验证需求：3.3**
        
        测试is_full方法的正确性。
        """
        buffer = ReservoirBuffer(capacity)
        
        # 添加样本直到满
        for i in range(capacity):
            assert not buffer.is_full(), f"添加{i}个样本后缓冲区不应满"
            
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, i)
        
        # 现在应该满了
        assert buffer.is_full(), "添加capacity个样本后缓冲区应满"
        
        # 继续添加，仍然满
        for i in range(10):
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, capacity + i)
            assert buffer.is_full(), "继续添加样本后缓冲区仍应满"


# ============================================================================
# Property 7: 缓冲区采样数量正确性
# **Feature: deep-cfr-refactor, Property 7: 缓冲区采样数量正确性**
# **验证需求：3.4**
# ============================================================================

class TestProperty7BufferSampleCount:
    """属性测试：缓冲区采样数量正确性。
    
    对于任何非空缓冲区和采样请求，返回的样本数量应该等于 min(请求数量, 缓冲区大小)。
    """
    
    @given(
        capacity=st.integers(min_value=1, max_value=100),
        num_samples=st.integers(min_value=1, max_value=100),
        batch_size=st.integers(min_value=1, max_value=200)
    )
    @settings(max_examples=100)
    def test_sample_count_correct(self, capacity, num_samples, batch_size):
        """
        **Feature: deep-cfr-refactor, Property 7: 缓冲区采样数量正确性**
        **验证需求：3.4**
        
        测试采样返回正确数量的样本。
        """
        buffer = ReservoirBuffer(capacity)
        
        # 添加样本
        actual_samples = min(num_samples, capacity)
        for i in range(num_samples):
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, i)
        
        # 采样
        states, targets, iterations = buffer.sample(batch_size)
        
        # 验证采样数量
        expected_count = min(batch_size, len(buffer))
        assert len(states) == expected_count, (
            f"采样数量应为min({batch_size}, {len(buffer)})={expected_count}，"
            f"实际为{len(states)}"
        )
        assert len(targets) == expected_count, (
            f"目标数量应为{expected_count}，实际为{len(targets)}"
        )
        assert len(iterations) == expected_count, (
            f"迭代编号数量应为{expected_count}，实际为{len(iterations)}"
        )
    
    @given(capacity=st.integers(min_value=1, max_value=100))
    @settings(max_examples=100)
    def test_sample_from_empty_buffer(self, capacity):
        """
        **Feature: deep-cfr-refactor, Property 7: 缓冲区采样数量正确性**
        **验证需求：3.4**
        
        测试从空缓冲区采样返回空数组。
        """
        buffer = ReservoirBuffer(capacity)
        
        states, targets, iterations = buffer.sample(10)
        
        assert len(states) == 0, "从空缓冲区采样应返回空数组"
        assert len(targets) == 0, "从空缓冲区采样应返回空数组"
        assert len(iterations) == 0, "从空缓冲区采样应返回空数组"
    
    @given(
        capacity=st.integers(min_value=10, max_value=100),
        num_samples=st.integers(min_value=5, max_value=50)
    )
    @settings(max_examples=100)
    def test_sample_less_than_buffer_size(self, capacity, num_samples):
        """
        **Feature: deep-cfr-refactor, Property 7: 缓冲区采样数量正确性**
        **验证需求：3.4**
        
        测试请求数量小于缓冲区大小时返回请求数量。
        """
        buffer = ReservoirBuffer(capacity)
        
        # 添加样本
        for i in range(num_samples):
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, i)
        
        # 请求少于缓冲区大小的样本
        batch_size = min(num_samples - 1, 1)
        states, targets, iterations = buffer.sample(batch_size)
        
        assert len(states) == batch_size, (
            f"请求{batch_size}个样本应返回{batch_size}个，实际返回{len(states)}个"
        )
    
    @given(
        capacity=st.integers(min_value=5, max_value=50),
        num_samples=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    def test_sample_more_than_buffer_size(self, capacity, num_samples):
        """
        **Feature: deep-cfr-refactor, Property 7: 缓冲区采样数量正确性**
        **验证需求：3.4**
        
        测试请求数量大于缓冲区大小时返回缓冲区大小。
        """
        buffer = ReservoirBuffer(capacity)
        
        # 添加样本
        actual_size = min(num_samples, capacity)
        for i in range(num_samples):
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, i)
        
        # 请求多于缓冲区大小的样本
        batch_size = len(buffer) + 10
        states, targets, iterations = buffer.sample(batch_size)
        
        assert len(states) == len(buffer), (
            f"请求{batch_size}个样本但缓冲区只有{len(buffer)}个，"
            f"应返回{len(buffer)}个，实际返回{len(states)}个"
        )


# ============================================================================
# 额外的单元测试
# ============================================================================

class TestReservoirBufferUnit:
    """ReservoirBuffer 单元测试。"""
    
    def test_invalid_capacity_raises_error(self):
        """测试无效容量抛出错误。"""
        with pytest.raises(ValueError):
            ReservoirBuffer(0)
        
        with pytest.raises(ValueError):
            ReservoirBuffer(-1)
    
    def test_clear_empties_buffer(self):
        """测试clear方法清空缓冲区。"""
        buffer = ReservoirBuffer(100)
        
        # 添加一些样本
        for i in range(50):
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, i)
        
        assert len(buffer) == 50
        
        # 清空
        buffer.clear()
        
        assert len(buffer) == 0
        assert buffer.get_total_seen() == 0
    
    def test_total_seen_tracks_all_samples(self):
        """测试total_seen跟踪所有见过的样本。"""
        buffer = ReservoirBuffer(10)
        
        # 添加100个样本
        for i in range(100):
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, i)
        
        # 缓冲区大小应为10
        assert len(buffer) == 10
        
        # 但total_seen应为100
        assert buffer.get_total_seen() == 100
    
    def test_sample_returns_correct_shapes(self):
        """测试采样返回正确的数组形状。"""
        buffer = ReservoirBuffer(100)
        
        # 添加样本
        for i in range(50):
            state = np.random.randn(state_dim)
            target = np.random.randn(target_dim)
            buffer.add(state, target, i)
        
        # 采样
        states, targets, iterations = buffer.sample(20)
        
        assert states.shape == (20, state_dim), f"状态形状应为(20, {state_dim})，实际为{states.shape}"
        assert targets.shape == (20, target_dim), f"目标形状应为(20, {target_dim})，实际为{targets.shape}"
        assert iterations.shape == (20,), f"迭代编号形状应为(20,)，实际为{iterations.shape}"
    
    def test_sample_data_integrity(self):
        """测试采样数据的完整性。"""
        buffer = ReservoirBuffer(100)
        
        # 添加已知样本
        known_states = []
        known_targets = []
        known_iterations = []
        
        for i in range(50):
            state = np.full(state_dim, float(i))
            target = np.full(target_dim, float(i * 2))
            buffer.add(state, target, i)
            known_states.append(state)
            known_targets.append(target)
            known_iterations.append(i)
        
        # 采样并验证数据来自已知样本
        states, targets, iterations = buffer.sample(20)
        
        for i in range(20):
            # 找到对应的已知样本
            iteration = iterations[i]
            assert 0 <= iteration < 50, f"迭代编号应在0-49范围内，实际为{iteration}"
            
            # 验证状态和目标与迭代编号匹配
            expected_state = np.full(state_dim, float(iteration))
            expected_target = np.full(target_dim, float(iteration * 2))
            
            np.testing.assert_array_almost_equal(
                states[i], expected_state,
                err_msg=f"状态数据不匹配，迭代编号{iteration}"
            )
            np.testing.assert_array_almost_equal(
                targets[i], expected_target,
                err_msg=f"目标数据不匹配，迭代编号{iteration}"
            )
