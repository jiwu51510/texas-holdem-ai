"""遗憾值处理器属性测试模块。

使用Hypothesis库进行属性测试，验证遗憾值处理器的核心正确性属性。
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from training.regret_processor import RegretProcessor, RegretProcessorConfig


# ============================================================================
# 测试策略（生成器）
# ============================================================================

# 生成有效的遗憾值数组
@st.composite
def regret_array_strategy(draw, min_size=1, max_size=100):
    """生成随机遗憾值数组。
    
    生成包含正负值的浮点数数组，模拟真实的遗憾值分布。
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # 生成范围在[-1000, 1000]的浮点数
    values = draw(st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, 
                  allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size
    ))
    return np.array(values, dtype=np.float64)


# 生成有效的衰减因子
valid_decay_factors = st.floats(min_value=0.0, max_value=1.0, 
                                 allow_nan=False, allow_infinity=False)

# 生成有效的裁剪阈值
valid_clip_thresholds = st.floats(min_value=0.01, max_value=10000.0,
                                   allow_nan=False, allow_infinity=False)


# ============================================================================
# Property 1: 正遗憾值截断保证非负
# **Feature: strategy-convergence-control, Property 1: 正遗憾值截断保证非负**
# **验证需求: 1.1, 6.2**
# ============================================================================

class TestProperty1PositiveTruncation:
    """属性测试：正遗憾值截断保证非负。
    
    对于任何遗憾值数组，应用正遗憾值截断后，所有元素应该非负。
    """
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_truncate_positive_all_non_negative(self, regrets):
        """
        **Feature: strategy-convergence-control, Property 1: 正遗憾值截断保证非负**
        **验证需求: 1.1, 6.2**
        
        测试截断后所有值都非负。
        """
        processor = RegretProcessor()
        truncated = processor.truncate_positive(regrets)
        
        assert np.all(truncated >= 0), (
            f"截断后应所有值非负，但存在负值: {truncated[truncated < 0]}"
        )
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_truncate_positive_preserves_positive_values(self, regrets):
        """
        **Feature: strategy-convergence-control, Property 1: 正遗憾值截断保证非负**
        **验证需求: 1.1, 6.2**
        
        测试截断保留正值不变。
        """
        processor = RegretProcessor()
        truncated = processor.truncate_positive(regrets)
        
        # 原始正值应该保持不变
        positive_mask = regrets > 0
        if np.any(positive_mask):
            np.testing.assert_array_almost_equal(
                truncated[positive_mask], 
                regrets[positive_mask],
                err_msg="正值应保持不变"
            )
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_truncate_positive_zeros_negative_values(self, regrets):
        """
        **Feature: strategy-convergence-control, Property 1: 正遗憾值截断保证非负**
        **验证需求: 1.1, 6.2**
        
        测试截断将负值变为零。
        """
        processor = RegretProcessor()
        truncated = processor.truncate_positive(regrets)
        
        # 原始负值应该变为0
        negative_mask = regrets < 0
        if np.any(negative_mask):
            assert np.all(truncated[negative_mask] == 0), (
                f"负值应被截断为0，但得到: {truncated[negative_mask]}"
            )
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_truncate_positive_preserves_shape(self, regrets):
        """
        **Feature: strategy-convergence-control, Property 1: 正遗憾值截断保证非负**
        **验证需求: 1.1, 6.2**
        
        测试截断保持数组形状不变。
        """
        processor = RegretProcessor()
        truncated = processor.truncate_positive(regrets)
        
        assert truncated.shape == regrets.shape, (
            f"形状应保持不变，原始: {regrets.shape}，截断后: {truncated.shape}"
        )
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_truncate_positive_idempotent(self, regrets):
        """
        **Feature: strategy-convergence-control, Property 1: 正遗憾值截断保证非负**
        **验证需求: 1.1, 6.2**
        
        测试截断操作是幂等的（多次应用结果相同）。
        """
        processor = RegretProcessor()
        truncated_once = processor.truncate_positive(regrets)
        truncated_twice = processor.truncate_positive(truncated_once)
        
        np.testing.assert_array_equal(
            truncated_once, truncated_twice,
            err_msg="截断操作应是幂等的"
        )


# ============================================================================
# Property 2: 遗憾值衰减正确性
# **Feature: strategy-convergence-control, Property 2: 遗憾值衰减正确性**
# **验证需求: 1.2**
# ============================================================================

class TestProperty2DecayCorrectness:
    """属性测试：遗憾值衰减正确性。
    
    对于任何遗憾值数组和衰减因子，衰减后的值应该等于原值乘以衰减因子。
    """
    
    @given(regret_array_strategy(), valid_decay_factors)
    @settings(max_examples=100)
    def test_decay_equals_multiplication(self, regrets, decay_factor):
        """
        **Feature: strategy-convergence-control, Property 2: 遗憾值衰减正确性**
        **验证需求: 1.2**
        
        测试衰减等于乘以衰减因子。
        """
        processor = RegretProcessor()
        decayed = processor.apply_decay(regrets, decay_factor)
        
        expected = regrets * decay_factor
        np.testing.assert_array_almost_equal(
            decayed, expected,
            err_msg=f"衰减后应等于原值乘以{decay_factor}"
        )
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_decay_with_factor_one_preserves_values(self, regrets):
        """
        **Feature: strategy-convergence-control, Property 2: 遗憾值衰减正确性**
        **验证需求: 1.2**
        
        测试衰减因子为1时保持值不变。
        """
        processor = RegretProcessor()
        decayed = processor.apply_decay(regrets, 1.0)
        
        np.testing.assert_array_almost_equal(
            decayed, regrets,
            err_msg="衰减因子为1时应保持值不变"
        )
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_decay_with_factor_zero_zeros_values(self, regrets):
        """
        **Feature: strategy-convergence-control, Property 2: 遗憾值衰减正确性**
        **验证需求: 1.2**
        
        测试衰减因子为0时所有值变为0。
        """
        processor = RegretProcessor()
        decayed = processor.apply_decay(regrets, 0.0)
        
        assert np.all(decayed == 0), (
            f"衰减因子为0时应所有值为0，但得到: {decayed}"
        )
    
    @given(regret_array_strategy(), valid_decay_factors)
    @settings(max_examples=100)
    def test_decay_preserves_shape(self, regrets, decay_factor):
        """
        **Feature: strategy-convergence-control, Property 2: 遗憾值衰减正确性**
        **验证需求: 1.2**
        
        测试衰减保持数组形状不变。
        """
        processor = RegretProcessor()
        decayed = processor.apply_decay(regrets, decay_factor)
        
        assert decayed.shape == regrets.shape, (
            f"形状应保持不变，原始: {regrets.shape}，衰减后: {decayed.shape}"
        )
    
    @given(regret_array_strategy(), valid_decay_factors)
    @settings(max_examples=100)
    def test_decay_preserves_sign(self, regrets, decay_factor):
        """
        **Feature: strategy-convergence-control, Property 2: 遗憾值衰减正确性**
        **验证需求: 1.2**
        
        测试衰减保持符号不变（除了零值和下溢情况）。
        """
        # 跳过衰减因子极小的情况（可能导致浮点数下溢）
        assume(decay_factor > 1e-100)
        
        processor = RegretProcessor()
        decayed = processor.apply_decay(regrets, decay_factor)
        
        # 非零值的符号应保持不变（排除下溢为零的情况）
        non_zero_mask = (regrets != 0) & (decayed != 0)
        if np.any(non_zero_mask):
            original_signs = np.sign(regrets[non_zero_mask])
            decayed_signs = np.sign(decayed[non_zero_mask])
            np.testing.assert_array_equal(
                original_signs, decayed_signs,
                err_msg="衰减应保持符号不变"
            )
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_decay_uses_config_default(self, regrets):
        """
        **Feature: strategy-convergence-control, Property 2: 遗憾值衰减正确性**
        **验证需求: 1.2**
        
        测试不指定衰减因子时使用配置默认值。
        """
        config = RegretProcessorConfig(decay_factor=0.95)
        processor = RegretProcessor(config)
        
        decayed = processor.apply_decay(regrets)
        expected = regrets * 0.95
        
        np.testing.assert_array_almost_equal(
            decayed, expected,
            err_msg="应使用配置的默认衰减因子"
        )


# ============================================================================
# Property 3: 遗憾值裁剪边界
# **Feature: strategy-convergence-control, Property 3: 遗憾值裁剪边界**
# **验证需求: 1.3**
# ============================================================================

class TestProperty3ClipBoundary:
    """属性测试：遗憾值裁剪边界。
    
    对于任何遗憾值数组和裁剪阈值，裁剪后所有元素的绝对值应该不超过阈值。
    """
    
    @given(regret_array_strategy(), valid_clip_thresholds)
    @settings(max_examples=100)
    def test_clip_absolute_value_within_threshold(self, regrets, threshold):
        """
        **Feature: strategy-convergence-control, Property 3: 遗憾值裁剪边界**
        **验证需求: 1.3**
        
        测试裁剪后绝对值不超过阈值。
        """
        processor = RegretProcessor()
        clipped = processor.clip_regrets(regrets, threshold)
        
        assert np.all(np.abs(clipped) <= threshold + 1e-10), (
            f"裁剪后绝对值应不超过{threshold}，"
            f"但最大绝对值为: {np.max(np.abs(clipped))}"
        )
    
    @given(regret_array_strategy(), valid_clip_thresholds)
    @settings(max_examples=100)
    def test_clip_preserves_values_within_threshold(self, regrets, threshold):
        """
        **Feature: strategy-convergence-control, Property 3: 遗憾值裁剪边界**
        **验证需求: 1.3**
        
        测试裁剪保留阈值内的值不变。
        """
        processor = RegretProcessor()
        clipped = processor.clip_regrets(regrets, threshold)
        
        # 绝对值在阈值内的值应保持不变
        within_threshold_mask = np.abs(regrets) <= threshold
        if np.any(within_threshold_mask):
            np.testing.assert_array_almost_equal(
                clipped[within_threshold_mask],
                regrets[within_threshold_mask],
                err_msg="阈值内的值应保持不变"
            )
    
    @given(regret_array_strategy(), valid_clip_thresholds)
    @settings(max_examples=100)
    def test_clip_preserves_sign(self, regrets, threshold):
        """
        **Feature: strategy-convergence-control, Property 3: 遗憾值裁剪边界**
        **验证需求: 1.3**
        
        测试裁剪保持符号不变。
        """
        processor = RegretProcessor()
        clipped = processor.clip_regrets(regrets, threshold)
        
        # 非零值的符号应保持不变
        non_zero_mask = regrets != 0
        if np.any(non_zero_mask):
            original_signs = np.sign(regrets[non_zero_mask])
            clipped_signs = np.sign(clipped[non_zero_mask])
            np.testing.assert_array_equal(
                original_signs, clipped_signs,
                err_msg="裁剪应保持符号不变"
            )
    
    @given(regret_array_strategy(), valid_clip_thresholds)
    @settings(max_examples=100)
    def test_clip_preserves_shape(self, regrets, threshold):
        """
        **Feature: strategy-convergence-control, Property 3: 遗憾值裁剪边界**
        **验证需求: 1.3**
        
        测试裁剪保持数组形状不变。
        """
        processor = RegretProcessor()
        clipped = processor.clip_regrets(regrets, threshold)
        
        assert clipped.shape == regrets.shape, (
            f"形状应保持不变，原始: {regrets.shape}，裁剪后: {clipped.shape}"
        )
    
    @given(regret_array_strategy(), valid_clip_thresholds)
    @settings(max_examples=100)
    def test_clip_idempotent(self, regrets, threshold):
        """
        **Feature: strategy-convergence-control, Property 3: 遗憾值裁剪边界**
        **验证需求: 1.3**
        
        测试裁剪操作是幂等的。
        """
        processor = RegretProcessor()
        clipped_once = processor.clip_regrets(regrets, threshold)
        clipped_twice = processor.clip_regrets(clipped_once, threshold)
        
        np.testing.assert_array_equal(
            clipped_once, clipped_twice,
            err_msg="裁剪操作应是幂等的"
        )
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_clip_uses_config_default(self, regrets):
        """
        **Feature: strategy-convergence-control, Property 3: 遗憾值裁剪边界**
        **验证需求: 1.3**
        
        测试不指定阈值时使用配置默认值。
        """
        config = RegretProcessorConfig(clip_threshold=50.0)
        processor = RegretProcessor(config)
        
        clipped = processor.clip_regrets(regrets)
        
        assert np.all(np.abs(clipped) <= 50.0 + 1e-10), (
            f"应使用配置的默认裁剪阈值50.0，"
            f"但最大绝对值为: {np.max(np.abs(clipped))}"
        )


# ============================================================================
# 配置验证测试
# ============================================================================

class TestConfigValidation:
    """配置验证测试。"""
    
    def test_invalid_decay_factor_raises_error(self):
        """测试无效衰减因子抛出错误。"""
        with pytest.raises(ValueError, match="衰减因子必须在0-1之间"):
            RegretProcessorConfig(decay_factor=1.5)
        
        with pytest.raises(ValueError, match="衰减因子必须在0-1之间"):
            RegretProcessorConfig(decay_factor=-0.1)
    
    def test_invalid_clip_threshold_raises_error(self):
        """测试无效裁剪阈值抛出错误。"""
        with pytest.raises(ValueError, match="裁剪阈值必须为正数"):
            RegretProcessorConfig(clip_threshold=0)
        
        with pytest.raises(ValueError, match="裁剪阈值必须为正数"):
            RegretProcessorConfig(clip_threshold=-10)
    
    def test_valid_config_creation(self):
        """测试有效配置创建成功。"""
        config = RegretProcessorConfig(
            use_positive_truncation=True,
            decay_factor=0.95,
            clip_threshold=50.0
        )
        assert config.use_positive_truncation is True
        assert config.decay_factor == 0.95
        assert config.clip_threshold == 50.0


# ============================================================================
# 完整处理流程测试
# ============================================================================

class TestProcessFlow:
    """完整处理流程测试。"""
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_process_applies_all_operations(self, regrets):
        """测试process方法应用所有操作。"""
        config = RegretProcessorConfig(
            use_positive_truncation=True,
            decay_factor=0.9,
            clip_threshold=50.0
        )
        processor = RegretProcessor(config)
        
        processed = processor.process(regrets)
        
        # 验证结果满足所有约束
        assert np.all(processed >= 0), "应用截断后应非负"
        assert np.all(np.abs(processed) <= 50.0 + 1e-10), "应用裁剪后应在阈值内"
    
    @given(regret_array_strategy())
    @settings(max_examples=100)
    def test_process_can_skip_operations(self, regrets):
        """测试process方法可以跳过操作。"""
        processor = RegretProcessor()
        
        # 只应用裁剪
        processed = processor.process(
            regrets, 
            apply_truncation=False, 
            apply_decay=False, 
            apply_clip=True
        )
        
        # 应该只有裁剪效果
        expected = processor.clip_regrets(regrets)
        np.testing.assert_array_almost_equal(processed, expected)
