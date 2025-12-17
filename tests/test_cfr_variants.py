"""CFR变体选择器属性测试模块。

使用Hypothesis库进行属性测试，验证CFR变体选择器的核心正确性属性。
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from training.cfr_variants import CFRVariant, CFRVariantConfig, CFRVariantSelector


# ============================================================================
# 测试策略（生成器）
# ============================================================================

# 生成有效的迭代次数（从1开始）
valid_iterations = st.integers(min_value=1, max_value=100000)

# 生成有效的DCFR参数
valid_dcfr_alpha = st.floats(min_value=0.0, max_value=10.0, 
                              allow_nan=False, allow_infinity=False)
valid_dcfr_beta = st.floats(min_value=0.0, max_value=10.0,
                             allow_nan=False, allow_infinity=False)
valid_dcfr_gamma = st.floats(min_value=0.0, max_value=10.0,
                              allow_nan=False, allow_infinity=False)


# ============================================================================
# Property 9: LCFR线性权重正确性
# **Feature: strategy-convergence-control, Property 9: LCFR线性权重正确性**
# **验证需求: 3.4, 6.3**
# ============================================================================

class TestProperty9LCFRLinearWeight:
    """属性测试：LCFR线性权重正确性。
    
    对于任何迭代次数t，LCFR权重应该与t成正比（实际上等于t）。
    """
    
    @given(valid_iterations)
    @settings(max_examples=100)
    def test_lcfr_weight_equals_iteration(self, iteration):
        """
        **Feature: strategy-convergence-control, Property 9: LCFR线性权重正确性**
        **验证需求: 3.4, 6.3**
        
        测试LCFR权重等于迭代次数。
        """
        selector = CFRVariantSelector()
        weight = selector.compute_lcfr_weight(iteration)
        
        assert weight == float(iteration), (
            f"LCFR权重应等于迭代次数{iteration}，但得到: {weight}"
        )
    
    @given(st.integers(min_value=1, max_value=10000), 
           st.integers(min_value=1, max_value=10000))
    @settings(max_examples=100)
    def test_lcfr_weight_proportional(self, iter1, iter2):
        """
        **Feature: strategy-convergence-control, Property 9: LCFR线性权重正确性**
        **验证需求: 3.4, 6.3**
        
        测试LCFR权重与迭代次数成正比。
        """
        selector = CFRVariantSelector()
        weight1 = selector.compute_lcfr_weight(iter1)
        weight2 = selector.compute_lcfr_weight(iter2)
        
        # 权重比应等于迭代次数比
        if iter2 != 0:
            expected_ratio = iter1 / iter2
            actual_ratio = weight1 / weight2
            assert abs(actual_ratio - expected_ratio) < 1e-10, (
                f"权重比应等于迭代次数比，期望: {expected_ratio}，实际: {actual_ratio}"
            )
    
    @given(st.integers(min_value=1, max_value=10000))
    @settings(max_examples=100)
    def test_lcfr_weight_monotonically_increasing(self, iteration):
        """
        **Feature: strategy-convergence-control, Property 9: LCFR线性权重正确性**
        **验证需求: 3.4, 6.3**
        
        测试LCFR权重随迭代次数单调递增。
        """
        selector = CFRVariantSelector()
        weight_current = selector.compute_lcfr_weight(iteration)
        weight_next = selector.compute_lcfr_weight(iteration + 1)
        
        assert weight_next > weight_current, (
            f"权重应单调递增，但iter={iteration}时权重={weight_current}，"
            f"iter={iteration+1}时权重={weight_next}"
        )
    
    @given(valid_iterations)
    @settings(max_examples=100)
    def test_lcfr_weight_positive(self, iteration):
        """
        **Feature: strategy-convergence-control, Property 9: LCFR线性权重正确性**
        **验证需求: 3.4, 6.3**
        
        测试LCFR权重始终为正。
        """
        selector = CFRVariantSelector()
        weight = selector.compute_lcfr_weight(iteration)
        
        assert weight > 0, f"LCFR权重应为正，但得到: {weight}"
    
    def test_lcfr_weight_invalid_iteration_raises_error(self):
        """
        **Feature: strategy-convergence-control, Property 9: LCFR线性权重正确性**
        **验证需求: 3.4, 6.3**
        
        测试无效迭代次数抛出错误。
        """
        selector = CFRVariantSelector()
        
        with pytest.raises(ValueError, match="迭代次数必须大于等于1"):
            selector.compute_lcfr_weight(0)
        
        with pytest.raises(ValueError, match="迭代次数必须大于等于1"):
            selector.compute_lcfr_weight(-1)



# ============================================================================
# Property 12: DCFR折扣因子正确性
# **Feature: strategy-convergence-control, Property 12: DCFR折扣因子正确性**
# **验证需求: 6.4**
# ============================================================================

class TestProperty12DCFRDiscount:
    """属性测试：DCFR折扣因子正确性。
    
    对于任何样本迭代次数和当前迭代次数，折扣因子应该随样本年龄增加而减小。
    """
    
    @given(st.integers(min_value=1, max_value=1000),
           st.integers(min_value=1, max_value=1000))
    @settings(max_examples=100)
    def test_dcfr_discount_decreases_with_age(self, sample_iter, extra_iters):
        """
        **Feature: strategy-convergence-control, Property 12: DCFR折扣因子正确性**
        **验证需求: 6.4**
        
        测试折扣因子随样本年龄增加而减小。
        """
        # 当前迭代 = 样本迭代 + 额外迭代
        current_iter = sample_iter + extra_iters
        
        selector = CFRVariantSelector()
        
        # 获取当前样本的折扣
        pos_discount, neg_discount, strategy_discount = selector.compute_dcfr_discount(
            sample_iter, current_iter
        )
        
        # 如果有更新的样本（更大的迭代次数），其折扣应该更大
        if sample_iter < current_iter:
            newer_sample_iter = sample_iter + 1
            pos_newer, neg_newer, strategy_newer = selector.compute_dcfr_discount(
                newer_sample_iter, current_iter
            )
            
            # 策略折扣应该随样本迭代次数增加而增加
            assert strategy_newer >= strategy_discount, (
                f"更新样本的策略折扣应更大，"
                f"旧样本(iter={sample_iter}): {strategy_discount}，"
                f"新样本(iter={newer_sample_iter}): {strategy_newer}"
            )
    
    @given(st.integers(min_value=1, max_value=10000))
    @settings(max_examples=100)
    def test_dcfr_discount_at_current_iteration(self, iteration):
        """
        **Feature: strategy-convergence-control, Property 12: DCFR折扣因子正确性**
        **验证需求: 6.4**
        
        测试当样本迭代等于当前迭代时，策略折扣为1。
        """
        selector = CFRVariantSelector()
        pos_discount, neg_discount, strategy_discount = selector.compute_dcfr_discount(
            iteration, iteration
        )
        
        # 当 t = T 时，策略折扣 (t/T)^gamma = 1
        assert abs(strategy_discount - 1.0) < 1e-10, (
            f"当样本迭代等于当前迭代时，策略折扣应为1，但得到: {strategy_discount}"
        )
    
    @given(st.integers(min_value=1, max_value=10000))
    @settings(max_examples=100)
    def test_dcfr_discount_in_valid_range(self, iteration):
        """
        **Feature: strategy-convergence-control, Property 12: DCFR折扣因子正确性**
        **验证需求: 6.4**
        
        测试所有折扣因子在[0, 1]范围内。
        """
        selector = CFRVariantSelector()
        current_iter = iteration + 100  # 确保有一些历史
        
        pos_discount, neg_discount, strategy_discount = selector.compute_dcfr_discount(
            iteration, current_iter
        )
        
        assert 0 <= pos_discount <= 1, (
            f"正遗憾折扣应在[0,1]范围内，但得到: {pos_discount}"
        )
        assert 0 <= neg_discount <= 1, (
            f"负遗憾折扣应在[0,1]范围内，但得到: {neg_discount}"
        )
        assert 0 <= strategy_discount <= 1, (
            f"策略折扣应在[0,1]范围内，但得到: {strategy_discount}"
        )
    
    @given(st.integers(min_value=1, max_value=1000),
           st.integers(min_value=1, max_value=1000))
    @settings(max_examples=100)
    def test_dcfr_positive_discount_formula(self, sample_iter, extra_iters):
        """
        **Feature: strategy-convergence-control, Property 12: DCFR折扣因子正确性**
        **验证需求: 6.4**
        
        测试正遗憾折扣公式正确性: t^alpha / (t^alpha + 1)。
        """
        current_iter = sample_iter + extra_iters
        config = CFRVariantConfig(discount_alpha=1.5)
        selector = CFRVariantSelector(config)
        
        pos_discount, _, _ = selector.compute_dcfr_discount(sample_iter, current_iter)
        
        # 验证公式: t^alpha / (t^alpha + 1)
        t = float(sample_iter)
        alpha = 1.5
        expected = (t ** alpha) / ((t ** alpha) + 1.0)
        
        assert abs(pos_discount - expected) < 1e-10, (
            f"正遗憾折扣公式不正确，期望: {expected}，实际: {pos_discount}"
        )
    
    @given(st.integers(min_value=1, max_value=1000),
           st.integers(min_value=1, max_value=1000))
    @settings(max_examples=100)
    def test_dcfr_strategy_discount_formula(self, sample_iter, extra_iters):
        """
        **Feature: strategy-convergence-control, Property 12: DCFR折扣因子正确性**
        **验证需求: 6.4**
        
        测试策略折扣公式正确性: (t/T)^gamma。
        """
        current_iter = sample_iter + extra_iters
        config = CFRVariantConfig(discount_gamma=2.0)
        selector = CFRVariantSelector(config)
        
        _, _, strategy_discount = selector.compute_dcfr_discount(sample_iter, current_iter)
        
        # 验证公式: (t/T)^gamma
        t = float(sample_iter)
        T = float(current_iter)
        gamma = 2.0
        expected = (t / T) ** gamma
        
        assert abs(strategy_discount - expected) < 1e-10, (
            f"策略折扣公式不正确，期望: {expected}，实际: {strategy_discount}"
        )
    
    def test_dcfr_discount_invalid_iterations_raise_error(self):
        """
        **Feature: strategy-convergence-control, Property 12: DCFR折扣因子正确性**
        **验证需求: 6.4**
        
        测试无效迭代次数抛出错误。
        """
        selector = CFRVariantSelector()
        
        # 样本迭代次数小于1
        with pytest.raises(ValueError, match="样本迭代次数必须大于等于1"):
            selector.compute_dcfr_discount(0, 10)
        
        # 当前迭代次数小于1
        with pytest.raises(ValueError, match="当前迭代次数必须大于等于1"):
            selector.compute_dcfr_discount(5, 0)
        
        # 样本迭代次数大于当前迭代次数
        with pytest.raises(ValueError, match="样本迭代次数.*不能大于当前迭代次数"):
            selector.compute_dcfr_discount(10, 5)


# ============================================================================
# 配置验证测试
# ============================================================================

class TestCFRVariantConfigValidation:
    """CFR变体配置验证测试。"""
    
    def test_invalid_discount_alpha_raises_error(self):
        """测试无效的discount_alpha抛出错误。"""
        with pytest.raises(ValueError, match="discount_alpha必须非负"):
            CFRVariantConfig(discount_alpha=-0.1)
    
    def test_invalid_discount_beta_raises_error(self):
        """测试无效的discount_beta抛出错误。"""
        with pytest.raises(ValueError, match="discount_beta必须非负"):
            CFRVariantConfig(discount_beta=-0.1)
    
    def test_invalid_discount_gamma_raises_error(self):
        """测试无效的discount_gamma抛出错误。"""
        with pytest.raises(ValueError, match="discount_gamma必须非负"):
            CFRVariantConfig(discount_gamma=-0.1)
    
    def test_valid_config_creation(self):
        """测试有效配置创建成功。"""
        config = CFRVariantConfig(
            variant=CFRVariant.DCFR,
            regret_floor=0.0,
            discount_alpha=1.5,
            discount_beta=0.5,
            discount_gamma=2.0
        )
        assert config.variant == CFRVariant.DCFR
        assert config.discount_alpha == 1.5
        assert config.discount_beta == 0.5
        assert config.discount_gamma == 2.0


# ============================================================================
# CFR变体选择器功能测试
# ============================================================================

class TestCFRVariantSelector:
    """CFR变体选择器功能测试。"""
    
    def test_standard_cfr_weight(self):
        """测试标准CFR权重始终为1。"""
        config = CFRVariantConfig(variant=CFRVariant.STANDARD)
        selector = CFRVariantSelector(config)
        
        for iteration in [1, 10, 100, 1000]:
            weight = selector.get_iteration_weight(iteration)
            assert weight == 1.0, f"标准CFR权重应为1，但得到: {weight}"
    
    def test_cfr_plus_weight(self):
        """测试CFR+权重始终为1。"""
        config = CFRVariantConfig(variant=CFRVariant.CFR_PLUS)
        selector = CFRVariantSelector(config)
        
        for iteration in [1, 10, 100, 1000]:
            weight = selector.get_iteration_weight(iteration)
            assert weight == 1.0, f"CFR+权重应为1，但得到: {weight}"
    
    def test_lcfr_weight_via_selector(self):
        """测试通过选择器获取LCFR权重。"""
        config = CFRVariantConfig(variant=CFRVariant.LCFR)
        selector = CFRVariantSelector(config)
        
        for iteration in [1, 10, 100]:
            weight = selector.get_iteration_weight(iteration)
            assert weight == float(iteration), (
                f"LCFR权重应等于迭代次数{iteration}，但得到: {weight}"
            )
    
    def test_dcfr_weight_via_selector(self):
        """测试通过选择器获取DCFR权重。"""
        config = CFRVariantConfig(variant=CFRVariant.DCFR, discount_gamma=2.0)
        selector = CFRVariantSelector(config)
        
        # DCFR需要current_iteration参数
        weight = selector.get_iteration_weight(5, current_iteration=10)
        expected = (5.0 / 10.0) ** 2.0  # (t/T)^gamma
        assert abs(weight - expected) < 1e-10, (
            f"DCFR权重应为{expected}，但得到: {weight}"
        )
    
    def test_dcfr_requires_current_iteration(self):
        """测试DCFR变体需要current_iteration参数。"""
        config = CFRVariantConfig(variant=CFRVariant.DCFR)
        selector = CFRVariantSelector(config)
        
        with pytest.raises(ValueError, match="DCFR变体需要提供current_iteration参数"):
            selector.get_iteration_weight(5)
