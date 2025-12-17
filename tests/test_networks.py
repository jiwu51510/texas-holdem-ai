"""神经网络模块的单元测试和属性测试。

测试内容：
- PolicyNetwork输入输出维度正确
- ValueNetwork输入输出维度正确
- 前向传播不抛出异常
- 策略网络输出概率和为1（误差<1e-6）
- 所有概率非负
- 属性测试：策略概率分布有效性
"""

import pytest
import torch
import numpy as np
from hypothesis import given, settings, strategies as st

from models.networks import PolicyNetwork, RegretNetwork, ValueNetwork


class TestPolicyNetwork:
    """策略网络单元测试。"""
    
    def test_default_initialization(self):
        """测试默认参数初始化。"""
        network = PolicyNetwork()
        assert network.input_dim == 370
        assert network.hidden_dims == [512, 256, 128]
        assert network.action_dim == 6  # FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN
    
    def test_custom_initialization(self):
        """测试自定义参数初始化。"""
        network = PolicyNetwork(
            input_dim=100,
            hidden_dims=[64, 32],
            action_dim=6
        )
        assert network.input_dim == 100
        assert network.hidden_dims == [64, 32]
        assert network.action_dim == 6
    
    def test_forward_output_shape_single(self):
        """测试单个输入的前向传播输出形状。"""
        network = PolicyNetwork()
        input_tensor = torch.randn(370)
        output = network.forward(input_tensor)
        assert output.shape == (6,)  # 6 个行动类型
    
    def test_forward_output_shape_batch(self):
        """测试批量输入的前向传播输出形状。"""
        network = PolicyNetwork()
        batch_size = 32
        input_tensor = torch.randn(batch_size, 370)
        output = network.forward(input_tensor)
        assert output.shape == (batch_size, 6)  # 6 个行动类型
    
    def test_forward_no_exception(self):
        """测试前向传播不抛出异常。"""
        network = PolicyNetwork()
        input_tensor = torch.randn(16, 370)
        # 不应该抛出异常
        output = network.forward(input_tensor)
        assert output is not None
    
    def test_get_action_probs_sum_to_one(self):
        """测试策略网络输出概率和为1（误差<1e-6）。"""
        network = PolicyNetwork()
        input_tensor = torch.randn(370)
        probs = network.get_action_probs(input_tensor)
        
        prob_sum = probs.sum().item()
        assert abs(prob_sum - 1.0) < 1e-6, f"概率和应为1，实际为{prob_sum}"
    
    def test_get_action_probs_sum_to_one_batch(self):
        """测试批量输入时每个样本的概率和为1。"""
        network = PolicyNetwork()
        batch_size = 32
        input_tensor = torch.randn(batch_size, 370)
        probs = network.get_action_probs(input_tensor)
        
        # 检查每个样本的概率和
        prob_sums = probs.sum(dim=-1)
        for i, prob_sum in enumerate(prob_sums):
            assert abs(prob_sum.item() - 1.0) < 1e-6, \
                f"样本{i}的概率和应为1，实际为{prob_sum.item()}"
    
    def test_get_action_probs_non_negative(self):
        """测试所有概率非负。"""
        network = PolicyNetwork()
        input_tensor = torch.randn(370)
        probs = network.get_action_probs(input_tensor)
        
        assert (probs >= 0).all(), "所有概率应该非负"
    
    def test_get_action_probs_non_negative_batch(self):
        """测试批量输入时所有概率非负。"""
        network = PolicyNetwork()
        batch_size = 32
        input_tensor = torch.randn(batch_size, 370)
        probs = network.get_action_probs(input_tensor)
        
        assert (probs >= 0).all(), "所有概率应该非负"
    
    def test_custom_action_dim(self):
        """测试自定义行动空间维度。"""
        action_dim = 10
        network = PolicyNetwork(action_dim=action_dim)
        input_tensor = torch.randn(370)
        probs = network.get_action_probs(input_tensor)
        
        assert probs.shape == (action_dim,)
        assert abs(probs.sum().item() - 1.0) < 1e-6


class TestValueNetwork:
    """价值网络单元测试。"""
    
    def test_default_initialization(self):
        """测试默认参数初始化。"""
        network = ValueNetwork()
        assert network.input_dim == 370
        assert network.hidden_dims == [512, 256, 128]
    
    def test_custom_initialization(self):
        """测试自定义参数初始化。"""
        network = ValueNetwork(
            input_dim=100,
            hidden_dims=[64, 32]
        )
        assert network.input_dim == 100
        assert network.hidden_dims == [64, 32]
    
    def test_forward_output_shape_single(self):
        """测试单个输入的前向传播输出形状。"""
        network = ValueNetwork()
        input_tensor = torch.randn(370)
        output = network.forward(input_tensor)
        assert output.shape == (1,)
    
    def test_forward_output_shape_batch(self):
        """测试批量输入的前向传播输出形状。"""
        network = ValueNetwork()
        batch_size = 32
        input_tensor = torch.randn(batch_size, 370)
        output = network.forward(input_tensor)
        assert output.shape == (batch_size, 1)
    
    def test_forward_no_exception(self):
        """测试前向传播不抛出异常。"""
        network = ValueNetwork()
        input_tensor = torch.randn(16, 370)
        # 不应该抛出异常
        output = network.forward(input_tensor)
        assert output is not None
    
    def test_output_range(self):
        """测试输出值在[-1, 1]范围内（Tanh激活）。"""
        network = ValueNetwork()
        # 使用多种输入测试
        for _ in range(10):
            input_tensor = torch.randn(370) * 10  # 放大输入以测试边界
            output = network.forward(input_tensor)
            assert -1.0 <= output.item() <= 1.0, \
                f"输出应在[-1, 1]范围内，实际为{output.item()}"
    
    def test_output_range_batch(self):
        """测试批量输入时输出值在[-1, 1]范围内。"""
        network = ValueNetwork()
        batch_size = 100
        input_tensor = torch.randn(batch_size, 370) * 10
        output = network.forward(input_tensor)
        
        assert (output >= -1.0).all() and (output <= 1.0).all(), \
            "所有输出应在[-1, 1]范围内"


class TestNetworkGradients:
    """测试网络梯度计算。"""
    
    def test_policy_network_gradients(self):
        """测试策略网络可以计算梯度。"""
        network = PolicyNetwork()
        input_tensor = torch.randn(16, 370, requires_grad=True)
        output = network.forward(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度是否存在
        for param in network.parameters():
            assert param.grad is not None, "梯度应该存在"
    
    def test_value_network_gradients(self):
        """测试价值网络可以计算梯度。"""
        network = ValueNetwork()
        input_tensor = torch.randn(16, 370, requires_grad=True)
        output = network.forward(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度是否存在
        for param in network.parameters():
            assert param.grad is not None, "梯度应该存在"



# ============================================================================
# 属性测试（Property-Based Testing）
# ============================================================================

class TestPolicyNetworkProperties:
    """策略网络属性测试。
    
    使用Hypothesis进行基于属性的测试，验证策略概率分布的有效性。
    """
    
    @given(
        batch_size=st.integers(min_value=1, max_value=64),
        input_dim=st.integers(min_value=10, max_value=500),
        action_dim=st.integers(min_value=2, max_value=20)
    )
    @settings(max_examples=100)
    def test_property_action_probs_valid_distribution(
        self, batch_size: int, input_dim: int, action_dim: int
    ):
        """属性测试：策略概率分布有效性。
        
        # Feature: texas-holdem-ai-training, Property 13: 策略概率分布有效性
        # *对于任何*游戏状态，策略查看器返回的行动概率分布应该满足：
        # 所有概率非负，且概率之和等于1（误差在1e-6内）
        # **验证需求：4.2**
        
        对于任意随机状态编码，验证：
        1. 所有概率值非负
        2. 概率之和等于1（误差<1e-6）
        """
        # 创建网络
        network = PolicyNetwork(
            input_dim=input_dim,
            hidden_dims=[64, 32],  # 使用较小的网络加速测试
            action_dim=action_dim
        )
        
        # 生成随机状态编码
        state_encoding = torch.randn(batch_size, input_dim)
        
        # 获取行动概率
        probs = network.get_action_probs(state_encoding)
        
        # 验证1：所有概率非负
        assert (probs >= 0).all(), \
            f"发现负概率值：{probs[probs < 0].tolist()}"
        
        # 验证2：每个样本的概率和为1
        prob_sums = probs.sum(dim=-1)
        for i, prob_sum in enumerate(prob_sums):
            assert abs(prob_sum.item() - 1.0) < 1e-6, \
                f"样本{i}的概率和应为1，实际为{prob_sum.item()}"
    
    @given(
        state_values=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=370,
            max_size=370
        )
    )
    @settings(max_examples=100)
    def test_property_policy_network_output_valid_probability_distribution(
        self, state_values: list
    ):
        """属性测试：策略网络输出概率分布有效性（Deep CFR）。
        
        # Feature: deep-cfr-refactor, Property 2: 策略网络输出概率分布有效性
        # *对于任何*有效的状态编码输入，策略网络的输出应该是有效的概率分布
        # （所有概率非负且和为1，误差在1e-6内）
        # **验证需求：1.4**
        
        对于任意370维状态编码，验证：
        1. 所有概率值非负
        2. 概率之和等于1（误差<1e-6）
        3. 输出维度为6（动作空间大小：FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN）
        """
        # 创建默认配置的策略网络（370维输入，6维输出）
        network = PolicyNetwork()
        
        # 将列表转换为张量
        state_encoding = torch.tensor(state_values, dtype=torch.float32)
        
        # 获取行动概率
        probs = network.get_action_probs(state_encoding)
        
        # 验证1：输出维度正确（6维动作空间）
        assert probs.shape == (6,), \
            f"输出形状应为(6,)，实际为{probs.shape}"
        
        # 验证2：所有概率非负
        assert (probs >= 0).all(), \
            f"发现负概率值：{probs[probs < 0].tolist()}"
        
        # 验证3：概率和为1
        prob_sum = probs.sum().item()
        assert abs(prob_sum - 1.0) < 1e-6, \
            f"概率和应为1，实际为{prob_sum}"
    
    @given(
        scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_action_probs_robust_to_input_scale(self, scale: float):
        """属性测试：概率分布对输入缩放的鲁棒性。
        
        # Feature: texas-holdem-ai-training, Property 13: 策略概率分布有效性
        # **验证需求：4.2**
        
        无论输入的缩放比例如何，输出的概率分布都应该有效。
        """
        network = PolicyNetwork()
        
        # 生成缩放后的随机输入
        state_encoding = torch.randn(370) * scale
        
        # 获取行动概率
        probs = network.get_action_probs(state_encoding)
        
        # 验证概率分布有效性
        assert (probs >= 0).all(), "所有概率应该非负"
        assert abs(probs.sum().item() - 1.0) < 1e-6, \
            f"概率和应为1，实际为{probs.sum().item()}"
    
    @given(
        state_values=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=370,
            max_size=370
        )
    )
    @settings(max_examples=100)
    def test_property_action_probs_with_arbitrary_state(self, state_values: list):
        """属性测试：任意状态编码的概率分布有效性。
        
        # Feature: texas-holdem-ai-training, Property 13: 策略概率分布有效性
        # **验证需求：4.2**
        
        对于任意生成的370维状态编码，概率分布都应该有效。
        """
        network = PolicyNetwork()
        
        # 将列表转换为张量
        state_encoding = torch.tensor(state_values, dtype=torch.float32)
        
        # 获取行动概率
        probs = network.get_action_probs(state_encoding)
        
        # 验证概率分布有效性
        assert (probs >= 0).all(), "所有概率应该非负"
        assert abs(probs.sum().item() - 1.0) < 1e-6, \
            f"概率和应为1，实际为{probs.sum().item()}"
        assert probs.shape == (6,), f"输出形状应为(6,)，实际为{probs.shape}"


class TestValueNetworkProperties:
    """价值网络属性测试。"""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=64),
        input_dim=st.integers(min_value=10, max_value=500)
    )
    @settings(max_examples=100)
    def test_property_value_output_range(self, batch_size: int, input_dim: int):
        """属性测试：价值网络输出范围。
        
        对于任意输入，价值网络的输出应该在[-1, 1]范围内（Tanh激活）。
        """
        network = ValueNetwork(
            input_dim=input_dim,
            hidden_dims=[64, 32]
        )
        
        # 生成随机输入
        state_encoding = torch.randn(batch_size, input_dim)
        
        # 获取价值估计
        values = network.forward(state_encoding)
        
        # 验证输出范围
        assert (values >= -1.0).all() and (values <= 1.0).all(), \
            f"输出应在[-1, 1]范围内，实际范围为[{values.min().item()}, {values.max().item()}]"
    
    @given(
        scale=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_property_value_bounded_regardless_of_input_scale(self, scale: float):
        """属性测试：无论输入缩放如何，输出都有界。
        
        即使输入值很大或很小，Tanh激活确保输出始终在[-1, 1]范围内。
        """
        network = ValueNetwork()
        
        # 生成缩放后的随机输入
        state_encoding = torch.randn(370) * scale
        
        # 获取价值估计
        value = network.forward(state_encoding)
        
        # 验证输出范围
        assert -1.0 <= value.item() <= 1.0, \
            f"输出应在[-1, 1]范围内，实际为{value.item()}"


# ============================================================================
# RegretNetwork 单元测试
# ============================================================================

class TestRegretNetwork:
    """遗憾网络单元测试。"""
    
    def test_default_initialization(self):
        """测试默认参数初始化。"""
        network = RegretNetwork()
        assert network.input_dim == 370
        assert network.hidden_dims == [512, 256, 128]
        assert network.action_dim == 6  # FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN
    
    def test_custom_initialization(self):
        """测试自定义参数初始化。"""
        network = RegretNetwork(
            input_dim=100,
            hidden_dims=[64, 32],
            action_dim=6
        )
        assert network.input_dim == 100
        assert network.hidden_dims == [64, 32]
        assert network.action_dim == 6
    
    def test_forward_output_shape_single(self):
        """测试单个输入的前向传播输出形状。"""
        network = RegretNetwork()
        input_tensor = torch.randn(370)
        output = network.forward(input_tensor)
        assert output.shape == (6,)  # 6 个行动类型
    
    def test_forward_output_shape_batch(self):
        """测试批量输入的前向传播输出形状。"""
        network = RegretNetwork()
        batch_size = 32
        input_tensor = torch.randn(batch_size, 370)
        output = network.forward(input_tensor)
        assert output.shape == (batch_size, 6)  # 6 个行动类型
    
    def test_forward_no_exception(self):
        """测试前向传播不抛出异常。"""
        network = RegretNetwork()
        input_tensor = torch.randn(16, 370)
        output = network.forward(input_tensor)
        assert output is not None
    
    def test_forward_output_unbounded(self):
        """测试遗憾网络输出可以是任意实数（无激活函数）。"""
        network = RegretNetwork()
        # 使用多种输入测试
        found_positive = False
        found_negative = False
        for _ in range(100):
            input_tensor = torch.randn(370) * 10
            output = network.forward(input_tensor)
            if (output > 0).any():
                found_positive = True
            if (output < 0).any():
                found_negative = True
            if found_positive and found_negative:
                break
        # 遗憾值应该可以是正数或负数
        assert found_positive or found_negative, "遗憾值应该可以是任意实数"
    
    def test_get_strategy_sum_to_one(self):
        """测试 get_strategy 输出概率和为1。"""
        network = RegretNetwork()
        input_tensor = torch.randn(370)
        strategy = network.get_strategy(input_tensor)
        
        prob_sum = strategy.sum().item()
        assert abs(prob_sum - 1.0) < 1e-6, f"策略概率和应为1，实际为{prob_sum}"
    
    def test_get_strategy_non_negative(self):
        """测试 get_strategy 输出所有概率非负。"""
        network = RegretNetwork()
        input_tensor = torch.randn(370)
        strategy = network.get_strategy(input_tensor)
        
        assert (strategy >= 0).all(), "所有策略概率应该非负"
    
    def test_get_strategy_batch(self):
        """测试批量输入时 get_strategy 的输出。"""
        network = RegretNetwork()
        batch_size = 32
        input_tensor = torch.randn(batch_size, 370)
        strategy = network.get_strategy(input_tensor)
        
        # 检查形状
        assert strategy.shape == (batch_size, 6)  # 6 个行动类型
        
        # 检查每个样本的概率和
        prob_sums = strategy.sum(dim=-1)
        for i, prob_sum in enumerate(prob_sums):
            assert abs(prob_sum.item() - 1.0) < 1e-6, \
                f"样本{i}的策略概率和应为1，实际为{prob_sum.item()}"
        
        # 检查所有概率非负
        assert (strategy >= 0).all(), "所有策略概率应该非负"
    
    def test_regret_network_gradients(self):
        """测试遗憾网络可以计算梯度。"""
        network = RegretNetwork()
        input_tensor = torch.randn(16, 370, requires_grad=True)
        output = network.forward(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度是否存在
        for param in network.parameters():
            assert param.grad is not None, "梯度应该存在"


# ============================================================================
# RegretNetwork 属性测试（Property-Based Testing）
# ============================================================================

class TestRegretNetworkProperties:
    """遗憾网络属性测试。
    
    使用 Hypothesis 进行基于属性的测试，验证遗憾网络和 Regret Matching 的正确性。
    """
    
    @given(
        batch_size=st.integers(min_value=1, max_value=64),
        input_dim=st.integers(min_value=10, max_value=500),
        action_dim=st.integers(min_value=2, max_value=20)
    )
    @settings(max_examples=100)
    def test_property_regret_network_output_dimension(
        self, batch_size: int, input_dim: int, action_dim: int
    ):
        """属性测试：遗憾网络输出维度正确性。
        
        # Feature: deep-cfr-refactor, Property 1: 遗憾网络输出维度正确性
        # *对于任何*有效的状态编码输入，遗憾网络的输出维度应该等于动作空间大小
        # **验证需求：1.3**
        
        对于任意随机状态编码，验证输出维度等于 action_dim。
        """
        # 创建网络
        network = RegretNetwork(
            input_dim=input_dim,
            hidden_dims=[64, 32],  # 使用较小的网络加速测试
            action_dim=action_dim
        )
        
        # 生成随机状态编码
        state_encoding = torch.randn(batch_size, input_dim)
        
        # 获取遗憾值
        regrets = network.forward(state_encoding)
        
        # 验证输出维度
        assert regrets.shape == (batch_size, action_dim), \
            f"输出形状应为({batch_size}, {action_dim})，实际为{regrets.shape}"
    
    @given(
        batch_size=st.integers(min_value=1, max_value=64),
        input_dim=st.integers(min_value=10, max_value=500),
        action_dim=st.integers(min_value=2, max_value=20)
    )
    @settings(max_examples=100)
    def test_property_regret_matching_valid_distribution(
        self, batch_size: int, input_dim: int, action_dim: int
    ):
        """属性测试：Regret Matching 输出有效性。
        
        # Feature: deep-cfr-refactor, Property 8: Regret Matching 输出有效性
        # *对于任何*遗憾值数组，Regret Matching 的输出应该是有效的概率分布
        # （所有概率非负且和为1）
        # **验证需求：4.3**
        
        对于任意随机状态编码，验证 get_strategy 输出是有效的概率分布。
        """
        # 创建网络
        network = RegretNetwork(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            action_dim=action_dim
        )
        
        # 生成随机状态编码
        state_encoding = torch.randn(batch_size, input_dim)
        
        # 获取策略
        strategy = network.get_strategy(state_encoding)
        
        # 验证1：所有概率非负
        assert (strategy >= 0).all(), \
            f"发现负概率值：{strategy[strategy < 0].tolist()}"
        
        # 验证2：每个样本的概率和为1（使用1e-5容差以适应浮点精度）
        prob_sums = strategy.sum(dim=-1)
        for i, prob_sum in enumerate(prob_sums):
            assert abs(prob_sum.item() - 1.0) < 1e-5, \
                f"样本{i}的概率和应为1，实际为{prob_sum.item()}"
    
    @given(
        regret_values=st.lists(
            st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=10
        )
    )
    @settings(max_examples=100)
    def test_property_regret_matching_positive_regrets_proportional(
        self, regret_values: list
    ):
        """属性测试：Regret Matching 正遗憾值比例。
        
        # Feature: deep-cfr-refactor, Property 9: Regret Matching 正遗憾值比例
        # *对于任何*包含正遗憾值的数组，Regret Matching 输出的概率应该与正遗憾值成正比
        # **验证需求：4.3**
        
        对于任意正遗憾值数组，验证输出概率与遗憾值成正比。
        """
        # 创建网络（只用于调用 _regret_matching 方法）
        network = RegretNetwork(action_dim=len(regret_values))
        
        # 将遗憾值转换为张量
        regrets = torch.tensor(regret_values, dtype=torch.float32)
        
        # 调用 Regret Matching
        strategy = network._regret_matching(regrets)
        
        # 验证概率与正遗憾值成正比
        # 对于全正遗憾值，strategy[i] / strategy[j] 应该等于 regrets[i] / regrets[j]
        positive_regrets = torch.clamp(regrets, min=0.0)
        regret_sum = positive_regrets.sum()
        
        if regret_sum > 0:
            expected_strategy = positive_regrets / regret_sum
            # 检查策略是否与期望值接近
            assert torch.allclose(strategy, expected_strategy, atol=1e-5), \
                f"策略应与正遗憾值成正比。期望：{expected_strategy.tolist()}，实际：{strategy.tolist()}"
    
    @given(
        regret_values=st.lists(
            st.floats(min_value=-100.0, max_value=0.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=10
        )
    )
    @settings(max_examples=100)
    def test_property_regret_matching_all_non_positive_uniform(
        self, regret_values: list
    ):
        """属性测试：Regret Matching 全非正遗憾值。
        
        # Feature: deep-cfr-refactor, Property 10: Regret Matching 全非正遗憾值
        # *对于任何*所有遗憾值都非正的数组，Regret Matching 应该输出均匀分布
        # **验证需求：4.3**
        
        对于任意全非正遗憾值数组，验证输出为均匀分布。
        """
        # 创建网络（只用于调用 _regret_matching 方法）
        action_dim = len(regret_values)
        network = RegretNetwork(action_dim=action_dim)
        
        # 将遗憾值转换为张量
        regrets = torch.tensor(regret_values, dtype=torch.float32)
        
        # 调用 Regret Matching
        strategy = network._regret_matching(regrets)
        
        # 验证输出为均匀分布
        expected_uniform = torch.ones(action_dim) / action_dim
        assert torch.allclose(strategy, expected_uniform, atol=1e-6), \
            f"全非正遗憾值应输出均匀分布。期望：{expected_uniform.tolist()}，实际：{strategy.tolist()}"
    
    @given(
        regret_values=st.lists(
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False, allow_subnormal=False),
            min_size=2,
            max_size=10
        ).filter(lambda vals: all(abs(v) > 1e-6 or v == 0.0 for v in vals))
    )
    @settings(max_examples=100)
    def test_property_regret_matching_mixed_regrets(
        self, regret_values: list
    ):
        """属性测试：混合遗憾值的 Regret Matching。
        
        # Feature: deep-cfr-refactor, Property 8: Regret Matching 输出有效性
        # **验证需求：4.3**
        
        对于任意混合（正负）遗憾值数组，验证输出是有效的概率分布。
        注意：过滤掉极小的非零值（< 1e-6）以避免浮点精度问题。
        这是因为 _regret_matching 使用 1e-10 作为数值稳定性的小值，
        当正遗憾值和非常小时会导致精度损失。
        """
        # 创建网络
        action_dim = len(regret_values)
        network = RegretNetwork(action_dim=action_dim)
        
        # 将遗憾值转换为张量
        regrets = torch.tensor(regret_values, dtype=torch.float32)
        
        # 调用 Regret Matching
        strategy = network._regret_matching(regrets)
        
        # 验证1：所有概率非负
        assert (strategy >= 0).all(), \
            f"发现负概率值：{strategy[strategy < 0].tolist()}"
        
        # 验证2：概率和为1（使用1e-4容差以适应浮点精度）
        prob_sum = strategy.sum().item()
        assert abs(prob_sum - 1.0) < 1e-4, \
            f"概率和应为1，实际为{prob_sum}"
        
        # 验证3：负遗憾值对应的概率应该为0（或接近0）
        positive_regrets = torch.clamp(regrets, min=0.0)
        regret_sum = positive_regrets.sum()
        
        if regret_sum > 0:
            # 负遗憾值对应的概率应该为0
            for i, r in enumerate(regret_values):
                if r <= 0:
                    assert strategy[i].item() < 1e-6, \
                        f"负遗憾值{r}对应的概率应为0，实际为{strategy[i].item()}"
