"""
EV预测神经网络测试模块

包含属性测试和单元测试，验证网络的正确性。
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings

from training.ev_prediction_network import EVPredictionNetwork


class TestEVPredictionNetworkProperties:
    """属性测试类"""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=64),
        num_actions=st.integers(min_value=2, max_value=10),
        hidden_dim=st.sampled_from([32, 64, 128])
    )
    @settings(max_examples=100)
    def test_output_dimensions(self, batch_size: int, num_actions: int, hidden_dim: int):
        """
        **Feature: ev-prediction-network, Property 1: 网络输出维度正确性**
        
        对于任意4维输入张量和任意动作数量N，网络应该输出：
        - 整体EV: [batch_size, 1] 维度
        - 动作EV: [batch_size, N] 维度
        - 策略: [batch_size, N] 维度
        
        **Validates: Requirements 1.1, 1.2, 1.3, 6.2, 6.3**
        """
        # 创建网络
        model = EVPredictionNetwork(num_actions=num_actions, hidden_dim=hidden_dim)
        model.eval()
        
        # 生成随机输入
        x = torch.randn(batch_size, 4)
        
        # 前向传播
        with torch.no_grad():
            ev, action_ev, strategy = model(x)
        
        # 验证输出维度
        assert ev.shape == (batch_size, 1), \
            f"EV维度错误：期望 ({batch_size}, 1)，实际 {ev.shape}"
        assert action_ev.shape == (batch_size, num_actions), \
            f"动作EV维度错误：期望 ({batch_size}, {num_actions})，实际 {action_ev.shape}"
        assert strategy.shape == (batch_size, num_actions), \
            f"策略维度错误：期望 ({batch_size}, {num_actions})，实际 {strategy.shape}"
    
    @given(
        batch_size=st.integers(min_value=1, max_value=64),
        num_actions=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=100)
    def test_strategy_probability_normalization(self, batch_size: int, num_actions: int):
        """
        **Feature: ev-prediction-network, Property 2: 策略概率归一化**
        
        对于任意有效的4维输入，网络输出的策略概率向量之和应该等于1.0（误差范围±0.001）
        
        **Validates: Requirements 1.4, 6.4**
        """
        # 创建网络
        model = EVPredictionNetwork(num_actions=num_actions)
        model.eval()
        
        # 生成随机输入（包括边界值）
        x = torch.randn(batch_size, 4)
        
        # 前向传播
        with torch.no_grad():
            _, _, strategy = model(x)
        
        # 验证概率和为1
        prob_sums = strategy.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=0.001), \
            f"策略概率和不为1：{prob_sums}"
        
        # 验证所有概率非负
        assert (strategy >= 0).all(), "存在负概率"
        
        # 验证所有概率不超过1
        assert (strategy <= 1).all(), "存在超过1的概率"
    
    @given(
        hero_equity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        range_equity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        solver_equity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        eqr=st.floats(min_value=0.0, max_value=3.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_valid_input_range(
        self, hero_equity: float, range_equity: float, 
        solver_equity: float, eqr: float
    ):
        """
        测试在有效输入范围内网络能正常工作
        
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        model = EVPredictionNetwork()
        model.eval()
        
        x = torch.tensor([[hero_equity, range_equity, solver_equity, eqr]])
        
        with torch.no_grad():
            ev, action_ev, strategy = model(x)
        
        # 验证输出不包含NaN或Inf
        assert not torch.isnan(ev).any(), "EV包含NaN"
        assert not torch.isinf(ev).any(), "EV包含Inf"
        assert not torch.isnan(action_ev).any(), "动作EV包含NaN"
        assert not torch.isinf(action_ev).any(), "动作EV包含Inf"
        assert not torch.isnan(strategy).any(), "策略包含NaN"


class TestEVPredictionNetworkUnit:
    """单元测试类"""
    
    def test_network_initialization(self):
        """测试网络初始化"""
        model = EVPredictionNetwork(num_actions=5, hidden_dim=64)
        
        assert model.num_actions == 5
        assert model.hidden_dim == 64
        assert model.input_dim == 4
    
    def test_invalid_input_dimension(self):
        """测试无效输入维度抛出异常"""
        model = EVPredictionNetwork()
        
        # 测试1维输入
        with pytest.raises(ValueError, match="输入维度错误"):
            model(torch.randn(4))
        
        # 测试错误的特征维度
        with pytest.raises(ValueError, match="输入维度错误"):
            model(torch.randn(2, 3))
        
        # 测试3维输入
        with pytest.raises(ValueError, match="输入维度错误"):
            model(torch.randn(2, 4, 1))
    
    def test_get_strategy_logits(self):
        """测试获取策略logits"""
        model = EVPredictionNetwork(num_actions=5)
        model.eval()
        
        x = torch.randn(3, 4)
        
        with torch.no_grad():
            logits = model.get_strategy_logits(x)
        
        assert logits.shape == (3, 5)
        # logits可以是任意实数，不需要归一化
    
    def test_forward_deterministic(self):
        """测试前向传播的确定性（eval模式下）"""
        model = EVPredictionNetwork()
        model.eval()
        
        x = torch.randn(2, 4)
        
        with torch.no_grad():
            ev1, action_ev1, strategy1 = model(x)
            ev2, action_ev2, strategy2 = model(x)
        
        assert torch.allclose(ev1, ev2)
        assert torch.allclose(action_ev1, action_ev2)
        assert torch.allclose(strategy1, strategy2)
    
    def test_gradient_flow(self):
        """测试梯度能够正常流动"""
        model = EVPredictionNetwork()
        model.train()
        
        x = torch.randn(4, 4, requires_grad=True)
        ev, action_ev, strategy = model(x)
        
        # 计算损失并反向传播
        loss = ev.sum() + action_ev.sum() + strategy.sum()
        loss.backward()
        
        # 验证梯度存在
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None
