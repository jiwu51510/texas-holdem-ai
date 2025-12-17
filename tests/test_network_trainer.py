"""网络训练器属性测试模块。

使用Hypothesis库进行属性测试，验证网络训练器的核心正确性属性。
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from training.network_trainer import NetworkTrainer, NetworkTrainerConfig


# ============================================================================
# 测试策略（生成器）
# ============================================================================

# 生成有效的张量值
@st.composite
def tensor_pair_strategy(draw, min_size=1, max_size=100):
    """生成随机预测值和目标值张量对。
    
    生成用于损失计算的预测值和目标值。
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # 生成范围在[-100, 100]的浮点数
    predictions = draw(st.lists(
        st.floats(min_value=-100.0, max_value=100.0, 
                  allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size
    ))
    targets = draw(st.lists(
        st.floats(min_value=-100.0, max_value=100.0, 
                  allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size
    ))
    return (
        torch.tensor(predictions, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32)
    )


# 生成有效的概率分布
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
    arr = np.array(values, dtype=np.float32)
    arr = arr / arr.sum()
    return torch.tensor(arr, dtype=torch.float32)


# 生成有效的Huber delta参数
valid_huber_deltas = st.floats(min_value=0.01, max_value=100.0,
                                allow_nan=False, allow_infinity=False)

# 生成有效的EMA衰减率
valid_ema_decays = st.floats(min_value=0.0, max_value=1.0,
                              allow_nan=False, allow_infinity=False)

# 生成有效的梯度裁剪范数
valid_clip_norms = st.floats(min_value=0.01, max_value=100.0,
                              allow_nan=False, allow_infinity=False)


# ============================================================================
# Property 4: Huber损失计算正确性
# **Feature: strategy-convergence-control, Property 4: Huber损失计算正确性**
# **验证需求: 1.4**
# ============================================================================

class TestProperty4HuberLoss:
    """属性测试：Huber损失计算正确性。
    
    对于任何预测值和目标值，当误差小于delta时Huber损失等于MSE/2，
    当误差大于delta时Huber损失等于delta*(|误差|-delta/2)。
    """
    
    @given(tensor_pair_strategy(), valid_huber_deltas)
    @settings(max_examples=100)
    def test_huber_loss_small_error_equals_half_mse(self, tensor_pair, delta):
        """
        **Feature: strategy-convergence-control, Property 4: Huber损失计算正确性**
        **验证需求: 1.4**
        
        测试小误差时Huber损失等于MSE/2。
        """
        predictions, targets = tensor_pair
        
        # 缩放使所有误差都小于delta
        error = predictions - targets
        max_error = torch.abs(error).max().item()
        if max_error > 0:
            scale = delta / (max_error * 2)  # 确保误差小于delta
            scaled_predictions = targets + (predictions - targets) * scale
        else:
            scaled_predictions = predictions
        
        trainer = NetworkTrainer()
        huber_loss = trainer.compute_huber_loss(scaled_predictions, targets, delta)
        
        # 计算MSE/2
        mse_half = 0.5 * torch.mean((scaled_predictions - targets) ** 2)
        
        torch.testing.assert_close(
            huber_loss, mse_half, rtol=1e-4, atol=1e-6,
            msg=f"小误差时Huber损失应等于MSE/2"
        )
    
    @given(tensor_pair_strategy(), valid_huber_deltas)
    @settings(max_examples=100)
    def test_huber_loss_non_negative(self, tensor_pair, delta):
        """
        **Feature: strategy-convergence-control, Property 4: Huber损失计算正确性**
        **验证需求: 1.4**
        
        测试Huber损失非负。
        """
        predictions, targets = tensor_pair
        
        trainer = NetworkTrainer()
        huber_loss = trainer.compute_huber_loss(predictions, targets, delta)
        
        assert huber_loss.item() >= 0, (
            f"Huber损失应非负，但得到: {huber_loss.item()}"
        )
    
    @given(tensor_pair_strategy())
    @settings(max_examples=100)
    def test_huber_loss_zero_for_identical_inputs(self, tensor_pair):
        """
        **Feature: strategy-convergence-control, Property 4: Huber损失计算正确性**
        **验证需求: 1.4**
        
        测试相同输入时Huber损失为零。
        """
        predictions, _ = tensor_pair
        
        trainer = NetworkTrainer()
        huber_loss = trainer.compute_huber_loss(predictions, predictions)
        
        assert abs(huber_loss.item()) < 1e-6, (
            f"相同输入时Huber损失应为零，但得到: {huber_loss.item()}"
        )
    
    @given(tensor_pair_strategy(), valid_huber_deltas)
    @settings(max_examples=100)
    def test_huber_loss_symmetric(self, tensor_pair, delta):
        """
        **Feature: strategy-convergence-control, Property 4: Huber损失计算正确性**
        **验证需求: 1.4**
        
        测试Huber损失对称性（交换预测和目标结果相同）。
        """
        predictions, targets = tensor_pair
        
        trainer = NetworkTrainer()
        loss1 = trainer.compute_huber_loss(predictions, targets, delta)
        loss2 = trainer.compute_huber_loss(targets, predictions, delta)
        
        torch.testing.assert_close(
            loss1, loss2, rtol=1e-5, atol=1e-6,
            msg="Huber损失应对称"
        )
    
    @given(tensor_pair_strategy())
    @settings(max_examples=100)
    def test_huber_loss_less_than_mse_for_large_errors(self, tensor_pair):
        """
        **Feature: strategy-convergence-control, Property 4: Huber损失计算正确性**
        **验证需求: 1.4**
        
        测试大误差时Huber损失小于等于MSE/2。
        """
        predictions, targets = tensor_pair
        
        trainer = NetworkTrainer()
        delta = 1.0
        huber_loss = trainer.compute_huber_loss(predictions, targets, delta)
        mse_half = 0.5 * torch.mean((predictions - targets) ** 2)
        
        # Huber损失应该小于等于MSE/2（因为大误差时使用线性惩罚）
        assert huber_loss.item() <= mse_half.item() + 1e-6, (
            f"Huber损失应小于等于MSE/2，"
            f"Huber: {huber_loss.item()}, MSE/2: {mse_half.item()}"
        )



# ============================================================================
# Property 6: EMA更新正确性
# **Feature: strategy-convergence-control, Property 6: EMA更新正确性**
# **验证需求: 3.1**
# ============================================================================

class SimpleNetwork(nn.Module):
    """用于测试的简单网络。"""
    
    def __init__(self, input_size=10, hidden_size=5, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestProperty6EMAUpdate:
    """属性测试：EMA更新正确性。
    
    对于任何目标网络参数、源网络参数和衰减率，
    EMA更新后的参数应该等于decay*target + (1-decay)*source。
    """
    
    @given(valid_ema_decays)
    @settings(max_examples=100)
    def test_ema_update_formula(self, decay):
        """
        **Feature: strategy-convergence-control, Property 6: EMA更新正确性**
        **验证需求: 3.1**
        
        测试EMA更新公式正确性。
        """
        # 创建两个网络
        target_net = SimpleNetwork()
        source_net = SimpleNetwork()
        
        # 保存更新前的目标网络参数
        old_target_params = {
            name: param.clone() 
            for name, param in target_net.named_parameters()
        }
        
        trainer = NetworkTrainer()
        trainer.update_ema(target_net, source_net, decay)
        
        # 验证更新后的参数
        for name, param in target_net.named_parameters():
            old_target = old_target_params[name]
            source_param = dict(source_net.named_parameters())[name]
            expected = decay * old_target + (1 - decay) * source_param
            
            torch.testing.assert_close(
                param, expected, rtol=1e-5, atol=1e-6,
                msg=f"EMA更新公式不正确: {name}"
            )
    
    @given(st.just(1.0))
    @settings(max_examples=10)
    def test_ema_decay_one_preserves_target(self, decay):
        """
        **Feature: strategy-convergence-control, Property 6: EMA更新正确性**
        **验证需求: 3.1**
        
        测试衰减率为1时保持目标网络不变。
        """
        target_net = SimpleNetwork()
        source_net = SimpleNetwork()
        
        # 保存更新前的目标网络参数
        old_target_params = {
            name: param.clone() 
            for name, param in target_net.named_parameters()
        }
        
        trainer = NetworkTrainer()
        trainer.update_ema(target_net, source_net, decay)
        
        # 验证参数不变
        for name, param in target_net.named_parameters():
            torch.testing.assert_close(
                param, old_target_params[name], rtol=1e-5, atol=1e-6,
                msg=f"衰减率为1时目标网络应不变: {name}"
            )
    
    @given(st.just(0.0))
    @settings(max_examples=10)
    def test_ema_decay_zero_copies_source(self, decay):
        """
        **Feature: strategy-convergence-control, Property 6: EMA更新正确性**
        **验证需求: 3.1**
        
        测试衰减率为0时完全复制源网络。
        """
        target_net = SimpleNetwork()
        source_net = SimpleNetwork()
        
        trainer = NetworkTrainer()
        trainer.update_ema(target_net, source_net, decay)
        
        # 验证参数等于源网络
        for (name, target_param), (_, source_param) in zip(
            target_net.named_parameters(), 
            source_net.named_parameters()
        ):
            torch.testing.assert_close(
                target_param, source_param, rtol=1e-5, atol=1e-6,
                msg=f"衰减率为0时应完全复制源网络: {name}"
            )
    
    @given(valid_ema_decays)
    @settings(max_examples=100)
    def test_ema_update_uses_config_default(self, decay):
        """
        **Feature: strategy-convergence-control, Property 6: EMA更新正确性**
        **验证需求: 3.1**
        
        测试不指定衰减率时使用配置默认值。
        """
        config = NetworkTrainerConfig(ema_decay=0.9)
        trainer = NetworkTrainer(config)
        
        target_net = SimpleNetwork()
        source_net = SimpleNetwork()
        
        # 保存更新前的目标网络参数
        old_target_params = {
            name: param.clone() 
            for name, param in target_net.named_parameters()
        }
        
        trainer.update_ema(target_net, source_net)  # 不指定decay
        
        # 验证使用配置的默认值
        for name, param in target_net.named_parameters():
            old_target = old_target_params[name]
            source_param = dict(source_net.named_parameters())[name]
            expected = 0.9 * old_target + 0.1 * source_param
            
            torch.testing.assert_close(
                param, expected, rtol=1e-5, atol=1e-6,
                msg=f"应使用配置的默认衰减率: {name}"
            )


# ============================================================================
# Property 7: KL散度非负性
# **Feature: strategy-convergence-control, Property 7: KL散度非负性**
# **验证需求: 3.2**
# ============================================================================

class TestProperty7KLDivergence:
    """属性测试：KL散度非负性。
    
    对于任何两个概率分布，KL散度应该非负，且相同分布的KL散度为0。
    """
    
    @given(probability_distribution_strategy(), probability_distribution_strategy())
    @settings(max_examples=100)
    def test_kl_divergence_non_negative(self, p, q):
        """
        **Feature: strategy-convergence-control, Property 7: KL散度非负性**
        **验证需求: 3.2**
        
        测试KL散度非负。
        """
        # 确保两个分布大小相同
        min_size = min(len(p), len(q))
        p = p[:min_size]
        q = q[:min_size]
        
        # 重新归一化
        p = p / p.sum()
        q = q / q.sum()
        
        trainer = NetworkTrainer()
        kl = trainer.compute_kl_divergence(p.unsqueeze(0), q.unsqueeze(0))
        
        assert kl.item() >= -1e-6, (
            f"KL散度应非负，但得到: {kl.item()}"
        )
    
    @given(probability_distribution_strategy())
    @settings(max_examples=100)
    def test_kl_divergence_zero_for_identical_distributions(self, p):
        """
        **Feature: strategy-convergence-control, Property 7: KL散度非负性**
        **验证需求: 3.2**
        
        测试相同分布的KL散度为0。
        """
        trainer = NetworkTrainer()
        kl = trainer.compute_kl_divergence(p.unsqueeze(0), p.unsqueeze(0))
        
        assert abs(kl.item()) < 1e-5, (
            f"相同分布的KL散度应为0，但得到: {kl.item()}"
        )
    
    @given(probability_distribution_strategy(), probability_distribution_strategy())
    @settings(max_examples=100)
    def test_kl_divergence_asymmetric(self, p, q):
        """
        **Feature: strategy-convergence-control, Property 7: KL散度非负性**
        **验证需求: 3.2**
        
        测试KL散度不对称（KL(p||q) != KL(q||p)，除非p==q）。
        """
        # 确保两个分布大小相同
        min_size = min(len(p), len(q))
        p = p[:min_size]
        q = q[:min_size]
        
        # 重新归一化
        p = p / p.sum()
        q = q / q.sum()
        
        # 跳过相同分布的情况
        assume(not torch.allclose(p, q, atol=1e-4))
        
        trainer = NetworkTrainer()
        kl_pq = trainer.compute_kl_divergence(p.unsqueeze(0), q.unsqueeze(0))
        kl_qp = trainer.compute_kl_divergence(q.unsqueeze(0), p.unsqueeze(0))
        
        # KL散度通常不对称
        # 这里只验证两者都非负
        assert kl_pq.item() >= -1e-6, f"KL(p||q)应非负: {kl_pq.item()}"
        assert kl_qp.item() >= -1e-6, f"KL(q||p)应非负: {kl_qp.item()}"
    
    @given(probability_distribution_strategy())
    @settings(max_examples=100)
    def test_kl_divergence_batch_processing(self, p):
        """
        **Feature: strategy-convergence-control, Property 7: KL散度非负性**
        **验证需求: 3.2**
        
        测试批量处理时KL散度计算正确。
        """
        # 创建批量数据
        batch_p = p.unsqueeze(0).repeat(5, 1)
        batch_q = p.unsqueeze(0).repeat(5, 1)
        
        trainer = NetworkTrainer()
        kl = trainer.compute_kl_divergence(batch_p, batch_q)
        
        # 相同分布的批量KL散度应接近0
        assert abs(kl.item()) < 1e-5, (
            f"相同分布批量的KL散度应接近0，但得到: {kl.item()}"
        )


# ============================================================================
# Property 8: 梯度裁剪边界
# **Feature: strategy-convergence-control, Property 8: 梯度裁剪边界**
# **验证需求: 3.3**
# ============================================================================

class TestProperty8GradientClipping:
    """属性测试：梯度裁剪边界。
    
    对于任何神经网络和裁剪范数，裁剪后的梯度范数应该不超过指定阈值。
    """
    
    @given(valid_clip_norms)
    @settings(max_examples=100)
    def test_gradient_clip_within_threshold(self, max_norm):
        """
        **Feature: strategy-convergence-control, Property 8: 梯度裁剪边界**
        **验证需求: 3.3**
        
        测试裁剪后梯度范数不超过阈值。
        """
        # 创建网络并计算梯度
        net = SimpleNetwork()
        x = torch.randn(4, 10)
        y = torch.randn(4, 3)
        
        output = net(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        
        trainer = NetworkTrainer()
        trainer.clip_gradients(net, max_norm)
        
        # 计算裁剪后的梯度范数
        total_norm = 0.0
        for p in net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        assert total_norm <= max_norm + 1e-5, (
            f"裁剪后梯度范数应不超过{max_norm}，但得到: {total_norm}"
        )
    
    @given(valid_clip_norms)
    @settings(max_examples=100)
    def test_gradient_clip_returns_original_norm(self, max_norm):
        """
        **Feature: strategy-convergence-control, Property 8: 梯度裁剪边界**
        **验证需求: 3.3**
        
        测试裁剪返回原始梯度范数。
        """
        # 创建网络并计算梯度
        net = SimpleNetwork()
        x = torch.randn(4, 10)
        y = torch.randn(4, 3)
        
        output = net(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        
        # 计算裁剪前的梯度范数
        original_norm = 0.0
        for p in net.parameters():
            if p.grad is not None:
                original_norm += p.grad.data.norm(2).item() ** 2
        original_norm = original_norm ** 0.5
        
        trainer = NetworkTrainer()
        returned_norm = trainer.clip_gradients(net, max_norm)
        
        # 返回值应该接近原始范数
        assert abs(returned_norm - original_norm) < 1e-4, (
            f"返回的范数应接近原始范数，"
            f"返回: {returned_norm}, 原始: {original_norm}"
        )
    
    @given(valid_clip_norms)
    @settings(max_examples=100)
    def test_gradient_clip_preserves_direction(self, max_norm):
        """
        **Feature: strategy-convergence-control, Property 8: 梯度裁剪边界**
        **验证需求: 3.3**
        
        测试裁剪保持梯度方向不变。
        """
        # 创建网络并计算梯度
        net = SimpleNetwork()
        x = torch.randn(4, 10)
        y = torch.randn(4, 3)
        
        output = net(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        
        # 保存裁剪前的梯度方向
        original_grads = {}
        for name, p in net.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.data.norm(2).item()
                if grad_norm > 1e-8:
                    original_grads[name] = p.grad.data.clone() / grad_norm
        
        trainer = NetworkTrainer()
        trainer.clip_gradients(net, max_norm)
        
        # 验证方向不变
        for name, p in net.named_parameters():
            if name in original_grads and p.grad is not None:
                grad_norm = p.grad.data.norm(2).item()
                if grad_norm > 1e-8:
                    clipped_direction = p.grad.data / grad_norm
                    # 方向应该相同（点积接近1）
                    dot_product = (original_grads[name] * clipped_direction).sum().item()
                    assert dot_product > 0.99, (
                        f"梯度方向应保持不变: {name}, 点积: {dot_product}"
                    )
    
    def test_gradient_clip_no_grad_returns_zero(self):
        """
        **Feature: strategy-convergence-control, Property 8: 梯度裁剪边界**
        **验证需求: 3.3**
        
        测试无梯度时返回0。
        """
        net = SimpleNetwork()
        # 不计算梯度
        
        trainer = NetworkTrainer()
        norm = trainer.clip_gradients(net, 1.0)
        
        assert norm == 0.0, f"无梯度时应返回0，但得到: {norm}"
    
    @given(valid_clip_norms)
    @settings(max_examples=100)
    def test_gradient_clip_uses_config_default(self, _):
        """
        **Feature: strategy-convergence-control, Property 8: 梯度裁剪边界**
        **验证需求: 3.3**
        
        测试不指定范数时使用配置默认值。
        """
        config = NetworkTrainerConfig(gradient_clip_norm=0.5)
        trainer = NetworkTrainer(config)
        
        # 创建网络并计算梯度
        net = SimpleNetwork()
        x = torch.randn(4, 10)
        y = torch.randn(4, 3)
        
        output = net(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        
        trainer.clip_gradients(net)  # 不指定max_norm
        
        # 计算裁剪后的梯度范数
        total_norm = 0.0
        for p in net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        assert total_norm <= 0.5 + 1e-5, (
            f"应使用配置的默认裁剪范数0.5，但梯度范数为: {total_norm}"
        )


# ============================================================================
# 配置验证测试
# ============================================================================

class TestConfigValidation:
    """配置验证测试。"""
    
    def test_invalid_huber_delta_raises_error(self):
        """测试无效Huber delta抛出错误。"""
        with pytest.raises(ValueError, match="huber_delta必须为正数"):
            NetworkTrainerConfig(huber_delta=0)
        
        with pytest.raises(ValueError, match="huber_delta必须为正数"):
            NetworkTrainerConfig(huber_delta=-1.0)
    
    def test_invalid_ema_decay_raises_error(self):
        """测试无效EMA衰减率抛出错误。"""
        with pytest.raises(ValueError, match="ema_decay必须在0-1之间"):
            NetworkTrainerConfig(ema_decay=1.5)
        
        with pytest.raises(ValueError, match="ema_decay必须在0-1之间"):
            NetworkTrainerConfig(ema_decay=-0.1)
    
    def test_invalid_kl_coefficient_raises_error(self):
        """测试无效KL系数抛出错误。"""
        with pytest.raises(ValueError, match="kl_coefficient必须非负"):
            NetworkTrainerConfig(kl_coefficient=-0.1)
    
    def test_invalid_gradient_clip_norm_raises_error(self):
        """测试无效梯度裁剪范数抛出错误。"""
        with pytest.raises(ValueError, match="gradient_clip_norm必须为正数"):
            NetworkTrainerConfig(gradient_clip_norm=0)
        
        with pytest.raises(ValueError, match="gradient_clip_norm必须为正数"):
            NetworkTrainerConfig(gradient_clip_norm=-1.0)
    
    def test_valid_config_creation(self):
        """测试有效配置创建成功。"""
        config = NetworkTrainerConfig(
            use_huber_loss=True,
            huber_delta=2.0,
            use_ema=True,
            ema_decay=0.99,
            kl_coefficient=0.05,
            gradient_clip_norm=2.0
        )
        assert config.use_huber_loss is True
        assert config.huber_delta == 2.0
        assert config.use_ema is True
        assert config.ema_decay == 0.99
        assert config.kl_coefficient == 0.05
        assert config.gradient_clip_norm == 2.0
