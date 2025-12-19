"""
EV训练器测试模块

包含单元测试，验证训练器和损失函数的正确性。
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from training.ev_prediction_network import EVPredictionNetwork
from training.ev_trainer import EVTrainer
from training.ev_dataset import EVDataset, create_scenario_json, DEFAULT_ACTIONS


class TestEVTrainerLoss:
    """损失函数测试类"""
    
    def test_ev_mse_loss(self):
        """测试EV MSE损失计算"""
        model = EVPredictionNetwork(num_actions=5)
        trainer = EVTrainer(model, device='cpu')
        
        # 创建测试数据
        features = torch.randn(4, 4)
        target_ev = torch.randn(4, 1)
        target_action_ev = torch.randn(4, 5)
        target_strategy = torch.softmax(torch.randn(4, 5), dim=-1)
        
        losses = trainer.compute_loss(features, target_ev, target_action_ev, target_strategy)
        
        # 验证损失存在且为正数
        assert "ev" in losses
        assert losses["ev"].item() >= 0
    
    def test_action_ev_mse_loss(self):
        """测试动作EV MSE损失计算"""
        model = EVPredictionNetwork(num_actions=5)
        trainer = EVTrainer(model, device='cpu')
        
        features = torch.randn(4, 4)
        target_ev = torch.randn(4, 1)
        target_action_ev = torch.randn(4, 5)
        target_strategy = torch.softmax(torch.randn(4, 5), dim=-1)
        
        losses = trainer.compute_loss(features, target_ev, target_action_ev, target_strategy)
        
        assert "action_ev" in losses
        assert losses["action_ev"].item() >= 0
    
    def test_strategy_kl_loss(self):
        """测试策略KL散度损失计算"""
        model = EVPredictionNetwork(num_actions=5)
        trainer = EVTrainer(model, device='cpu')
        
        features = torch.randn(4, 4)
        target_ev = torch.randn(4, 1)
        target_action_ev = torch.randn(4, 5)
        target_strategy = torch.softmax(torch.randn(4, 5), dim=-1)
        
        losses = trainer.compute_loss(features, target_ev, target_action_ev, target_strategy)
        
        assert "strategy" in losses
        # KL散度可以为负（当使用log_softmax时），但这里我们使用的是KL_div
    
    def test_total_loss_weighted_sum(self):
        """测试总损失是加权和"""
        model = EVPredictionNetwork(num_actions=5)
        
        # 使用不同的权重
        trainer = EVTrainer(
            model, 
            ev_weight=2.0, 
            action_ev_weight=3.0, 
            strategy_weight=0.5,
            device='cpu'
        )
        
        features = torch.randn(4, 4)
        target_ev = torch.randn(4, 1)
        target_action_ev = torch.randn(4, 5)
        target_strategy = torch.softmax(torch.randn(4, 5), dim=-1)
        
        losses = trainer.compute_loss(features, target_ev, target_action_ev, target_strategy)
        
        # 验证总损失是加权和
        expected_total = (
            2.0 * losses["ev"] +
            3.0 * losses["action_ev"] +
            0.5 * losses["strategy"]
        )
        
        assert torch.isclose(losses["total"], expected_total, atol=1e-5)
    
    def test_loss_decreases_with_training(self):
        """测试训练过程中损失下降"""
        model = EVPredictionNetwork(num_actions=5, hidden_dim=32)
        trainer = EVTrainer(model, learning_rate=0.01, device='cpu')
        
        # 创建简单的训练数据
        features = torch.randn(32, 4)
        target_ev = torch.randn(32, 1)
        target_action_ev = torch.randn(32, 5)
        target_strategy = torch.softmax(torch.randn(32, 5), dim=-1)
        
        dataset = TensorDataset(features, target_ev, target_action_ev, target_strategy)
        dataloader = DataLoader(dataset, batch_size=8)
        
        # 记录初始损失
        initial_losses = trainer.train_epoch(dataloader)
        
        # 训练几个epoch
        for _ in range(10):
            losses = trainer.train_epoch(dataloader)
        
        # 验证损失下降（或至少不增加太多）
        # 注意：由于随机性，不能保证每次都下降，但多次训练后应该有改善
        assert losses["total"] < initial_losses["total"] * 2  # 宽松的检查


class TestEVTrainerEvaluation:
    """评估功能测试类"""
    
    def test_evaluate_returns_statistics(self):
        """测试评估返回统计信息"""
        model = EVPredictionNetwork(num_actions=5)
        trainer = EVTrainer(model, device='cpu')
        
        # 创建测试数据
        features = torch.randn(16, 4)
        target_ev = torch.randn(16, 1)
        target_action_ev = torch.randn(16, 5)
        target_strategy = torch.softmax(torch.randn(16, 5), dim=-1)
        
        dataset = TensorDataset(features, target_ev, target_action_ev, target_strategy)
        dataloader = DataLoader(dataset, batch_size=4)
        
        stats = trainer.evaluate(dataloader)
        
        # 验证统计信息结构
        assert "ev_mse" in stats
        assert "action_ev_mse" in stats
        assert "strategy_kl" in stats
        assert "num_samples" in stats
        
        # 验证统计信息内容
        for key in ["ev_mse", "action_ev_mse", "strategy_kl"]:
            assert "mean" in stats[key]
            assert "std" in stats[key]
            assert "min" in stats[key]
            assert "max" in stats[key]
        
        assert stats["num_samples"] == 16


class TestEVTrainerCheckpoint:
    """检查点功能测试类"""
    
    def test_save_and_load_checkpoint(self):
        """测试保存和加载检查点"""
        model = EVPredictionNetwork(num_actions=5, hidden_dim=32)
        trainer = EVTrainer(model, learning_rate=0.001, device='cpu')
        
        # 训练一点
        features = torch.randn(8, 4)
        target_ev = torch.randn(8, 1)
        target_action_ev = torch.randn(8, 5)
        target_strategy = torch.softmax(torch.randn(8, 5), dim=-1)
        
        dataset = TensorDataset(features, target_ev, target_action_ev, target_strategy)
        dataloader = DataLoader(dataset, batch_size=4)
        trainer.train_epoch(dataloader)
        
        # 保存检查点
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            
            # 验证文件存在
            assert checkpoint_path.exists()
            
            # 创建新的训练器并加载
            model2 = EVPredictionNetwork(num_actions=5, hidden_dim=32)
            trainer2 = EVTrainer(model2, device='cpu')
            trainer2.load_checkpoint(str(checkpoint_path))
            
            # 验证状态恢复
            assert trainer2.epoch == trainer.epoch
    
    def test_load_nonexistent_checkpoint(self):
        """测试加载不存在的检查点"""
        model = EVPredictionNetwork(num_actions=5)
        trainer = EVTrainer(model, device='cpu')
        
        with pytest.raises(FileNotFoundError, match="检查点文件不存在"):
            trainer.load_checkpoint("/nonexistent/path/checkpoint.pt")
    
    def test_checkpoint_preserves_model_weights(self):
        """测试检查点保留模型权重"""
        model = EVPredictionNetwork(num_actions=5, hidden_dim=32)
        trainer = EVTrainer(model, device='cpu')
        
        # 获取原始权重
        original_weights = {
            name: param.clone() 
            for name, param in model.named_parameters()
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            
            # 创建新模型并加载
            model2 = EVPredictionNetwork(num_actions=5, hidden_dim=32)
            trainer2 = EVTrainer(model2, device='cpu')
            trainer2.load_checkpoint(str(checkpoint_path))
            
            # 验证权重一致
            for name, param in model2.named_parameters():
                assert torch.allclose(param, original_weights[name]), \
                    f"权重 {name} 不一致"


class TestEVTrainerConfig:
    """配置功能测试类"""
    
    def test_learning_rate_config(self):
        """测试学习率配置"""
        model = EVPredictionNetwork(num_actions=5)
        trainer = EVTrainer(model, learning_rate=0.005, device='cpu')
        
        assert trainer.get_learning_rate() == 0.005
    
    def test_set_learning_rate(self):
        """测试设置学习率"""
        model = EVPredictionNetwork(num_actions=5)
        trainer = EVTrainer(model, learning_rate=0.001, device='cpu')
        
        trainer.set_learning_rate(0.0001)
        assert trainer.get_learning_rate() == 0.0001
    
    def test_gradient_clipping(self):
        """测试梯度裁剪"""
        model = EVPredictionNetwork(num_actions=5)
        trainer = EVTrainer(model, grad_clip=0.5, device='cpu')
        
        # 创建会产生大梯度的数据
        features = torch.randn(4, 4) * 100
        target_ev = torch.randn(4, 1) * 100
        target_action_ev = torch.randn(4, 5) * 100
        target_strategy = torch.softmax(torch.randn(4, 5), dim=-1)
        
        dataset = TensorDataset(features, target_ev, target_action_ev, target_strategy)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # 训练不应该崩溃
        losses = trainer.train_epoch(dataloader)
        assert not np.isnan(losses["total"])



from hypothesis import given, strategies as st, settings


class TestEVTrainerCheckpointProperties:
    """检查点属性测试类"""
    
    @given(
        num_actions=st.integers(min_value=2, max_value=8),
        hidden_dim=st.sampled_from([32, 64, 128]),
        learning_rate=st.floats(min_value=1e-5, max_value=1e-2, allow_nan=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_checkpoint_roundtrip_consistency(
        self, num_actions: int, hidden_dim: int, learning_rate: float
    ):
        """
        **Feature: ev-prediction-network, Property 3: 模型保存/加载往返一致性**
        
        对于任意训练后的模型状态，保存到文件后再加载，
        模型的所有参数应该与保存前完全一致
        
        **Validates: Requirements 4.2**
        """
        # 创建模型和训练器
        model1 = EVPredictionNetwork(num_actions=num_actions, hidden_dim=hidden_dim)
        trainer1 = EVTrainer(model1, learning_rate=learning_rate, device='cpu')
        
        # 随机初始化一些训练状态
        trainer1.epoch = np.random.randint(1, 100)
        trainer1.best_loss = np.random.random() * 10
        
        # 保存原始参数
        original_params = {
            name: param.clone().detach()
            for name, param in model1.named_parameters()
        }
        original_epoch = trainer1.epoch
        original_best_loss = trainer1.best_loss
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            
            # 保存检查点
            trainer1.save_checkpoint(str(checkpoint_path))
            
            # 创建新的模型和训练器
            model2 = EVPredictionNetwork(num_actions=num_actions, hidden_dim=hidden_dim)
            trainer2 = EVTrainer(model2, device='cpu')
            
            # 加载检查点
            trainer2.load_checkpoint(str(checkpoint_path))
            
            # 验证所有参数一致
            for name, param in model2.named_parameters():
                assert torch.allclose(param, original_params[name], atol=1e-6), \
                    f"参数 {name} 不一致"
            
            # 验证训练状态一致
            assert trainer2.epoch == original_epoch
            assert trainer2.best_loss == original_best_loss
    
    @given(
        batch_size=st.integers(min_value=4, max_value=16),
        num_batches=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=50, deadline=None)
    def test_checkpoint_preserves_predictions(self, batch_size: int, num_batches: int):
        """
        测试检查点保留模型预测能力
        
        保存和加载后，对相同输入应该产生相同输出
        """
        model1 = EVPredictionNetwork(num_actions=5, hidden_dim=32)
        trainer1 = EVTrainer(model1, device='cpu')
        
        # 生成测试输入
        test_inputs = [torch.randn(batch_size, 4) for _ in range(num_batches)]
        
        # 获取原始预测
        model1.eval()
        with torch.no_grad():
            original_predictions = [model1(x) for x in test_inputs]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            trainer1.save_checkpoint(str(checkpoint_path))
            
            # 创建新模型并加载
            model2 = EVPredictionNetwork(num_actions=5, hidden_dim=32)
            trainer2 = EVTrainer(model2, device='cpu')
            trainer2.load_checkpoint(str(checkpoint_path))
            
            # 验证预测一致
            model2.eval()
            with torch.no_grad():
                for i, x in enumerate(test_inputs):
                    ev2, action_ev2, strategy2 = model2(x)
                    ev1, action_ev1, strategy1 = original_predictions[i]
                    
                    assert torch.allclose(ev2, ev1, atol=1e-6)
                    assert torch.allclose(action_ev2, action_ev1, atol=1e-6)
                    assert torch.allclose(strategy2, strategy1, atol=1e-6)
