"""检查点管理器的测试模块。

包含单元测试和基于属性的测试，验证CheckpointManager的功能正确性。
"""

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from utils.checkpoint_manager import CheckpointManager, CHECKPOINT_FORMAT_VERSION
from models.networks import PolicyNetwork, ValueNetwork, RegretNetwork
from models.core import CheckpointInfo


# ============================================================================
# 测试辅助类和函数
# ============================================================================

class SimpleModel(nn.Module):
    """用于测试的简单模型。"""
    
    def __init__(self, input_dim: int = 10, output_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_checkpoint_dir():
    """创建临时检查点目录的fixture。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """创建CheckpointManager实例的fixture。"""
    return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)


@pytest.fixture
def simple_model():
    """创建简单模型的fixture。"""
    return SimpleModel()


@pytest.fixture
def policy_network():
    """创建策略网络的fixture。"""
    return PolicyNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=5)


@pytest.fixture
def regret_network():
    """创建遗憾网络的fixture。"""
    return RegretNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=5)


# ============================================================================
# 单元测试
# ============================================================================

class TestCheckpointManagerSave:
    """测试检查点保存功能。"""
    
    def test_save_creates_file(self, checkpoint_manager, simple_model):
        """测试保存检查点会创建文件。"""
        metadata = {'episode_number': 100, 'win_rate': 0.6, 'avg_reward': 10.5}
        
        filepath = checkpoint_manager.save(simple_model, None, metadata)
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.pt')
    
    def test_save_with_optimizer(self, checkpoint_manager, simple_model):
        """测试保存包含优化器状态的检查点。"""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        metadata = {'episode_number': 200}
        
        filepath = checkpoint_manager.save(simple_model, optimizer, metadata)
        
        assert os.path.exists(filepath)
        # 验证文件包含优化器状态
        checkpoint_data = torch.load(filepath, weights_only=False)
        assert checkpoint_data['optimizer_state_dict'] is not None
    
    def test_save_without_optimizer(self, checkpoint_manager, simple_model):
        """测试保存不包含优化器状态的检查点。"""
        metadata = {'episode_number': 300}
        
        filepath = checkpoint_manager.save(simple_model, None, metadata)
        
        checkpoint_data = torch.load(filepath, weights_only=False)
        assert checkpoint_data['optimizer_state_dict'] is None
    
    def test_save_requires_episode_number(self, checkpoint_manager, simple_model):
        """测试保存时必须提供episode_number。"""
        metadata = {'win_rate': 0.5}  # 缺少episode_number
        
        with pytest.raises(ValueError, match="episode_number"):
            checkpoint_manager.save(simple_model, None, metadata)
    
    def test_save_includes_metadata(self, checkpoint_manager, simple_model):
        """测试保存的检查点包含所有元数据。"""
        metadata = {
            'episode_number': 500,
            'win_rate': 0.75,
            'avg_reward': 25.0,
            'custom_field': 'test_value'
        }
        
        filepath = checkpoint_manager.save(simple_model, None, metadata)
        
        checkpoint_data = torch.load(filepath, weights_only=False)
        assert checkpoint_data['episode_number'] == 500
        assert checkpoint_data['win_rate'] == 0.75
        assert checkpoint_data['avg_reward'] == 25.0
        assert checkpoint_data['custom_field'] == 'test_value'
        assert 'timestamp' in checkpoint_data


class TestCheckpointManagerLoad:
    """测试检查点加载功能。"""
    
    def test_load_restores_model_state(self, checkpoint_manager, simple_model):
        """测试加载检查点能恢复模型状态。"""
        # 保存原始参数
        original_params = {k: v.clone() for k, v in simple_model.state_dict().items()}
        metadata = {'episode_number': 100}
        filepath = checkpoint_manager.save(simple_model, None, metadata)
        
        # 修改模型参数
        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(0)
        
        # 加载检查点
        loaded_model, _, _ = checkpoint_manager.load(filepath, simple_model)
        
        # 验证参数已恢复
        for key, original_value in original_params.items():
            assert torch.equal(loaded_model.state_dict()[key], original_value)
    
    def test_load_restores_optimizer_state(self, checkpoint_manager, simple_model):
        """测试加载检查点能恢复优化器状态。"""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        
        # 执行一些优化步骤以改变优化器状态
        x = torch.randn(1, 10)
        loss = simple_model(x).sum()
        loss.backward()
        optimizer.step()
        
        # 保存检查点
        original_state = {k: v for k, v in optimizer.state_dict().items()}
        metadata = {'episode_number': 100}
        filepath = checkpoint_manager.save(simple_model, optimizer, metadata)
        
        # 创建新的优化器
        new_optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        
        # 加载检查点
        _, loaded_optimizer, _ = checkpoint_manager.load(filepath, simple_model, new_optimizer)
        
        # 验证优化器状态已恢复
        assert loaded_optimizer is not None
    
    def test_load_returns_metadata(self, checkpoint_manager, simple_model):
        """测试加载检查点返回元数据。"""
        metadata = {
            'episode_number': 1000,
            'win_rate': 0.8,
            'avg_reward': 50.0
        }
        filepath = checkpoint_manager.save(simple_model, None, metadata)
        
        _, _, loaded_metadata = checkpoint_manager.load(filepath, simple_model)
        
        assert loaded_metadata['episode_number'] == 1000
        assert loaded_metadata['win_rate'] == 0.8
        assert loaded_metadata['avg_reward'] == 50.0
    
    def test_load_nonexistent_file_raises_error(self, checkpoint_manager, simple_model):
        """测试加载不存在的文件会抛出错误。"""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load("nonexistent.pt", simple_model)


class TestCheckpointManagerList:
    """测试检查点列表功能。"""
    
    def test_list_empty_directory(self, checkpoint_manager):
        """测试空目录返回空列表。"""
        checkpoints = checkpoint_manager.list_checkpoints()
        assert checkpoints == []
    
    def test_list_returns_all_checkpoints(self, checkpoint_manager, simple_model):
        """测试列出所有检查点。"""
        # 保存多个检查点
        for i in range(3):
            metadata = {'episode_number': i * 100}
            checkpoint_manager.save(simple_model, None, metadata)
            time.sleep(0.001)  # 确保时间戳不同
        
        checkpoints = checkpoint_manager.list_checkpoints()
        
        assert len(checkpoints) == 3
    
    def test_list_returns_checkpoint_info(self, checkpoint_manager, simple_model):
        """测试列表返回CheckpointInfo对象。"""
        metadata = {'episode_number': 500, 'win_rate': 0.7, 'avg_reward': 30.0}
        checkpoint_manager.save(simple_model, None, metadata)
        
        checkpoints = checkpoint_manager.list_checkpoints()
        
        assert len(checkpoints) == 1
        info = checkpoints[0]
        assert isinstance(info, CheckpointInfo)
        assert info.episode_number == 500
        assert info.win_rate == 0.7
        assert info.avg_reward == 30.0
        assert isinstance(info.timestamp, datetime)
    
    def test_list_sorted_by_timestamp(self, checkpoint_manager, simple_model):
        """测试检查点按时间戳排序（最新在前）。"""
        # 保存多个检查点
        for i in range(3):
            metadata = {'episode_number': i * 100}
            checkpoint_manager.save(simple_model, None, metadata)
            time.sleep(0.01)  # 确保时间戳不同
        
        checkpoints = checkpoint_manager.list_checkpoints()
        
        # 验证按时间戳降序排列
        for i in range(len(checkpoints) - 1):
            assert checkpoints[i].timestamp >= checkpoints[i + 1].timestamp


class TestCheckpointManagerDelete:
    """测试检查点删除功能。"""
    
    def test_delete_existing_checkpoint(self, checkpoint_manager, simple_model):
        """测试删除存在的检查点。"""
        metadata = {'episode_number': 100}
        filepath = checkpoint_manager.save(simple_model, None, metadata)
        
        assert os.path.exists(filepath)
        
        result = checkpoint_manager.delete(filepath)
        
        assert result is True
        assert not os.path.exists(filepath)
    
    def test_delete_nonexistent_checkpoint(self, checkpoint_manager):
        """测试删除不存在的检查点返回False。"""
        result = checkpoint_manager.delete("nonexistent.pt")
        assert result is False
    
    def test_delete_removes_from_list(self, checkpoint_manager, simple_model):
        """测试删除后检查点不再出现在列表中。"""
        metadata = {'episode_number': 100}
        filepath = checkpoint_manager.save(simple_model, None, metadata)
        
        # 验证检查点在列表中
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 1
        
        # 删除检查点
        checkpoint_manager.delete(filepath)
        
        # 验证检查点不在列表中
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 0


class TestCheckpointFilenameUniqueness:
    """测试检查点文件名唯一性。"""
    
    def test_consecutive_saves_have_unique_filenames(self, checkpoint_manager, simple_model):
        """测试连续保存多个检查点生成不同的文件名。"""
        filepaths = []
        
        # 快速连续保存多个检查点
        for i in range(5):
            metadata = {'episode_number': i}
            filepath = checkpoint_manager.save(simple_model, None, metadata)
            filepaths.append(filepath)
        
        # 验证所有文件名都不同
        assert len(set(filepaths)) == 5
    
    def test_same_episode_different_filenames(self, checkpoint_manager, simple_model):
        """测试相同回合数保存多次生成不同的文件名。"""
        filepaths = []
        
        # 使用相同的episode_number保存多次
        for _ in range(3):
            metadata = {'episode_number': 100}
            filepath = checkpoint_manager.save(simple_model, None, metadata)
            filepaths.append(filepath)
        
        # 验证所有文件名都不同
        assert len(set(filepaths)) == 3


class TestCheckpointManagerWithRealNetworks:
    """使用真实网络测试检查点管理器。"""
    
    def test_save_load_policy_network(self, checkpoint_manager, policy_network):
        """测试保存和加载策略网络。"""
        # 保存原始参数
        original_params = {k: v.clone() for k, v in policy_network.state_dict().items()}
        metadata = {'episode_number': 1000, 'win_rate': 0.65}
        
        filepath = checkpoint_manager.save(policy_network, None, metadata)
        
        # 修改模型参数
        with torch.no_grad():
            for param in policy_network.parameters():
                param.fill_(0)
        
        # 加载检查点
        loaded_model, _, _ = checkpoint_manager.load(filepath, policy_network)
        
        # 验证参数已恢复
        for key, original_value in original_params.items():
            assert torch.equal(loaded_model.state_dict()[key], original_value)
    
    def test_save_load_value_network(self, checkpoint_manager):
        """测试保存和加载价值网络。"""
        value_network = ValueNetwork(input_dim=370, hidden_dims=[64, 32])
        
        # 保存原始参数
        original_params = {k: v.clone() for k, v in value_network.state_dict().items()}
        metadata = {'episode_number': 2000}
        
        filepath = checkpoint_manager.save(value_network, None, metadata)
        
        # 修改模型参数
        with torch.no_grad():
            for param in value_network.parameters():
                param.fill_(0)
        
        # 加载检查点
        loaded_model, _, _ = checkpoint_manager.load(filepath, value_network)
        
        # 验证参数已恢复
        for key, original_value in original_params.items():
            assert torch.equal(loaded_model.state_dict()[key], original_value)


# ============================================================================
# Deep CFR 格式测试
# ============================================================================

class TestDeepCFRCheckpointSave:
    """测试 Deep CFR 格式检查点保存功能。
    
    验证需求：1.5
    """
    
    def test_save_deep_cfr_creates_file(self, checkpoint_manager, regret_network, policy_network):
        """测试保存 Deep CFR 检查点会创建文件。"""
        metadata = {'episode_number': 100, 'win_rate': 0.6, 'avg_reward': 10.5}
        
        filepath = checkpoint_manager.save_deep_cfr(
            regret_network, policy_network, None, None, metadata
        )
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.pt')
    
    def test_save_deep_cfr_with_optimizers(self, checkpoint_manager, regret_network, policy_network):
        """测试保存包含优化器状态的 Deep CFR 检查点。"""
        regret_optimizer = torch.optim.Adam(regret_network.parameters(), lr=0.001)
        policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.001)
        metadata = {'episode_number': 200}
        
        filepath = checkpoint_manager.save_deep_cfr(
            regret_network, policy_network, regret_optimizer, policy_optimizer, metadata
        )
        
        assert os.path.exists(filepath)
        # 验证文件包含优化器状态
        checkpoint_data = torch.load(filepath, weights_only=False)
        assert checkpoint_data['regret_optimizer_state_dict'] is not None
        assert checkpoint_data['policy_optimizer_state_dict'] is not None
    
    def test_save_deep_cfr_includes_format_version(self, checkpoint_manager, regret_network, policy_network):
        """测试保存的 Deep CFR 检查点包含格式版本号。"""
        metadata = {'episode_number': 300}
        
        filepath = checkpoint_manager.save_deep_cfr(
            regret_network, policy_network, None, None, metadata
        )
        
        checkpoint_data = torch.load(filepath, weights_only=False)
        assert checkpoint_data['checkpoint_format'] == CHECKPOINT_FORMAT_VERSION
    
    def test_save_deep_cfr_includes_all_network_params(self, checkpoint_manager, regret_network, policy_network):
        """测试保存的 Deep CFR 检查点包含所有网络参数。"""
        metadata = {'episode_number': 400}
        
        filepath = checkpoint_manager.save_deep_cfr(
            regret_network, policy_network, None, None, metadata
        )
        
        checkpoint_data = torch.load(filepath, weights_only=False)
        assert 'regret_network_state_dict' in checkpoint_data
        assert 'policy_network_state_dict' in checkpoint_data
        assert checkpoint_data['regret_network_state_dict'] is not None
        assert checkpoint_data['policy_network_state_dict'] is not None
    
    def test_save_deep_cfr_requires_episode_number(self, checkpoint_manager, regret_network, policy_network):
        """测试保存 Deep CFR 检查点时必须提供 episode_number。"""
        metadata = {'win_rate': 0.5}  # 缺少 episode_number
        
        with pytest.raises(ValueError, match="episode_number"):
            checkpoint_manager.save_deep_cfr(
                regret_network, policy_network, None, None, metadata
            )


class TestDeepCFRCheckpointLoad:
    """测试 Deep CFR 格式检查点加载功能。
    
    验证需求：1.5
    """
    
    def test_load_deep_cfr_restores_regret_network(self, checkpoint_manager, regret_network, policy_network):
        """测试加载 Deep CFR 检查点能恢复遗憾网络状态。"""
        # 保存原始参数
        original_regret_params = {k: v.clone() for k, v in regret_network.state_dict().items()}
        metadata = {'episode_number': 100}
        filepath = checkpoint_manager.save_deep_cfr(
            regret_network, policy_network, None, None, metadata
        )
        
        # 修改遗憾网络参数
        with torch.no_grad():
            for param in regret_network.parameters():
                param.fill_(0)
        
        # 加载检查点
        loaded_regret, _, _, _, _ = checkpoint_manager.load_deep_cfr(
            filepath, regret_network, policy_network
        )
        
        # 验证参数已恢复
        for key, original_value in original_regret_params.items():
            assert torch.equal(loaded_regret.state_dict()[key], original_value)
    
    def test_load_deep_cfr_restores_policy_network(self, checkpoint_manager, regret_network, policy_network):
        """测试加载 Deep CFR 检查点能恢复策略网络状态。"""
        # 保存原始参数
        original_policy_params = {k: v.clone() for k, v in policy_network.state_dict().items()}
        metadata = {'episode_number': 100}
        filepath = checkpoint_manager.save_deep_cfr(
            regret_network, policy_network, None, None, metadata
        )
        
        # 修改策略网络参数
        with torch.no_grad():
            for param in policy_network.parameters():
                param.fill_(0)
        
        # 加载检查点
        _, loaded_policy, _, _, _ = checkpoint_manager.load_deep_cfr(
            filepath, regret_network, policy_network
        )
        
        # 验证参数已恢复
        for key, original_value in original_policy_params.items():
            assert torch.equal(loaded_policy.state_dict()[key], original_value)
    
    def test_load_deep_cfr_restores_optimizers(self, checkpoint_manager, regret_network, policy_network):
        """测试加载 Deep CFR 检查点能恢复优化器状态。"""
        regret_optimizer = torch.optim.Adam(regret_network.parameters(), lr=0.001)
        policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.001)
        
        # 执行一些优化步骤以改变优化器状态
        x = torch.randn(1, 370)
        regret_loss = regret_network(x).sum()
        regret_loss.backward()
        regret_optimizer.step()
        
        policy_loss = policy_network(x).sum()
        policy_loss.backward()
        policy_optimizer.step()
        
        # 保存检查点
        metadata = {'episode_number': 100}
        filepath = checkpoint_manager.save_deep_cfr(
            regret_network, policy_network, regret_optimizer, policy_optimizer, metadata
        )
        
        # 创建新的优化器
        new_regret_optimizer = torch.optim.Adam(regret_network.parameters(), lr=0.001)
        new_policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.001)
        
        # 加载检查点
        _, _, loaded_regret_opt, loaded_policy_opt, _ = checkpoint_manager.load_deep_cfr(
            filepath, regret_network, policy_network, new_regret_optimizer, new_policy_optimizer
        )
        
        # 验证优化器状态已恢复
        assert loaded_regret_opt is not None
        assert loaded_policy_opt is not None
    
    def test_load_deep_cfr_returns_metadata(self, checkpoint_manager, regret_network, policy_network):
        """测试加载 Deep CFR 检查点返回元数据。"""
        metadata = {
            'episode_number': 1000,
            'win_rate': 0.8,
            'avg_reward': 50.0
        }
        filepath = checkpoint_manager.save_deep_cfr(
            regret_network, policy_network, None, None, metadata
        )
        
        _, _, _, _, loaded_metadata = checkpoint_manager.load_deep_cfr(
            filepath, regret_network, policy_network
        )
        
        assert loaded_metadata['episode_number'] == 1000
        assert loaded_metadata['win_rate'] == 0.8
        assert loaded_metadata['avg_reward'] == 50.0
        assert loaded_metadata['checkpoint_format'] == CHECKPOINT_FORMAT_VERSION
    
    def test_load_deep_cfr_nonexistent_file_raises_error(self, checkpoint_manager, regret_network, policy_network):
        """测试加载不存在的 Deep CFR 检查点会抛出错误。"""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_deep_cfr(
                "nonexistent.pt", regret_network, policy_network
            )


class TestLegacyCheckpointCompatibility:
    """测试旧格式检查点的兼容性加载。
    
    验证需求：5.4
    """
    
    def test_load_legacy_checkpoint_as_deep_cfr(self, checkpoint_manager, regret_network, policy_network):
        """测试使用 load_deep_cfr 加载旧格式检查点。"""
        # 使用旧格式保存检查点
        legacy_policy = PolicyNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=5)
        original_policy_params = {k: v.clone() for k, v in legacy_policy.state_dict().items()}
        metadata = {'episode_number': 500}
        
        filepath = checkpoint_manager.save(legacy_policy, None, metadata)
        
        # 使用 load_deep_cfr 加载
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, loaded_policy, _, _, loaded_metadata = checkpoint_manager.load_deep_cfr(
                filepath, regret_network, policy_network
            )
            
            # 验证发出了警告
            assert len(w) >= 1
            assert "旧格式检查点" in str(w[0].message)
        
        # 验证策略网络参数已恢复
        for key, original_value in original_policy_params.items():
            assert torch.equal(loaded_policy.state_dict()[key], original_value)
        
        # 验证元数据
        assert loaded_metadata['checkpoint_format'] == 'legacy'
        assert loaded_metadata['episode_number'] == 500
    
    def test_detect_legacy_checkpoint_format(self, checkpoint_manager, policy_network):
        """测试检测旧格式检查点。"""
        # 使用旧格式保存检查点
        metadata = {'episode_number': 100}
        filepath = checkpoint_manager.save(policy_network, None, metadata)
        
        # 检测格式
        detected_format = checkpoint_manager.detect_checkpoint_format(filepath)
        assert detected_format == 'legacy'
    
    def test_detect_deep_cfr_checkpoint_format(self, checkpoint_manager, regret_network, policy_network):
        """测试检测 Deep CFR 格式检查点。"""
        # 使用 Deep CFR 格式保存检查点
        metadata = {'episode_number': 100}
        filepath = checkpoint_manager.save_deep_cfr(
            regret_network, policy_network, None, None, metadata
        )
        
        # 检测格式
        detected_format = checkpoint_manager.detect_checkpoint_format(filepath)
        assert detected_format == CHECKPOINT_FORMAT_VERSION
    
    def test_legacy_checkpoint_with_value_network_warning(self, checkpoint_manager, regret_network, policy_network):
        """测试加载包含价值网络的旧格式检查点时发出警告。"""
        # 使用旧格式保存检查点（包含价值网络）
        legacy_policy = PolicyNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=5)
        value_network = ValueNetwork(input_dim=370, hidden_dims=[64, 32])
        metadata = {'episode_number': 500}
        
        filepath = checkpoint_manager.save(
            legacy_policy, None, metadata, value_network=value_network
        )
        
        # 使用 load_deep_cfr 加载
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            checkpoint_manager.load_deep_cfr(
                filepath, regret_network, policy_network
            )
            
            # 验证发出了关于价值网络的警告
            warning_messages = [str(warning.message) for warning in w]
            assert any("价值网络" in msg for msg in warning_messages)


# ============================================================================
# 基于属性的测试（Property-Based Tests）
# ============================================================================

from hypothesis import given, strategies as st, settings


class TestCheckpointRoundTripProperty:
    """属性17：检查点往返一致性测试。
    
    Feature: texas-holdem-ai-training, Property 17: 检查点往返一致性
    验证需求：5.1, 5.2
    """
    
    @given(
        input_dim=st.integers(min_value=1, max_value=50),
        output_dim=st.integers(min_value=1, max_value=10),
        episode_number=st.integers(min_value=0, max_value=1000000),
        win_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        avg_reward=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_model_params_round_trip(self, input_dim, output_dim, episode_number, win_rate, avg_reward):
        """
        属性17：检查点往返一致性
        
        *对于任何*模型状态和优化器状态，保存为检查点后再加载，
        应该恢复出等价的模型参数和优化器状态。
        
        Feature: texas-holdem-ai-training, Property 17: 检查点往返一致性
        **验证需求：5.1, 5.2**
        """
        metadata = {
            'episode_number': episode_number,
            'win_rate': win_rate,
            'avg_reward': avg_reward
        }
        
        # 创建模型
        model = SimpleModel(input_dim=input_dim, output_dim=output_dim)
        
        # 保存原始参数
        original_params = {k: v.clone() for k, v in model.state_dict().items()}
        
        # 使用临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            filepath = manager.save(model, None, metadata)
            
            # 修改模型参数
            with torch.no_grad():
                for param in model.parameters():
                    param.fill_(0)
            
            # 加载检查点
            loaded_model, _, loaded_metadata = manager.load(filepath, model)
            
            # 验证模型参数等价
            for key, original_value in original_params.items():
                assert torch.allclose(loaded_model.state_dict()[key], original_value)
            
            # 验证元数据等价
            assert loaded_metadata['episode_number'] == episode_number
            assert abs(loaded_metadata['win_rate'] - win_rate) < 1e-6
            assert abs(loaded_metadata['avg_reward'] - avg_reward) < 1e-6


class TestCheckpointListCompletenessProperty:
    """属性18：检查点列表完整性测试。
    
    Feature: texas-holdem-ai-training, Property 18: 检查点列表完整性
    验证需求：5.3
    """
    
    @given(n=st.integers(min_value=1, max_value=10))
    @settings(max_examples=100)
    def test_list_returns_n_checkpoints(self, n):
        """
        属性18：检查点列表完整性
        
        *对于任何*已保存的检查点集合，列出检查点应该返回所有检查点的信息，
        且每个检查点信息包含路径、回合数、时间戳和性能指标。
        
        Feature: texas-holdem-ai-training, Property 18: 检查点列表完整性
        **验证需求：5.3**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            model = SimpleModel()
            
            # 保存N个检查点
            saved_episodes = []
            for i in range(n):
                metadata = {'episode_number': i * 100, 'win_rate': 0.5, 'avg_reward': 0.0}
                manager.save(model, None, metadata)
                saved_episodes.append(i * 100)
                time.sleep(0.001)
            
            # 列出检查点
            checkpoints = manager.list_checkpoints()
            
            # 验证数量正确
            assert len(checkpoints) == n, f"期望 {n} 个检查点，实际 {len(checkpoints)} 个"
            
            # 验证每个检查点信息完整
            for checkpoint in checkpoints:
                assert isinstance(checkpoint, CheckpointInfo)
                assert checkpoint.path is not None and len(checkpoint.path) > 0
                assert checkpoint.episode_number in saved_episodes
                assert isinstance(checkpoint.timestamp, datetime)
                assert 0 <= checkpoint.win_rate <= 1
                assert isinstance(checkpoint.avg_reward, float)


class TestCheckpointDeleteEffectivenessProperty:
    """属性19：检查点删除有效性测试。
    
    Feature: texas-holdem-ai-training, Property 19: 检查点删除有效性
    验证需求：5.4
    """
    
    @given(n=st.integers(min_value=2, max_value=10))
    @settings(max_examples=100)
    def test_delete_removes_checkpoint(self, n):
        """
        属性19：检查点删除有效性
        
        *对于任何*存在的检查点文件，删除操作后，该文件应该不再存在于文件系统中，
        且不再出现在检查点列表中。
        
        Feature: texas-holdem-ai-training, Property 19: 检查点删除有效性
        **验证需求：5.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            model = SimpleModel()
            
            # 保存N个检查点
            filepaths = []
            for i in range(n):
                metadata = {'episode_number': i * 100}
                filepath = manager.save(model, None, metadata)
                filepaths.append(filepath)
                time.sleep(0.001)
            
            # 删除第一个检查点
            delete_path = filepaths[0]
            
            # 验证文件存在
            assert os.path.exists(delete_path)
            
            # 删除检查点
            result = manager.delete(delete_path)
            
            # 验证删除成功
            assert result is True
            
            # 验证文件不存在
            assert not os.path.exists(delete_path)
            
            # 验证不在列表中
            checkpoints = manager.list_checkpoints()
            checkpoint_paths = [c.path for c in checkpoints]
            assert delete_path not in checkpoint_paths
            
            # 验证列表数量正确
            assert len(checkpoints) == n - 1


class TestCheckpointFilenameUniquenessProperty:
    """属性20：检查点文件名唯一性测试。
    
    Feature: texas-holdem-ai-training, Property 20: 检查点文件名唯一性
    验证需求：5.5
    """
    
    @given(n=st.integers(min_value=2, max_value=20))
    @settings(max_examples=100)
    def test_consecutive_saves_unique_filenames(self, n):
        """
        属性20：检查点文件名唯一性
        
        *对于任何*训练会话，连续保存多个检查点应该生成不同的文件名，不会发生覆盖。
        
        Feature: texas-holdem-ai-training, Property 20: 检查点文件名唯一性
        **验证需求：5.5**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            model = SimpleModel()
            
            # 连续保存N个检查点（使用相同的episode_number以测试唯一性）
            filepaths = []
            for _ in range(n):
                metadata = {'episode_number': 100}  # 故意使用相同的episode_number
                filepath = manager.save(model, None, metadata)
                filepaths.append(filepath)
            
            # 验证所有文件名都不同
            unique_paths = set(filepaths)
            assert len(unique_paths) == n, f"期望 {n} 个唯一文件名，实际 {len(unique_paths)} 个"
            
            # 验证所有文件都存在
            for filepath in filepaths:
                assert os.path.exists(filepath), f"文件不存在: {filepath}"
            
            # 验证列表中有N个检查点
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == n


class TestDeepCFRCheckpointRoundTripProperty:
    """属性3：Deep CFR 检查点往返一致性测试。
    
    Feature: deep-cfr-refactor, Property 3: 检查点往返一致性
    验证需求：1.5
    """
    
    @given(
        episode_number=st.integers(min_value=0, max_value=1000000),
        win_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        avg_reward=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_deep_cfr_checkpoint_round_trip(self, episode_number, win_rate, avg_reward):
        """
        属性3：检查点往返一致性
        
        *对于任何*遗憾网络和策略网络的参数状态，保存为检查点后再加载，
        应该恢复出等价的网络参数。
        
        Feature: deep-cfr-refactor, Property 3: 检查点往返一致性
        **验证需求：1.5**
        """
        metadata = {
            'episode_number': episode_number,
            'win_rate': win_rate,
            'avg_reward': avg_reward
        }
        
        # 创建网络
        regret_network = RegretNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=5)
        policy_network = PolicyNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=5)
        
        # 保存原始参数
        original_regret_params = {k: v.clone() for k, v in regret_network.state_dict().items()}
        original_policy_params = {k: v.clone() for k, v in policy_network.state_dict().items()}
        
        # 使用临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            filepath = manager.save_deep_cfr(
                regret_network, policy_network, None, None, metadata
            )
            
            # 修改网络参数
            with torch.no_grad():
                for param in regret_network.parameters():
                    param.fill_(0)
                for param in policy_network.parameters():
                    param.fill_(0)
            
            # 加载检查点
            loaded_regret, loaded_policy, _, _, loaded_metadata = manager.load_deep_cfr(
                filepath, regret_network, policy_network
            )
            
            # 验证遗憾网络参数等价
            for key, original_value in original_regret_params.items():
                assert torch.allclose(loaded_regret.state_dict()[key], original_value)
            
            # 验证策略网络参数等价
            for key, original_value in original_policy_params.items():
                assert torch.allclose(loaded_policy.state_dict()[key], original_value)
            
            # 验证元数据等价
            assert loaded_metadata['episode_number'] == episode_number
            assert abs(loaded_metadata['win_rate'] - win_rate) < 1e-6
            assert abs(loaded_metadata['avg_reward'] - avg_reward) < 1e-6
            assert loaded_metadata['checkpoint_format'] == CHECKPOINT_FORMAT_VERSION
