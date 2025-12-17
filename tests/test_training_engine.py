"""训练引擎的单元测试和属性测试（Deep CFR架构）。

测试内容：
- 训练引擎初始化（Deep CFR架构）
- 自我对弈回合生成Episode
- Deep CFR训练循环
- 检查点保存和加载（新旧格式兼容性）
- 属性测试（训练初始化完整性、终止安全性、检查点往返一致性、配置参数应用）
"""

import os
import tempfile
import shutil
from pathlib import Path

import pytest
import torch
import numpy as np
from hypothesis import given, settings, strategies as st

from models.core import TrainingConfig, Episode, Action, ActionType
from training.training_engine import TrainingEngine


class TestTrainingEngineInit:
    """测试训练引擎初始化（Deep CFR架构）。"""
    
    def test_init_with_default_config(self):
        """测试使用默认配置初始化训练引擎。"""
        config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 验证Deep CFR组件已初始化
            assert engine.env is not None
            assert engine.state_encoder is not None
            assert engine.regret_network is not None  # Deep CFR: 遗憾网络
            assert engine.policy_network is not None  # Deep CFR: 策略网络
            assert engine.regret_optimizer is not None  # Deep CFR: 遗憾网络优化器
            assert engine.policy_optimizer is not None
            assert engine.regret_buffer is not None  # Deep CFR: 遗憾缓冲区
            assert engine.strategy_buffer is not None  # Deep CFR: 策略缓冲区
            assert engine.deep_cfr_trainer is not None  # Deep CFR训练器
            assert engine.checkpoint_manager is not None
            
            # 验证初始状态
            assert engine.current_episode == 0
            assert engine.win_count == 0
            assert len(engine.total_rewards) == 0
            assert engine.should_stop == False
    
    def test_init_with_custom_config(self):
        """测试使用自定义配置初始化训练引擎。"""
        config = TrainingConfig(
            learning_rate=0.0001,
            batch_size=64,
            num_episodes=5000,
            discount_factor=0.95,
            network_architecture=[256, 128],
            checkpoint_interval=500,
            initial_stack=2000,
            small_blind=10,
            big_blind=20,
            # Deep CFR特有参数
            regret_buffer_size=100000,
            strategy_buffer_size=100000,
            cfr_iterations_per_update=100,
            network_train_steps=50
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 验证配置已应用
            assert engine.config.learning_rate == 0.0001
            assert engine.config.batch_size == 64
            assert engine.config.num_episodes == 5000
            assert engine.config.discount_factor == 0.95
            assert engine.config.network_architecture == [256, 128]
            assert engine.config.checkpoint_interval == 500
            assert engine.config.initial_stack == 2000
            assert engine.config.small_blind == 10
            assert engine.config.big_blind == 20
            
            # 验证Deep CFR特有配置
            assert engine.config.regret_buffer_size == 100000
            assert engine.config.strategy_buffer_size == 100000
            assert engine.config.cfr_iterations_per_update == 100
            assert engine.config.network_train_steps == 50
            
            # 验证环境使用了正确的配置
            assert engine.env.initial_stack == 2000
            assert engine.env.small_blind == 10
            assert engine.env.big_blind == 20
    
    def test_regret_network_parameters_not_empty(self):
        """测试遗憾网络参数非空（Deep CFR）。"""
        config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 验证遗憾网络有参数
            params = list(engine.regret_network.parameters())
            assert len(params) > 0
            
            # 验证参数不全为零
            has_nonzero = False
            for param in params:
                if torch.any(param != 0):
                    has_nonzero = True
                    break
            assert has_nonzero, "遗憾网络参数不应全为零"
    
    def test_policy_network_parameters_not_empty(self):
        """测试策略网络参数非空。"""
        config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 验证策略网络有参数
            params = list(engine.policy_network.parameters())
            assert len(params) > 0
            
            # 验证参数不全为零
            has_nonzero = False
            for param in params:
                if torch.any(param != 0):
                    has_nonzero = True
                    break
            assert has_nonzero, "策略网络参数不应全为零"


class TestSelfPlayEpisode:
    """测试自我对弈回合。"""
    
    def test_self_play_returns_two_episodes(self):
        """测试自我对弈返回两个Episode对象。"""
        config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            episode_p0, episode_p1 = engine.self_play_episode()
            
            # 验证返回类型
            assert isinstance(episode_p0, Episode)
            assert isinstance(episode_p1, Episode)
            
            # 验证玩家ID
            assert episode_p0.player_id == 0
            assert episode_p1.player_id == 1
    
    def test_episode_structure_valid(self):
        """测试Episode结构有效。"""
        config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            episode_p0, episode_p1 = engine.self_play_episode()
            
            # 验证Episode结构：len(states) == len(actions) + 1
            assert len(episode_p0.states) == len(episode_p0.actions) + 1
            assert len(episode_p1.states) == len(episode_p1.actions) + 1
            
            # 验证奖励数量与行动数量匹配
            assert len(episode_p0.rewards) == len(episode_p0.actions)
            assert len(episode_p1.rewards) == len(episode_p1.actions)
    
    def test_final_rewards_sum_to_zero(self):
        """测试最终奖励之和为零（零和游戏）。"""
        config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 运行多次以确保稳定性
            for _ in range(5):
                episode_p0, episode_p1 = engine.self_play_episode()
                
                # 零和游戏：两个玩家的最终奖励之和应为0
                total_reward = episode_p0.final_reward + episode_p1.final_reward
                assert abs(total_reward) < 1e-6, f"最终奖励之和应为0，实际为{total_reward}"


class TestTrainingLoop:
    """测试训练循环（Deep CFR）。"""
    
    @pytest.mark.slow
    def test_train_runs_specified_episodes(self):
        """测试训练循环运行指定迭代数。"""
        config = TrainingConfig(
            num_episodes=2,
            checkpoint_interval=100,  # 设置较大间隔避免保存检查点
            cfr_iterations_per_update=1,  # 最小化CFR迭代次数以加速测试
            network_train_steps=1  # 最小化网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            result = engine.train(num_episodes=2)
            
            # 验证运行了指定迭代数
            assert engine.current_episode == 2
            assert result['total_episodes'] == 2
    
    @pytest.mark.slow
    def test_train_updates_metrics(self):
        """测试训练更新指标。"""
        config = TrainingConfig(
            num_episodes=3,
            checkpoint_interval=100,
            cfr_iterations_per_update=1,  # 最小化CFR迭代次数以加速测试
            network_train_steps=1  # 最小化网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            result = engine.train(num_episodes=3)
            
            # 验证指标已更新
            assert 'win_rate' in result
            assert 'avg_reward' in result
            assert 'total_rewards' in result
            assert len(result['total_rewards']) == 3
    
    @pytest.mark.slow
    def test_train_accumulates_episodes(self):
        """测试多次训练累积迭代数。"""
        config = TrainingConfig(
            num_episodes=2,
            checkpoint_interval=100,
            cfr_iterations_per_update=1,  # 最小化CFR迭代次数以加速测试
            network_train_steps=1  # 最小化网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 第一次训练
            engine.train(num_episodes=2)
            assert engine.current_episode == 2
            
            # 第二次训练
            engine.train(num_episodes=2)
            assert engine.current_episode == 4


class TestCheckpointSaving:
    """测试检查点保存（Deep CFR格式）。"""
    
    @pytest.mark.slow
    def test_checkpoint_saved_at_interval(self):
        """测试按间隔保存检查点。"""
        config = TrainingConfig(
            num_episodes=4,
            checkpoint_interval=2,
            cfr_iterations_per_update=1,  # 最小化CFR迭代次数以加速测试
            network_train_steps=1  # 最小化网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            engine.train(num_episodes=4)
            
            # 验证检查点已保存
            checkpoints = engine.checkpoint_manager.list_checkpoints()
            # 应该有至少2个检查点（第2迭代和第4迭代）
            assert len(checkpoints) >= 2
    
    @pytest.mark.slow
    def test_checkpoint_contains_metadata(self):
        """测试检查点包含元数据。"""
        config = TrainingConfig(
            num_episodes=3,
            checkpoint_interval=3,
            cfr_iterations_per_update=1,  # 最小化CFR迭代次数以加速测试
            network_train_steps=1  # 最小化网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            engine.train(num_episodes=3)
            
            # 获取最新检查点
            checkpoints = engine.checkpoint_manager.list_checkpoints()
            assert len(checkpoints) > 0
            
            latest = checkpoints[0]
            assert latest.episode_number == 3
            assert 0 <= latest.win_rate <= 1
    
    def test_load_checkpoint_restores_state(self):
        """测试加载检查点恢复状态（Deep CFR格式）。"""
        config = TrainingConfig(
            num_episodes=3,
            checkpoint_interval=3,
            cfr_iterations_per_update=1,
            network_train_steps=1
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 第一个引擎：手动保存检查点（不运行完整训练以节省时间）
            engine1 = TrainingEngine(config, checkpoint_dir=tmpdir)
            engine1.current_episode = 5
            checkpoint_path = engine1._save_checkpoint()
            
            # 第二个引擎加载检查点
            engine2 = TrainingEngine(config, checkpoint_dir=tmpdir)
            engine2.load_checkpoint(checkpoint_path)
            
            # 验证状态已恢复
            assert engine2.current_episode == 5


class TestUpdatePolicy:
    """测试策略更新（Deep CFR架构）。"""
    
    def test_update_policy_with_episodes(self):
        """测试使用Episode更新策略（Deep CFR）。"""
        config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 先运行一些CFR迭代以填充缓冲区
            for _ in range(5):
                engine.deep_cfr_trainer.run_cfr_iteration()
            
            # 生成一些Episode（用于兼容性测试）
            episodes = []
            for _ in range(3):
                ep0, ep1 = engine.self_play_episode()
                episodes.extend([ep0, ep1])
            
            # 更新策略（在Deep CFR中会调用train_networks）
            metrics = engine.update_policy(episodes)
            
            # 验证返回指标（Deep CFR返回regret_loss和policy_loss）
            assert 'policy_loss' in metrics
            assert 'regret_loss' in metrics
    
    def test_update_policy_with_empty_list(self):
        """测试使用空列表更新策略。"""
        config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 使用空列表更新
            metrics = engine.update_policy([])
            
            # 应该返回零损失
            assert metrics['policy_loss'] == 0.0
            assert metrics['regret_loss'] == 0.0


class TestGetCurrentMetrics:
    """测试获取当前指标。"""
    
    def test_get_metrics_initial(self):
        """测试初始状态的指标。"""
        config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            metrics = engine.get_current_metrics()
            
            assert metrics['episode'] == 0
            assert metrics['win_rate'] == 0
            assert metrics['avg_reward'] == 0
            assert metrics['win_count'] == 0
    
    @pytest.mark.slow
    def test_get_metrics_after_training(self):
        """测试训练后的指标。"""
        config = TrainingConfig(
            num_episodes=2,
            checkpoint_interval=100,
            cfr_iterations_per_update=1,
            network_train_steps=1
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            engine.train(num_episodes=2)
            
            metrics = engine.get_current_metrics()
            
            assert metrics['episode'] == 2
            assert 0 <= metrics['win_rate'] <= 1


# ============================================================================
# 属性测试
# ============================================================================

class TestTrainingEngineProperties:
    """训练引擎的属性测试。"""
    
    @given(
        learning_rate=st.floats(min_value=1e-5, max_value=0.1),
        batch_size=st.integers(min_value=1, max_value=128),
        discount_factor=st.floats(min_value=0.0, max_value=1.0),
        checkpoint_interval=st.integers(min_value=1, max_value=1000),
        initial_stack=st.integers(min_value=100, max_value=10000),
        small_blind=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_training_initialization_completeness(
        self,
        learning_rate: float,
        batch_size: int,
        discount_factor: float,
        checkpoint_interval: int,
        initial_stack: int,
        small_blind: int
    ):
        """属性1：训练初始化完整性
        
        *对于任何*有效的训练配置，启动训练会话应该成功初始化策略网络
        （参数非空）并开始执行训练循环
        
        **Feature: texas-holdem-ai-training, Property 1: 训练初始化完整性**
        **验证需求：1.1**
        """
        # 确保big_blind > small_blind
        big_blind = small_blind * 2
        
        # 确保initial_stack >= big_blind
        if initial_stack < big_blind:
            initial_stack = big_blind * 10
        
        config = TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_episodes=10,
            discount_factor=discount_factor,
            checkpoint_interval=checkpoint_interval,
            initial_stack=initial_stack,
            small_blind=small_blind,
            big_blind=big_blind
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 验证策略网络参数非空
            params = list(engine.policy_network.parameters())
            assert len(params) > 0, "策略网络应该有参数"
            
            # 验证至少有一个参数不全为零
            has_nonzero = False
            for param in params:
                if torch.any(param != 0):
                    has_nonzero = True
                    break
            assert has_nonzero, "策略网络参数不应全为零"
            
            # 验证所有Deep CFR组件已初始化
            assert engine.env is not None
            assert engine.state_encoder is not None
            assert engine.regret_network is not None  # Deep CFR
            assert engine.policy_network is not None
            assert engine.deep_cfr_trainer is not None  # Deep CFR
            assert engine.checkpoint_manager is not None
    
    @pytest.mark.slow
    @given(num_episodes=st.integers(min_value=1, max_value=2))
    @settings(max_examples=3, deadline=None)
    def test_property_training_termination_safety(self, num_episodes: int):
        """属性2：训练终止安全性
        
        *对于任何*训练会话，无论是达到指定迭代数还是手动停止，
        系统都应该保存最终模型状态且文件可被成功加载
        
        **Feature: texas-holdem-ai-training, Property 2: 训练终止安全性**
        **验证需求：1.5**
        """
        config = TrainingConfig(
            num_episodes=num_episodes,
            checkpoint_interval=max(1, num_episodes),  # 确保至少保存一次
            cfr_iterations_per_update=1,  # 减少CFR迭代次数以加速测试
            network_train_steps=1  # 减少网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 运行训练
            engine.train(num_episodes=num_episodes)
            
            # 验证检查点已保存
            checkpoints = engine.checkpoint_manager.list_checkpoints()
            assert len(checkpoints) > 0, "训练终止后应该有检查点"
            
            # 验证检查点可以被加载
            latest_checkpoint = checkpoints[0]
            
            # 创建新引擎并加载检查点
            engine2 = TrainingEngine(config, checkpoint_dir=tmpdir)
            engine2.load_checkpoint(latest_checkpoint.path)
            
            # 验证状态已恢复
            assert engine2.current_episode == engine.current_episode
    
    @pytest.mark.slow
    @given(
        num_episodes=st.integers(min_value=2, max_value=3),
        checkpoint_interval=st.integers(min_value=1, max_value=2)
    )
    @settings(max_examples=3, deadline=None)
    def test_property_checkpoint_interval_consistency(
        self, 
        num_episodes: int, 
        checkpoint_interval: int
    ):
        """属性：检查点保存间隔一致性
        
        *对于任何*指定的检查点间隔N，运行N个迭代的训练应该至少创建一个检查点文件
        
        **Feature: texas-holdem-ai-training, Property: 检查点保存间隔一致性**
        **验证需求：1.3**
        """
        config = TrainingConfig(
            num_episodes=num_episodes,
            checkpoint_interval=checkpoint_interval,
            cfr_iterations_per_update=1,  # 减少CFR迭代次数以加速测试
            network_train_steps=1  # 减少网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 运行训练
            engine.train(num_episodes=num_episodes)
            
            # 验证检查点数量
            checkpoints = engine.checkpoint_manager.list_checkpoints()
            
            # 预期检查点数量：至少 num_episodes // checkpoint_interval 个
            # 加上最终保存的一个
            expected_min = num_episodes // checkpoint_interval
            if num_episodes % checkpoint_interval != 0:
                expected_min += 1  # 最终保存
            
            assert len(checkpoints) >= 1, \
                f"运行{num_episodes}回合（间隔{checkpoint_interval}）应该至少有1个检查点"
    
    @given(
        learning_rate=st.floats(min_value=1e-5, max_value=0.1),
        batch_size=st.integers(min_value=1, max_value=128),
        discount_factor=st.floats(min_value=0.0, max_value=1.0),
        initial_stack=st.integers(min_value=100, max_value=10000),
        small_blind=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_config_parameters_applied_correctly(
        self,
        learning_rate: float,
        batch_size: int,
        discount_factor: float,
        initial_stack: int,
        small_blind: int
    ):
        """属性4：配置参数应用正确性
        
        *对于任何*有效的训练配置参数（学习率、批次大小等），
        启动训练后，训练器内部使用的参数应该与配置中指定的参数相匹配
        
        **Feature: texas-holdem-ai-training, Property 4: 配置参数应用正确性**
        **验证需求：1.4**
        """
        # 确保big_blind > small_blind
        big_blind = small_blind * 2
        
        # 确保initial_stack >= big_blind
        if initial_stack < big_blind:
            initial_stack = big_blind * 10
        
        config = TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_episodes=10,
            discount_factor=discount_factor,
            initial_stack=initial_stack,
            small_blind=small_blind,
            big_blind=big_blind
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 验证配置参数已正确应用
            assert engine.config.learning_rate == learning_rate
            assert engine.config.batch_size == batch_size
            assert engine.config.discount_factor == discount_factor
            assert engine.config.initial_stack == initial_stack
            assert engine.config.small_blind == small_blind
            assert engine.config.big_blind == big_blind
            
            # 验证环境使用了正确的配置
            assert engine.env.initial_stack == initial_stack
            assert engine.env.small_blind == small_blind
            assert engine.env.big_blind == big_blind
            
            # 验证优化器使用了正确的学习率（Deep CFR）
            for param_group in engine.policy_optimizer.param_groups:
                assert param_group['lr'] == learning_rate
            for param_group in engine.regret_optimizer.param_groups:
                assert param_group['lr'] == learning_rate



class TestDeepCFRIntegration:
    """测试Deep CFR集成功能。"""
    
    @pytest.mark.slow
    def test_deep_cfr_training_runs(self):
        """测试Deep CFR训练能够运行。"""
        config = TrainingConfig(
            num_episodes=3,
            cfr_iterations_per_update=1,  # 减少CFR迭代次数以加速测试
            network_train_steps=1  # 减少网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 运行训练
            result = engine.train(num_episodes=3)
            
            # 验证训练完成
            assert result['total_episodes'] == 3
    
    @pytest.mark.slow
    def test_deep_cfr_buffers_filled(self):
        """测试Deep CFR缓冲区被填充。"""
        config = TrainingConfig(
            num_episodes=5,
            cfr_iterations_per_update=2,  # 减少CFR迭代次数以加速测试
            network_train_steps=1  # 减少网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 运行训练
            engine.train(num_episodes=5)
            
            # 验证缓冲区被填充
            metrics = engine.get_current_metrics()
            assert metrics['regret_buffer_size'] > 0
            assert metrics['strategy_buffer_size'] > 0
    
    def test_update_policy_returns_deep_cfr_metrics(self):
        """测试update_policy返回Deep CFR指标。"""
        config = TrainingConfig(
            num_episodes=5,
            cfr_iterations_per_update=2,  # 减少CFR迭代次数以加速测试
            network_train_steps=1  # 减少网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 先运行一些CFR迭代以填充缓冲区
            for _ in range(5):
                engine.deep_cfr_trainer.run_cfr_iteration()
            
            # 生成一些训练数据
            episode_p0, episode_p1 = engine.self_play_episode()
            
            # 更新策略
            metrics = engine.update_policy([episode_p0, episode_p1])
            
            # 验证返回的Deep CFR指标
            assert 'policy_loss' in metrics
            assert 'regret_loss' in metrics


class TestCheckpointRoundTrip:
    """测试检查点往返一致性（属性3）。"""
    
    def test_checkpoint_round_trip_basic(self):
        """基本检查点往返测试。
        
        **Feature: deep-cfr-refactor, Property 3: 检查点往返一致性**
        **验证需求：1.5**
        """
        config = TrainingConfig(
            num_episodes=3,
            checkpoint_interval=3,
            cfr_iterations_per_update=1,
            network_train_steps=1
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建第一个引擎
            engine1 = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 手动修改网络参数以确保它们不是初始值
            with torch.no_grad():
                for param in engine1.regret_network.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
                for param in engine1.policy_network.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            
            # 保存原始网络参数
            original_regret_params = {
                name: param.clone() 
                for name, param in engine1.regret_network.named_parameters()
            }
            original_policy_params = {
                name: param.clone() 
                for name, param in engine1.policy_network.named_parameters()
            }
            
            # 手动保存检查点
            engine1.current_episode = 5
            checkpoint_path = engine1._save_checkpoint()
            
            # 创建新引擎并加载检查点
            engine2 = TrainingEngine(config, checkpoint_dir=tmpdir)
            engine2.load_checkpoint(checkpoint_path)
            
            # 验证遗憾网络参数一致
            for name, param in engine2.regret_network.named_parameters():
                assert name in original_regret_params, f"缺少参数: {name}"
                assert torch.allclose(param, original_regret_params[name], atol=1e-6), \
                    f"遗憾网络参数 {name} 不一致"
            
            # 验证策略网络参数一致
            for name, param in engine2.policy_network.named_parameters():
                assert name in original_policy_params, f"缺少参数: {name}"
                assert torch.allclose(param, original_policy_params[name], atol=1e-6), \
                    f"策略网络参数 {name} 不一致"
            
            # 验证训练状态一致
            assert engine2.current_episode == engine1.current_episode
    
    @given(
        learning_rate=st.floats(min_value=1e-4, max_value=0.01),
        batch_size=st.integers(min_value=16, max_value=64)
    )
    @settings(max_examples=5, deadline=None)
    def test_property_checkpoint_round_trip_consistency(
        self,
        learning_rate: float,
        batch_size: int
    ):
        """属性3：检查点往返一致性
        
        *对于任何*遗憾网络和策略网络的参数状态，保存为检查点后再加载，
        应该恢复出等价的网络参数
        
        **Feature: deep-cfr-refactor, Property 3: 检查点往返一致性**
        **验证需求：1.5**
        """
        config = TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_episodes=3,
            checkpoint_interval=3,
            cfr_iterations_per_update=1,
            network_train_steps=1
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建第一个引擎
            engine1 = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 手动修改网络参数
            with torch.no_grad():
                for param in engine1.regret_network.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
                for param in engine1.policy_network.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            
            # 保存原始网络参数
            original_regret_params = {
                name: param.clone() 
                for name, param in engine1.regret_network.named_parameters()
            }
            original_policy_params = {
                name: param.clone() 
                for name, param in engine1.policy_network.named_parameters()
            }
            
            # 保存检查点
            engine1.current_episode = 5
            checkpoint_path = engine1._save_checkpoint()
            
            # 创建新引擎并加载检查点
            engine2 = TrainingEngine(config, checkpoint_dir=tmpdir)
            engine2.load_checkpoint(checkpoint_path)
            
            # 验证遗憾网络参数一致
            for name, param in engine2.regret_network.named_parameters():
                assert name in original_regret_params, f"缺少参数: {name}"
                assert torch.allclose(param, original_regret_params[name], atol=1e-6), \
                    f"遗憾网络参数 {name} 不一致"
            
            # 验证策略网络参数一致
            for name, param in engine2.policy_network.named_parameters():
                assert name in original_policy_params, f"缺少参数: {name}"
                assert torch.allclose(param, original_policy_params[name], atol=1e-6), \
                    f"策略网络参数 {name} 不一致"
            
            # 验证训练状态一致
            assert engine2.current_episode == engine1.current_episode


class TestLegacyCheckpointCompatibility:
    """测试旧检查点格式兼容性（需求5.4）。"""
    
    def test_load_legacy_checkpoint_format(self):
        """测试加载旧格式检查点（包含policy_network和value_network）。
        
        旧格式检查点包含：
        - model_state_dict: 策略网络参数
        - optimizer_state_dict: 策略网络优化器参数
        - has_value_network: 是否包含价值网络
        - value_network_state_dict: 价值网络参数（可选）
        
        **验证需求：5.4**
        """
        config = TrainingConfig(
            num_episodes=3,
            checkpoint_interval=3,
            cfr_iterations_per_update=1,
            network_train_steps=1
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建引擎
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 手动修改策略网络参数
            with torch.no_grad():
                for param in engine.policy_network.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            
            # 保存原始策略网络参数
            original_policy_params = {
                name: param.clone() 
                for name, param in engine.policy_network.named_parameters()
            }
            
            # 创建旧格式检查点
            legacy_checkpoint_path = Path(tmpdir) / "legacy_checkpoint.pt"
            legacy_checkpoint_data = {
                'model_state_dict': engine.policy_network.state_dict(),
                'optimizer_state_dict': engine.policy_optimizer.state_dict(),
                'episode_number': 100,
                'win_rate': 0.55,
                'avg_reward': 10.5,
                'win_count': 55,
                'has_value_network': True,
                'value_network_state_dict': {},  # 空的价值网络参数（模拟旧格式）
            }
            torch.save(legacy_checkpoint_data, legacy_checkpoint_path)
            
            # 创建新引擎并加载旧格式检查点
            engine2 = TrainingEngine(config, checkpoint_dir=tmpdir)
            engine2.load_checkpoint(str(legacy_checkpoint_path))
            
            # 验证策略网络参数已正确加载
            for name, param in engine2.policy_network.named_parameters():
                assert name in original_policy_params, f"缺少参数: {name}"
                assert torch.allclose(param, original_policy_params[name], atol=1e-6), \
                    f"策略网络参数 {name} 不一致"
            
            # 验证训练状态已恢复
            assert engine2.current_episode == 100
            assert engine2.win_count == 55
    
    def test_load_legacy_checkpoint_without_value_network(self):
        """测试加载不包含价值网络的旧格式检查点。
        
        **验证需求：5.4**
        """
        config = TrainingConfig(
            num_episodes=3,
            checkpoint_interval=3,
            cfr_iterations_per_update=1,
            network_train_steps=1
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建引擎
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 创建旧格式检查点（不包含价值网络）
            legacy_checkpoint_path = Path(tmpdir) / "legacy_checkpoint_no_value.pt"
            legacy_checkpoint_data = {
                'model_state_dict': engine.policy_network.state_dict(),
                'optimizer_state_dict': engine.policy_optimizer.state_dict(),
                'episode_number': 50,
                'win_rate': 0.45,
                'avg_reward': 5.0,
                'win_count': 22,
                'has_value_network': False,
            }
            torch.save(legacy_checkpoint_data, legacy_checkpoint_path)
            
            # 创建新引擎并加载旧格式检查点
            engine2 = TrainingEngine(config, checkpoint_dir=tmpdir)
            engine2.load_checkpoint(str(legacy_checkpoint_path))
            
            # 验证训练状态已恢复
            assert engine2.current_episode == 50
            assert engine2.win_count == 22
    
    def test_new_checkpoint_format_detected(self):
        """测试新格式检查点被正确识别。
        
        **验证需求：1.5**
        """
        config = TrainingConfig(
            num_episodes=3,
            checkpoint_interval=3,
            cfr_iterations_per_update=1,
            network_train_steps=1
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建引擎并保存新格式检查点
            engine1 = TrainingEngine(config, checkpoint_dir=tmpdir)
            engine1.current_episode = 10
            checkpoint_path = engine1._save_checkpoint()
            
            # 验证检查点包含新格式标记
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
            assert checkpoint_data.get('checkpoint_format') == 'deep_cfr_v1'
            assert 'regret_network_state_dict' in checkpoint_data
            assert 'policy_network_state_dict' in checkpoint_data
    
    @pytest.mark.slow
    def test_training_continues_after_loading_legacy_checkpoint(self):
        """测试加载旧格式检查点后可以继续训练。
        
        **验证需求：5.4, 7.4**
        """
        config = TrainingConfig(
            num_episodes=2,
            checkpoint_interval=2,
            cfr_iterations_per_update=1,  # 减少CFR迭代次数以加速测试
            network_train_steps=1  # 减少网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建引擎
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 创建旧格式检查点
            legacy_checkpoint_path = Path(tmpdir) / "legacy_checkpoint.pt"
            legacy_checkpoint_data = {
                'model_state_dict': engine.policy_network.state_dict(),
                'optimizer_state_dict': engine.policy_optimizer.state_dict(),
                'episode_number': 10,
                'win_rate': 0.5,
                'avg_reward': 0.0,
                'win_count': 5,
                'has_value_network': False,
            }
            torch.save(legacy_checkpoint_data, legacy_checkpoint_path)
            
            # 创建新引擎并加载旧格式检查点
            engine2 = TrainingEngine(config, checkpoint_dir=tmpdir)
            engine2.load_checkpoint(str(legacy_checkpoint_path))
            
            # 验证初始状态
            assert engine2.current_episode == 10
            
            # 继续训练
            result = engine2.train(num_episodes=2)
            
            # 验证训练完成
            assert engine2.current_episode == 12
            assert result['total_episodes'] == 12


class TestNetworkInitialization:
    """测试网络初始化（需求1.1, 1.2）。"""
    
    def test_regret_network_created_correctly(self):
        """测试遗憾网络正确创建。
        
        **验证需求：1.1**
        """
        config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 验证遗憾网络存在
            assert engine.regret_network is not None
            
            # 验证网络架构
            # 输入维度应该是370（状态编码维度）
            # 输出维度应该是6（动作空间大小：FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN）
            test_input = torch.randn(1, 370)
            output = engine.regret_network(test_input)
            assert output.shape == (1, 6), f"遗憾网络输出维度应为(1, 6)，实际为{output.shape}"
    
    def test_policy_network_created_correctly(self):
        """测试策略网络正确创建。
        
        **验证需求：1.2**
        """
        config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 验证策略网络存在
            assert engine.policy_network is not None
            
            # 验证网络架构
            # 输出维度应该是6（动作空间大小：FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN）
            test_input = torch.randn(1, 370)
            output = engine.policy_network.get_action_probs(test_input)
            assert output.shape == (1, 6), f"策略网络输出维度应为(1, 6)，实际为{output.shape}"
            
            # 验证输出是有效的概率分布
            assert torch.all(output >= 0), "策略网络输出应该非负"
            assert torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-6), \
                "策略网络输出应该和为1"
    
    def test_networks_use_correct_architecture(self):
        """测试网络使用正确的架构配置。
        
        **验证需求：7.1, 7.2**
        """
        custom_architecture = [256, 128, 64]
        config = TrainingConfig(network_architecture=custom_architecture)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 验证配置已应用
            assert engine.config.network_architecture == custom_architecture
            
            # 验证网络可以正常工作
            # 输出维度应该是6（动作空间大小：FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN）
            test_input = torch.randn(1, 370)
            
            regret_output = engine.regret_network(test_input)
            assert regret_output.shape == (1, 6)
            
            policy_output = engine.policy_network.get_action_probs(test_input)
            assert policy_output.shape == (1, 6)


class TestTrainingFlowExecution:
    """测试训练流程执行（需求7.3）。"""
    
    def test_deep_cfr_trainer_integrated(self):
        """测试Deep CFR训练器已集成。
        
        **验证需求：7.3**
        """
        config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 验证Deep CFR训练器存在
            assert engine.deep_cfr_trainer is not None
            
            # 验证缓冲区存在
            assert engine.regret_buffer is not None
            assert engine.strategy_buffer is not None
    
    @pytest.mark.slow
    def test_training_uses_deep_cfr_flow(self):
        """测试训练使用Deep CFR流程。
        
        **验证需求：7.3**
        """
        config = TrainingConfig(
            num_episodes=2,
            cfr_iterations_per_update=1,  # 减少CFR迭代次数以加速测试
            network_train_steps=1  # 减少网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 运行训练
            result = engine.train(num_episodes=2)
            
            # 验证Deep CFR指标
            assert 'regret_buffer_size' in result
            assert 'strategy_buffer_size' in result
            
            # 验证缓冲区被填充
            assert result['regret_buffer_size'] > 0
            assert result['strategy_buffer_size'] > 0
    
    @pytest.mark.slow
    def test_network_sync_between_engine_and_trainer(self):
        """测试引擎和训练器之间的网络同步。
        
        **验证需求：7.3**
        """
        config = TrainingConfig(
            num_episodes=2,
            cfr_iterations_per_update=1,  # 减少CFR迭代次数以加速测试
            network_train_steps=1  # 减少网络训练步数以加速测试
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(config, checkpoint_dir=tmpdir)
            
            # 运行训练
            engine.train(num_episodes=2)
            
            # 验证网络参数已同步
            for (name1, param1), (name2, param2) in zip(
                engine.regret_network.named_parameters(),
                engine.deep_cfr_trainer.regret_network.named_parameters()
            ):
                assert name1 == name2
                assert torch.allclose(param1, param2, atol=1e-6), \
                    f"遗憾网络参数 {name1} 未同步"
            
            for (name1, param1), (name2, param2) in zip(
                engine.policy_network.named_parameters(),
                engine.deep_cfr_trainer.policy_network.named_parameters()
            ):
                assert name1 == name2
                assert torch.allclose(param1, param2, atol=1e-6), \
                    f"策略网络参数 {name1} 未同步"
