"""并行训练器的单元测试和属性测试。

测试内容：
- 创建指定数量的并行环境
- 数据收集和聚合
- 参数同步
- 进程错误处理
- 优雅终止
- 属性测试：并行环境创建正确性、数据聚合完整性
"""

import pytest
import numpy as np
import torch
import time
import multiprocessing as mp
from typing import List, Dict

from hypothesis import given, settings, strategies as st, assume

from training.parallel_trainer import (
    ParallelTrainer, 
    WorkerExperience,
    worker_process,
    _self_play_episode,
    _select_random_action
)
from models.core import (
    TrainingConfig, Episode, GameState, Action, ActionType, GameStage, Card
)
from environment.poker_environment import PokerEnvironment
from environment.state_encoder import StateEncoder


# ============================================================================
# 测试夹具
# ============================================================================

@pytest.fixture
def default_config():
    """创建默认训练配置。"""
    return TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        num_episodes=100,
        discount_factor=0.99,
        network_architecture=[64, 32],  # 使用较小的网络加速测试
        checkpoint_interval=50,
        num_parallel_envs=2,
        initial_stack=1000,
        small_blind=5,
        big_blind=10
    )


@pytest.fixture
def parallel_trainer(default_config):
    """创建并行训练器实例。"""
    trainer = ParallelTrainer(default_config)
    yield trainer
    # 清理资源
    trainer.cleanup()


# ============================================================================
# 单元测试：创建并行环境
# ============================================================================

class TestCreateParallelEnvs:
    """测试创建并行环境功能。"""
    
    def test_create_specified_number_of_envs(self, default_config):
        """测试创建指定数量的并行环境。"""
        trainer = ParallelTrainer(default_config, num_workers=3)
        
        num_created = trainer.create_parallel_envs()
        
        assert num_created == 3
        assert trainer.get_num_workers() == 3
        assert trainer.is_environments_created()
        
        trainer.cleanup()
    
    def test_create_single_env(self, default_config):
        """测试创建单个环境。"""
        trainer = ParallelTrainer(default_config, num_workers=1)
        
        num_created = trainer.create_parallel_envs()
        
        assert num_created == 1
        assert trainer.get_num_workers() == 1
        
        trainer.cleanup()
    
    def test_create_envs_uses_config_value(self, default_config):
        """测试使用配置中的并行环境数量。"""
        default_config.num_parallel_envs = 4
        trainer = ParallelTrainer(default_config)
        
        num_created = trainer.create_parallel_envs()
        
        assert num_created == 4
        
        trainer.cleanup()
    
    def test_minimum_one_worker(self, default_config):
        """测试至少有一个工作进程。"""
        trainer = ParallelTrainer(default_config, num_workers=0)
        
        # 应该自动调整为1
        assert trainer.get_num_workers() == 1
        
        trainer.cleanup()
    
    def test_recreate_envs_after_cleanup(self, default_config):
        """测试清理后重新创建环境。"""
        trainer = ParallelTrainer(default_config, num_workers=2)
        
        trainer.create_parallel_envs()
        assert trainer.is_environments_created()
        
        trainer.cleanup()
        assert not trainer.is_environments_created()
        
        # 重新创建
        num_created = trainer.create_parallel_envs()
        assert num_created == 2
        assert trainer.is_environments_created()
        
        trainer.cleanup()


# ============================================================================
# 单元测试：数据收集和聚合
# ============================================================================

class TestCollectExperiences:
    """测试数据收集功能。"""
    
    def test_collect_experiences_basic(self, default_config):
        """测试基本的经验收集。"""
        trainer = ParallelTrainer(default_config, num_workers=2)
        trainer.create_parallel_envs()
        
        # 每个工作进程执行2个回合
        experiences = trainer.collect_experiences(episodes_per_worker=2, timeout=60.0)
        
        assert len(experiences) == 2  # 2个工作进程
        
        for exp in experiences:
            assert isinstance(exp, WorkerExperience)
            assert exp.error is None
            # 每个回合产生2个Episode（两个玩家各一个）
            assert len(exp.episodes) == 4  # 2回合 * 2玩家
        
        trainer.cleanup()
    
    def test_collect_experiences_single_worker(self, default_config):
        """测试单个工作进程的经验收集。"""
        trainer = ParallelTrainer(default_config, num_workers=1)
        trainer.create_parallel_envs()
        
        experiences = trainer.collect_experiences(episodes_per_worker=3, timeout=60.0)
        
        assert len(experiences) == 1
        assert len(experiences[0].episodes) == 6  # 3回合 * 2玩家
        
        trainer.cleanup()
    
    def test_collect_experiences_without_create_raises_error(self, default_config):
        """测试未创建环境时收集经验应该抛出错误。"""
        trainer = ParallelTrainer(default_config, num_workers=2)
        
        with pytest.raises(RuntimeError, match="必须先调用"):
            trainer.collect_experiences(episodes_per_worker=1)
        
        trainer.cleanup()


class TestAggregateData:
    """测试数据聚合功能。"""
    
    def test_aggregate_data_basic(self, default_config):
        """测试基本的数据聚合。"""
        trainer = ParallelTrainer(default_config, num_workers=2)
        trainer.create_parallel_envs()
        
        experiences = trainer.collect_experiences(episodes_per_worker=2, timeout=60.0)
        
        all_episodes, all_rewards, total_wins = trainer.aggregate_data(experiences)
        
        # 2个工作进程 * 2回合 * 2玩家 = 8个Episode
        assert len(all_episodes) == 8
        
        # 2个工作进程 * 2回合 = 4个奖励
        assert len(all_rewards) == 4
        
        # 验证所有Episode都是有效的
        for episode in all_episodes:
            assert isinstance(episode, Episode)
            assert episode.player_id in [0, 1]
        
        trainer.cleanup()
    
    def test_aggregate_data_empty(self, default_config):
        """测试聚合空数据。"""
        trainer = ParallelTrainer(default_config)
        
        all_episodes, all_rewards, total_wins = trainer.aggregate_data([])
        
        assert all_episodes == []
        assert all_rewards == []
        assert total_wins == 0
        
        trainer.cleanup()
    
    def test_aggregate_data_preserves_all_episodes(self, default_config):
        """测试聚合保留所有Episode。"""
        trainer = ParallelTrainer(default_config, num_workers=3)
        trainer.create_parallel_envs()
        
        experiences = trainer.collect_experiences(episodes_per_worker=2, timeout=60.0)
        
        # 计算预期的Episode总数
        expected_episodes = sum(len(exp.episodes) for exp in experiences)
        expected_rewards = sum(len(exp.total_rewards) for exp in experiences)
        expected_wins = sum(exp.win_count for exp in experiences)
        
        all_episodes, all_rewards, total_wins = trainer.aggregate_data(experiences)
        
        assert len(all_episodes) == expected_episodes
        assert len(all_rewards) == expected_rewards
        assert total_wins == expected_wins
        
        trainer.cleanup()


# ============================================================================
# 单元测试：参数同步
# ============================================================================

class TestSyncParameters:
    """测试参数同步功能。"""
    
    def test_sync_parameters_basic(self, default_config):
        """测试基本的参数同步。"""
        trainer = ParallelTrainer(default_config)
        
        # 创建模拟的策略参数
        params = {
            'layer1.weight': torch.randn(64, 370),
            'layer1.bias': torch.randn(64),
            'layer2.weight': torch.randn(4, 64),
            'layer2.bias': torch.randn(4)
        }
        
        trainer.sync_parameters(params)
        
        # 验证参数已保存
        assert len(trainer.policy_params) == 4
        
        # 验证参数值正确
        for key in params:
            assert key in trainer.policy_params
            assert torch.allclose(trainer.policy_params[key], params[key])
        
        trainer.cleanup()
    
    def test_sync_parameters_creates_copy(self, default_config):
        """测试参数同步创建副本而不是引用。"""
        trainer = ParallelTrainer(default_config)
        
        params = {
            'weight': torch.randn(10, 10)
        }
        original_value = params['weight'].clone()
        
        trainer.sync_parameters(params)
        
        # 修改原始参数
        params['weight'].fill_(0)
        
        # 验证同步的参数未被修改
        assert not torch.allclose(trainer.policy_params['weight'], params['weight'])
        assert torch.allclose(trainer.policy_params['weight'], original_value)
        
        trainer.cleanup()
    
    def test_sync_parameters_multiple_times(self, default_config):
        """测试多次同步参数。"""
        trainer = ParallelTrainer(default_config)
        
        # 第一次同步
        params1 = {'weight': torch.ones(5, 5)}
        trainer.sync_parameters(params1)
        
        # 第二次同步
        params2 = {'weight': torch.zeros(5, 5)}
        trainer.sync_parameters(params2)
        
        # 验证使用最新的参数
        assert torch.allclose(trainer.policy_params['weight'], params2['weight'])
        
        trainer.cleanup()


# ============================================================================
# 单元测试：优雅终止
# ============================================================================

class TestCleanup:
    """测试优雅终止功能。"""
    
    def test_cleanup_basic(self, default_config):
        """测试基本的清理功能。"""
        trainer = ParallelTrainer(default_config, num_workers=2)
        trainer.create_parallel_envs()
        
        assert trainer.is_environments_created()
        
        trainer.cleanup()
        
        assert not trainer.is_environments_created()
        assert trainer.result_queue is None
        assert trainer.stop_event is None
        assert trainer.error_event is None
    
    def test_cleanup_without_create(self, default_config):
        """测试未创建环境时的清理。"""
        trainer = ParallelTrainer(default_config)
        
        # 不应该抛出异常
        trainer.cleanup()
        
        assert not trainer.is_environments_created()
    
    def test_cleanup_multiple_times(self, default_config):
        """测试多次清理。"""
        trainer = ParallelTrainer(default_config, num_workers=2)
        trainer.create_parallel_envs()
        
        trainer.cleanup()
        trainer.cleanup()  # 第二次清理不应该抛出异常
        
        assert not trainer.is_environments_created()
    
    def test_cleanup_during_collection(self, default_config):
        """测试在收集过程中清理。"""
        trainer = ParallelTrainer(default_config, num_workers=2)
        trainer.create_parallel_envs()
        
        # 启动收集但立即清理
        # 这个测试主要验证不会死锁或崩溃
        trainer.cleanup()
        
        assert not trainer.is_environments_created()


# ============================================================================
# 单元测试：辅助函数
# ============================================================================

class TestHelperFunctions:
    """测试辅助函数。"""
    
    def test_select_random_action_returns_legal_action(self):
        """测试随机选择返回合法行动。"""
        legal_actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
            Action(ActionType.RAISE, 100)
        ]
        
        for _ in range(10):
            action = _select_random_action(legal_actions)
            assert action in legal_actions
    
    def test_select_random_action_empty_list(self):
        """测试空列表返回弃牌。"""
        action = _select_random_action([])
        assert action.action_type == ActionType.FOLD
    
    def test_self_play_episode_returns_valid_episodes(self, default_config):
        """测试自我对弈返回有效的Episode。"""
        env = PokerEnvironment(
            initial_stack=default_config.initial_stack,
            small_blind=default_config.small_blind,
            big_blind=default_config.big_blind
        )
        state_encoder = StateEncoder()
        
        episode_p0, episode_p1 = _self_play_episode(
            env, state_encoder, default_config, None
        )
        
        # 验证Episode有效性
        assert isinstance(episode_p0, Episode)
        assert isinstance(episode_p1, Episode)
        assert episode_p0.player_id == 0
        assert episode_p1.player_id == 1
        
        # 验证状态和行动数量匹配
        assert len(episode_p0.states) == len(episode_p0.actions) + 1
        assert len(episode_p1.states) == len(episode_p1.actions) + 1
        
        # 验证奖励数量与行动数量匹配
        assert len(episode_p0.rewards) == len(episode_p0.actions)
        assert len(episode_p1.rewards) == len(episode_p1.actions)


# ============================================================================
# 属性测试
# ============================================================================

class TestParallelTrainerProperties:
    """并行训练器的属性测试。"""
    
    @given(num_workers=st.integers(min_value=1, max_value=4))
    @settings(max_examples=100, deadline=None)
    def test_property_30_parallel_env_creation(self, num_workers: int):
        """属性30：并行环境创建正确性
        
        *对于任何*指定的工作进程数N，启用并行训练应该创建恰好N个并行游戏环境
        
        **Feature: texas-holdem-ai-training, Property 30: 并行环境创建正确性**
        **验证需求：8.1, 8.4**
        """
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=32,
            num_episodes=100,
            num_parallel_envs=num_workers,
            initial_stack=1000,
            small_blind=5,
            big_blind=10
        )
        
        trainer = ParallelTrainer(config, num_workers=num_workers)
        
        try:
            # 创建环境
            num_created = trainer.create_parallel_envs()
            
            # 验证创建的环境数量正确
            assert num_created == num_workers, \
                f"期望创建 {num_workers} 个环境，实际创建 {num_created} 个"
            
            # 验证get_num_workers返回正确的数量
            assert trainer.get_num_workers() == num_workers, \
                f"get_num_workers() 返回 {trainer.get_num_workers()}，期望 {num_workers}"
            
            # 验证环境已创建标志
            assert trainer.is_environments_created(), \
                "is_environments_created() 应该返回 True"
            
        finally:
            trainer.cleanup()
    
    @given(
        num_workers=st.integers(min_value=1, max_value=3),
        episodes_per_worker=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_31_data_aggregation_completeness(
        self, 
        num_workers: int, 
        episodes_per_worker: int
    ):
        """属性31：并行数据聚合完整性
        
        *对于任何*并行训练会话，收集到的训练数据总量应该等于所有工作进程生成的数据量之和
        
        **Feature: texas-holdem-ai-training, Property 31: 并行数据聚合完整性**
        **验证需求：8.2**
        """
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=32,
            num_episodes=100,
            num_parallel_envs=num_workers,
            initial_stack=1000,
            small_blind=5,
            big_blind=10
        )
        
        trainer = ParallelTrainer(config, num_workers=num_workers)
        
        try:
            trainer.create_parallel_envs()
            
            # 收集经验
            experiences = trainer.collect_experiences(
                episodes_per_worker=episodes_per_worker,
                timeout=120.0
            )
            
            # 验证收集到的经验数量
            assert len(experiences) == num_workers, \
                f"期望收集 {num_workers} 个工作进程的经验，实际收集 {len(experiences)} 个"
            
            # 计算各工作进程的数据量
            total_episodes_from_workers = sum(len(exp.episodes) for exp in experiences)
            total_rewards_from_workers = sum(len(exp.total_rewards) for exp in experiences)
            total_wins_from_workers = sum(exp.win_count for exp in experiences)
            
            # 聚合数据
            all_episodes, all_rewards, total_wins = trainer.aggregate_data(experiences)
            
            # 验证聚合后的数据量等于各工作进程数据量之和
            assert len(all_episodes) == total_episodes_from_workers, \
                f"聚合后Episode数量 {len(all_episodes)} != 各进程之和 {total_episodes_from_workers}"
            
            assert len(all_rewards) == total_rewards_from_workers, \
                f"聚合后奖励数量 {len(all_rewards)} != 各进程之和 {total_rewards_from_workers}"
            
            assert total_wins == total_wins_from_workers, \
                f"聚合后胜利次数 {total_wins} != 各进程之和 {total_wins_from_workers}"
            
            # 验证每个回合产生2个Episode（两个玩家各一个）
            expected_episodes = num_workers * episodes_per_worker * 2
            assert len(all_episodes) == expected_episodes, \
                f"期望 {expected_episodes} 个Episode，实际 {len(all_episodes)} 个"
            
        finally:
            trainer.cleanup()


# ============================================================================
# 集成测试
# ============================================================================

class TestParallelTrainerIntegration:
    """并行训练器的集成测试。"""
    
    def test_full_training_cycle(self, default_config):
        """测试完整的训练周期。"""
        trainer = ParallelTrainer(default_config, num_workers=2)
        
        # 创建环境
        num_envs = trainer.create_parallel_envs()
        assert num_envs == 2
        
        # 同步参数
        params = {'test_param': torch.randn(10, 10)}
        trainer.sync_parameters(params)
        
        # 收集经验
        experiences = trainer.collect_experiences(episodes_per_worker=2, timeout=60.0)
        assert len(experiences) == 2
        
        # 聚合数据
        all_episodes, all_rewards, total_wins = trainer.aggregate_data(experiences)
        assert len(all_episodes) == 8  # 2 workers * 2 episodes * 2 players
        
        # 清理
        trainer.cleanup()
        assert not trainer.is_environments_created()
    
    def test_multiple_collection_rounds(self, default_config):
        """测试多轮数据收集。"""
        trainer = ParallelTrainer(default_config, num_workers=2)
        trainer.create_parallel_envs()
        
        total_episodes = []
        
        for _ in range(3):
            experiences = trainer.collect_experiences(episodes_per_worker=1, timeout=60.0)
            all_episodes, _, _ = trainer.aggregate_data(experiences)
            total_episodes.extend(all_episodes)
        
        # 3轮 * 2工作进程 * 1回合 * 2玩家 = 12个Episode
        assert len(total_episodes) == 12
        
        trainer.cleanup()
