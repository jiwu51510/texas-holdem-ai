"""训练引擎模块 - 管理德州扑克AI的训练流程。

本模块实现了训练引擎的核心功能：
- 初始化训练环境、神经网络、优化器和Deep CFR训练器
- 执行Deep CFR训练流程
- 管理训练循环和检查点保存
- 支持从检查点恢复训练
- 实现优雅终止（Ctrl+C时保存状态）
- 支持卡牌抽象集成，减少状态空间

架构说明（Deep CFR）：
- RegretNetwork（遗憾网络）：学习每个动作的即时遗憾值
- PolicyNetwork（策略网络）：学习长期平均策略
- 使用蓄水池采样的经验回放缓冲区
- 可选的卡牌抽象支持，大幅减少信息集数量
"""

import signal
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.core import (
    TrainingConfig, Episode, GameState, Action, ActionType, GameStage
)
from models.networks import PolicyNetwork, RegretNetwork
from environment.poker_environment import PokerEnvironment
from environment.state_encoder import StateEncoder
from training.deep_cfr_trainer import DeepCFRTrainer
from training.reservoir_buffer import ReservoirBuffer
from training.parallel_trainer import ParallelTrainer
from utils.checkpoint_manager import CheckpointManager

# 可选的卡牌抽象支持
try:
    from abstraction.card_abstraction import CardAbstraction
    from abstraction.data_classes import AbstractionConfig
    ABSTRACTION_AVAILABLE = True
except ImportError:
    ABSTRACTION_AVAILABLE = False
    CardAbstraction = None
    AbstractionConfig = None

# 可选的TensorBoard支持
try:
    from monitoring.tensorboard_logger import TensorBoardLogger, is_tensorboard_available
    TENSORBOARD_AVAILABLE = is_tensorboard_available()
except ImportError:
    TENSORBOARD_AVAILABLE = False
    TensorBoardLogger = None


class TrainingEngine:
    """训练引擎 - 管理德州扑克AI的完整训练流程（Deep CFR架构）。
    
    提供以下功能：
    - 初始化训练组件（环境、遗憾网络、策略网络、缓冲区）
    - 执行Deep CFR训练流程
    - 管理训练循环和进度显示
    - 按间隔自动保存检查点
    - 支持从检查点恢复训练
    - 实现优雅终止（Ctrl+C时保存状态）
    
    Deep CFR架构：
    - RegretNetwork：学习每个动作的即时遗憾值
    - PolicyNetwork：学习长期平均策略
    - ReservoirBuffer：使用蓄水池采样的经验回放缓冲区
    
    Attributes:
        config: 训练配置
        env: 游戏环境
        state_encoder: 状态编码器
        regret_network: 遗憾网络
        policy_network: 策略网络
        regret_optimizer: 遗憾网络优化器
        policy_optimizer: 策略网络优化器
        regret_buffer: 遗憾值缓冲区
        strategy_buffer: 策略缓冲区
        deep_cfr_trainer: Deep CFR训练器
        checkpoint_manager: 检查点管理器
        current_episode: 当前训练回合数（CFR迭代次数）
        total_rewards: 累积奖励列表
        win_count: 胜利次数
        should_stop: 是否应该停止训练的标志
    """
    
    def __init__(
        self, 
        config: TrainingConfig,
        checkpoint_dir: str = "checkpoints",
        tensorboard_dir: str = "runs",
        enable_tensorboard: bool = True,
        experiment_name: Optional[str] = None
    ):
        """初始化训练引擎（Deep CFR架构）。
        
        Args:
            config: 训练配置对象
            checkpoint_dir: 检查点保存目录
            tensorboard_dir: TensorBoard日志目录
            enable_tensorboard: 是否启用TensorBoard
            experiment_name: 实验名称（用于TensorBoard）
        """
        self.config = config
        
        # 初始化TensorBoard日志记录器
        self.tensorboard_logger: Optional[TensorBoardLogger] = None
        if enable_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                self.tensorboard_logger = TensorBoardLogger(
                    log_dir=tensorboard_dir,
                    experiment_name=experiment_name
                )
                print(f"TensorBoard已启用，日志目录: {self.tensorboard_logger.log_dir}")
            except Exception as e:
                print(f"TensorBoard初始化失败: {e}")
                self.tensorboard_logger = None
        elif enable_tensorboard and not TENSORBOARD_AVAILABLE:
            print("TensorBoard未安装，跳过。安装命令: pip install tensorboard")
        
        # 初始化卡牌抽象（如果配置启用）
        self.card_abstraction: Optional['CardAbstraction'] = None
        self._init_card_abstraction()
        
        # 初始化游戏环境
        self.env = PokerEnvironment(
            initial_stack=config.initial_stack,
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            max_raises_per_street=config.max_raises_per_street
        )
        
        # 初始化状态编码器（传入卡牌抽象）
        self.state_encoder = StateEncoder(card_abstraction=self.card_abstraction)
        
        # 初始化神经网络（Deep CFR架构）
        input_dim = self.state_encoder.encoding_dim  # 370
        action_dim = 6  # FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN
        
        # 遗憾网络：学习每个动作的即时遗憾值
        self.regret_network = RegretNetwork(
            input_dim=input_dim,
            hidden_dims=config.network_architecture,
            action_dim=action_dim
        )
        
        # 策略网络：学习长期平均策略
        self.policy_network = PolicyNetwork(
            input_dim=input_dim,
            hidden_dims=config.network_architecture,
            action_dim=action_dim
        )
        
        # 初始化优化器
        self.regret_optimizer = optim.Adam(
            self.regret_network.parameters(),
            lr=config.learning_rate
        )
        
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate
        )
        
        # 初始化经验回放缓冲区（蓄水池采样）
        self.regret_buffer = ReservoirBuffer(config.regret_buffer_size)
        self.strategy_buffer = ReservoirBuffer(config.strategy_buffer_size)
        
        # 初始化Deep CFR训练器
        self.deep_cfr_trainer = DeepCFRTrainer(config)
        
        # 如果启用了抽象，将抽象对象传递给Deep CFR训练器的状态编码器
        if self.card_abstraction is not None:
            self.deep_cfr_trainer.state_encoder.set_card_abstraction(self.card_abstraction)
        
        # 初始化并行训练器（如果配置了多个并行环境）
        self.parallel_trainer: Optional[ParallelTrainer] = None
        if config.num_parallel_envs > 1:
            self.parallel_trainer = ParallelTrainer(config, num_workers=config.num_parallel_envs)
            print(f"已启用并行训练，工作进程数: {config.num_parallel_envs}")
        
        # 初始化检查点管理器
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # 训练状态
        self.current_episode = 0
        self.total_rewards: List[float] = []
        self.win_count = 0
        self.should_stop = False
        
        # 设置信号处理器以实现优雅终止
        self._setup_signal_handlers()
    
    def _init_card_abstraction(self) -> None:
        """初始化卡牌抽象模块。
        
        根据配置加载预计算的抽象或创建新的抽象对象。
        """
        if not self.config.use_abstraction:
            print("卡牌抽象未启用")
            return
        
        if not ABSTRACTION_AVAILABLE:
            print("警告：卡牌抽象模块不可用，跳过抽象初始化")
            return
        
        # 创建抽象配置
        abstraction_config = None
        if self.config.abstraction_config:
            abstraction_config = AbstractionConfig.from_dict(self.config.abstraction_config)
        else:
            abstraction_config = AbstractionConfig()
        
        # 创建抽象对象
        self.card_abstraction = CardAbstraction(abstraction_config)
        
        # 如果指定了抽象文件路径，尝试加载
        if self.config.abstraction_path:
            if os.path.exists(self.config.abstraction_path):
                try:
                    self.card_abstraction.load(self.config.abstraction_path)
                    print(f"已加载卡牌抽象: {self.config.abstraction_path}")
                    
                    # 检查配置是否匹配
                    if not self.card_abstraction.config_matches(abstraction_config):
                        print("警告：加载的抽象配置与当前配置不匹配")
                        self._log_abstraction_config_mismatch(abstraction_config)
                    
                    # 输出抽象统计信息
                    self._log_abstraction_stats()
                except Exception as e:
                    print(f"加载卡牌抽象失败: {e}")
                    self.card_abstraction = None
            else:
                print(f"警告：抽象文件路径不存在: {self.config.abstraction_path}")
                print("请先使用 generate-abstraction 命令生成抽象")
                self.card_abstraction = None
        else:
            print("警告：启用了抽象但未指定抽象文件路径")
            self.card_abstraction = None
    
    def _log_abstraction_config_mismatch(self, expected_config: 'AbstractionConfig') -> None:
        """记录抽象配置不匹配的详细信息。
        
        Args:
            expected_config: 期望的配置
        """
        if self.card_abstraction is None or self.card_abstraction.result is None:
            return
        
        loaded_config = self.card_abstraction.result.config
        
        mismatches = []
        if loaded_config.preflop_buckets != expected_config.preflop_buckets:
            mismatches.append(f"preflop_buckets: 加载={loaded_config.preflop_buckets}, 期望={expected_config.preflop_buckets}")
        if loaded_config.flop_buckets != expected_config.flop_buckets:
            mismatches.append(f"flop_buckets: 加载={loaded_config.flop_buckets}, 期望={expected_config.flop_buckets}")
        if loaded_config.turn_buckets != expected_config.turn_buckets:
            mismatches.append(f"turn_buckets: 加载={loaded_config.turn_buckets}, 期望={expected_config.turn_buckets}")
        if loaded_config.river_buckets != expected_config.river_buckets:
            mismatches.append(f"river_buckets: 加载={loaded_config.river_buckets}, 期望={expected_config.river_buckets}")
        if loaded_config.use_potential_aware != expected_config.use_potential_aware:
            mismatches.append(f"use_potential_aware: 加载={loaded_config.use_potential_aware}, 期望={expected_config.use_potential_aware}")
        
        if mismatches:
            print("配置不匹配详情:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")
    
    def _log_abstraction_stats(self) -> None:
        """记录抽象统计信息。"""
        if self.card_abstraction is None:
            return
        
        try:
            stats = self.card_abstraction.get_abstraction_stats()
            print("抽象统计信息:")
            print(f"  - 生成耗时: {stats.get('generation_time', 0):.2f}秒")
            
            stages = stats.get('stages', {})
            for stage_name, stage_stats in stages.items():
                count = stage_stats.get('count', 0)
                avg_size = stage_stats.get('avg_size', 0)
                print(f"  - {stage_name}: {count}个桶, 平均大小={avg_size:.1f}")
        except Exception as e:
            print(f"获取抽象统计信息失败: {e}")
    
    def check_abstraction_config_change(self) -> bool:
        """检查抽象配置是否发生变化。
        
        Returns:
            如果配置发生变化，返回True
        """
        if not self.config.use_abstraction:
            return False
        
        if self.card_abstraction is None:
            return False
        
        if not self.config.abstraction_config:
            return False
        
        expected_config = AbstractionConfig.from_dict(self.config.abstraction_config)
        return not self.card_abstraction.config_matches(expected_config)
    
    def is_abstraction_enabled(self) -> bool:
        """检查抽象是否已启用并加载。
        
        Returns:
            如果抽象已启用并加载，返回True
        """
        return (self.config.use_abstraction and 
                self.card_abstraction is not None and 
                self.card_abstraction.is_loaded())
    
    def _setup_signal_handlers(self) -> None:
        """设置信号处理器以实现优雅终止。"""
        def signal_handler(signum, frame):
            print("\n收到终止信号，正在保存状态...")
            self.should_stop = True
        
        # 注册SIGINT（Ctrl+C）和SIGTERM信号处理器
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def self_play_episode(self) -> Tuple[Episode, Episode]:
        """执行一次自我对弈回合。
        
        两个AI代理互相对弈一局完整的德州扑克游戏。
        
        Returns:
            Tuple[Episode, Episode]: 两个玩家的Episode对象
        """
        # 重置环境
        state = self.env.reset()
        
        # 记录两个玩家的轨迹
        states_p0: List[GameState] = [deepcopy(state)]
        states_p1: List[GameState] = [deepcopy(state)]
        actions_p0: List[Action] = []
        actions_p1: List[Action] = []
        rewards_p0: List[float] = []
        rewards_p1: List[float] = []
        
        done = False
        
        while not done:
            current_player = state.current_player
            
            # 获取合法行动
            legal_actions = self.env.get_legal_actions(state)
            
            # 使用策略网络选择行动
            action = self._select_action(state, current_player, legal_actions)
            
            # 执行行动
            next_state, reward, done = self.env.step(action)
            
            # 记录轨迹
            if current_player == 0:
                actions_p0.append(action)
                rewards_p0.append(reward)
                if not done:
                    states_p0.append(deepcopy(next_state))
            else:
                actions_p1.append(action)
                rewards_p1.append(reward)
                if not done:
                    states_p1.append(deepcopy(next_state))
            
            state = next_state
        
        # 计算最终奖励
        final_reward_p0 = state.player_stacks[0] - self.config.initial_stack
        final_reward_p1 = state.player_stacks[1] - self.config.initial_stack
        
        # 确保状态和行动数量匹配
        # Episode要求: len(states) == len(actions) + 1
        if len(states_p0) != len(actions_p0) + 1:
            states_p0.append(deepcopy(state))
        if len(states_p1) != len(actions_p1) + 1:
            states_p1.append(deepcopy(state))
        
        # 确保奖励数量与行动数量匹配
        while len(rewards_p0) < len(actions_p0):
            rewards_p0.append(0.0)
        while len(rewards_p1) < len(actions_p1):
            rewards_p1.append(0.0)
        
        # 创建Episode对象
        episode_p0 = Episode(
            states=states_p0,
            actions=actions_p0,
            rewards=rewards_p0,
            player_id=0,
            final_reward=float(final_reward_p0)
        )
        
        episode_p1 = Episode(
            states=states_p1,
            actions=actions_p1,
            rewards=rewards_p1,
            player_id=1,
            final_reward=float(final_reward_p1)
        )
        
        return episode_p0, episode_p1
    
    def _select_action(
        self, 
        state: GameState, 
        player_id: int, 
        legal_actions: List[Action]
    ) -> Action:
        """使用遗憾网络生成策略选择行动（Deep CFR）。
        
        遗憾网络输出每个动作的遗憾值，通过Regret Matching转换为策略概率。
        对于同一类型的多个合法行动（如不同金额的RAISE），
        我们将该类型的概率平均分配给所有该类型的合法行动。
        
        Args:
            state: 当前游戏状态
            player_id: 玩家ID
            legal_actions: 合法行动列表
            
        Returns:
            选择的行动
        """
        if not legal_actions:
            # 如果没有合法行动，返回弃牌
            return Action(ActionType.FOLD)
        
        # 编码状态
        state_encoding = self.state_encoder.encode(state, player_id)
        state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
        
        # 使用遗憾网络的Regret Matching获取策略概率
        with torch.no_grad():
            action_probs = self.regret_network.get_strategy(state_tensor)
            action_probs = action_probs.squeeze(0).numpy()
        
        # 统计每种行动类型的合法行动数量
        action_type_counts = {}
        for action in legal_actions:
            action_type = action.action_type
            action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
        
        # 为每个合法行动分配概率
        # 同一类型的多个行动平分该类型的概率
        legal_probs = []
        for action in legal_actions:
            action_idx = self._action_to_index(action)
            type_prob = action_probs[action_idx]
            count = action_type_counts[action.action_type]
            # 平分该类型的概率
            legal_probs.append(type_prob / count)
        
        # 归一化概率
        legal_probs = np.array(legal_probs)
        total_prob = legal_probs.sum()
        
        if total_prob > 0:
            legal_probs = legal_probs / total_prob
        else:
            # 如果所有概率为0，使用均匀分布
            legal_probs = np.ones(len(legal_actions)) / len(legal_actions)
        
        # 按概率选择行动
        choice = np.random.choice(len(legal_actions), p=legal_probs)
        return legal_actions[choice]
    
    def _action_to_index(self, action: Action) -> int:
        """将行动转换为索引。
        
        Args:
            action: 行动对象
            
        Returns:
            行动索引
        """
        action_type_to_idx = {
            ActionType.FOLD: 0,
            ActionType.CHECK: 1,
            ActionType.CALL: 2,
            ActionType.RAISE_SMALL: 3,
            ActionType.RAISE_BIG: 4,
            ActionType.RAISE: 3  # 向后兼容
        }
        return action_type_to_idx.get(action.action_type, 0)
    
    def update_policy(self, episodes: List[Episode]) -> Dict[str, float]:
        """使用Deep CFR训练器更新网络（兼容旧接口）。
        
        注意：在Deep CFR架构中，主要的训练逻辑在train方法中通过
        deep_cfr_trainer执行。此方法保留用于兼容性，但实际上
        会委托给deep_cfr_trainer.train_networks()。
        
        Args:
            episodes: 训练回合列表（在Deep CFR中不直接使用）
            
        Returns:
            训练指标字典（损失值等）
        """
        if not episodes:
            return {'regret_loss': 0.0, 'policy_loss': 0.0}
        
        # 在Deep CFR中，训练是通过CFR迭代和网络训练分开进行的
        # 这里我们调用deep_cfr_trainer的训练方法
        metrics = self.deep_cfr_trainer.train_networks()
        
        # 同步网络参数（从deep_cfr_trainer到本地）
        self._sync_networks_from_trainer()
        
        # 记录到TensorBoard
        if self.tensorboard_logger is not None:
            step = self.current_episode
            self.tensorboard_logger.log_scalar('Loss/regret', metrics.get('regret_loss', 0.0), step)
            self.tensorboard_logger.log_scalar('Loss/policy', metrics.get('policy_loss', 0.0), step)
        
        return metrics
    
    def _sync_networks_from_trainer(self) -> None:
        """从deep_cfr_trainer同步网络参数到本地网络。"""
        self.regret_network.load_state_dict(
            self.deep_cfr_trainer.regret_network.state_dict()
        )
        self.policy_network.load_state_dict(
            self.deep_cfr_trainer.policy_network.state_dict()
        )
    
    def _sync_networks_to_trainer(self) -> None:
        """将本地网络参数同步到deep_cfr_trainer。"""
        self.deep_cfr_trainer.regret_network.load_state_dict(
            self.regret_network.state_dict()
        )
        self.deep_cfr_trainer.policy_network.load_state_dict(
            self.policy_network.state_dict()
        )
    
    def train(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """执行Deep CFR主训练循环。
        
        Deep CFR训练流程：
        1. 执行cfr_iterations_per_update次CFR迭代，收集样本
        2. 训练遗憾网络和策略网络
        3. 重复直到达到目标迭代次数
        
        Args:
            num_episodes: CFR迭代次数，如果为None则使用配置中的值
            
        Returns:
            训练结果字典
        """
        if num_episodes is None:
            num_episodes = self.config.num_episodes
        
        # Deep CFR使用单进程训练（CFR遍历本身就是计算密集型）
        return self._train_deep_cfr(num_episodes)
    
    def _train_deep_cfr(self, num_iterations: int) -> Dict[str, Any]:
        """Deep CFR训练循环。
        
        Args:
            num_iterations: CFR迭代次数
            
        Returns:
            训练结果字典
        """
        target_iteration = self.current_episode + num_iterations
        cfr_iterations_per_update = self.config.cfr_iterations_per_update
        
        print(f"开始Deep CFR训练，目标迭代数: {num_iterations}")
        print(f"当前迭代: {self.current_episode}, 目标迭代: {target_iteration}")
        print(f"每次网络更新前的CFR迭代次数: {cfr_iterations_per_update}")
        print("-" * 50)
        
        # 同步网络参数到trainer
        self._sync_networks_to_trainer()
        
        try:
            while self.current_episode < target_iteration and not self.should_stop:
                # 执行CFR迭代，收集样本
                cfr_metrics_list = []
                for _ in range(min(cfr_iterations_per_update, target_iteration - self.current_episode)):
                    if self.should_stop:
                        break
                    
                    # 每10次迭代输出一次详细采样信息
                    verbose = (self.current_episode % 10 == 0)
                    cfr_metrics = self.deep_cfr_trainer.run_cfr_iteration(verbose=verbose)
                    cfr_metrics_list.append(cfr_metrics)
                    
                    # 记录收益（用于统计）
                    utility = cfr_metrics.get('utility_p0', 0.0)
                    self.total_rewards.append(utility)
                    if utility > 0:
                        self.win_count += 1
                    
                    self.current_episode += 1
                    
                    # 显示进度
                    if self.current_episode % 100 == 0:
                        self._log_progress(target_iteration)
                
                # 训练网络（在保存检查点之前）
                should_save_checkpoint = (
                    self.current_episode % self.config.checkpoint_interval == 0 and 
                    cfr_metrics_list
                )
                
                if cfr_metrics_list:
                    train_metrics = self.deep_cfr_trainer.train_networks()
                    
                    # 同步网络参数
                    self._sync_networks_from_trainer()
                    
                    # 保存检查点（在网络训练之后）
                    if should_save_checkpoint:
                        self._save_checkpoint()
                    
                    # 记录到TensorBoard
                    if self.tensorboard_logger is not None:
                        self.tensorboard_logger.log_scalar(
                            'Loss/regret', train_metrics.get('regret_loss', 0.0), self.current_episode
                        )
                        self.tensorboard_logger.log_scalar(
                            'Loss/policy', train_metrics.get('policy_loss', 0.0), self.current_episode
                        )
                        self.tensorboard_logger.log_scalar(
                            'Buffer/regret_size', len(self.deep_cfr_trainer.regret_buffer), self.current_episode
                        )
                        self.tensorboard_logger.log_scalar(
                            'Buffer/strategy_size', len(self.deep_cfr_trainer.strategy_buffer), self.current_episode
                        )
        
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            self._sync_networks_from_trainer()
            self._save_checkpoint()
            raise
        
        # 同步最终网络参数
        self._sync_networks_from_trainer()
        
        return self._finalize_training([])
    
    def _train_single(self, num_episodes: int) -> Dict[str, Any]:
        """单进程训练循环（兼容旧接口，实际调用Deep CFR训练）。
        
        Args:
            num_episodes: 训练回合数
            
        Returns:
            训练结果字典
        """
        return self._train_deep_cfr(num_episodes)
    
    def _train_parallel(self, num_episodes: int) -> Dict[str, Any]:
        """多进程并行训练循环（Deep CFR不支持，回退到单进程）。
        
        注意：Deep CFR的CFR遍历需要完整的游戏树访问，
        不适合简单的并行化。因此回退到单进程训练。
        
        Args:
            num_episodes: 训练回合数
            
        Returns:
            训练结果字典
        """
        print("警告：Deep CFR架构不支持并行训练，回退到单进程模式")
        return self._train_deep_cfr(num_episodes)
    
    def _log_progress(self, target_episode: int) -> None:
        """记录训练进度。
        
        Args:
            target_episode: 目标迭代数
        """
        win_rate = self.win_count / self.current_episode if self.current_episode > 0 else 0
        avg_reward = np.mean(self.total_rewards[-100:]) if self.total_rewards else 0
        
        # 获取缓冲区大小
        regret_buffer_size = len(self.deep_cfr_trainer.regret_buffer)
        strategy_buffer_size = len(self.deep_cfr_trainer.strategy_buffer)
        
        print(f"迭代 {self.current_episode}/{target_episode} | "
              f"胜率: {win_rate:.2%} | "
              f"平均收益: {avg_reward:.2f} | "
              f"缓冲区: R={regret_buffer_size}, S={strategy_buffer_size}")
        
        # 记录到TensorBoard
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_scalar('Metrics/win_rate', win_rate, self.current_episode)
            self.tensorboard_logger.log_scalar('Metrics/avg_reward', avg_reward, self.current_episode)
            self.tensorboard_logger.log_scalar('Progress/iterations', self.current_episode, self.current_episode)
            
            # 记录Deep CFR信息
            self.tensorboard_logger.log_scalar(
                'DeepCFR/regret_buffer_size', 
                regret_buffer_size, 
                self.current_episode
            )
            self.tensorboard_logger.log_scalar(
                'DeepCFR/strategy_buffer_size', 
                strategy_buffer_size, 
                self.current_episode
            )
    
    def _finalize_training(self, remaining_episodes: List[Episode]) -> Dict[str, Any]:
        """完成训练并返回结果。
        
        Args:
            remaining_episodes: 剩余未处理的回合（Deep CFR中不使用）
            
        Returns:
            训练结果字典
        """
        # 保存最终检查点
        self._save_checkpoint()
        
        # 计算最终统计
        win_rate = self.win_count / self.current_episode if self.current_episode > 0 else 0
        avg_reward = np.mean(self.total_rewards) if self.total_rewards else 0
        
        # 获取Deep CFR指标
        cfr_metrics = self.deep_cfr_trainer.get_metrics()
        
        print("-" * 50)
        print(f"Deep CFR训练完成！")
        print(f"总迭代数: {self.current_episode}")
        print(f"最终胜率: {win_rate:.2%}")
        print(f"平均收益: {avg_reward:.2f}")
        print(f"遗憾缓冲区大小: {cfr_metrics['regret_buffer_size']}")
        print(f"策略缓冲区大小: {cfr_metrics['strategy_buffer_size']}")
        
        # 关闭TensorBoard日志记录器
        if self.tensorboard_logger is not None:
            # 记录最终超参数和指标
            self.tensorboard_logger.log_hparams(
                hparam_dict={
                    'learning_rate': self.config.learning_rate,
                    'batch_size': self.config.batch_size,
                    'cfr_iterations_per_update': self.config.cfr_iterations_per_update,
                    'network_train_steps': self.config.network_train_steps,
                    'regret_buffer_size': self.config.regret_buffer_size,
                    'strategy_buffer_size': self.config.strategy_buffer_size,
                },
                metric_dict={
                    'hparam/win_rate': win_rate,
                    'hparam/avg_reward': avg_reward,
                }
            )
            self.tensorboard_logger.flush()
        
        return {
            'total_episodes': self.current_episode,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'total_rewards': self.total_rewards,
            'regret_buffer_size': cfr_metrics['regret_buffer_size'],
            'strategy_buffer_size': cfr_metrics['strategy_buffer_size']
        }
    
    def _save_checkpoint(self) -> str:
        """保存当前训练状态到检查点。
        
        保存遗憾网络和策略网络的参数（Deep CFR架构）。
        
        Returns:
            检查点文件路径
        """
        win_rate = self.win_count / self.current_episode if self.current_episode > 0 else 0
        avg_reward = np.mean(self.total_rewards[-100:]) if self.total_rewards else 0
        
        # 获取Deep CFR指标
        cfr_metrics = self.deep_cfr_trainer.get_metrics()
        
        metadata = {
            'episode_number': self.current_episode,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'win_count': self.win_count,
            'total_rewards_count': len(self.total_rewards),
            'cfr_iterations': cfr_metrics['iteration'],
            'regret_buffer_size': cfr_metrics['regret_buffer_size'],
            'strategy_buffer_size': cfr_metrics['strategy_buffer_size'],
            'checkpoint_format': 'deep_cfr_v1'  # 标记检查点格式版本
        }
        
        # 保存遗憾网络和策略网络（使用新的检查点格式）
        checkpoint_path = self._save_deep_cfr_checkpoint(metadata)
        
        print(f"检查点已保存: {checkpoint_path}")
        return checkpoint_path
    
    def _save_deep_cfr_checkpoint(self, metadata: Dict[str, Any]) -> str:
        """保存Deep CFR检查点（新格式）。
        
        Args:
            metadata: 元数据字典
            
        Returns:
            检查点文件路径
        """
        import time
        from datetime import datetime
        from pathlib import Path
        
        episode_number = metadata['episode_number']
        timestamp = int(time.time() * 1000000)
        filename = f"checkpoint_{timestamp}_{episode_number}.pt"
        filepath = Path(self.checkpoint_manager.checkpoint_dir) / filename
        
        # 获取动作配置信息
        action_dim = 6  # 与初始化时一致
        action_names = ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG', 'ALL_IN']
        
        checkpoint_data = {
            # 网络参数
            'regret_network_state_dict': self.regret_network.state_dict(),
            'policy_network_state_dict': self.policy_network.state_dict(),
            # 优化器参数
            'regret_optimizer_state_dict': self.regret_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            # 元数据
            'episode_number': episode_number,
            'timestamp': datetime.now().isoformat(),
            'win_rate': metadata.get('win_rate', 0.0),
            'avg_reward': metadata.get('avg_reward', 0.0),
            'checkpoint_format': 'deep_cfr_v1',
            # 动作配置（需求 3.1, 3.2）
            'action_config': {
                'action_names': action_names,
                'action_dim': action_dim,
            },
            # 其他元数据
            **{k: v for k, v in metadata.items() if k not in ['episode_number', 'win_rate', 'avg_reward', 'checkpoint_format']}
        }
        
        torch.save(checkpoint_data, filepath)
        return str(filepath)
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """保存训练检查点（公共接口）。
        
        Args:
            path: 可选的保存路径（如果为None则使用默认命名）
            
        Returns:
            检查点文件路径
        """
        return self._save_checkpoint()
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """从检查点加载训练状态。
        
        支持两种检查点格式：
        1. Deep CFR格式（新）：包含regret_network和policy_network
        2. 旧格式：包含policy_network和value_network（兼容性处理）
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        from pathlib import Path
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        # 加载检查点数据
        checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        
        # 检测检查点格式
        checkpoint_format = checkpoint_data.get('checkpoint_format', 'legacy')
        
        if checkpoint_format == 'deep_cfr_v1':
            # 新格式：Deep CFR
            self._load_deep_cfr_checkpoint(checkpoint_data)
        else:
            # 旧格式：兼容性处理
            self._load_legacy_checkpoint(checkpoint_data)
        
        # 恢复训练状态
        self.current_episode = checkpoint_data.get('episode_number', 0)
        self.win_count = checkpoint_data.get('win_count', 0)
        
        # 同步网络参数到trainer
        self._sync_networks_to_trainer()
        
        # 恢复Deep CFR训练器状态
        self.deep_cfr_trainer.iteration = checkpoint_data.get('cfr_iterations', 0)
        
        print(f"从检查点恢复训练状态:")
        print(f"  格式: {checkpoint_format}")
        print(f"  迭代数: {self.current_episode}")
        print(f"  胜率: {checkpoint_data.get('win_rate', 0):.2%}")
        print(f"  平均收益: {checkpoint_data.get('avg_reward', 0):.2f}")
    
    def _load_deep_cfr_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """加载Deep CFR格式的检查点。
        
        Args:
            checkpoint_data: 检查点数据字典
        """
        # 加载遗憾网络
        if 'regret_network_state_dict' in checkpoint_data:
            self.regret_network.load_state_dict(checkpoint_data['regret_network_state_dict'])
        
        # 加载策略网络
        if 'policy_network_state_dict' in checkpoint_data:
            self.policy_network.load_state_dict(checkpoint_data['policy_network_state_dict'])
        
        # 加载优化器状态
        if 'regret_optimizer_state_dict' in checkpoint_data:
            self.regret_optimizer.load_state_dict(checkpoint_data['regret_optimizer_state_dict'])
        
        if 'policy_optimizer_state_dict' in checkpoint_data:
            self.policy_optimizer.load_state_dict(checkpoint_data['policy_optimizer_state_dict'])
    
    def _load_legacy_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """加载旧格式的检查点（兼容性处理）。
        
        旧格式包含policy_network和value_network。
        我们只加载policy_network，遗憾网络保持初始化状态。
        
        Args:
            checkpoint_data: 检查点数据字典
        """
        print("警告：检测到旧格式检查点，仅加载策略网络参数")
        
        # 加载策略网络（旧格式中的model_state_dict）
        if 'model_state_dict' in checkpoint_data:
            self.policy_network.load_state_dict(checkpoint_data['model_state_dict'])
        
        # 加载策略网络优化器
        if 'optimizer_state_dict' in checkpoint_data:
            self.policy_optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # 旧格式的value_network不再使用，跳过
        if checkpoint_data.get('has_value_network'):
            print("  注意：旧检查点中的价值网络参数已被忽略（Deep CFR不使用价值网络）")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """获取当前训练指标。
        
        Returns:
            训练指标字典
        """
        win_rate = self.win_count / self.current_episode if self.current_episode > 0 else 0
        avg_reward = np.mean(self.total_rewards[-100:]) if self.total_rewards else 0
        
        # 获取Deep CFR指标
        cfr_metrics = self.deep_cfr_trainer.get_metrics()
        
        return {
            'episode': self.current_episode,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'win_count': self.win_count,
            'cfr_iterations': cfr_metrics['iteration'],
            'regret_buffer_size': cfr_metrics['regret_buffer_size'],
            'strategy_buffer_size': cfr_metrics['strategy_buffer_size']
        }

