"""并行训练器模块 - 支持多进程并行训练德州扑克AI。

本模块实现了并行训练的核心功能：
- 创建多个并行游戏环境
- 从所有进程收集经验数据
- 聚合所有进程的数据
- 参数同步机制
- 错误处理和优雅终止
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Manager
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch
import traceback
import time
import signal
from copy import deepcopy

from models.core import (
    TrainingConfig, Episode, GameState, Action, ActionType, GameStage
)
from environment.poker_environment import PokerEnvironment
from environment.state_encoder import StateEncoder


@dataclass
class WorkerExperience:
    """工作进程收集的经验数据。
    
    Attributes:
        worker_id: 工作进程ID
        episodes: 收集的Episode列表
        total_rewards: 累积奖励列表
        win_count: 胜利次数
        error: 如果发生错误，包含错误信息
    """
    worker_id: int
    episodes: List[Episode]
    total_rewards: List[float]
    win_count: int
    error: Optional[str] = None


def worker_process(
    worker_id: int,
    config: TrainingConfig,
    policy_params: Dict[str, torch.Tensor],
    num_episodes: int,
    result_queue: Queue,
    stop_event: Event,
    error_event: Event
) -> None:
    """工作进程函数 - 在独立进程中运行游戏模拟。
    
    Args:
        worker_id: 工作进程ID
        config: 训练配置
        policy_params: 策略网络参数（用于同步）
        num_episodes: 要执行的回合数
        result_queue: 结果队列，用于返回经验数据
        stop_event: 停止事件，用于优雅终止
        error_event: 错误事件，用于通知主进程发生错误
    """
    try:
        # 初始化游戏环境
        env = PokerEnvironment(
            initial_stack=config.initial_stack,
            small_blind=config.small_blind,
            big_blind=config.big_blind
        )
        
        # 初始化状态编码器
        state_encoder = StateEncoder()
        
        # 收集经验数据
        episodes = []
        total_rewards = []
        win_count = 0
        
        for episode_idx in range(num_episodes):
            # 检查是否应该停止
            if stop_event.is_set():
                break
            
            # 执行一次自我对弈
            episode_p0, episode_p1 = _self_play_episode(
                env, state_encoder, config, policy_params
            )
            
            episodes.append(episode_p0)
            episodes.append(episode_p1)
            
            # 记录奖励
            total_rewards.append(episode_p0.final_reward)
            if episode_p0.final_reward > 0:
                win_count += 1
        
        # 将结果放入队列
        experience = WorkerExperience(
            worker_id=worker_id,
            episodes=episodes,
            total_rewards=total_rewards,
            win_count=win_count,
            error=None
        )
        result_queue.put(experience)
        
    except Exception as e:
        # 发生错误，设置错误事件并将错误信息放入队列
        error_event.set()
        error_msg = f"Worker {worker_id} error: {str(e)}\n{traceback.format_exc()}"
        experience = WorkerExperience(
            worker_id=worker_id,
            episodes=[],
            total_rewards=[],
            win_count=0,
            error=error_msg
        )
        result_queue.put(experience)


def _self_play_episode(
    env: PokerEnvironment,
    state_encoder: StateEncoder,
    config: TrainingConfig,
    policy_params: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[Episode, Episode]:
    """执行一次自我对弈回合。
    
    Args:
        env: 游戏环境
        state_encoder: 状态编码器
        config: 训练配置
        policy_params: 策略网络参数（可选，如果为None则使用随机策略）
        
    Returns:
        Tuple[Episode, Episode]: 两个玩家的Episode对象
    """
    # 重置环境
    state = env.reset()
    
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
        legal_actions = env.get_legal_actions(state)
        
        # 使用随机策略选择行动（简化版本，不使用神经网络）
        # 在实际使用中，可以加载策略网络参数并使用
        action = _select_random_action(legal_actions)
        
        # 执行行动
        next_state, reward, done = env.step(action)
        
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
    final_reward_p0 = state.player_stacks[0] - config.initial_stack
    final_reward_p1 = state.player_stacks[1] - config.initial_stack
    
    # 确保状态和行动数量匹配
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


def _select_random_action(legal_actions: List[Action]) -> Action:
    """随机选择一个合法行动。
    
    Args:
        legal_actions: 合法行动列表
        
    Returns:
        选择的行动
    """
    if not legal_actions:
        return Action(ActionType.FOLD)
    
    # 使用加权随机选择，偏向于非弃牌行动
    weights = []
    for action in legal_actions:
        if action.action_type == ActionType.FOLD:
            weights.append(0.1)  # 弃牌权重较低
        elif action.action_type == ActionType.CHECK:
            weights.append(0.3)
        elif action.action_type == ActionType.CALL:
            weights.append(0.3)
        else:  # RAISE
            weights.append(0.3)
    
    # 归一化权重
    total = sum(weights)
    weights = [w / total for w in weights]
    
    # 按权重选择
    choice = np.random.choice(len(legal_actions), p=weights)
    return legal_actions[choice]


class ParallelTrainer:
    """并行训练器 - 支持多进程并行训练德州扑克AI。
    
    提供以下功能：
    - 创建N个并行游戏环境
    - 从所有进程收集经验数据
    - 聚合所有进程的数据
    - 参数同步机制（主进程更新后广播到工作进程）
    - 错误处理（捕获进程异常）
    - 优雅终止所有进程
    
    Attributes:
        config: 训练配置
        num_workers: 工作进程数量
        workers: 工作进程列表
        result_queue: 结果队列
        stop_event: 停止事件
        error_event: 错误事件
        policy_params: 当前策略网络参数
        is_running: 是否正在运行
    """
    
    def __init__(self, config: TrainingConfig, num_workers: Optional[int] = None):
        """初始化并行训练器。
        
        Args:
            config: 训练配置
            num_workers: 工作进程数量，如果为None则使用配置中的值
        """
        self.config = config
        self.num_workers = num_workers if num_workers is not None else config.num_parallel_envs
        
        # 确保至少有1个工作进程
        if self.num_workers < 1:
            self.num_workers = 1
        
        # 进程管理
        self.workers: List[Process] = []
        self.result_queue: Optional[Queue] = None
        self.stop_event: Optional[Event] = None
        self.error_event: Optional[Event] = None
        
        # 策略参数（用于同步）
        self.policy_params: Dict[str, torch.Tensor] = {}
        
        # 状态标志
        self.is_running = False
        self._environments_created = False
    
    def create_parallel_envs(self) -> int:
        """创建N个并行游戏环境。
        
        初始化多进程所需的队列和事件。
        
        Returns:
            创建的环境数量
        """
        if self._environments_created:
            # 如果已经创建，先清理
            self.cleanup()
        
        # 创建多进程通信组件
        self.result_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.error_event = mp.Event()
        
        self._environments_created = True
        
        return self.num_workers
    
    def collect_experiences(
        self, 
        episodes_per_worker: int,
        timeout: float = 300.0
    ) -> List[WorkerExperience]:
        """从所有进程收集经验数据。
        
        启动工作进程，等待它们完成，并收集结果。
        
        Args:
            episodes_per_worker: 每个工作进程要执行的回合数
            timeout: 超时时间（秒）
            
        Returns:
            所有工作进程的经验数据列表
            
        Raises:
            RuntimeError: 如果环境未创建或发生进程错误
        """
        if not self._environments_created:
            raise RuntimeError("必须先调用 create_parallel_envs() 创建环境")
        
        # 清空之前的结果
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except:
                break
        
        # 重置事件
        self.stop_event.clear()
        self.error_event.clear()
        
        # 启动工作进程
        self.workers = []
        self.is_running = True
        
        for worker_id in range(self.num_workers):
            p = Process(
                target=worker_process,
                args=(
                    worker_id,
                    self.config,
                    self.policy_params,
                    episodes_per_worker,
                    self.result_queue,
                    self.stop_event,
                    self.error_event
                )
            )
            p.start()
            self.workers.append(p)
        
        # 收集结果
        experiences = []
        errors = []
        start_time = time.time()
        
        while len(experiences) < self.num_workers:
            # 检查超时
            if time.time() - start_time > timeout:
                self._terminate_workers()
                raise RuntimeError(f"收集经验数据超时（{timeout}秒）")
            
            # 检查错误事件
            if self.error_event.is_set():
                # 继续收集以获取错误信息
                pass
            
            try:
                # 尝试从队列获取结果
                experience = self.result_queue.get(timeout=1.0)
                experiences.append(experience)
                
                if experience.error:
                    errors.append(experience.error)
                    
            except:
                # 队列为空，继续等待
                pass
        
        # 等待所有进程结束
        for p in self.workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        
        self.workers = []
        self.is_running = False
        
        # 如果有错误，抛出异常
        if errors:
            raise RuntimeError(f"工作进程发生错误:\n" + "\n".join(errors))
        
        return experiences
    
    def aggregate_data(
        self, 
        experiences: List[WorkerExperience]
    ) -> Tuple[List[Episode], List[float], int]:
        """聚合所有进程的数据。
        
        将所有工作进程收集的经验数据合并为单一数据集。
        
        Args:
            experiences: 工作进程经验数据列表
            
        Returns:
            Tuple包含：
            - 所有Episode的列表
            - 所有奖励的列表
            - 总胜利次数
        """
        all_episodes = []
        all_rewards = []
        total_wins = 0
        
        for exp in experiences:
            all_episodes.extend(exp.episodes)
            all_rewards.extend(exp.total_rewards)
            total_wins += exp.win_count
        
        return all_episodes, all_rewards, total_wins
    
    def sync_parameters(self, policy_params: Dict[str, torch.Tensor]) -> None:
        """同步策略网络参数。
        
        将主进程的策略网络参数保存，以便下次启动工作进程时使用。
        
        Args:
            policy_params: 策略网络参数字典
        """
        # 深拷贝参数以避免引用问题
        self.policy_params = {
            k: v.clone().detach() for k, v in policy_params.items()
        }
    
    def cleanup(self) -> None:
        """优雅终止所有进程并清理资源。"""
        # 设置停止事件
        if self.stop_event is not None:
            self.stop_event.set()
        
        # 终止所有工作进程
        self._terminate_workers()
        
        # 清理队列
        if self.result_queue is not None:
            try:
                while not self.result_queue.empty():
                    self.result_queue.get_nowait()
            except:
                pass
            self.result_queue = None
        
        # 重置状态
        self.stop_event = None
        self.error_event = None
        self._environments_created = False
        self.is_running = False
    
    def _terminate_workers(self) -> None:
        """终止所有工作进程。"""
        for p in self.workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive():
                    # 强制终止
                    try:
                        p.kill()
                    except:
                        pass
        
        self.workers = []
    
    def get_num_workers(self) -> int:
        """获取工作进程数量。
        
        Returns:
            工作进程数量
        """
        return self.num_workers
    
    def is_environments_created(self) -> bool:
        """检查环境是否已创建。
        
        Returns:
            如果环境已创建返回True
        """
        return self._environments_created
    
    def __del__(self):
        """析构函数，确保清理资源。"""
        self.cleanup()
