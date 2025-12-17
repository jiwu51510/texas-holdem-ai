"""指标收集器模块 - 收集和聚合训练指标。

该模块实现了MetricsCollector类，用于：
- 记录训练回合数据（状态、行动、奖励）
- 计算训练指标（胜率、平均奖励、损失值、回合数）
- 维护滑动窗口统计
"""

from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import threading

from models.core import Episode


@dataclass
class EpisodeRecord:
    """单个回合的记录数据。
    
    Attributes:
        episode_number: 回合编号
        timestamp: 记录时间戳
        player_id: 玩家ID
        final_reward: 最终奖励
        num_actions: 行动数量
        won: 是否获胜
    """
    episode_number: int
    timestamp: datetime
    player_id: int
    final_reward: float
    num_actions: int
    won: bool


class MetricsCollector:
    """训练指标收集器。
    
    收集和聚合训练过程中的各种指标，包括：
    - 胜率（win_rate）
    - 平均奖励（avg_reward）
    - 损失值（loss）
    - 已完成回合数（episodes_completed）
    
    支持滑动窗口统计，默认窗口大小为100回合。
    
    Attributes:
        window_size: 滑动窗口大小
        episodes_completed: 已完成的回合总数
    """
    
    def __init__(self, window_size: int = 100):
        """初始化指标收集器。
        
        Args:
            window_size: 滑动窗口大小，用于计算最近N回合的统计数据
        """
        if window_size <= 0:
            raise ValueError(f"窗口大小必须为正数，收到 {window_size}")
        
        self._window_size = window_size
        self._episodes_completed = 0
        
        # 滑动窗口存储最近的回合记录
        self._recent_episodes: deque[EpisodeRecord] = deque(maxlen=window_size)
        
        # 全局统计
        self._total_wins = 0
        self._total_reward = 0.0
        
        # 损失值历史（由外部训练器更新）
        self._loss_history: List[float] = []
        self._recent_losses: deque[float] = deque(maxlen=window_size)
        
        # 指标历史记录
        self._metric_history: Dict[str, List[float]] = {
            'win_rate': [],
            'avg_reward': [],
            'loss': [],
            'episodes_completed': []
        }
        
        # 线程安全锁
        self._lock = threading.Lock()
    
    @property
    def window_size(self) -> int:
        """获取滑动窗口大小。"""
        return self._window_size
    
    @property
    def episodes_completed(self) -> int:
        """获取已完成的回合总数。"""
        with self._lock:
            return self._episodes_completed
    
    def record_episode(self, episode: Episode) -> None:
        """记录一个训练回合的数据。
        
        Args:
            episode: 训练回合数据
        """
        with self._lock:
            self._episodes_completed += 1
            
            # 判断是否获胜（最终奖励为正表示获胜）
            won = episode.final_reward > 0
            
            # 创建回合记录
            record = EpisodeRecord(
                episode_number=self._episodes_completed,
                timestamp=datetime.now(),
                player_id=episode.player_id,
                final_reward=episode.final_reward,
                num_actions=len(episode.actions),
                won=won
            )
            
            # 更新滑动窗口
            self._recent_episodes.append(record)
            
            # 更新全局统计
            if won:
                self._total_wins += 1
            self._total_reward += episode.final_reward
            
            # 更新指标历史
            self._update_metric_history()
    
    def record_loss(self, loss: float) -> None:
        """记录训练损失值。
        
        Args:
            loss: 损失值
        """
        with self._lock:
            self._loss_history.append(loss)
            self._recent_losses.append(loss)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """获取当前训练指标。
        
        Returns:
            包含以下指标的字典：
            - win_rate: 胜率（最近window_size回合）
            - avg_reward: 平均奖励（最近window_size回合）
            - loss: 最近的损失值（如果有）
            - episodes_completed: 已完成回合数
        """
        with self._lock:
            metrics = {
                'win_rate': self._calculate_window_win_rate(),
                'avg_reward': self._calculate_window_avg_reward(),
                'loss': self._get_recent_loss(),
                'episodes_completed': float(self._episodes_completed)
            }
            return metrics
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """获取指定指标的历史数据。
        
        Args:
            metric_name: 指标名称（'win_rate', 'avg_reward', 'loss', 'episodes_completed'）
            
        Returns:
            指标的历史数据列表
            
        Raises:
            ValueError: 如果指标名称无效
        """
        with self._lock:
            if metric_name not in self._metric_history:
                valid_names = list(self._metric_history.keys())
                raise ValueError(f"无效的指标名称: {metric_name}。有效名称: {valid_names}")
            return self._metric_history[metric_name].copy()
    
    def get_global_stats(self) -> Dict[str, float]:
        """获取全局统计数据（从训练开始到现在）。
        
        Returns:
            包含全局统计的字典：
            - total_episodes: 总回合数
            - total_wins: 总胜利数
            - global_win_rate: 全局胜率
            - global_avg_reward: 全局平均奖励
        """
        with self._lock:
            total_episodes = self._episodes_completed
            global_win_rate = self._total_wins / total_episodes if total_episodes > 0 else 0.0
            global_avg_reward = self._total_reward / total_episodes if total_episodes > 0 else 0.0
            
            return {
                'total_episodes': float(total_episodes),
                'total_wins': float(self._total_wins),
                'global_win_rate': global_win_rate,
                'global_avg_reward': global_avg_reward
            }
    
    def get_window_stats(self) -> Dict[str, float]:
        """获取滑动窗口内的统计数据。
        
        Returns:
            包含窗口统计的字典：
            - window_episodes: 窗口内回合数
            - window_wins: 窗口内胜利数
            - window_win_rate: 窗口内胜率
            - window_avg_reward: 窗口内平均奖励
        """
        with self._lock:
            window_episodes = len(self._recent_episodes)
            window_wins = sum(1 for ep in self._recent_episodes if ep.won)
            window_win_rate = window_wins / window_episodes if window_episodes > 0 else 0.0
            window_avg_reward = (
                sum(ep.final_reward for ep in self._recent_episodes) / window_episodes
                if window_episodes > 0 else 0.0
            )
            
            return {
                'window_episodes': float(window_episodes),
                'window_wins': float(window_wins),
                'window_win_rate': window_win_rate,
                'window_avg_reward': window_avg_reward
            }
    
    def reset(self) -> None:
        """重置所有指标和统计数据。"""
        with self._lock:
            self._episodes_completed = 0
            self._recent_episodes.clear()
            self._total_wins = 0
            self._total_reward = 0.0
            self._loss_history.clear()
            self._recent_losses.clear()
            for key in self._metric_history:
                self._metric_history[key].clear()
    
    def _calculate_window_win_rate(self) -> float:
        """计算滑动窗口内的胜率。"""
        if not self._recent_episodes:
            return 0.0
        wins = sum(1 for ep in self._recent_episodes if ep.won)
        return wins / len(self._recent_episodes)
    
    def _calculate_window_avg_reward(self) -> float:
        """计算滑动窗口内的平均奖励。"""
        if not self._recent_episodes:
            return 0.0
        total = sum(ep.final_reward for ep in self._recent_episodes)
        return total / len(self._recent_episodes)
    
    def _get_recent_loss(self) -> float:
        """获取最近的损失值。"""
        if not self._recent_losses:
            return 0.0
        return self._recent_losses[-1]
    
    def _update_metric_history(self) -> None:
        """更新指标历史记录。"""
        self._metric_history['win_rate'].append(self._calculate_window_win_rate())
        self._metric_history['avg_reward'].append(self._calculate_window_avg_reward())
        self._metric_history['loss'].append(self._get_recent_loss())
        self._metric_history['episodes_completed'].append(float(self._episodes_completed))
