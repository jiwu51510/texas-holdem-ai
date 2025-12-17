"""缓冲区管理器模块。

提供改进的缓冲区采样和管理功能，包括时间衰减采样、
重要性采样、分层采样和过旧样本清理。
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from training.reservoir_buffer import ReservoirBuffer


@dataclass
class BufferManagerConfig:
    """缓冲区管理器配置。
    
    Attributes:
        time_decay_factor: 时间衰减因子，控制近期样本的采样偏好（0-1之间）
        importance_threshold: 重要性阈值，低于此值的样本可能被替换
        max_sample_age: 最大样本年龄（迭代次数），超过此年龄的样本将被清理
        stratified_sampling: 是否使用分层采样
    """
    time_decay_factor: float = 0.99
    importance_threshold: float = 0.1
    max_sample_age: int = 10000
    stratified_sampling: bool = True
    
    def __post_init__(self):
        """验证配置参数。"""
        if not 0.0 <= self.time_decay_factor <= 1.0:
            raise ValueError(
                f"时间衰减因子必须在0-1之间，收到: {self.time_decay_factor}"
            )
        if self.importance_threshold < 0:
            raise ValueError(
                f"重要性阈值必须非负，收到: {self.importance_threshold}"
            )
        if self.max_sample_age <= 0:
            raise ValueError(
                f"最大样本年龄必须为正整数，收到: {self.max_sample_age}"
            )


class BufferManager:
    """缓冲区管理器。
    
    提供改进的缓冲区采样和管理功能，支持：
    - 时间衰减采样：近期样本有更高的采样概率
    - 重要性采样：基于样本重要性的采样
    - 分层采样：确保不同游戏阶段的样本均衡
    - 过旧样本清理：定期清理过旧的样本
    """
    
    def __init__(self, config: Optional[BufferManagerConfig] = None):
        """初始化缓冲区管理器。
        
        Args:
            config: 管理器配置，如果为None则使用默认配置
        """
        self.config = config or BufferManagerConfig()
    
    def sample_with_time_decay(
        self, 
        buffer: ReservoirBuffer,
        batch_size: int,
        current_iteration: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用时间衰减采样。
        
        近期样本有更高的采样概率。采样概率与样本年龄的指数衰减成正比：
        P(sample) ∝ decay_factor^(current_iteration - sample_iteration)
        
        Args:
            buffer: 蓄水池缓冲区
            batch_size: 请求的批次大小
            current_iteration: 当前迭代次数
            
        Returns:
            (states, targets, iterations) 三元组
            - states: 状态编码数组，形状为 (n, state_dim)
            - targets: 目标值数组，形状为 (n, target_dim)
            - iterations: 迭代编号数组，形状为 (n,)
            
            其中 n = min(batch_size, len(buffer))
        """
        if len(buffer) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # 计算每个样本的采样权重
        weights = self._compute_time_decay_weights(buffer, current_iteration)
        
        # 归一化权重为概率分布
        probabilities = weights / np.sum(weights)
        
        # 实际采样数量
        actual_size = min(batch_size, len(buffer))
        
        # 按概率采样索引（不放回）
        indices = np.random.choice(
            len(buffer),
            size=actual_size,
            replace=False,
            p=probabilities
        )
        
        # 收集样本
        return self._collect_samples(buffer, indices)
    
    def _compute_time_decay_weights(
        self, 
        buffer: ReservoirBuffer, 
        current_iteration: int
    ) -> np.ndarray:
        """计算时间衰减权重。
        
        权重 = decay_factor^(current_iteration - sample_iteration)
        
        Args:
            buffer: 蓄水池缓冲区
            current_iteration: 当前迭代次数
            
        Returns:
            每个样本的权重数组
        """
        weights = np.zeros(len(buffer))
        decay_factor = self.config.time_decay_factor
        
        for i, (_, _, iteration) in enumerate(buffer.buffer):
            age = max(0, current_iteration - iteration)
            weights[i] = decay_factor ** age
        
        # 确保权重非零（防止数值下溢）
        weights = np.maximum(weights, 1e-10)
        
        return weights
    
    def _collect_samples(
        self, 
        buffer: ReservoirBuffer, 
        indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从缓冲区收集指定索引的样本。
        
        Args:
            buffer: 蓄水池缓冲区
            indices: 要收集的样本索引
            
        Returns:
            (states, targets, iterations) 三元组
        """
        states = []
        targets = []
        iterations = []
        
        for idx in indices:
            state, target, iteration = buffer.buffer[idx]
            states.append(state)
            targets.append(target)
            iterations.append(iteration)
        
        return np.array(states), np.array(targets), np.array(iterations)
    
    def sample_stratified(
        self,
        buffers: Dict[str, ReservoirBuffer],
        batch_size: int,
        stage_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """分层采样。
        
        确保不同游戏阶段的样本均衡。
        
        Args:
            buffers: 按阶段分组的缓冲区字典
            batch_size: 总批次大小
            stage_weights: 各阶段的采样权重，如果为None则均匀分配
            
        Returns:
            按阶段分组的采样结果字典
        """
        if not buffers:
            return {}
        
        # 计算各阶段的采样数量
        if stage_weights is None:
            # 均匀分配
            stage_weights = {stage: 1.0 for stage in buffers}
        
        total_weight = sum(stage_weights.get(stage, 1.0) for stage in buffers)
        stage_sizes = {}
        
        remaining = batch_size
        stages = list(buffers.keys())
        
        for i, stage in enumerate(stages):
            if i == len(stages) - 1:
                # 最后一个阶段获取剩余数量
                stage_sizes[stage] = remaining
            else:
                weight = stage_weights.get(stage, 1.0)
                size = int(batch_size * weight / total_weight)
                stage_sizes[stage] = size
                remaining -= size
        
        # 从各阶段采样
        results = {}
        for stage, buffer in buffers.items():
            size = stage_sizes.get(stage, 0)
            if size > 0 and len(buffer) > 0:
                results[stage] = buffer.sample(min(size, len(buffer)))
            else:
                results[stage] = (np.array([]), np.array([]), np.array([]))
        
        return results
    
    def cleanup_old_samples(
        self,
        buffer: ReservoirBuffer,
        current_iteration: int,
        max_age: Optional[int] = None
    ) -> int:
        """清理过旧样本。
        
        移除年龄超过阈值的样本。
        
        Args:
            buffer: 蓄水池缓冲区
            current_iteration: 当前迭代次数
            max_age: 最大样本年龄，如果为None则使用配置值
            
        Returns:
            清理的样本数量
        """
        if max_age is None:
            max_age = self.config.max_sample_age
        
        # 找出需要保留的样本
        samples_to_keep = []
        removed_count = 0
        
        for sample in buffer.buffer:
            state, target, iteration = sample
            age = current_iteration - iteration
            
            if age <= max_age:
                samples_to_keep.append(sample)
            else:
                removed_count += 1
        
        # 更新缓冲区
        buffer.buffer = samples_to_keep
        
        return removed_count
    
    def compute_sample_importance(
        self,
        target: np.ndarray,
        method: str = "magnitude"
    ) -> float:
        """计算样本重要性。
        
        Args:
            target: 目标值（遗憾值或策略概率）
            method: 计算方法，支持 "magnitude"（幅度）和 "variance"（方差）
            
        Returns:
            样本重要性分数
        """
        if method == "magnitude":
            return float(np.max(np.abs(target)))
        elif method == "variance":
            return float(np.var(target))
        else:
            raise ValueError(f"不支持的重要性计算方法: {method}")
    
    def get_buffer_statistics(
        self,
        buffer: ReservoirBuffer,
        current_iteration: int
    ) -> Dict[str, Any]:
        """获取缓冲区统计信息。
        
        Args:
            buffer: 蓄水池缓冲区
            current_iteration: 当前迭代次数
            
        Returns:
            包含各项统计信息的字典
        """
        if len(buffer) == 0:
            return {
                "size": 0,
                "capacity": buffer.capacity,
                "fill_ratio": 0.0,
                "avg_age": 0.0,
                "min_age": 0,
                "max_age": 0,
                "total_seen": buffer.total_seen
            }
        
        ages = [current_iteration - sample[2] for sample in buffer.buffer]
        
        return {
            "size": len(buffer),
            "capacity": buffer.capacity,
            "fill_ratio": len(buffer) / buffer.capacity,
            "avg_age": float(np.mean(ages)),
            "min_age": int(np.min(ages)),
            "max_age": int(np.max(ages)),
            "total_seen": buffer.total_seen
        }
