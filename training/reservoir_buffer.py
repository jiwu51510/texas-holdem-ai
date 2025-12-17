"""蓄水池缓冲区（Reservoir Buffer）实现。

使用蓄水池采样算法保持固定大小的经验回放缓冲区，
确保每个样本被保留的概率相等。
"""

import random
from typing import Tuple, List, Optional
import numpy as np


class ReservoirBuffer:
    """使用蓄水池采样的经验回放缓冲区。
    
    蓄水池采样确保：
    - 缓冲区大小固定
    - 每个样本被保留的概率相等
    - 内存使用可控
    
    Attributes:
        capacity: 缓冲区最大容量
        buffer: 存储样本的列表
        total_seen: 已见过的样本总数
    """
    
    def __init__(self, capacity: int):
        """初始化缓冲区。
        
        Args:
            capacity: 缓冲区最大容量，必须为正整数
            
        Raises:
            ValueError: 如果容量不是正整数
        """
        if capacity <= 0:
            raise ValueError(f"缓冲区容量必须为正整数，收到: {capacity}")
        
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, np.ndarray, int]] = []
        self.total_seen = 0
    
    def add(self, state: np.ndarray, target: np.ndarray, iteration: int) -> None:
        """添加样本到缓冲区。
        
        使用蓄水池采样：
        - 如果缓冲区未满，直接添加
        - 如果缓冲区已满，以 capacity/n 的概率替换随机样本
        
        Args:
            state: 状态编码
            target: 目标值（遗憾值或策略概率）
            iteration: CFR 迭代编号
        """
        self.total_seen += 1
        sample = (state.copy(), target.copy(), iteration)
        
        if len(self.buffer) < self.capacity:
            # 缓冲区未满，直接添加
            self.buffer.append(sample)
        else:
            # 缓冲区已满，使用蓄水池采样
            # 以 capacity/total_seen 的概率替换随机位置
            j = random.randint(0, self.total_seen - 1)
            if j < self.capacity:
                self.buffer[j] = sample
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从缓冲区随机采样。
        
        Args:
            batch_size: 请求的样本数量
            
        Returns:
            (states, targets, iterations) 三元组
            - states: 状态编码数组，形状为 (n, state_dim)
            - targets: 目标值数组，形状为 (n, target_dim)
            - iterations: 迭代编号数组，形状为 (n,)
            
            其中 n = min(batch_size, len(buffer))
        """
        if len(self.buffer) == 0:
            # 空缓冲区返回空数组
            return np.array([]), np.array([]), np.array([])
        
        # 实际采样数量
        actual_size = min(batch_size, len(self.buffer))
        
        # 随机选择索引
        indices = random.sample(range(len(self.buffer)), actual_size)
        
        # 收集样本
        states = []
        targets = []
        iterations = []
        
        for idx in indices:
            state, target, iteration = self.buffer[idx]
            states.append(state)
            targets.append(target)
            iterations.append(iteration)
        
        return np.array(states), np.array(targets), np.array(iterations)
    
    def __len__(self) -> int:
        """返回当前缓冲区大小。
        
        Returns:
            缓冲区中的样本数量
        """
        return len(self.buffer)
    
    def clear(self) -> None:
        """清空缓冲区。"""
        self.buffer.clear()
        self.total_seen = 0
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满。
        
        Returns:
            如果缓冲区已满返回 True，否则返回 False
        """
        return len(self.buffer) >= self.capacity
    
    def get_total_seen(self) -> int:
        """获取已见过的样本总数。
        
        Returns:
            已见过的样本总数
        """
        return self.total_seen
