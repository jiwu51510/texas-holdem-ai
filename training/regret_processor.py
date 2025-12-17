"""遗憾值处理器模块。

本模块提供遗憾值的截断、衰减、裁剪等处理功能，
用于改进Deep CFR训练中的策略收敛。

主要组件：
- RegretProcessorConfig: 遗憾值处理器配置
- RegretProcessor: 遗憾值处理器
"""

from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class RegretProcessorConfig:
    """遗憾值处理器配置。
    
    Attributes:
        use_positive_truncation: 是否使用正遗憾截断（CFR+）
        decay_factor: 遗憾值衰减因子（0-1之间）
        clip_threshold: 遗憾值裁剪阈值（正数）
    """
    use_positive_truncation: bool = True
    decay_factor: float = 0.99
    clip_threshold: float = 100.0
    
    def __post_init__(self):
        """验证配置参数。"""
        if not 0.0 <= self.decay_factor <= 1.0:
            raise ValueError(f"衰减因子必须在0-1之间，当前值: {self.decay_factor}")
        if self.clip_threshold <= 0:
            raise ValueError(f"裁剪阈值必须为正数，当前值: {self.clip_threshold}")


class RegretProcessor:
    """遗憾值处理器。
    
    提供遗憾值的截断、衰减、裁剪等处理功能，
    用于稳定CFR训练过程中的遗憾值累积。
    """
    
    def __init__(self, config: RegretProcessorConfig = None):
        """初始化遗憾值处理器。
        
        Args:
            config: 处理器配置，如果为None则使用默认配置
        """
        self.config = config or RegretProcessorConfig()
    
    def truncate_positive(self, regrets: np.ndarray) -> np.ndarray:
        """正遗憾值截断（CFR+）。
        
        将负遗憾值截断为0，只保留正遗憾值。
        这是CFR+算法的核心改进之一。
        
        Args:
            regrets: 原始遗憾值数组
            
        Returns:
            截断后的遗憾值数组（所有值非负）
        """
        return np.maximum(regrets, 0.0)
    
    def apply_decay(self, regrets: np.ndarray, 
                    decay_factor: float = None) -> np.ndarray:
        """应用遗憾值衰减。
        
        通过乘以衰减因子来减少历史遗憾值的影响，
        防止遗憾值无限累积。
        
        Args:
            regrets: 遗憾值数组
            decay_factor: 衰减因子（0-1之间），如果为None则使用配置值
            
        Returns:
            衰减后的遗憾值数组
        """
        if decay_factor is None:
            decay_factor = self.config.decay_factor
        return regrets * decay_factor
    
    def clip_regrets(self, regrets: np.ndarray, 
                     max_value: float = None) -> np.ndarray:
        """裁剪遗憾值。
        
        将遗憾值的绝对值限制在阈值内，防止数值爆炸。
        
        Args:
            regrets: 遗憾值数组
            max_value: 最大绝对值，如果为None则使用配置值
            
        Returns:
            裁剪后的遗憾值数组
        """
        if max_value is None:
            max_value = self.config.clip_threshold
        return np.clip(regrets, -max_value, max_value)
    
    def process(self, regrets: np.ndarray, 
                apply_truncation: bool = None,
                apply_decay: bool = True,
                apply_clip: bool = True) -> np.ndarray:
        """完整的遗憾值处理流程。
        
        按顺序应用截断、衰减、裁剪操作。
        
        Args:
            regrets: 原始遗憾值数组
            apply_truncation: 是否应用正遗憾截断，如果为None则使用配置值
            apply_decay: 是否应用衰减
            apply_clip: 是否应用裁剪
            
        Returns:
            处理后的遗憾值数组
        """
        result = regrets.copy()
        
        # 正遗憾值截断
        if apply_truncation is None:
            apply_truncation = self.config.use_positive_truncation
        if apply_truncation:
            result = self.truncate_positive(result)
        
        # 遗憾值衰减
        if apply_decay:
            result = self.apply_decay(result)
        
        # 遗憾值裁剪
        if apply_clip:
            result = self.clip_regrets(result)
        
        return result
