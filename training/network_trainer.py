"""网络训练器模块。

本模块提供改进的神经网络训练功能，包括：
- Huber损失计算（减少异常值影响）
- EMA目标网络更新（平滑策略更新）
- KL散度计算（策略变化监控）
- 梯度裁剪（防止梯度爆炸）

主要组件：
- NetworkTrainerConfig: 网络训练器配置
- NetworkTrainer: 网络训练器
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import numpy as np


@dataclass
class NetworkTrainerConfig:
    """网络训练器配置。
    
    Attributes:
        use_huber_loss: 是否使用Huber损失
        huber_delta: Huber损失的delta参数
        use_ema: 是否使用EMA更新
        ema_decay: EMA衰减率（0-1之间）
        kl_coefficient: KL散度正则化系数
        gradient_clip_norm: 梯度裁剪范数（正数）
    """
    use_huber_loss: bool = True
    huber_delta: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.995
    kl_coefficient: float = 0.01
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        """验证配置参数。"""
        if self.huber_delta <= 0:
            raise ValueError(f"huber_delta必须为正数，当前值: {self.huber_delta}")
        if not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError(f"ema_decay必须在0-1之间，当前值: {self.ema_decay}")
        if self.kl_coefficient < 0:
            raise ValueError(f"kl_coefficient必须非负，当前值: {self.kl_coefficient}")
        if self.gradient_clip_norm <= 0:
            raise ValueError(f"gradient_clip_norm必须为正数，当前值: {self.gradient_clip_norm}")


class NetworkTrainer:
    """网络训练器。
    
    提供改进的网络训练功能，包括Huber损失、EMA更新、
    KL散度计算和梯度裁剪。
    """
    
    def __init__(self, config: NetworkTrainerConfig = None):
        """初始化网络训练器。
        
        Args:
            config: 训练器配置，如果为None则使用默认配置
        """
        self.config = config or NetworkTrainerConfig()
    
    def compute_huber_loss(self, predictions: torch.Tensor,
                           targets: torch.Tensor,
                           delta: float = None) -> torch.Tensor:
        """计算Huber损失。
        
        Huber损失在误差较小时使用MSE，误差较大时使用MAE，
        减少异常值的影响。
        
        公式:
        - 当 |error| <= delta: loss = 0.5 * error^2
        - 当 |error| > delta: loss = delta * (|error| - 0.5 * delta)
        
        Args:
            predictions: 预测值张量
            targets: 目标值张量
            delta: Huber损失的delta参数，如果为None则使用配置值
            
        Returns:
            Huber损失值（标量张量）
        """
        if delta is None:
            delta = self.config.huber_delta
        
        error = predictions - targets
        abs_error = torch.abs(error)
        
        # 分段计算Huber损失
        quadratic = torch.clamp(abs_error, max=delta)
        linear = abs_error - quadratic
        
        # loss = 0.5 * quadratic^2 + delta * linear
        loss = 0.5 * quadratic ** 2 + delta * linear
        
        return loss.mean()
    
    def compute_kl_divergence(self, p: torch.Tensor, 
                               q: torch.Tensor,
                               epsilon: float = 1e-10) -> torch.Tensor:
        """计算KL散度。
        
        KL(p||q) = sum(p * log(p/q))
        
        使用数值稳定的实现，添加epsilon防止log(0)。
        
        Args:
            p: 概率分布P（参考分布）
            q: 概率分布Q（近似分布）
            epsilon: 数值稳定性的小常数
            
        Returns:
            KL散度值（标量张量）
        """
        # 确保概率分布有效
        p_safe = torch.clamp(p, min=epsilon, max=1.0)
        q_safe = torch.clamp(q, min=epsilon, max=1.0)
        
        # 归一化（确保和为1）
        p_normalized = p_safe / p_safe.sum(dim=-1, keepdim=True)
        q_normalized = q_safe / q_safe.sum(dim=-1, keepdim=True)
        
        # KL散度: sum(p * log(p/q))
        kl = p_normalized * (torch.log(p_normalized) - torch.log(q_normalized))
        
        # 对最后一个维度求和，然后取平均
        return kl.sum(dim=-1).mean()
    
    def update_ema(self, target_network: nn.Module,
                   source_network: nn.Module,
                   decay: float = None) -> None:
        """使用EMA更新目标网络。
        
        target = decay * target + (1 - decay) * source
        
        Args:
            target_network: 目标网络（将被更新）
            source_network: 源网络（提供新参数）
            decay: EMA衰减率，如果为None则使用配置值
        """
        if decay is None:
            decay = self.config.ema_decay
        
        with torch.no_grad():
            for target_param, source_param in zip(
                target_network.parameters(), 
                source_network.parameters()
            ):
                target_param.data.mul_(decay).add_(
                    source_param.data, alpha=1.0 - decay
                )
    
    def clip_gradients(self, network: nn.Module,
                       max_norm: float = None) -> float:
        """裁剪梯度。
        
        使用L2范数裁剪梯度，防止梯度爆炸。
        
        Args:
            network: 神经网络
            max_norm: 最大梯度范数，如果为None则使用配置值
            
        Returns:
            裁剪前的梯度范数
        """
        if max_norm is None:
            max_norm = self.config.gradient_clip_norm
        
        # 获取所有需要梯度的参数
        parameters = [p for p in network.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        # 计算裁剪前的梯度范数
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        
        return float(total_norm)
    
    def compute_policy_loss_with_kl(self, 
                                     predictions: torch.Tensor,
                                     targets: torch.Tensor,
                                     old_policy: torch.Tensor = None) -> torch.Tensor:
        """计算带KL正则化的策略损失。
        
        如果提供了旧策略，则添加KL散度正则化项，
        限制策略变化幅度。
        
        Args:
            predictions: 预测的策略分布
            targets: 目标策略分布
            old_policy: 旧策略分布（用于KL正则化）
            
        Returns:
            总损失值
        """
        # 基础损失（使用Huber或MSE）
        if self.config.use_huber_loss:
            base_loss = self.compute_huber_loss(predictions, targets)
        else:
            base_loss = torch.nn.functional.mse_loss(predictions, targets)
        
        # 如果提供了旧策略，添加KL正则化
        if old_policy is not None and self.config.kl_coefficient > 0:
            kl_loss = self.compute_kl_divergence(predictions, old_policy)
            total_loss = base_loss + self.config.kl_coefficient * kl_loss
        else:
            total_loss = base_loss
        
        return total_loss
