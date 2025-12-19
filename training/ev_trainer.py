"""
EV预测训练器模块

该模块实现了训练EV预测网络的功能，包括：
- 多任务损失计算（EV MSE + 动作EV MSE + 策略交叉熵）
- 训练循环和评估
- 模型保存和加载
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from training.ev_prediction_network import EVPredictionNetwork

# 配置日志
logger = logging.getLogger(__name__)


class EVTrainer:
    """
    EV预测训练器
    
    支持多任务学习，同时优化EV预测、动作EV预测和策略预测。
    """
    
    def __init__(
        self,
        model: EVPredictionNetwork,
        learning_rate: float = 1e-3,
        ev_weight: float = 1.0,
        action_ev_weight: float = 1.0,
        strategy_weight: float = 1.0,
        grad_clip: float = 1.0,
        device: Optional[str] = None
    ):
        """
        初始化训练器
        
        参数:
            model: 神经网络模型
            learning_rate: 学习率
            ev_weight: EV损失权重
            action_ev_weight: 动作EV损失权重
            strategy_weight: 策略损失权重
            grad_clip: 梯度裁剪阈值
            device: 计算设备（'cuda' 或 'cpu'）
        """
        self.model = model
        self.learning_rate = learning_rate
        self.ev_weight = ev_weight
        self.action_ev_weight = action_ev_weight
        self.strategy_weight = strategy_weight
        self.grad_clip = grad_clip
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练状态
        self.epoch = 0
        self.best_loss = float('inf')
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
    
    def compute_loss(
        self,
        features: torch.Tensor,
        target_ev: torch.Tensor,
        target_action_ev: torch.Tensor,
        target_strategy: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        
        参数:
            features: 输入特征 [batch_size, 4]
            target_ev: 目标EV [batch_size, 1]
            target_action_ev: 目标动作EV [batch_size, num_actions]
            target_strategy: 目标策略 [batch_size, num_actions]
        
        返回:
            包含各项损失的字典
        """
        # 前向传播
        pred_ev, pred_action_ev, pred_strategy = self.model(features)
        
        # EV损失（MSE）
        ev_loss = self.mse_loss(pred_ev, target_ev)
        
        # 动作EV损失（MSE）
        action_ev_loss = self.mse_loss(pred_action_ev, target_action_ev)
        
        # 策略损失（KL散度）
        # 使用KL散度：KL(target || pred) = sum(target * log(target / pred))
        # 为了数值稳定性，添加小的epsilon
        eps = 1e-8
        strategy_loss = F.kl_div(
            torch.log(pred_strategy + eps),
            target_strategy,
            reduction='batchmean'
        )
        
        # 总损失
        total_loss = (
            self.ev_weight * ev_loss +
            self.action_ev_weight * action_ev_loss +
            self.strategy_weight * strategy_loss
        )
        
        return {
            "total": total_loss,
            "ev": ev_loss,
            "action_ev": action_ev_loss,
            "strategy": strategy_loss
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch
        
        参数:
            dataloader: 训练数据加载器
        
        返回:
            各项损失的平均值
        """
        self.model.train()
        
        total_losses = {"total": 0.0, "ev": 0.0, "action_ev": 0.0, "strategy": 0.0}
        num_batches = 0
        
        for batch in dataloader:
            features, target_ev, target_action_ev, target_strategy = batch
            
            # 移动到设备
            features = features.to(self.device)
            target_ev = target_ev.to(self.device)
            target_action_ev = target_action_ev.to(self.device)
            target_strategy = target_strategy.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 计算损失
            losses = self.compute_loss(features, target_ev, target_action_ev, target_strategy)
            
            # 检查损失是否有效
            if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
                logger.warning("检测到NaN或Inf损失，跳过此批次")
                continue
            
            # 反向传播
            losses["total"].backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # 更新参数
            self.optimizer.step()
            
            # 累积损失
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1
        
        self.epoch += 1
        
        # 计算平均损失
        if num_batches > 0:
            for key in total_losses:
                total_losses[key] /= num_batches
        
        return total_losses
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        评估模型
        
        参数:
            dataloader: 评估数据加载器
        
        返回:
            各项指标的统计摘要
        """
        self.model.eval()
        
        all_ev_errors = []
        all_action_ev_errors = []
        all_strategy_kl = []
        
        with torch.no_grad():
            for batch in dataloader:
                features, target_ev, target_action_ev, target_strategy = batch
                
                features = features.to(self.device)
                target_ev = target_ev.to(self.device)
                target_action_ev = target_action_ev.to(self.device)
                target_strategy = target_strategy.to(self.device)
                
                # 前向传播
                pred_ev, pred_action_ev, pred_strategy = self.model(features)
                
                # 计算误差
                ev_error = (pred_ev - target_ev).pow(2).cpu().numpy()
                action_ev_error = (pred_action_ev - target_action_ev).pow(2).mean(dim=1).cpu().numpy()
                
                eps = 1e-8
                kl = (target_strategy * torch.log((target_strategy + eps) / (pred_strategy + eps))).sum(dim=1).cpu().numpy()
                
                all_ev_errors.extend(ev_error.flatten().tolist())
                all_action_ev_errors.extend(action_ev_error.tolist())
                all_strategy_kl.extend(kl.tolist())
        
        # 计算统计摘要
        ev_errors = np.array(all_ev_errors)
        action_ev_errors = np.array(all_action_ev_errors)
        strategy_kl = np.array(all_strategy_kl)
        
        return {
            "ev_mse": {
                "mean": float(ev_errors.mean()),
                "std": float(ev_errors.std()),
                "min": float(ev_errors.min()),
                "max": float(ev_errors.max())
            },
            "action_ev_mse": {
                "mean": float(action_ev_errors.mean()),
                "std": float(action_ev_errors.std()),
                "min": float(action_ev_errors.min()),
                "max": float(action_ev_errors.max())
            },
            "strategy_kl": {
                "mean": float(strategy_kl.mean()),
                "std": float(strategy_kl.std()),
                "min": float(strategy_kl.min()),
                "max": float(strategy_kl.max())
            },
            "num_samples": len(all_ev_errors)
        }
    
    def save_checkpoint(self, path: str):
        """
        保存模型检查点
        
        参数:
            path: 保存路径
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": {
                "learning_rate": self.learning_rate,
                "ev_weight": self.ev_weight,
                "action_ev_weight": self.action_ev_weight,
                "strategy_weight": self.strategy_weight,
                "grad_clip": self.grad_clip,
                "num_actions": self.model.num_actions,
                "hidden_dim": self.model.hidden_dim
            }
        }
        
        # 确保目录存在
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, path)
        logger.info(f"模型检查点已保存到: {path}")
    
    def load_checkpoint(self, path: str):
        """
        加载模型检查点
        
        参数:
            path: 检查点路径
        
        异常:
            FileNotFoundError: 如果文件不存在
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点文件不存在: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        
        logger.info(f"模型检查点已从 {path} 加载，epoch={self.epoch}")
    
    def get_learning_rate(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def set_learning_rate(self, lr: float):
        """设置学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
