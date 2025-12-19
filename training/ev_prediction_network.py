"""
EV预测神经网络模块

该模块实现了一个多任务学习网络，用于预测扑克场景中的：
- 整体期望值（EV）
- 每个动作的期望值（Action EV）
- 动作策略概率分布（Strategy）

输入特征为4个标量：hero_equity, range_equity, solver_equity, eqr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EVPredictionNetwork(nn.Module):
    """
    EV预测神经网络
    
    采用多任务学习架构：
    - 共享编码器：将4维输入映射到高维特征空间
    - EV预测头：预测整体期望值（标量）
    - 动作EV预测头：预测每个动作的期望值（N维向量）
    - 策略预测头：预测动作概率分布（N维向量，经softmax归一化）
    """
    
    def __init__(self, num_actions: int = 5, hidden_dim: int = 64):
        """
        初始化网络
        
        参数:
            num_actions: 动作数量（默认5：Check, Bet33, Bet50, Bet75, Bet120）
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.input_dim = 4  # hero_equity, range_equity, solver_equity, eqr
        
        # 共享编码器：4 -> hidden_dim -> hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # EV预测头：hidden_dim -> 1
        self.ev_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 动作EV预测头：hidden_dim -> num_actions
        self.action_ev_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # 策略预测头：hidden_dim -> num_actions（输出logits，forward中应用softmax）
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, 4]
               - x[:, 0]: hero_equity
               - x[:, 1]: range_equity  
               - x[:, 2]: solver_equity
               - x[:, 3]: eqr
        
        返回:
            ev: 整体EV预测 [batch_size, 1]
            action_ev: 动作EV预测 [batch_size, num_actions]
            strategy: 策略概率 [batch_size, num_actions]（经softmax归一化）
        
        异常:
            ValueError: 如果输入维度不正确
        """
        # 验证输入维度
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"输入维度错误：期望 [batch_size, {self.input_dim}]，"
                f"实际 {list(x.shape)}"
            )
        
        # 共享编码
        features = self.encoder(x)
        
        # 三个预测头
        ev = self.ev_head(features)
        action_ev = self.action_ev_head(features)
        strategy_logits = self.strategy_head(features)
        
        # 对策略输出应用softmax确保概率有效性
        strategy = F.softmax(strategy_logits, dim=-1)
        
        return ev, action_ev, strategy
    
    def get_strategy_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取策略的原始logits（用于计算交叉熵损失）
        
        参数:
            x: 输入特征 [batch_size, 4]
        
        返回:
            strategy_logits: 策略logits [batch_size, num_actions]
        """
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"输入维度错误：期望 [batch_size, {self.input_dim}]，"
                f"实际 {list(x.shape)}"
            )
        
        features = self.encoder(x)
        return self.strategy_head(features)
