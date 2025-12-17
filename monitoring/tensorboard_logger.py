"""TensorBoard日志记录器模块。

本模块提供TensorBoard集成功能，用于可视化训练过程中的各种指标：
- 损失值曲线
- 胜率曲线
- 策略分布
- 网络权重直方图
- 学习率变化
"""

from typing import Dict, Optional, Any, List
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class TensorBoardLogger:
    """TensorBoard日志记录器。
    
    提供以下功能：
    - 记录标量指标（损失、胜率、奖励等）
    - 记录直方图（网络权重、策略分布等）
    - 记录图像（策略热图等）
    - 支持多个实验对比
    
    Attributes:
        log_dir: TensorBoard日志目录
        writer: SummaryWriter实例
    """
    
    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: Optional[str] = None,
        comment: str = ""
    ):
        """初始化TensorBoard日志记录器。
        
        Args:
            log_dir: 日志根目录
            experiment_name: 实验名称，如果为None则自动生成
            comment: 附加注释，会添加到目录名后
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError(
                "TensorBoard未安装。请运行: pip install tensorboard"
            )
        
        # 生成实验名称
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"poker_training_{timestamp}"
        
        # 构建日志目录路径
        if comment:
            experiment_name = f"{experiment_name}_{comment}"
        
        self._log_dir = Path(log_dir) / experiment_name
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建SummaryWriter
        self._writer = SummaryWriter(log_dir=str(self._log_dir))
        
        # 记录步数
        self._global_step = 0
    
    @property
    def log_dir(self) -> Path:
        """获取日志目录路径。"""
        return self._log_dir
    
    @property
    def global_step(self) -> int:
        """获取当前全局步数。"""
        return self._global_step
    
    def set_global_step(self, step: int) -> None:
        """设置全局步数。
        
        Args:
            step: 步数
        """
        self._global_step = step
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """记录标量值。
        
        Args:
            tag: 指标名称（如 "Loss/train", "Metrics/win_rate"）
            value: 标量值
            step: 步数，如果为None则使用全局步数
        """
        if step is None:
            step = self._global_step
        self._writer.add_scalar(tag, value, step)
    
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """记录多个标量值到同一图表。
        
        Args:
            main_tag: 主标签名称
            tag_scalar_dict: 标签-值字典
            step: 步数
        """
        if step is None:
            step = self._global_step
        self._writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(
        self,
        tag: str,
        values: np.ndarray,
        step: Optional[int] = None,
        bins: str = 'tensorflow'
    ) -> None:
        """记录直方图。
        
        Args:
            tag: 标签名称
            values: 数值数组
            step: 步数
            bins: 分箱方式
        """
        if step is None:
            step = self._global_step
        self._writer.add_histogram(tag, values, step, bins=bins)
    
    def log_training_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """记录训练指标。
        
        这是一个便捷方法，用于记录常见的训练指标。
        
        Args:
            metrics: 指标字典，支持以下键：
                - loss: 总损失
                - policy_loss: 策略损失
                - value_loss: 价值损失
                - cfr_loss: CFR损失
                - entropy_loss: 熵损失
                - win_rate: 胜率
                - avg_reward: 平均奖励
                - episodes_completed: 已完成回合数
            step: 步数
        """
        if step is None:
            step = self._global_step
        
        # 损失相关指标
        if 'loss' in metrics:
            self.log_scalar('Loss/total', metrics['loss'], step)
        if 'policy_loss' in metrics:
            self.log_scalar('Loss/policy', metrics['policy_loss'], step)
        if 'value_loss' in metrics:
            self.log_scalar('Loss/value', metrics['value_loss'], step)
        if 'cfr_loss' in metrics:
            self.log_scalar('Loss/cfr', metrics['cfr_loss'], step)
        if 'entropy_loss' in metrics:
            self.log_scalar('Loss/entropy', metrics['entropy_loss'], step)
        
        # 性能指标
        if 'win_rate' in metrics:
            self.log_scalar('Metrics/win_rate', metrics['win_rate'], step)
        if 'avg_reward' in metrics:
            self.log_scalar('Metrics/avg_reward', metrics['avg_reward'], step)
        
        # 训练进度
        if 'episodes_completed' in metrics:
            self.log_scalar('Progress/episodes', metrics['episodes_completed'], step)
    
    def log_strategy_distribution(
        self,
        action_probs: Dict[str, float],
        hand_label: str = "average",
        step: Optional[int] = None
    ) -> None:
        """记录策略分布。
        
        Args:
            action_probs: 行动概率字典 {action_name: probability}
            hand_label: 手牌标签
            step: 步数
        """
        if step is None:
            step = self._global_step
        
        for action, prob in action_probs.items():
            self.log_scalar(f'Strategy/{hand_label}/{action}', prob, step)
    
    def log_network_weights(
        self,
        model: Any,
        step: Optional[int] = None
    ) -> None:
        """记录网络权重直方图。
        
        Args:
            model: PyTorch模型
            step: 步数
        """
        if step is None:
            step = self._global_step
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.log_histogram(f'Weights/{name}', param.data.cpu().numpy(), step)
                if param.grad is not None:
                    self.log_histogram(
                        f'Gradients/{name}', 
                        param.grad.data.cpu().numpy(), 
                        step
                    )
    
    def log_learning_rate(
        self,
        lr: float,
        step: Optional[int] = None
    ) -> None:
        """记录学习率。
        
        Args:
            lr: 学习率
            step: 步数
        """
        if step is None:
            step = self._global_step
        self.log_scalar('Training/learning_rate', lr, step)
    
    def log_regret(
        self,
        avg_regret: float,
        max_regret: float,
        step: Optional[int] = None
    ) -> None:
        """记录CFR遗憾值。
        
        Args:
            avg_regret: 平均遗憾值
            max_regret: 最大遗憾值
            step: 步数
        """
        if step is None:
            step = self._global_step
        self.log_scalar('CFR/avg_regret', avg_regret, step)
        self.log_scalar('CFR/max_regret', max_regret, step)
    
    def log_action_frequencies(
        self,
        frequencies: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """记录行动频率。
        
        Args:
            frequencies: 行动频率字典
            step: 步数
        """
        if step is None:
            step = self._global_step
        
        for action, freq in frequencies.items():
            self.log_scalar(f'Actions/{action}_frequency', freq, step)
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None
    ) -> None:
        """记录文本。
        
        Args:
            tag: 标签
            text: 文本内容
            step: 步数
        """
        if step is None:
            step = self._global_step
        self._writer.add_text(tag, text, step)
    
    def log_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, float]
    ) -> None:
        """记录超参数和对应的指标。
        
        用于超参数搜索时对比不同配置的效果。
        
        Args:
            hparam_dict: 超参数字典
            metric_dict: 指标字典
        """
        self._writer.add_hparams(hparam_dict, metric_dict)
    
    def flush(self) -> None:
        """刷新缓冲区，确保数据写入磁盘。"""
        self._writer.flush()
    
    def close(self) -> None:
        """关闭日志记录器。"""
        self._writer.close()
    
    def __enter__(self):
        """上下文管理器入口。"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口。"""
        self.close()
        return False


def is_tensorboard_available() -> bool:
    """检查TensorBoard是否可用。
    
    Returns:
        TensorBoard是否已安装
    """
    return TENSORBOARD_AVAILABLE
