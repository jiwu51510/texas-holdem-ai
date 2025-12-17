"""收敛监控器模块。

本模块提供训练过程中的收敛状态监控功能，包括：
- 策略熵计算（监控策略确定性）
- 遗憾值统计（监控遗憾值分布）
- 策略变化监控（检测策略震荡）
- 收敛报告生成

主要组件：
- ConvergenceMonitorConfig: 收敛监控器配置
- ConvergenceMonitor: 收敛监控器
- ConvergenceMetrics: 收敛指标数据类

需求: 4.1, 4.2, 4.3, 4.4
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from collections import deque


@dataclass
class ConvergenceMonitorConfig:
    """收敛监控器配置。
    
    Attributes:
        entropy_window: 熵值监控窗口大小（用于震荡检测）
        oscillation_threshold: 震荡检测阈值（熵值变化幅度）
        kl_warning_threshold: KL散度警告阈值
        monitor_interval: 监控间隔（迭代次数）
    """
    entropy_window: int = 100
    oscillation_threshold: float = 0.1
    kl_warning_threshold: float = 0.5
    monitor_interval: int = 1000
    
    def __post_init__(self):
        """验证配置参数。"""
        if self.entropy_window <= 0:
            raise ValueError(f"entropy_window必须为正整数，当前值: {self.entropy_window}")
        if self.oscillation_threshold <= 0:
            raise ValueError(f"oscillation_threshold必须为正数，当前值: {self.oscillation_threshold}")
        if self.kl_warning_threshold <= 0:
            raise ValueError(f"kl_warning_threshold必须为正数，当前值: {self.kl_warning_threshold}")
        if self.monitor_interval <= 0:
            raise ValueError(f"monitor_interval必须为正整数，当前值: {self.monitor_interval}")


@dataclass
class ConvergenceMetrics:
    """收敛指标。
    
    Attributes:
        iteration: 迭代次数
        avg_entropy: 平均策略熵
        regret_mean: 遗憾值均值
        regret_std: 遗憾值标准差
        regret_max: 遗憾值最大值
        policy_kl: 策略KL散度
        is_oscillating: 是否震荡
    """
    iteration: int
    avg_entropy: float
    regret_mean: float
    regret_std: float
    regret_max: float
    policy_kl: float
    is_oscillating: bool


class ConvergenceMonitor:
    """收敛监控器。
    
    监控训练过程中的收敛状态，包括策略熵、遗憾值统计、
    策略变化和震荡检测。
    """
    
    def __init__(self, config: ConvergenceMonitorConfig = None):
        """初始化收敛监控器。
        
        Args:
            config: 监控器配置，如果为None则使用默认配置
        """
        self.config = config or ConvergenceMonitorConfig()
        
        # 熵值历史（用于震荡检测）
        self._entropy_history: deque = deque(maxlen=self.config.entropy_window)
        
        # 上一次的策略（用于KL散度计算）
        self._last_policy: Optional[np.ndarray] = None
        
        # 监控指标历史
        self._metrics_history: List[ConvergenceMetrics] = []
        
        # 当前迭代次数
        self._current_iteration: int = 0
    
    def compute_entropy(self, strategy: np.ndarray, 
                        epsilon: float = 1e-10) -> float:
        """计算策略熵。
        
        H(p) = -sum(p * log(p))
        
        熵值越高表示策略越不确定（更接近均匀分布），
        熵值越低表示策略越确定（更接近确定性策略）。
        
        Args:
            strategy: 策略概率分布（一维或多维数组）
            epsilon: 数值稳定性的小常数
            
        Returns:
            熵值（非负）
        """
        # 确保输入是numpy数组
        strategy = np.asarray(strategy, dtype=np.float64)
        
        # 处理空数组
        if strategy.size == 0:
            return 0.0
        
        # 展平为一维（如果是多维）
        if strategy.ndim > 1:
            # 对每行计算熵，然后取平均
            return np.mean([self.compute_entropy(row, epsilon) 
                           for row in strategy])
        
        # 确保概率有效（非负且和为1）
        strategy_safe = np.clip(strategy, epsilon, 1.0)
        strategy_normalized = strategy_safe / np.sum(strategy_safe)
        
        # 计算熵: H(p) = -sum(p * log(p))
        entropy = -np.sum(strategy_normalized * np.log(strategy_normalized))
        
        return float(entropy)
    
    def compute_regret_stats(self, regrets: np.ndarray) -> Dict[str, float]:
        """计算遗憾值统计信息。
        
        Args:
            regrets: 遗憾值数组
            
        Returns:
            包含均值、方差、标准差、最大值、最小值的字典
        """
        # 确保输入是numpy数组
        regrets = np.asarray(regrets, dtype=np.float64)
        
        # 处理空数组
        if regrets.size == 0:
            return {
                'mean': 0.0,
                'variance': 0.0,
                'std': 0.0,
                'max': 0.0,
                'min': 0.0,
                'abs_mean': 0.0,
                'abs_max': 0.0
            }
        
        # 展平数组
        regrets_flat = regrets.flatten()
        
        return {
            'mean': float(np.mean(regrets_flat)),
            'variance': float(np.var(regrets_flat)),
            'std': float(np.std(regrets_flat)),
            'max': float(np.max(regrets_flat)),
            'min': float(np.min(regrets_flat)),
            'abs_mean': float(np.mean(np.abs(regrets_flat))),
            'abs_max': float(np.max(np.abs(regrets_flat)))
        }
    
    def detect_oscillation(self, entropy_history: List[float] = None) -> bool:
        """检测策略震荡。
        
        通过分析熵值历史检测是否存在震荡。
        震荡的特征是熵值在高低之间反复波动。
        
        检测方法：
        1. 计算熵值的一阶差分
        2. 检测差分符号的变化次数
        3. 如果变化次数超过阈值，认为存在震荡
        
        Args:
            entropy_history: 熵值历史列表，如果为None则使用内部历史
            
        Returns:
            是否检测到震荡
        """
        if entropy_history is None:
            entropy_history = list(self._entropy_history)
        
        # 需要至少3个数据点才能检测震荡
        if len(entropy_history) < 3:
            return False
        
        entropy_array = np.array(entropy_history)
        
        # 计算一阶差分
        diff = np.diff(entropy_array)
        
        # 计算差分的符号
        signs = np.sign(diff)
        
        # 计算符号变化次数（从正变负或从负变正）
        sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
        
        # 计算熵值的标准差
        entropy_std = np.std(entropy_array)
        
        # 震荡检测条件：
        # 1. 符号变化次数超过历史长度的一半
        # 2. 熵值标准差超过阈值
        oscillation_ratio = sign_changes / (len(entropy_history) - 2)
        
        is_oscillating = (
            oscillation_ratio > 0.5 and 
            entropy_std > self.config.oscillation_threshold
        )
        
        return bool(is_oscillating)
    
    def compute_kl_divergence(self, p: np.ndarray, q: np.ndarray,
                               epsilon: float = 1e-10) -> float:
        """计算KL散度。
        
        KL(p||q) = sum(p * log(p/q))
        
        Args:
            p: 概率分布P（参考分布）
            q: 概率分布Q（近似分布）
            epsilon: 数值稳定性的小常数
            
        Returns:
            KL散度值（非负）
        """
        # 确保输入是numpy数组
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        
        # 处理空数组
        if p.size == 0 or q.size == 0:
            return 0.0
        
        # 确保形状相同
        if p.shape != q.shape:
            raise ValueError(f"分布形状不匹配: {p.shape} vs {q.shape}")
        
        # 展平为一维
        p_flat = p.flatten()
        q_flat = q.flatten()
        
        # 确保概率有效
        p_safe = np.clip(p_flat, epsilon, 1.0)
        q_safe = np.clip(q_flat, epsilon, 1.0)
        
        # 归一化
        p_normalized = p_safe / np.sum(p_safe)
        q_normalized = q_safe / np.sum(q_safe)
        
        # KL散度: sum(p * log(p/q))
        kl = np.sum(p_normalized * np.log(p_normalized / q_normalized))
        
        return float(max(0.0, kl))  # 确保非负
    
    def update(self, iteration: int, strategy: np.ndarray = None,
               regrets: np.ndarray = None) -> Optional[ConvergenceMetrics]:
        """更新监控状态。
        
        Args:
            iteration: 当前迭代次数
            strategy: 当前策略分布
            regrets: 当前遗憾值
            
        Returns:
            如果达到监控间隔，返回收敛指标；否则返回None
        """
        self._current_iteration = iteration
        
        # 计算并记录熵值
        if strategy is not None:
            entropy = self.compute_entropy(strategy)
            self._entropy_history.append(entropy)
        
        # 检查是否达到监控间隔
        if iteration % self.config.monitor_interval != 0:
            return None
        
        # 计算各项指标
        avg_entropy = np.mean(list(self._entropy_history)) if self._entropy_history else 0.0
        
        regret_stats = self.compute_regret_stats(regrets) if regrets is not None else {
            'mean': 0.0, 'std': 0.0, 'max': 0.0
        }
        
        # 计算策略KL散度
        policy_kl = 0.0
        if strategy is not None and self._last_policy is not None:
            try:
                if strategy.shape == self._last_policy.shape:
                    policy_kl = self.compute_kl_divergence(strategy, self._last_policy)
            except ValueError:
                pass
        
        # 更新上一次策略
        if strategy is not None:
            self._last_policy = strategy.copy()
        
        # 检测震荡
        is_oscillating = self.detect_oscillation()
        
        # 创建指标
        metrics = ConvergenceMetrics(
            iteration=iteration,
            avg_entropy=float(avg_entropy),
            regret_mean=regret_stats['mean'],
            regret_std=regret_stats['std'],
            regret_max=regret_stats['max'],
            policy_kl=policy_kl,
            is_oscillating=is_oscillating
        )
        
        self._metrics_history.append(metrics)
        
        # 输出警告
        if is_oscillating:
            print(f"[警告] 迭代 {iteration}: 检测到策略震荡，建议调整学习率或增加正则化")
        
        if policy_kl > self.config.kl_warning_threshold:
            print(f"[警告] 迭代 {iteration}: 策略变化过大 (KL={policy_kl:.4f})，建议降低学习率")
        
        return metrics
    
    def get_convergence_report(self) -> Dict[str, Any]:
        """获取收敛报告。
        
        Returns:
            包含各项监控指标的报告
        """
        report = {
            'current_iteration': self._current_iteration,
            'total_metrics_recorded': len(self._metrics_history),
            'entropy_history_size': len(self._entropy_history),
            'config': {
                'entropy_window': self.config.entropy_window,
                'oscillation_threshold': self.config.oscillation_threshold,
                'kl_warning_threshold': self.config.kl_warning_threshold,
                'monitor_interval': self.config.monitor_interval
            }
        }
        
        # 添加最新指标
        if self._metrics_history:
            latest = self._metrics_history[-1]
            report['latest_metrics'] = {
                'iteration': latest.iteration,
                'avg_entropy': latest.avg_entropy,
                'regret_mean': latest.regret_mean,
                'regret_std': latest.regret_std,
                'regret_max': latest.regret_max,
                'policy_kl': latest.policy_kl,
                'is_oscillating': latest.is_oscillating
            }
        
        # 添加熵值统计
        if self._entropy_history:
            entropy_array = np.array(list(self._entropy_history))
            report['entropy_stats'] = {
                'mean': float(np.mean(entropy_array)),
                'std': float(np.std(entropy_array)),
                'min': float(np.min(entropy_array)),
                'max': float(np.max(entropy_array))
            }
        
        # 添加震荡检测结果
        report['is_oscillating'] = self.detect_oscillation()
        
        # 添加收敛趋势分析
        if len(self._metrics_history) >= 2:
            recent_entropies = [m.avg_entropy for m in self._metrics_history[-10:]]
            if len(recent_entropies) >= 2:
                entropy_trend = recent_entropies[-1] - recent_entropies[0]
                report['entropy_trend'] = 'decreasing' if entropy_trend < -0.01 else (
                    'increasing' if entropy_trend > 0.01 else 'stable'
                )
        
        return report
    
    def reset(self):
        """重置监控器状态。"""
        self._entropy_history.clear()
        self._last_policy = None
        self._metrics_history.clear()
        self._current_iteration = 0
