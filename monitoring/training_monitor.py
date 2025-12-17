"""训练监控器模块 - 实时监控训练进度。

该模块实现了TrainingMonitor类，用于：
- 实时监控训练进度
- 更新和显示训练指标
- 绘制指标曲线
- 检测训练异常
- 持久化日志到文件
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict

from monitoring.metrics_collector import MetricsCollector


@dataclass
class LogEntry:
    """日志条目数据结构。
    
    Attributes:
        timestamp: 时间戳（ISO格式字符串）
        episode_number: 回合编号
        metrics: 指标字典
    """
    timestamp: str
    episode_number: int
    metrics: Dict[str, float]


class TrainingMonitor:
    """训练监控器。
    
    实时监控训练进度，提供以下功能：
    - 启动后台监控线程
    - 定期更新和显示训练指标
    - 绘制指标曲线（使用matplotlib）
    - 检测训练异常（如损失值发散）
    - 将日志持久化到JSON Lines格式文件
    
    Attributes:
        metrics_collector: 指标收集器实例
        update_interval: 更新间隔（秒）
        log_file_path: 日志文件路径
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        update_interval: float = 1.0,
        log_file_path: Optional[str] = None,
        loss_threshold: float = 100.0,
        loss_increase_threshold: float = 10.0
    ):
        """初始化训练监控器。
        
        Args:
            metrics_collector: 指标收集器实例
            update_interval: 更新间隔（秒），默认1秒
            log_file_path: 日志文件路径，默认为None（不写入文件）
            loss_threshold: 损失值阈值，超过此值视为异常
            loss_increase_threshold: 损失值增长阈值，连续增长超过此倍数视为发散
        """
        if update_interval <= 0:
            raise ValueError(f"更新间隔必须为正数，收到 {update_interval}")
        
        self._metrics_collector = metrics_collector
        self._update_interval = update_interval
        self._log_file_path = Path(log_file_path) if log_file_path else None
        self._loss_threshold = loss_threshold
        self._loss_increase_threshold = loss_increase_threshold
        
        # 监控线程相关
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()
        
        # 最近的指标快照
        self._latest_metrics: Dict[str, float] = {}
        self._metrics_lock = threading.Lock()
        
        # 异常检测相关
        self._anomalies: List[str] = []
        self._previous_losses: List[float] = []
        
        # 回调函数
        self._on_update_callback: Optional[Callable[[Dict[str, float]], None]] = None
        self._on_anomaly_callback: Optional[Callable[[List[str]], None]] = None
        
        # 上次更新时间
        self._last_update_time: Optional[datetime] = None
    
    @property
    def metrics_collector(self) -> MetricsCollector:
        """获取指标收集器。"""
        return self._metrics_collector
    
    @property
    def update_interval(self) -> float:
        """获取更新间隔。"""
        return self._update_interval
    
    @property
    def log_file_path(self) -> Optional[Path]:
        """获取日志文件路径。"""
        return self._log_file_path
    
    @property
    def is_running(self) -> bool:
        """检查监控器是否正在运行。"""
        return self._running
    
    @property
    def last_update_time(self) -> Optional[datetime]:
        """获取上次更新时间。"""
        return self._last_update_time
    
    def set_on_update_callback(self, callback: Callable[[Dict[str, float]], None]) -> None:
        """设置指标更新回调函数。
        
        Args:
            callback: 回调函数，接收指标字典作为参数
        """
        self._on_update_callback = callback
    
    def set_on_anomaly_callback(self, callback: Callable[[List[str]], None]) -> None:
        """设置异常检测回调函数。
        
        Args:
            callback: 回调函数，接收异常列表作为参数
        """
        self._on_anomaly_callback = callback
    
    def start(self) -> None:
        """启动监控线程。
        
        如果监控器已经在运行，则不执行任何操作。
        """
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """停止监控线程。
        
        等待监控线程安全退出。
        """
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=self._update_interval * 2)
    
    def update(self, metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """更新显示的指标。
        
        Args:
            metrics: 可选的指标字典，如果为None则从收集器获取
            
        Returns:
            更新后的指标字典
        """
        if metrics is None:
            metrics = self._metrics_collector.get_current_metrics()
        
        with self._metrics_lock:
            self._latest_metrics = metrics.copy()
            self._last_update_time = datetime.now()
        
        # 写入日志文件
        if self._log_file_path:
            self._write_log_entry(metrics)
        
        # 检测异常
        anomalies = self.check_anomalies(metrics)
        if anomalies and self._on_anomaly_callback:
            self._on_anomaly_callback(anomalies)
        
        # 调用更新回调
        if self._on_update_callback:
            self._on_update_callback(metrics)
        
        return metrics
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """获取最新的指标快照。
        
        Returns:
            最新的指标字典
        """
        with self._metrics_lock:
            return self._latest_metrics.copy()
    
    def check_anomalies(self, metrics: Dict[str, float]) -> List[str]:
        """检测训练异常。
        
        检测以下异常情况：
        - 损失值超过阈值
        - 损失值持续发散（连续增长）
        - 损失值为NaN或Inf
        
        Args:
            metrics: 当前指标字典
            
        Returns:
            检测到的异常列表
        """
        anomalies = []
        loss = metrics.get('loss', 0.0)
        
        # 检查NaN或Inf
        if loss != loss:  # NaN检查
            anomalies.append("损失值为NaN")
        elif abs(loss) == float('inf'):
            anomalies.append("损失值为无穷大")
        else:
            # 检查损失值是否超过阈值
            if loss > self._loss_threshold:
                anomalies.append(f"损失值 ({loss:.4f}) 超过阈值 ({self._loss_threshold})")
            
            # 检查损失值是否发散
            self._previous_losses.append(loss)
            if len(self._previous_losses) > 10:
                self._previous_losses.pop(0)
            
            if len(self._previous_losses) >= 5:
                # 检查最近5个损失值是否持续增长
                recent = self._previous_losses[-5:]
                if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                    # 检查增长幅度
                    if recent[-1] > recent[0] * self._loss_increase_threshold:
                        anomalies.append(
                            f"损失值发散：从 {recent[0]:.4f} 增长到 {recent[-1]:.4f}"
                        )
        
        self._anomalies = anomalies
        return anomalies
    
    def get_anomalies(self) -> List[str]:
        """获取最近检测到的异常列表。
        
        Returns:
            异常列表
        """
        return self._anomalies.copy()
    
    def plot_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Any:
        """绘制指标曲线。
        
        使用matplotlib绘制指定指标的历史曲线。
        
        Args:
            metric_names: 要绘制的指标名称列表，默认绘制所有指标
            save_path: 保存图片的路径，默认不保存
            show: 是否显示图片，默认True
            
        Returns:
            matplotlib的Figure对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("需要安装matplotlib才能绘制图表：pip install matplotlib")
        
        if metric_names is None:
            metric_names = ['win_rate', 'avg_reward', 'loss']
        
        # 获取指标历史
        histories = {}
        for name in metric_names:
            try:
                histories[name] = self._metrics_collector.get_metric_history(name)
            except ValueError:
                continue
        
        if not histories:
            raise ValueError("没有可绘制的指标数据")
        
        # 创建子图
        num_metrics = len(histories)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
        
        if num_metrics == 1:
            axes = [axes]
        
        # 指标名称的中文映射
        name_mapping = {
            'win_rate': '胜率',
            'avg_reward': '平均奖励',
            'loss': '损失值',
            'episodes_completed': '已完成回合数'
        }
        
        for ax, (name, history) in zip(axes, histories.items()):
            if history:
                ax.plot(history, label=name_mapping.get(name, name))
                ax.set_xlabel('回合')
                ax.set_ylabel(name_mapping.get(name, name))
                ax.set_title(f'{name_mapping.get(name, name)}变化曲线')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def read_log_file(self) -> List[LogEntry]:
        """读取日志文件内容。
        
        Returns:
            日志条目列表
            
        Raises:
            FileNotFoundError: 如果日志文件不存在
            ValueError: 如果未设置日志文件路径
        """
        if not self._log_file_path:
            raise ValueError("未设置日志文件路径")
        
        if not self._log_file_path.exists():
            return []
        
        entries = []
        with open(self._log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    entries.append(LogEntry(
                        timestamp=data['timestamp'],
                        episode_number=data['episode_number'],
                        metrics=data['metrics']
                    ))
        
        return entries
    
    def _monitor_loop(self) -> None:
        """监控线程的主循环。"""
        while not self._stop_event.is_set():
            try:
                self.update()
            except Exception as e:
                # 记录错误但继续运行
                print(f"监控更新错误: {e}")
            
            # 等待下一次更新
            self._stop_event.wait(self._update_interval)
    
    def _write_log_entry(self, metrics: Dict[str, float]) -> None:
        """写入日志条目到文件。
        
        Args:
            metrics: 指标字典
        """
        if not self._log_file_path:
            return
        
        # 确保目录存在
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            episode_number=int(metrics.get('episodes_completed', 0)),
            metrics=metrics
        )
        
        with open(self._log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + '\n')
