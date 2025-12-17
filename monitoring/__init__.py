"""训练监控和指标收集模块。

该模块提供训练过程的监控功能，包括：
- MetricsCollector: 指标收集器，收集和聚合训练指标
- TrainingMonitor: 训练监控器，实时监控训练进度
- TensorBoardLogger: TensorBoard日志记录器，可视化训练曲线
"""

from monitoring.metrics_collector import MetricsCollector, EpisodeRecord
from monitoring.training_monitor import TrainingMonitor, LogEntry

# TensorBoard支持（可选）
try:
    from monitoring.tensorboard_logger import TensorBoardLogger, is_tensorboard_available
    __all__ = [
        'MetricsCollector', 'EpisodeRecord', 'TrainingMonitor', 'LogEntry',
        'TensorBoardLogger', 'is_tensorboard_available'
    ]
except ImportError:
    __all__ = ['MetricsCollector', 'EpisodeRecord', 'TrainingMonitor', 'LogEntry']
