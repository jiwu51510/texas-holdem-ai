"""训练引擎和 CFR 算法实现。

本模块提供 Deep CFR 训练所需的核心组件：

Deep CFR 组件：
- DeepCFRTrainer: Deep CFR 训练器，实现标准 Deep CFR 算法
- ReservoirBuffer: 蓄水池缓冲区，用于经验回放

收敛控制组件：
- RegretProcessor: 遗憾值处理器
- RegretProcessorConfig: 遗憾值处理器配置
- CFRVariant: CFR变体枚举
- CFRVariantConfig: CFR变体配置
- CFRVariantSelector: CFR变体选择器
- BufferManager: 缓冲区管理器
- BufferManagerConfig: 缓冲区管理器配置
- ConvergenceMonitor: 收敛监控器
- ConvergenceMonitorConfig: 收敛监控器配置
- ConvergenceMetrics: 收敛指标数据类

辅助组件：
- CFRTrainer: 基础 CFR 训练器
- InfoSet: 信息集数据结构
- ParallelTrainer: 并行训练器
- WorkerExperience: 工作进程经验数据
"""

from training.cfr_trainer import CFRTrainer, InfoSet
from training.parallel_trainer import ParallelTrainer, WorkerExperience
from training.reservoir_buffer import ReservoirBuffer
from training.deep_cfr_trainer import DeepCFRTrainer
from training.regret_processor import RegretProcessor, RegretProcessorConfig
from training.cfr_variants import CFRVariant, CFRVariantConfig, CFRVariantSelector
from training.buffer_manager import BufferManager, BufferManagerConfig
from training.convergence_monitor import (
    ConvergenceMonitor, 
    ConvergenceMonitorConfig, 
    ConvergenceMetrics
)

__all__ = [
    # Deep CFR 核心组件
    'DeepCFRTrainer',
    'ReservoirBuffer',
    # 收敛控制组件
    'RegretProcessor',
    'RegretProcessorConfig',
    'CFRVariant',
    'CFRVariantConfig',
    'CFRVariantSelector',
    'BufferManager',
    'BufferManagerConfig',
    'ConvergenceMonitor',
    'ConvergenceMonitorConfig',
    'ConvergenceMetrics',
    # 辅助组件
    'CFRTrainer', 
    'InfoSet', 
    'ParallelTrainer', 
    'WorkerExperience', 
]
