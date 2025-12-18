"""可视化模块。

本模块实现实验结果的可视化功能。
"""

from typing import Dict, List, Optional, Tuple
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np

from experiments.equity_solver_validation.data_models import (
    ComparisonResult,
    BatchExperimentResult,
)


class Visualizer:
    """可视化器。
    
    生成实验结果的各种图表。
    """
    
    def __init__(self, output_dir: str = 'experiments/results'):
        """初始化可视化器。
        
        Args:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if not HAS_MATPLOTLIB:
            print("警告: matplotlib未安装，可视化功能不可用")
    
    def plot_equity_vs_strategy(
        self,
        comparison: ComparisonResult,
        action: str = 'bet',
        filename: str = 'equity_vs_strategy.png'
    ) -> Optional[str]:
        """绘制胜率-策略散点图。
        
        Args:
            comparison: 对比结果
            action: 要绘制的动作
            filename: 输出文件名
            
        Returns:
            图表文件路径，如果matplotlib不可用则返回None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        equities = []
        strategy_probs = []
        
        for hand, equity in comparison.equity_vector.items():
            if hand in comparison.solver_strategy:
                prob = comparison.solver_strategy[hand].get(action, 0)
                equities.append(equity)
                strategy_probs.append(prob)
        
        if not equities:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(equities, strategy_probs, alpha=0.6, s=50)
        ax.set_xlabel('胜率 (Equity)', fontsize=12)
        ax.set_ylabel(f'{action}概率', fontsize=12)
        ax.set_title(f'胜率 vs {action}策略', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(equities) > 2:
            z = np.polyfit(equities, strategy_probs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, 1, 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'趋势线')
            ax.legend()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_action_distribution(
        self,
        comparison: ComparisonResult,
        filename: str = 'action_distribution.png'
    ) -> Optional[str]:
        """绘制动作分布对比图。
        
        Args:
            comparison: 对比结果
            filename: 输出文件名
            
        Returns:
            图表文件路径
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if not comparison.action_distribution:
            return None
        
        actions = list(comparison.action_distribution.keys())
        equity_freqs = [comparison.action_distribution[a][0] for a in actions]
        solver_freqs = [comparison.action_distribution[a][1] for a in actions]
        
        x = np.arange(len(actions))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, equity_freqs, width, label='胜率策略', color='steelblue')
        bars2 = ax.bar(x + width/2, solver_freqs, width, label='Solver策略', color='coral')
        
        ax.set_xlabel('动作', fontsize=12)
        ax.set_ylabel('频率', fontsize=12)
        ax.set_title('动作分布对比', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(actions)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_batch_summary(
        self,
        batch_result: BatchExperimentResult,
        filename: str = 'batch_summary.png'
    ) -> Optional[str]:
        """绘制批量实验汇总图。
        
        Args:
            batch_result: 批量实验结果
            filename: 输出文件名
            
        Returns:
            图表文件路径
        """
        if not HAS_MATPLOTLIB:
            return None
        
        successful_results = [r for r in batch_result.results if r.success]
        
        if not successful_results:
            return None
        
        # 收集数据
        scenario_names = []
        tvd_values = []
        agreement_values = []
        
        for result in successful_results:
            scenario_names.append(result.scenario.name[:15])  # 截断名称
            tvd_values.append(result.comparison.metrics.total_variation_distance)
            agreement_values.append(result.comparison.metrics.action_agreement_rate)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 总变差距离
        ax1 = axes[0]
        bars1 = ax1.barh(scenario_names, tvd_values, color='steelblue')
        ax1.set_xlabel('总变差距离', fontsize=12)
        ax1.set_title('各场景策略差异', fontsize=14)
        ax1.set_xlim(0, 1)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 动作一致率
        ax2 = axes[1]
        bars2 = ax2.barh(scenario_names, agreement_values, color='coral')
        ax2.set_xlabel('动作一致率', fontsize=12)
        ax2.set_title('各场景动作一致性', fontsize=14)
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_tvd_histogram(
        self,
        batch_result: BatchExperimentResult,
        filename: str = 'tvd_histogram.png'
    ) -> Optional[str]:
        """绘制总变差距离直方图。
        
        Args:
            batch_result: 批量实验结果
            filename: 输出文件名
            
        Returns:
            图表文件路径
        """
        if not HAS_MATPLOTLIB:
            return None
        
        tvd_values = [
            r.comparison.metrics.total_variation_distance
            for r in batch_result.results
            if r.success
        ]
        
        if not tvd_values:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(tvd_values, bins=20, color='steelblue', edgecolor='white', alpha=0.7)
        ax.axvline(np.mean(tvd_values), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(tvd_values):.3f}')
        ax.set_xlabel('总变差距离', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title('策略差异分布', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
