"""转牌阶段Potential直方图可视化模块。

本模块实现了转牌验证实验结果的可视化功能，包括：
- Potential直方图热力图生成
- 按动作类型分组显示
- 支持PNG和SVG格式导出

Requirements: 7.1, 7.2, 7.3
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np

from experiments.turn_potential_validation.data_models import (
    TurnExperimentResult,
    TurnBatchExperimentResult,
    CorrelationResult,
    ClusteringComparisonResult,
    TurnValidationMetrics,
)


class TurnVisualizer:
    """转牌阶段Potential直方图可视化器。
    
    生成转牌验证实验结果的各种图表，包括：
    - Potential直方图热力图
    - 按动作类型分组的直方图显示
    - EMD距离矩阵热力图
    - 聚类结果可视化
    - 批量实验汇总图表
    
    Attributes:
        output_dir: 图表输出目录
        num_bins: Potential直方图的区间数量
        dpi: 图像分辨率
    """
    
    def __init__(
        self,
        output_dir: str = 'experiments/results/turn_validation',
        num_bins: int = 50,
        dpi: int = 150
    ):
        """初始化可视化器。
        
        Args:
            output_dir: 图表输出目录
            num_bins: Potential直方图的区间数量
            dpi: 图像分辨率
        """
        self.output_dir = output_dir
        self.num_bins = num_bins
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        
        if not HAS_MATPLOTLIB:
            print("警告: matplotlib未安装，可视化功能不可用")
        
        # 定义动作颜色映射
        self._action_colors = {
            'fold': '#FF6B6B',      # 红色
            'check': '#4ECDC4',     # 青色
            'call': '#45B7D1',      # 蓝色
            'bet': '#96CEB4',       # 绿色
            'raise': '#FFEAA7',     # 黄色
            'allin': '#DDA0DD',     # 紫色
        }
    
    def _get_action_color(self, action: str) -> str:
        """获取动作对应的颜色。
        
        Args:
            action: 动作名称
            
        Returns:
            颜色代码
        """
        action_lower = action.lower()
        for key, color in self._action_colors.items():
            if key in action_lower:
                return color
        return '#808080'  # 默认灰色
    
    def _save_figure(
        self,
        fig: Any,
        filename: str,
        format: str = 'png'
    ) -> str:
        """保存图表到文件。
        
        Args:
            fig: matplotlib图表对象
            filename: 文件名（不含扩展名）
            format: 输出格式（'png' 或 'svg'）
            
        Returns:
            保存的文件路径
        """
        filepath = os.path.join(self.output_dir, f"{filename}.{format}")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', format=format)
        plt.close(fig)
        return filepath

    def plot_potential_histogram_heatmap(
        self,
        potential_histograms: Dict[str, np.ndarray],
        solver_strategies: Optional[Dict[str, Dict[str, float]]] = None,
        title: str = 'Potential直方图热力图',
        filename: str = 'potential_histogram_heatmap',
        format: str = 'png',
        sort_by: str = 'mean_equity'
    ) -> Optional[str]:
        """绘制Potential直方图热力图。
        
        每行代表一个手牌，每列代表一个Equity区间，
        颜色深浅表示该区间的概率密度。
        
        Args:
            potential_histograms: 每个手牌的Potential直方图
            solver_strategies: Solver策略（可选，用于排序和标注）
            title: 图表标题
            filename: 输出文件名（不含扩展名）
            format: 输出格式（'png' 或 'svg'）
            sort_by: 排序方式（'mean_equity', 'variance', 'hand'）
            
        Returns:
            图表文件路径，如果matplotlib不可用则返回None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if not potential_histograms:
            return None
        
        # 获取手牌列表并排序
        hands = list(potential_histograms.keys())
        
        if sort_by == 'mean_equity':
            # 按平均Equity排序
            def get_mean_equity(hand):
                hist = potential_histograms[hand]
                bin_centers = np.linspace(0.01, 0.99, len(hist))
                return np.sum(hist * bin_centers)
            hands.sort(key=get_mean_equity, reverse=True)
        elif sort_by == 'variance':
            # 按方差排序
            def get_variance(hand):
                hist = potential_histograms[hand]
                bin_centers = np.linspace(0.01, 0.99, len(hist))
                mean = np.sum(hist * bin_centers)
                return np.sum(hist * (bin_centers - mean) ** 2)
            hands.sort(key=get_variance, reverse=True)
        else:
            # 按手牌名称排序
            hands.sort()
        
        # 构建热力图矩阵
        n_hands = len(hands)
        n_bins = len(potential_histograms[hands[0]])
        heatmap_data = np.zeros((n_hands, n_bins))
        
        for i, hand in enumerate(hands):
            heatmap_data[i, :] = potential_histograms[hand]
        
        # 创建图表
        fig_height = max(8, n_hands * 0.15)
        fig, ax = plt.subplots(figsize=(14, fig_height))
        
        # 绘制热力图
        im = ax.imshow(
            heatmap_data,
            aspect='auto',
            cmap='YlOrRd',
            interpolation='nearest',
            vmin=0,
            vmax=np.percentile(heatmap_data, 95)  # 使用95百分位数作为最大值
        )
        
        # 设置坐标轴
        ax.set_xlabel('Equity区间', fontsize=12)
        ax.set_ylabel('手牌', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # 设置X轴刻度
        x_ticks = np.linspace(0, n_bins - 1, 11).astype(int)
        x_labels = [f'{i/n_bins:.1f}' for i in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        
        # 设置Y轴刻度（如果手牌数量不太多）
        if n_hands <= 50:
            ax.set_yticks(range(n_hands))
            ax.set_yticklabels(hands, fontsize=8)
        else:
            # 只显示部分标签
            step = n_hands // 20
            y_ticks = range(0, n_hands, step)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([hands[i] for i in y_ticks], fontsize=8)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('概率密度', fontsize=10)
        
        # 如果有策略信息，在右侧添加主要动作标注
        if solver_strategies:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            
            action_labels = []
            for hand in hands:
                if hand in solver_strategies:
                    strategy = solver_strategies[hand]
                    main_action = max(strategy.items(), key=lambda x: x[1])[0]
                    action_labels.append(main_action[:6])  # 截断动作名称
                else:
                    action_labels.append('')
            
            if n_hands <= 50:
                ax2.set_yticks(range(n_hands))
                ax2.set_yticklabels(action_labels, fontsize=7)
            ax2.set_ylabel('主要动作', fontsize=10)
        
        return self._save_figure(fig, filename, format)

    def plot_histograms_by_action(
        self,
        potential_histograms: Dict[str, np.ndarray],
        solver_strategies: Dict[str, Dict[str, float]],
        title: str = '按动作类型分组的Potential直方图',
        filename: str = 'histograms_by_action',
        format: str = 'png'
    ) -> Optional[str]:
        """绘制按动作类型分组的Potential直方图。
        
        将手牌按其主要动作分组，每组显示平均直方图和分布范围。
        
        Args:
            potential_histograms: 每个手牌的Potential直方图
            solver_strategies: Solver策略
            title: 图表标题
            filename: 输出文件名（不含扩展名）
            format: 输出格式（'png' 或 'svg'）
            
        Returns:
            图表文件路径
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if not potential_histograms or not solver_strategies:
            return None
        
        # 按主要动作分组
        action_groups: Dict[str, List[np.ndarray]] = defaultdict(list)
        
        for hand, histogram in potential_histograms.items():
            if hand in solver_strategies:
                strategy = solver_strategies[hand]
                main_action = max(strategy.items(), key=lambda x: x[1])[0]
                action_groups[main_action].append(histogram)
        
        if not action_groups:
            return None
        
        # 创建子图
        n_actions = len(action_groups)
        fig, axes = plt.subplots(
            n_actions, 1,
            figsize=(12, 3 * n_actions),
            sharex=True
        )
        
        if n_actions == 1:
            axes = [axes]
        
        # X轴：Equity区间中点
        n_bins = len(list(potential_histograms.values())[0])
        x = np.linspace(0.01, 0.99, n_bins)
        
        for ax, (action, histograms) in zip(axes, sorted(action_groups.items())):
            histograms_array = np.array(histograms)
            
            # 计算平均值和标准差
            mean_hist = np.mean(histograms_array, axis=0)
            std_hist = np.std(histograms_array, axis=0)
            
            # 获取动作颜色
            color = self._get_action_color(action)
            
            # 绘制平均直方图
            ax.fill_between(
                x,
                mean_hist - std_hist,
                mean_hist + std_hist,
                alpha=0.3,
                color=color,
                label='±1标准差'
            )
            ax.plot(x, mean_hist, color=color, linewidth=2, label='平均值')
            
            # 添加标注
            ax.set_ylabel('概率密度', fontsize=10)
            ax.set_title(f'{action} (n={len(histograms)})', fontsize=11)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
        
        axes[-1].set_xlabel('Equity', fontsize=12)
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, filename, format)
    
    def plot_action_histogram_comparison(
        self,
        potential_histograms: Dict[str, np.ndarray],
        solver_strategies: Dict[str, Dict[str, float]],
        title: str = '不同动作的Potential直方图对比',
        filename: str = 'action_histogram_comparison',
        format: str = 'png'
    ) -> Optional[str]:
        """绘制不同动作的平均Potential直方图对比图。
        
        在同一图表中对比不同动作的平均直方图分布。
        
        Args:
            potential_histograms: 每个手牌的Potential直方图
            solver_strategies: Solver策略
            title: 图表标题
            filename: 输出文件名（不含扩展名）
            format: 输出格式（'png' 或 'svg'）
            
        Returns:
            图表文件路径
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if not potential_histograms or not solver_strategies:
            return None
        
        # 按主要动作分组
        action_groups: Dict[str, List[np.ndarray]] = defaultdict(list)
        
        for hand, histogram in potential_histograms.items():
            if hand in solver_strategies:
                strategy = solver_strategies[hand]
                main_action = max(strategy.items(), key=lambda x: x[1])[0]
                action_groups[main_action].append(histogram)
        
        if not action_groups:
            return None
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # X轴：Equity区间中点
        n_bins = len(list(potential_histograms.values())[0])
        x = np.linspace(0.01, 0.99, n_bins)
        
        # 绘制每个动作的平均直方图
        for action in sorted(action_groups.keys()):
            histograms = action_groups[action]
            mean_hist = np.mean(histograms, axis=0)
            color = self._get_action_color(action)
            ax.plot(
                x, mean_hist,
                color=color,
                linewidth=2,
                label=f'{action} (n={len(histograms)})'
            )
        
        ax.set_xlabel('Equity', fontsize=12)
        ax.set_ylabel('概率密度', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        return self._save_figure(fig, filename, format)

    def plot_emd_distance_matrix(
        self,
        hands: List[str],
        distance_matrix: np.ndarray,
        solver_strategies: Optional[Dict[str, Dict[str, float]]] = None,
        title: str = 'EMD距离矩阵',
        filename: str = 'emd_distance_matrix',
        format: str = 'png'
    ) -> Optional[str]:
        """绘制EMD距离矩阵热力图。
        
        Args:
            hands: 手牌列表
            distance_matrix: EMD距离矩阵
            solver_strategies: Solver策略（可选，用于排序）
            title: 图表标题
            filename: 输出文件名（不含扩展名）
            format: 输出格式（'png' 或 'svg'）
            
        Returns:
            图表文件路径
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if len(hands) == 0 or distance_matrix.size == 0:
            return None
        
        # 如果有策略信息，按主要动作排序
        if solver_strategies:
            action_order = []
            for hand in hands:
                if hand in solver_strategies:
                    strategy = solver_strategies[hand]
                    main_action = max(strategy.items(), key=lambda x: x[1])[0]
                    action_order.append((hand, main_action))
                else:
                    action_order.append((hand, 'unknown'))
            
            # 按动作排序
            action_order.sort(key=lambda x: x[1])
            sorted_hands = [h for h, _ in action_order]
            
            # 重新排列距离矩阵
            indices = [hands.index(h) for h in sorted_hands]
            sorted_matrix = distance_matrix[np.ix_(indices, indices)]
            hands = sorted_hands
            distance_matrix = sorted_matrix
        
        # 创建图表
        n = len(hands)
        fig_size = max(8, n * 0.2)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # 绘制热力图
        im = ax.imshow(
            distance_matrix,
            cmap='Blues',
            interpolation='nearest'
        )
        
        # 设置坐标轴
        ax.set_title(title, fontsize=14)
        
        # 设置刻度
        if n <= 30:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(hands, rotation=90, fontsize=7)
            ax.set_yticklabels(hands, fontsize=7)
        else:
            step = n // 15
            ticks = range(0, n, step)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([hands[i] for i in ticks], rotation=90, fontsize=7)
            ax.set_yticklabels([hands[i] for i in ticks], fontsize=7)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('EMD距离', fontsize=10)
        
        return self._save_figure(fig, filename, format)
    
    def plot_clustering_result(
        self,
        cluster_labels: Dict[str, int],
        potential_histograms: Dict[str, np.ndarray],
        solver_strategies: Optional[Dict[str, Dict[str, float]]] = None,
        title: str = '聚类结果可视化',
        filename: str = 'clustering_result',
        format: str = 'png'
    ) -> Optional[str]:
        """绘制聚类结果可视化图。
        
        显示每个聚类的平均直方图和手牌分布。
        
        Args:
            cluster_labels: 聚类标签
            potential_histograms: Potential直方图
            solver_strategies: Solver策略（可选）
            title: 图表标题
            filename: 输出文件名（不含扩展名）
            format: 输出格式（'png' 或 'svg'）
            
        Returns:
            图表文件路径
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if not cluster_labels or not potential_histograms:
            return None
        
        # 按聚类分组
        cluster_histograms: Dict[int, List[np.ndarray]] = defaultdict(list)
        cluster_actions: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for hand, cluster in cluster_labels.items():
            if hand in potential_histograms:
                cluster_histograms[cluster].append(potential_histograms[hand])
                
                if solver_strategies and hand in solver_strategies:
                    strategy = solver_strategies[hand]
                    main_action = max(strategy.items(), key=lambda x: x[1])[0]
                    cluster_actions[cluster][main_action] += 1
        
        n_clusters = len(cluster_histograms)
        if n_clusters == 0:
            return None
        
        # 创建子图
        fig, axes = plt.subplots(
            n_clusters, 2,
            figsize=(14, 3 * n_clusters),
            gridspec_kw={'width_ratios': [2, 1]}
        )
        
        if n_clusters == 1:
            axes = [axes]
        
        # X轴：Equity区间中点
        n_bins = len(list(potential_histograms.values())[0])
        x = np.linspace(0.01, 0.99, n_bins)
        
        # 定义聚类颜色
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for idx, cluster in enumerate(sorted(cluster_histograms.keys())):
            histograms = cluster_histograms[cluster]
            histograms_array = np.array(histograms)
            
            # 左图：直方图分布
            ax_hist = axes[idx][0]
            mean_hist = np.mean(histograms_array, axis=0)
            std_hist = np.std(histograms_array, axis=0)
            
            ax_hist.fill_between(
                x,
                mean_hist - std_hist,
                mean_hist + std_hist,
                alpha=0.3,
                color=cluster_colors[idx]
            )
            ax_hist.plot(x, mean_hist, color=cluster_colors[idx], linewidth=2)
            ax_hist.set_ylabel('概率密度', fontsize=10)
            ax_hist.set_title(f'聚类 {cluster} (n={len(histograms)})', fontsize=11)
            ax_hist.grid(True, alpha=0.3)
            ax_hist.set_xlim(0, 1)
            
            if idx == n_clusters - 1:
                ax_hist.set_xlabel('Equity', fontsize=10)
            
            # 右图：动作分布饼图
            ax_pie = axes[idx][1]
            if cluster in cluster_actions and cluster_actions[cluster]:
                actions = list(cluster_actions[cluster].keys())
                counts = list(cluster_actions[cluster].values())
                colors = [self._get_action_color(a) for a in actions]
                
                ax_pie.pie(
                    counts,
                    labels=actions,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax_pie.set_title('动作分布', fontsize=10)
            else:
                ax_pie.text(0.5, 0.5, '无策略数据', ha='center', va='center')
                ax_pie.set_xlim(0, 1)
                ax_pie.set_ylim(0, 1)
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, filename, format)

    def plot_correlation_analysis(
        self,
        correlation_result: CorrelationResult,
        title: str = '相关性分析结果',
        filename: str = 'correlation_analysis',
        format: str = 'png'
    ) -> Optional[str]:
        """绘制相关性分析结果图。
        
        Args:
            correlation_result: 相关性分析结果
            title: 图表标题
            filename: 输出文件名（不含扩展名）
            format: 输出格式（'png' 或 'svg'）
            
        Returns:
            图表文件路径
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 特征相关性条形图
        ax1 = axes[0]
        features = ['平均Equity', '方差', '熵']
        correlations = [
            correlation_result.mean_equity_correlation,
            correlation_result.variance_correlation,
            correlation_result.histogram_entropy_correlation
        ]
        colors = ['#4ECDC4' if c >= 0 else '#FF6B6B' for c in correlations]
        
        bars = ax1.barh(features, correlations, color=colors)
        ax1.set_xlabel('Spearman相关系数', fontsize=10)
        ax1.set_title('直方图特征与策略相关性', fontsize=11)
        ax1.set_xlim(-1, 1)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for bar, corr in zip(bars, correlations):
            width = bar.get_width()
            ax1.annotate(
                f'{corr:.3f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5 if width >= 0 else -5, 0),
                textcoords='offset points',
                ha='left' if width >= 0 else 'right',
                va='center',
                fontsize=9
            )
        
        # 2. 同一动作内EMD距离
        ax2 = axes[1]
        if correlation_result.intra_action_emd:
            actions = list(correlation_result.intra_action_emd.keys())
            emd_values = list(correlation_result.intra_action_emd.values())
            colors = [self._get_action_color(a) for a in actions]
            
            ax2.bar(actions, emd_values, color=colors)
            ax2.set_xlabel('动作', fontsize=10)
            ax2.set_ylabel('平均EMD距离', fontsize=10)
            ax2.set_title('同一动作内EMD距离', fontsize=11)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, '无数据', ha='center', va='center')
        
        # 3. 不同动作间EMD距离
        ax3 = axes[2]
        if correlation_result.inter_action_emd:
            pairs = list(correlation_result.inter_action_emd.keys())
            emd_values = list(correlation_result.inter_action_emd.values())
            
            ax3.barh(pairs, emd_values, color='#45B7D1')
            ax3.set_xlabel('平均EMD距离', fontsize=10)
            ax3.set_title('不同动作间EMD距离', fontsize=11)
            ax3.grid(True, alpha=0.3, axis='x')
        else:
            ax3.text(0.5, 0.5, '无数据', ha='center', va='center')
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, filename, format)
    
    def plot_metrics_summary(
        self,
        metrics: TurnValidationMetrics,
        title: str = '验证指标汇总',
        filename: str = 'metrics_summary',
        format: str = 'png'
    ) -> Optional[str]:
        """绘制验证指标汇总图。
        
        Args:
            metrics: 验证指标
            title: 图表标题
            filename: 输出文件名（不含扩展名）
            format: 输出格式（'png' 或 'svg'）
            
        Returns:
            图表文件路径
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. 直方图质量指标
        ax1 = axes[0]
        quality_metrics = {
            '平均熵': metrics.avg_histogram_entropy,
            '稀疏度': metrics.histogram_sparsity,
        }
        
        names = list(quality_metrics.keys())
        values = list(quality_metrics.values())
        
        bars = ax1.bar(names, values, color=['#4ECDC4', '#96CEB4'])
        ax1.set_ylabel('值', fontsize=10)
        ax1.set_title('直方图质量指标', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax1.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                fontsize=9
            )
        
        # 2. 策略一致性指标
        ax2 = axes[1]
        consistency_metrics = {
            '策略相关性': metrics.strategy_correlation,
            'EV相关性': metrics.ev_correlation,
            '聚类纯度': metrics.clustering_purity,
            '动作一致率': metrics.action_agreement_rate,
        }
        
        names = list(consistency_metrics.keys())
        values = list(consistency_metrics.values())
        colors = ['#4ECDC4' if v >= 0 else '#FF6B6B' for v in values]
        
        bars = ax2.barh(names, values, color=colors)
        ax2.set_xlabel('值', fontsize=10)
        ax2.set_title('策略一致性指标', fontsize=11)
        ax2.set_xlim(-1, 1)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax2.annotate(
                f'{val:.3f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5 if width >= 0 else -5, 0),
                textcoords='offset points',
                ha='left' if width >= 0 else 'right',
                va='center',
                fontsize=9
            )
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, filename, format)

    def plot_batch_summary(
        self,
        batch_result: TurnBatchExperimentResult,
        title: str = '批量实验汇总',
        filename: str = 'batch_summary',
        format: str = 'png'
    ) -> Optional[str]:
        """绘制批量实验汇总图。
        
        Args:
            batch_result: 批量实验结果
            title: 图表标题
            filename: 输出文件名（不含扩展名）
            format: 输出格式（'png' 或 'svg'）
            
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
        strategy_corr_values = []
        purity_values = []
        agreement_values = []
        
        for result in successful_results:
            scenario_names.append(result.scenario.name[:20])  # 截断名称
            if result.metrics:
                strategy_corr_values.append(result.metrics.strategy_correlation)
                purity_values.append(result.metrics.clustering_purity)
                agreement_values.append(result.metrics.action_agreement_rate)
            else:
                strategy_corr_values.append(0)
                purity_values.append(0)
                agreement_values.append(0)
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        
        # 1. 策略相关性
        ax1 = axes[0]
        colors = ['#4ECDC4' if v >= 0 else '#FF6B6B' for v in strategy_corr_values]
        bars1 = ax1.barh(scenario_names, strategy_corr_values, color=colors)
        ax1.set_xlabel('策略相关性', fontsize=10)
        ax1.set_title('各场景策略相关性', fontsize=11)
        ax1.set_xlim(-1, 1)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. 聚类纯度
        ax2 = axes[1]
        bars2 = ax2.barh(scenario_names, purity_values, color='#96CEB4')
        ax2.set_xlabel('聚类纯度', fontsize=10)
        ax2.set_title('各场景聚类纯度', fontsize=11)
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. 动作一致率
        ax3 = axes[2]
        bars3 = ax3.barh(scenario_names, agreement_values, color='#45B7D1')
        ax3.set_xlabel('动作一致率', fontsize=10)
        ax3.set_title('各场景动作一致率', fontsize=11)
        ax3.set_xlim(0, 1)
        ax3.grid(True, alpha=0.3, axis='x')
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, filename, format)
    
    def plot_experiment_result(
        self,
        result: TurnExperimentResult,
        prefix: str = '',
        format: str = 'png'
    ) -> Dict[str, Optional[str]]:
        """为单个实验结果生成所有相关图表。
        
        Args:
            result: 实验结果
            prefix: 文件名前缀
            format: 输出格式（'png' 或 'svg'）
            
        Returns:
            图表文件路径字典
        """
        if not result.success:
            return {}
        
        file_prefix = f"{prefix}_{result.scenario.name}" if prefix else result.scenario.name
        file_prefix = file_prefix.replace(' ', '_').replace('/', '_')
        
        paths = {}
        
        # 获取Solver策略（如果有）
        solver_strategies = None
        if result.solver_result and result.solver_result.strategies:
            solver_strategies = result.solver_result.strategies
        
        # 1. Potential直方图热力图
        if result.potential_histograms:
            paths['heatmap'] = self.plot_potential_histogram_heatmap(
                result.potential_histograms,
                solver_strategies,
                title=f'{result.scenario.name} - Potential直方图热力图',
                filename=f'{file_prefix}_heatmap',
                format=format
            )
        
        # 2. 按动作分组的直方图
        if result.potential_histograms and solver_strategies:
            paths['by_action'] = self.plot_histograms_by_action(
                result.potential_histograms,
                solver_strategies,
                title=f'{result.scenario.name} - 按动作分组',
                filename=f'{file_prefix}_by_action',
                format=format
            )
            
            paths['action_comparison'] = self.plot_action_histogram_comparison(
                result.potential_histograms,
                solver_strategies,
                title=f'{result.scenario.name} - 动作对比',
                filename=f'{file_prefix}_action_comparison',
                format=format
            )
        
        # 3. 相关性分析
        if result.correlation_result:
            paths['correlation'] = self.plot_correlation_analysis(
                result.correlation_result,
                title=f'{result.scenario.name} - 相关性分析',
                filename=f'{file_prefix}_correlation',
                format=format
            )
        
        # 4. 验证指标
        if result.metrics:
            paths['metrics'] = self.plot_metrics_summary(
                result.metrics,
                title=f'{result.scenario.name} - 验证指标',
                filename=f'{file_prefix}_metrics',
                format=format
            )
        
        return paths
    
    def generate_all_visualizations(
        self,
        batch_result: TurnBatchExperimentResult,
        format: str = 'png'
    ) -> Dict[str, Any]:
        """为批量实验结果生成所有可视化图表。
        
        Args:
            batch_result: 批量实验结果
            format: 输出格式（'png' 或 'svg'）
            
        Returns:
            所有图表文件路径的字典
        """
        all_paths = {
            'batch_summary': None,
            'individual_results': {}
        }
        
        # 生成批量汇总图
        all_paths['batch_summary'] = self.plot_batch_summary(
            batch_result,
            filename='batch_summary',
            format=format
        )
        
        # 为每个成功的实验生成图表
        for result in batch_result.results:
            if result.success:
                paths = self.plot_experiment_result(
                    result,
                    format=format
                )
                all_paths['individual_results'][result.scenario.name] = paths
        
        return all_paths
