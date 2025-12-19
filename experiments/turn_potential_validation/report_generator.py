"""转牌阶段Potential直方图验证实验报告生成器。

本模块实现了实验结果的详细分析和报告生成功能，包括：
- 对比报告生成
- 统计汇总
- Markdown格式报告

Requirements: 5.3, 6.3, 7.1
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json

import numpy as np

from experiments.turn_potential_validation.data_models import (
    TurnExperimentResult,
    TurnBatchExperimentResult,
    TurnValidationMetrics,
    CorrelationResult,
    ClusteringComparisonResult,
)


class TurnReportGenerator:
    """转牌实验报告生成器。
    
    生成转牌阶段Potential直方图验证实验的详细分析报告。
    支持文本格式和Markdown格式输出。
    """
    
    def __init__(self, output_dir: str = 'experiments/results/turn_validation'):
        """初始化报告生成器。
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_text_report(
        self,
        batch_result: TurnBatchExperimentResult,
        title: str = '转牌阶段Potential直方图验证实验报告'
    ) -> str:
        """生成文本格式报告。
        
        Args:
            batch_result: 批量实验结果
            title: 报告标题
            
        Returns:
            报告文本
        """
        lines = [
            "=" * 70,
            title,
            "=" * 70,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # 1. 实验概述
        lines.extend(self._generate_overview_section(batch_result))
        
        # 2. 汇总统计
        lines.extend(self._generate_summary_section(batch_result))
        
        # 3. 各场景详情
        lines.extend(self._generate_scenario_details_section(batch_result))
        
        # 4. 结论和建议
        lines.extend(self._generate_conclusions_section(batch_result))
        
        lines.append("=" * 70)
        
        return "\n".join(lines)

    def _generate_overview_section(
        self,
        batch_result: TurnBatchExperimentResult
    ) -> List[str]:
        """生成实验概述部分。"""
        lines = [
            "",
            "-" * 70,
            "1. 实验概述",
            "-" * 70,
            "",
            f"实验数量: {len(batch_result.results)}",
            f"成功数量: {batch_result.success_count}",
            f"失败数量: {batch_result.failure_count}",
            f"成功率: {batch_result.success_count / len(batch_result.results) * 100:.1f}%",
            f"总耗时: {batch_result.total_execution_time:.2f} 秒",
            f"平均耗时: {batch_result.total_execution_time / len(batch_result.results):.2f} 秒/场景",
            "",
        ]
        
        # 按标签统计
        tag_counts: Dict[str, int] = {}
        for result in batch_result.results:
            for tag in result.scenario.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        if tag_counts:
            lines.append("场景标签分布:")
            for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - {tag}: {count}")
        
        return lines
    
    def _generate_summary_section(
        self,
        batch_result: TurnBatchExperimentResult
    ) -> List[str]:
        """生成汇总统计部分。"""
        lines = [
            "",
            "-" * 70,
            "2. 汇总统计",
            "-" * 70,
            "",
        ]
        
        # 收集成功实验的指标
        successful_results = [r for r in batch_result.results if r.success]
        
        if not successful_results:
            lines.append("无成功的实验结果")
            return lines
        
        # 收集各指标
        metrics_data = {
            '策略相关性': [],
            '聚类纯度': [],
            '动作一致率': [],
            '平均直方图熵': [],
            '直方图稀疏度': [],
            '同一动作内EMD': [],
            '不同动作间EMD': [],
        }
        
        for result in successful_results:
            if result.metrics:
                metrics_data['策略相关性'].append(result.metrics.strategy_correlation)
                metrics_data['聚类纯度'].append(result.metrics.clustering_purity)
                metrics_data['动作一致率'].append(result.metrics.action_agreement_rate)
                metrics_data['平均直方图熵'].append(result.metrics.avg_histogram_entropy)
                metrics_data['直方图稀疏度'].append(result.metrics.histogram_sparsity)
                metrics_data['同一动作内EMD'].append(result.metrics.avg_intra_action_emd)
                metrics_data['不同动作间EMD'].append(result.metrics.avg_inter_action_emd)
        
        # 计算统计量
        lines.append("指标统计:")
        lines.append("")
        lines.append(f"{'指标名称':<20} {'平均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
        lines.append("-" * 62)
        
        for name, values in metrics_data.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                lines.append(
                    f"{name:<20} {mean_val:>10.4f} {std_val:>10.4f} "
                    f"{min_val:>10.4f} {max_val:>10.4f}"
                )
        
        # 按场景类型分组统计
        lines.extend(self._generate_tag_based_statistics(successful_results))
        
        return lines
    
    def _generate_tag_based_statistics(
        self,
        results: List[TurnExperimentResult]
    ) -> List[str]:
        """生成按标签分组的统计。"""
        lines = [
            "",
            "按场景类型分组统计:",
            "",
        ]
        
        # 按主要标签分组
        tag_groups: Dict[str, List[TurnExperimentResult]] = {}
        primary_tags = ['dry_board', 'wet_board', 'draw_board', 'paired_board', 'special']
        
        for result in results:
            for tag in primary_tags:
                if tag in result.scenario.tags:
                    if tag not in tag_groups:
                        tag_groups[tag] = []
                    tag_groups[tag].append(result)
                    break
        
        for tag, group_results in sorted(tag_groups.items()):
            if not group_results:
                continue
            
            corr_values = [r.metrics.strategy_correlation for r in group_results if r.metrics]
            purity_values = [r.metrics.clustering_purity for r in group_results if r.metrics]
            agreement_values = [r.metrics.action_agreement_rate for r in group_results if r.metrics]
            
            lines.append(f"\n{tag} ({len(group_results)} 个场景):")
            if corr_values:
                lines.append(f"  策略相关性: {np.mean(corr_values):.4f} ± {np.std(corr_values):.4f}")
            if purity_values:
                lines.append(f"  聚类纯度: {np.mean(purity_values):.4f} ± {np.std(purity_values):.4f}")
            if agreement_values:
                lines.append(f"  动作一致率: {np.mean(agreement_values):.4f} ± {np.std(agreement_values):.4f}")
        
        return lines

    def _generate_scenario_details_section(
        self,
        batch_result: TurnBatchExperimentResult
    ) -> List[str]:
        """生成各场景详情部分。"""
        lines = [
            "",
            "-" * 70,
            "3. 各场景详情",
            "-" * 70,
        ]
        
        for i, result in enumerate(batch_result.results, 1):
            lines.append("")
            status = "✓ 成功" if result.success else "✗ 失败"
            lines.append(f"场景 {i}: {result.scenario.name} [{status}]")
            lines.append(f"  描述: {result.scenario.description}")
            lines.append(f"  公共牌: {result.scenario.get_turn_community_str()}")
            lines.append(f"  标签: {', '.join(result.scenario.tags)}")
            lines.append(f"  执行时间: {result.execution_time:.2f} 秒")
            
            if result.error:
                lines.append(f"  错误: {result.error}")
                continue
            
            if result.metrics:
                lines.append("")
                lines.append("  验证指标:")
                lines.append(f"    策略相关性: {result.metrics.strategy_correlation:.4f}")
                lines.append(f"    聚类纯度: {result.metrics.clustering_purity:.4f}")
                lines.append(f"    动作一致率: {result.metrics.action_agreement_rate:.4f}")
                lines.append(f"    平均直方图熵: {result.metrics.avg_histogram_entropy:.4f}")
                lines.append(f"    直方图稀疏度: {result.metrics.histogram_sparsity:.4f}")
            
            if result.correlation_result:
                lines.append("")
                lines.append("  相关性分析:")
                lines.append(f"    平均Equity相关性: {result.correlation_result.mean_equity_correlation:.4f}")
                lines.append(f"    方差相关性: {result.correlation_result.variance_correlation:.4f}")
                
                if result.correlation_result.intra_action_emd:
                    lines.append("    同一动作内EMD:")
                    for action, emd in result.correlation_result.intra_action_emd.items():
                        lines.append(f"      {action}: {emd:.4f}")
            
            if result.clustering_result:
                lines.append("")
                lines.append("  聚类分析:")
                lines.append(f"    聚类数量: {result.clustering_result.num_clusters}")
                lines.append(f"    聚类纯度: {result.clustering_result.purity:.4f}")
                lines.append(f"    归一化互信息: {result.clustering_result.normalized_mutual_info:.4f}")
            
            if result.potential_histograms:
                lines.append("")
                lines.append(f"  Potential直方图数量: {len(result.potential_histograms)}")
        
        return lines
    
    def _generate_conclusions_section(
        self,
        batch_result: TurnBatchExperimentResult
    ) -> List[str]:
        """生成结论和建议部分。"""
        lines = [
            "",
            "-" * 70,
            "4. 结论和建议",
            "-" * 70,
            "",
        ]
        
        successful_results = [r for r in batch_result.results if r.success]
        
        if not successful_results:
            lines.append("由于没有成功的实验结果，无法生成结论。")
            return lines
        
        # 计算平均指标
        avg_corr = np.mean([r.metrics.strategy_correlation for r in successful_results if r.metrics])
        avg_purity = np.mean([r.metrics.clustering_purity for r in successful_results if r.metrics])
        avg_agreement = np.mean([r.metrics.action_agreement_rate for r in successful_results if r.metrics])
        
        # 生成结论
        lines.append("主要发现:")
        lines.append("")
        
        # 策略相关性评估
        if avg_corr > 0.5:
            lines.append(f"  1. Potential直方图特征与Solver策略呈现较强正相关 ({avg_corr:.4f})")
            lines.append("     这表明Potential直方图能够有效捕获手牌的策略价值。")
        elif avg_corr > 0:
            lines.append(f"  1. Potential直方图特征与Solver策略呈现弱正相关 ({avg_corr:.4f})")
            lines.append("     Potential直方图对策略有一定预测能力，但可能需要更多特征。")
        else:
            lines.append(f"  1. Potential直方图特征与Solver策略相关性较弱 ({avg_corr:.4f})")
            lines.append("     可能需要考虑其他因素来改进抽象方法。")
        
        lines.append("")
        
        # 聚类质量评估
        if avg_purity > 0.7:
            lines.append(f"  2. 基于Potential的聚类质量较高 (纯度: {avg_purity:.4f})")
            lines.append("     同一聚类内的手牌倾向于采取相似的策略。")
        elif avg_purity > 0.5:
            lines.append(f"  2. 基于Potential的聚类质量中等 (纯度: {avg_purity:.4f})")
            lines.append("     聚类有一定的策略一致性，但存在改进空间。")
        else:
            lines.append(f"  2. 基于Potential的聚类质量较低 (纯度: {avg_purity:.4f})")
            lines.append("     聚类内策略差异较大，可能需要调整聚类方法或特征。")
        
        lines.append("")
        
        # 动作预测评估
        if avg_agreement > 0.6:
            lines.append(f"  3. 基于Potential的动作预测准确率较高 ({avg_agreement:.4f})")
            lines.append("     Potential直方图可以较好地预测激进/被动动作。")
        else:
            lines.append(f"  3. 基于Potential的动作预测准确率一般 ({avg_agreement:.4f})")
            lines.append("     简单的均值阈值方法可能不足以准确预测动作。")
        
        # 建议
        lines.extend([
            "",
            "建议:",
            "",
            "  1. 考虑使用更多直方图特征（如偏度、峰度）来提高预测能力",
            "  2. 尝试不同的聚类算法（如层次聚类、DBSCAN）",
            "  3. 在更多样化的场景下验证方法的鲁棒性",
            "  4. 考虑结合其他信息（如位置、筹码深度）来改进抽象",
        ])
        
        return lines

    def generate_markdown_report(
        self,
        batch_result: TurnBatchExperimentResult,
        title: str = '转牌阶段Potential直方图验证实验报告'
    ) -> str:
        """生成Markdown格式报告。
        
        Args:
            batch_result: 批量实验结果
            title: 报告标题
            
        Returns:
            Markdown格式报告
        """
        lines = [
            f"# {title}",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # 1. 实验概述
        lines.extend(self._generate_markdown_overview(batch_result))
        
        # 2. 汇总统计
        lines.extend(self._generate_markdown_summary(batch_result))
        
        # 3. 各场景详情
        lines.extend(self._generate_markdown_details(batch_result))
        
        # 4. 结论
        lines.extend(self._generate_markdown_conclusions(batch_result))
        
        return "\n".join(lines)
    
    def _generate_markdown_overview(
        self,
        batch_result: TurnBatchExperimentResult
    ) -> List[str]:
        """生成Markdown格式的实验概述。"""
        success_rate = batch_result.success_count / len(batch_result.results) * 100
        
        lines = [
            "## 1. 实验概述",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 实验数量 | {len(batch_result.results)} |",
            f"| 成功数量 | {batch_result.success_count} |",
            f"| 失败数量 | {batch_result.failure_count} |",
            f"| 成功率 | {success_rate:.1f}% |",
            f"| 总耗时 | {batch_result.total_execution_time:.2f}秒 |",
            "",
        ]
        
        return lines
    
    def _generate_markdown_summary(
        self,
        batch_result: TurnBatchExperimentResult
    ) -> List[str]:
        """生成Markdown格式的汇总统计。"""
        lines = [
            "## 2. 汇总统计",
            "",
        ]
        
        successful_results = [r for r in batch_result.results if r.success]
        
        if not successful_results:
            lines.append("*无成功的实验结果*")
            return lines
        
        # 收集指标
        metrics_data = {}
        for result in successful_results:
            if result.metrics:
                for name, value in [
                    ('策略相关性', result.metrics.strategy_correlation),
                    ('聚类纯度', result.metrics.clustering_purity),
                    ('动作一致率', result.metrics.action_agreement_rate),
                    ('平均直方图熵', result.metrics.avg_histogram_entropy),
                ]:
                    if name not in metrics_data:
                        metrics_data[name] = []
                    metrics_data[name].append(value)
        
        # 生成表格
        lines.extend([
            "| 指标 | 平均值 | 标准差 | 最小值 | 最大值 |",
            "|------|--------|--------|--------|--------|",
        ])
        
        for name, values in metrics_data.items():
            if values:
                lines.append(
                    f"| {name} | {np.mean(values):.4f} | {np.std(values):.4f} | "
                    f"{np.min(values):.4f} | {np.max(values):.4f} |"
                )
        
        lines.append("")
        
        return lines
    
    def _generate_markdown_details(
        self,
        batch_result: TurnBatchExperimentResult
    ) -> List[str]:
        """生成Markdown格式的场景详情。"""
        lines = [
            "## 3. 各场景详情",
            "",
        ]
        
        for result in batch_result.results:
            status = "✅" if result.success else "❌"
            lines.append(f"### {status} {result.scenario.name}")
            lines.append("")
            lines.append(f"- **描述**: {result.scenario.description}")
            lines.append(f"- **公共牌**: `{result.scenario.get_turn_community_str()}`")
            lines.append(f"- **标签**: {', '.join(result.scenario.tags)}")
            lines.append(f"- **执行时间**: {result.execution_time:.2f}秒")
            
            if result.error:
                lines.append(f"- **错误**: {result.error}")
            elif result.metrics:
                lines.append("")
                lines.append("| 指标 | 值 |")
                lines.append("|------|-----|")
                lines.append(f"| 策略相关性 | {result.metrics.strategy_correlation:.4f} |")
                lines.append(f"| 聚类纯度 | {result.metrics.clustering_purity:.4f} |")
                lines.append(f"| 动作一致率 | {result.metrics.action_agreement_rate:.4f} |")
            
            lines.append("")
        
        return lines
    
    def _generate_markdown_conclusions(
        self,
        batch_result: TurnBatchExperimentResult
    ) -> List[str]:
        """生成Markdown格式的结论。"""
        lines = [
            "## 4. 结论",
            "",
        ]
        
        successful_results = [r for r in batch_result.results if r.success]
        
        if not successful_results:
            lines.append("*由于没有成功的实验结果，无法生成结论。*")
            return lines
        
        avg_corr = np.mean([r.metrics.strategy_correlation for r in successful_results if r.metrics])
        avg_purity = np.mean([r.metrics.clustering_purity for r in successful_results if r.metrics])
        
        lines.extend([
            f"- 平均策略相关性: **{avg_corr:.4f}**",
            f"- 平均聚类纯度: **{avg_purity:.4f}**",
            "",
            "### 主要发现",
            "",
        ])
        
        if avg_corr > 0.3:
            lines.append("1. Potential直方图特征与Solver策略存在正相关关系")
        else:
            lines.append("1. Potential直方图特征与Solver策略相关性较弱")
        
        if avg_purity > 0.5:
            lines.append("2. 基于Potential的聚类能够较好地分组相似策略的手牌")
        else:
            lines.append("2. 基于Potential的聚类质量有待提高")
        
        lines.append("")
        
        return lines
    
    def save_report(
        self,
        batch_result: TurnBatchExperimentResult,
        filename: Optional[str] = None,
        format: str = 'text'
    ) -> str:
        """保存报告到文件。
        
        Args:
            batch_result: 批量实验结果
            filename: 文件名（不含扩展名）
            format: 报告格式（'text' 或 'markdown'）
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_report_{timestamp}"
        
        if format == 'markdown':
            content = self.generate_markdown_report(batch_result)
            filepath = os.path.join(self.output_dir, f"{filename}.md")
        else:
            content = self.generate_text_report(batch_result)
            filepath = os.path.join(self.output_dir, f"{filename}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def generate_comparison_report(
        self,
        results_list: List[TurnBatchExperimentResult],
        labels: List[str],
        title: str = '实验对比报告'
    ) -> str:
        """生成多个实验结果的对比报告。
        
        Args:
            results_list: 多个批量实验结果
            labels: 每个结果的标签
            title: 报告标题
            
        Returns:
            对比报告文本
        """
        lines = [
            "=" * 70,
            title,
            "=" * 70,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 70,
            "实验对比",
            "-" * 70,
            "",
        ]
        
        # 表头
        header = f"{'指标':<25}"
        for label in labels:
            header += f" {label:>15}"
        lines.append(header)
        lines.append("-" * (25 + 16 * len(labels)))
        
        # 收集各实验的汇总指标
        metrics_names = [
            'avg_strategy_correlation',
            'avg_clustering_purity',
            'avg_action_agreement_rate',
            'avg_histogram_entropy',
        ]
        
        display_names = {
            'avg_strategy_correlation': '平均策略相关性',
            'avg_clustering_purity': '平均聚类纯度',
            'avg_action_agreement_rate': '平均动作一致率',
            'avg_histogram_entropy': '平均直方图熵',
        }
        
        for metric_name in metrics_names:
            row = f"{display_names.get(metric_name, metric_name):<25}"
            for result in results_list:
                value = result.summary_metrics.get(metric_name, 0.0)
                row += f" {value:>15.4f}"
            lines.append(row)
        
        # 成功率
        row = f"{'成功率':<25}"
        for result in results_list:
            rate = result.success_count / len(result.results) * 100
            row += f" {rate:>14.1f}%"
        lines.append(row)
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
