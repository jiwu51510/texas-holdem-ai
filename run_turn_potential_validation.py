#!/usr/bin/env python3
"""转牌阶段Potential直方图验证实验主脚本。

本脚本用于运行转牌阶段的Potential-Aware抽象验证实验。
支持命令行参数配置实验场景、Solver参数和输出选项。

使用示例:
    # 运行所有预定义场景
    python run_turn_potential_validation.py
    
    # 运行特定类型的场景
    python run_turn_potential_validation.py --tag dry_board
    
    # 运行单个场景
    python run_turn_potential_validation.py --scenario "干燥牌面_K724"
    
    # 自定义Solver参数
    python run_turn_potential_validation.py --iterations 1000 --clusters 8
    
    # 生成可视化图表
    python run_turn_potential_validation.py --visualize --format svg

Requirements: 6.1, 6.2
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from experiments.turn_potential_validation import (
    TurnExperimentRunner,
    TurnVisualizer,
    TurnReportGenerator,
    get_all_turn_scenarios,
    get_turn_scenarios_by_tag,
    get_scenario_by_name,
    list_all_scenario_names,
    list_all_tags,
)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='转牌阶段Potential直方图验证实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                           # 运行所有场景
  %(prog)s --tag dry_board           # 运行干燥牌面场景
  %(prog)s --scenario "干燥牌面_K724" # 运行指定场景
  %(prog)s --list-scenarios          # 列出所有场景
  %(prog)s --list-tags               # 列出所有标签
  %(prog)s --visualize --format svg  # 生成SVG格式图表
        """
    )
    
    # 场景选择
    scenario_group = parser.add_argument_group('场景选择')
    scenario_group.add_argument(
        '--scenario', '-s',
        type=str,
        help='运行指定名称的场景'
    )
    scenario_group.add_argument(
        '--tag', '-t',
        type=str,
        help='运行指定标签的所有场景'
    )
    scenario_group.add_argument(
        '--all', '-a',
        action='store_true',
        default=True,
        help='运行所有预定义场景（默认）'
    )

    # 信息查询
    info_group = parser.add_argument_group('信息查询')
    info_group.add_argument(
        '--list-scenarios',
        action='store_true',
        help='列出所有可用场景名称'
    )
    info_group.add_argument(
        '--list-tags',
        action='store_true',
        help='列出所有可用标签'
    )
    
    # Solver参数
    solver_group = parser.add_argument_group('Solver参数')
    solver_group.add_argument(
        '--iterations', '-i',
        type=int,
        default=500,
        help='Solver迭代次数（默认: 500）'
    )
    solver_group.add_argument(
        '--clusters', '-c',
        type=int,
        default=5,
        help='聚类数量（默认: 5）'
    )
    solver_group.add_argument(
        '--bins', '-b',
        type=int,
        default=50,
        help='Potential直方图区间数量（默认: 50）'
    )
    
    # 输出选项
    output_group = parser.add_argument_group('输出选项')
    output_group.add_argument(
        '--output-dir', '-o',
        type=str,
        default='experiments/results/turn_validation',
        help='输出目录（默认: experiments/results/turn_validation）'
    )
    output_group.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='生成可视化图表'
    )
    output_group.add_argument(
        '--format', '-f',
        type=str,
        choices=['png', 'svg'],
        default='png',
        help='图表输出格式（默认: png）'
    )
    output_group.add_argument(
        '--save-results',
        action='store_true',
        default=True,
        help='保存实验结果到JSON文件（默认: True）'
    )
    output_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，减少输出'
    )
    
    return parser.parse_args()


def list_scenarios_info():
    """列出所有场景信息。"""
    print("\n可用场景列表:")
    print("=" * 60)
    
    scenarios = get_all_turn_scenarios()
    for scenario in scenarios:
        print(f"\n名称: {scenario.name}")
        print(f"  描述: {scenario.description}")
        print(f"  公共牌: {scenario.get_turn_community_str()}")
        print(f"  标签: {', '.join(scenario.tags)}")
    
    print(f"\n共 {len(scenarios)} 个场景")


def list_tags_info():
    """列出所有标签信息。"""
    print("\n可用标签列表:")
    print("=" * 60)
    
    tags = list_all_tags()
    for tag in tags:
        scenarios = get_turn_scenarios_by_tag(tag)
        print(f"\n{tag}: {len(scenarios)} 个场景")
        for s in scenarios:
            print(f"  - {s.name}")


def run_experiments(args):
    """运行实验。"""
    # 确定要运行的场景
    if args.scenario:
        try:
            scenarios = [get_scenario_by_name(args.scenario)]
            print(f"\n运行单个场景: {args.scenario}")
        except ValueError as e:
            print(f"错误: {e}")
            print("\n可用场景:")
            for name in list_all_scenario_names():
                print(f"  - {name}")
            return None
    elif args.tag:
        scenarios = get_turn_scenarios_by_tag(args.tag)
        if not scenarios:
            print(f"错误: 找不到标签为 '{args.tag}' 的场景")
            print("\n可用标签:")
            for tag in list_all_tags():
                print(f"  - {tag}")
            return None
        print(f"\n运行标签 '{args.tag}' 的 {len(scenarios)} 个场景")
    else:
        scenarios = get_all_turn_scenarios()
        print(f"\n运行所有 {len(scenarios)} 个预定义场景")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建实验运行器
    runner = TurnExperimentRunner(
        num_histogram_bins=args.bins,
        default_solver_iterations=args.iterations,
        default_num_clusters=args.clusters,
    )
    
    # 运行实验
    print("\n" + "=" * 60)
    print("开始运行实验")
    print("=" * 60)
    
    batch_result = runner.run_batch_experiments(
        scenarios=scenarios,
        solver_iterations=args.iterations,
        num_clusters=args.clusters,
        verbose=not args.quiet,
    )
    
    return batch_result


def generate_report(batch_result, args):
    """生成实验报告。"""
    report_generator = TurnReportGenerator(output_dir=args.output_dir)
    
    # 生成文本报告
    text_report = report_generator.generate_text_report(batch_result)
    print("\n" + text_report)
    
    # 保存文本报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_path = report_generator.save_report(
        batch_result,
        filename=f"report_{timestamp}",
        format='text'
    )
    print(f"\n文本报告已保存到: {text_path}")
    
    # 保存Markdown报告
    md_path = report_generator.save_report(
        batch_result,
        filename=f"report_{timestamp}",
        format='markdown'
    )
    print(f"Markdown报告已保存到: {md_path}")
    
    return text_path


def save_results(batch_result, args):
    """保存实验结果。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"results_{timestamp}.json")
    
    batch_result.save_to_file(results_path)
    print(f"结果已保存到: {results_path}")
    
    return results_path


def generate_visualizations(batch_result, args):
    """生成可视化图表。"""
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    
    visualizer = TurnVisualizer(
        output_dir=args.output_dir,
        num_bins=args.bins,
    )
    
    generated_files = []
    
    # 为每个成功的实验生成图表
    for result in batch_result.results:
        if result.success:
            print(f"\n生成 {result.scenario.name} 的图表...")
            paths = visualizer.plot_experiment_result(
                result,
                format=args.format,
            )
            for name, path in paths.items():
                if path:
                    generated_files.append(path)
                    if not args.quiet:
                        print(f"  - {name}: {path}")
    
    # 生成批量汇总图表
    if len(batch_result.results) > 1:
        print("\n生成批量汇总图表...")
        summary_path = visualizer.plot_batch_summary(
            batch_result,
            filename='batch_summary',
            format=args.format,
        )
        if summary_path:
            generated_files.append(summary_path)
            print(f"  - 批量汇总: {summary_path}")
    
    print(f"\n共生成 {len(generated_files)} 个图表文件")
    
    return generated_files


def print_summary_statistics(batch_result):
    """打印汇总统计信息。"""
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)
    
    print(f"\n实验总数: {len(batch_result.results)}")
    print(f"成功数量: {batch_result.success_count}")
    print(f"失败数量: {batch_result.failure_count}")
    print(f"总耗时: {batch_result.total_execution_time:.2f} 秒")
    
    if batch_result.summary_metrics:
        print("\n汇总指标:")
        for metric, value in batch_result.summary_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # 打印各场景简要结果
    print("\n各场景结果:")
    for result in batch_result.results:
        status = "✓" if result.success else "✗"
        if result.success and result.metrics:
            print(f"  {status} {result.scenario.name}")
            print(f"      策略相关性: {result.metrics.strategy_correlation:.4f}")
            print(f"      聚类纯度: {result.metrics.clustering_purity:.4f}")
            print(f"      动作一致率: {result.metrics.action_agreement_rate:.4f}")
        else:
            print(f"  {status} {result.scenario.name}: {result.error}")


def main():
    """主函数。"""
    args = parse_args()
    
    # 处理信息查询命令
    if args.list_scenarios:
        list_scenarios_info()
        return 0
    
    if args.list_tags:
        list_tags_info()
        return 0
    
    # 运行实验
    batch_result = run_experiments(args)
    
    if batch_result is None:
        return 1
    
    # 打印汇总统计
    print_summary_statistics(batch_result)
    
    # 生成报告
    generate_report(batch_result, args)
    
    # 保存结果
    if args.save_results:
        save_results(batch_result, args)
    
    # 生成可视化
    if args.visualize:
        generate_visualizations(batch_result, args)
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
