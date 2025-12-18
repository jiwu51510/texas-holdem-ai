#!/usr/bin/env python3
"""胜率-Solver验证实验运行脚本。

本脚本运行胜率方法与Solver方法的对比验证实验。

用法:
    python run_equity_solver_validation.py [--scenarios SCENARIO_NAMES] [--verbose]
    
示例:
    python run_equity_solver_validation.py  # 运行所有场景
    python run_equity_solver_validation.py --scenarios dry_board wet_board
    python run_equity_solver_validation.py --verbose
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.equity_solver_validation.scenarios import (
    get_all_scenarios,
    create_dry_board_scenario,
    create_wet_board_scenario,
    create_paired_board_scenario,
    create_flush_board_scenario,
    create_polarized_vs_condensed_scenario,
)
from experiments.equity_solver_validation.experiment_runner import ExperimentRunner
from experiments.equity_solver_validation.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(
        description='运行胜率-Solver验证实验'
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        choices=['all', 'dry', 'wet', 'paired', 'flush', 'polarized'],
        default=['all'],
        help='要运行的场景'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='输出详细信息'
    )
    parser.add_argument(
        '--output-dir',
        default='experiments/results',
        help='结果输出目录'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='不生成可视化图表'
    )
    
    args = parser.parse_args()
    
    # 选择场景
    scenario_map = {
        'dry': create_dry_board_scenario,
        'wet': create_wet_board_scenario,
        'paired': create_paired_board_scenario,
        'flush': create_flush_board_scenario,
        'polarized': create_polarized_vs_condensed_scenario,
    }
    
    if 'all' in args.scenarios:
        scenarios = get_all_scenarios()
    else:
        scenarios = [scenario_map[s]() for s in args.scenarios if s in scenario_map]
    
    if not scenarios:
        print("错误: 没有选择任何场景")
        return 1
    
    print("=" * 60)
    print("胜率-Solver验证实验")
    print("=" * 60)
    print(f"\n选择的场景数量: {len(scenarios)}")
    for s in scenarios:
        print(f"  - {s.name}")
    print()
    
    # 运行实验
    runner = ExperimentRunner(output_dir=args.output_dir)
    batch_result = runner.run_batch_experiments(scenarios, verbose=args.verbose)
    
    # 保存结果
    results_path = runner.save_results(batch_result)
    print(f"\n结果已保存到: {results_path}")
    
    # 生成报告
    report_path = runner.generate_report(batch_result)
    print(f"报告已保存到: {report_path}")
    
    # 生成可视化
    if not args.no_visualize:
        visualizer = Visualizer(output_dir=args.output_dir)
        
        # 批量汇总图
        summary_path = visualizer.plot_batch_summary(batch_result)
        if summary_path:
            print(f"汇总图已保存到: {summary_path}")
        
        # TVD直方图
        hist_path = visualizer.plot_tvd_histogram(batch_result)
        if hist_path:
            print(f"直方图已保存到: {hist_path}")
        
        # 为每个成功的场景生成详细图表
        for result in batch_result.results:
            if result.success:
                safe_name = result.scenario.name.replace(' ', '_')
                
                # 胜率-策略散点图
                scatter_path = visualizer.plot_equity_vs_strategy(
                    result.comparison,
                    action='bet',
                    filename=f'{safe_name}_equity_vs_bet.png'
                )
                
                # 动作分布图
                dist_path = visualizer.plot_action_distribution(
                    result.comparison,
                    filename=f'{safe_name}_action_dist.png'
                )
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("实验汇总")
    print("=" * 60)
    print(f"成功: {batch_result.success_count}/{len(batch_result.results)}")
    
    if batch_result.summary_metrics:
        print("\n关键指标:")
        if 'avg_total_variation_distance' in batch_result.summary_metrics:
            print(f"  平均总变差距离: {batch_result.summary_metrics['avg_total_variation_distance']:.4f}")
        if 'avg_action_agreement_rate' in batch_result.summary_metrics:
            print(f"  平均动作一致率: {batch_result.summary_metrics['avg_action_agreement_rate']:.4f}")
    
    print("\n实验结论:")
    if batch_result.summary_metrics.get('avg_total_variation_distance', 1) < 0.3:
        print("  胜率方法与Solver方法的策略差异较小，")
        print("  胜率标量可以作为策略的有效近似。")
    else:
        print("  胜率方法与Solver方法存在显著差异，")
        print("  需要更复杂的方法来近似最优策略。")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
