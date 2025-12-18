"""实验运行器模块。

本模块实现实验的批量运行和结果汇总功能。
"""

from typing import Dict, List, Optional
import time
import json
import os
import numpy as np

from models.core import Card
from experiments.equity_solver_validation.data_models import (
    SolverConfig,
    SolverResult,
    ValidationMetrics,
    ComparisonResult,
    ExperimentScenario,
    ExperimentResult,
    BatchExperimentResult,
)
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    RangeVsRangeCalculator,
)
from experiments.equity_solver_validation.solver_wrapper import SimpleCFRSolver
from experiments.equity_solver_validation.strategy_comparator import StrategyComparator


class ExperimentRunner:
    """实验运行器。
    
    运行胜率-Solver验证实验并收集结果。
    """
    
    def __init__(self, output_dir: str = 'experiments/results'):
        """初始化运行器。
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = output_dir
        self.calculator = RangeVsRangeCalculator()
        os.makedirs(output_dir, exist_ok=True)
    
    def run_single_experiment(
        self,
        scenario: ExperimentScenario,
        verbose: bool = False
    ) -> ExperimentResult:
        """运行单个实验场景。
        
        Args:
            scenario: 实验场景
            verbose: 是否输出详细信息
            
        Returns:
            ExperimentResult实例
        """
        start_time = time.time()
        
        try:
            if verbose:
                print(f"运行场景: {scenario.name}")
                print(f"  公共牌: {scenario.get_community_cards_str()}")
            
            # 1. 计算胜率向量
            equity_dict = self.calculator.calculate_range_vs_range_equity(
                scenario.oop_range,
                scenario.ip_range,
                scenario.community_cards
            )
            
            if verbose:
                print(f"  计算了 {len(equity_dict)} 个手牌的胜率")
            
            # 2. 运行Solver
            solver = SimpleCFRSolver(scenario.solver_config)
            solver_result = solver.solve(
                scenario.community_cards,
                scenario.oop_range,
                scenario.ip_range
            )
            
            if verbose:
                print(f"  Solver迭代: {solver_result.iterations}")
            
            # 3. 将胜率转换为策略
            comparator = StrategyComparator(
                pot_size=scenario.solver_config.pot_size,
                bet_size=scenario.solver_config.pot_size * 0.5
            )
            
            equity_strategy = comparator.equity_to_strategy(equity_dict, 'oop_root')
            
            # 4. 获取Solver策略
            solver_strategy = solver_result.strategies.get('root', {})
            
            # 5. 对比策略
            comparison = comparator.compare_strategies(equity_strategy, solver_strategy)
            comparison.equity_vector = equity_dict
            
            execution_time = time.time() - start_time
            
            if verbose:
                print(f"  总变差距离: {comparison.metrics.total_variation_distance:.4f}")
                print(f"  动作一致率: {comparison.metrics.action_agreement_rate:.4f}")
                print(f"  执行时间: {execution_time:.2f}s")
            
            return ExperimentResult(
                scenario=scenario,
                comparison=comparison,
                solver_result=solver_result,
                execution_time=execution_time,
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExperimentResult(
                scenario=scenario,
                error=str(e),
                execution_time=execution_time,
            )
    
    def run_batch_experiments(
        self,
        scenarios: List[ExperimentScenario],
        verbose: bool = True
    ) -> BatchExperimentResult:
        """运行批量实验。
        
        Args:
            scenarios: 实验场景列表
            verbose: 是否输出详细信息
            
        Returns:
            BatchExperimentResult实例
        """
        start_time = time.time()
        results = []
        
        if verbose:
            print(f"开始运行 {len(scenarios)} 个实验场景...")
            print("=" * 60)
        
        for i, scenario in enumerate(scenarios):
            if verbose:
                print(f"\n[{i+1}/{len(scenarios)}]")
            
            result = self.run_single_experiment(scenario, verbose)
            results.append(result)
        
        total_time = time.time() - start_time
        
        batch_result = BatchExperimentResult(
            results=results,
            total_execution_time=total_time,
        )
        batch_result.compute_summary()
        
        if verbose:
            print("\n" + "=" * 60)
            print("实验完成!")
            print(f"  成功: {batch_result.success_count}")
            print(f"  失败: {batch_result.failure_count}")
            print(f"  总时间: {total_time:.2f}s")
            
            if batch_result.summary_metrics:
                print("\n汇总指标:")
                for key, value in batch_result.summary_metrics.items():
                    print(f"  {key}: {value:.4f}")
        
        return batch_result
    
    def save_results(
        self,
        batch_result: BatchExperimentResult,
        filename: str = 'experiment_results.json'
    ) -> str:
        """保存实验结果。
        
        Args:
            batch_result: 批量实验结果
            filename: 输出文件名
            
        Returns:
            保存的文件路径
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_result.to_dict(), f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_results(self, filename: str) -> BatchExperimentResult:
        """加载实验结果。
        
        Args:
            filename: 结果文件名
            
        Returns:
            BatchExperimentResult实例
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return BatchExperimentResult.from_dict(data)
    
    def generate_report(
        self,
        batch_result: BatchExperimentResult,
        filename: str = 'experiment_report.txt'
    ) -> str:
        """生成实验报告。
        
        Args:
            batch_result: 批量实验结果
            filename: 报告文件名
            
        Returns:
            报告文件路径
        """
        filepath = os.path.join(self.output_dir, filename)
        
        lines = [
            "=" * 60,
            "胜率-Solver验证实验报告",
            "=" * 60,
            "",
            f"实验数量: {len(batch_result.results)}",
            f"成功: {batch_result.success_count}",
            f"失败: {batch_result.failure_count}",
            f"总执行时间: {batch_result.total_execution_time:.2f}s",
            "",
            "-" * 60,
            "汇总指标",
            "-" * 60,
        ]
        
        for key, value in batch_result.summary_metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        
        lines.extend([
            "",
            "-" * 60,
            "各场景详情",
            "-" * 60,
        ])
        
        for result in batch_result.results:
            lines.append(f"\n场景: {result.scenario.name}")
            lines.append(f"  标签: {', '.join(result.scenario.tags)}")
            
            if result.success:
                lines.append(f"  状态: 成功")
                lines.append(f"  总变差距离: {result.comparison.metrics.total_variation_distance:.4f}")
                lines.append(f"  动作一致率: {result.comparison.metrics.action_agreement_rate:.4f}")
            else:
                lines.append(f"  状态: 失败")
                lines.append(f"  错误: {result.error}")
            
            lines.append(f"  执行时间: {result.execution_time:.2f}s")
        
        lines.extend([
            "",
            "=" * 60,
            "报告结束",
            "=" * 60,
        ])
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return filepath
