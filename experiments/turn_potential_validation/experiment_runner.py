"""转牌实验运行器模块。

本模块实现了转牌阶段Potential直方图验证实验的运行器。
支持单个场景和批量场景的实验运行。
"""

import time
import json
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

from models.core import Card
from experiments.equity_solver_validation.data_models import SolverConfig
from experiments.turn_potential_validation.data_models import (
    TurnScenario,
    CorrelationResult,
    ClusteringComparisonResult,
    TurnValidationMetrics,
    TurnExperimentResult,
    TurnBatchExperimentResult,
)
from experiments.turn_potential_validation.potential_histogram import (
    PotentialHistogramCalculator,
)
from experiments.turn_potential_validation.turn_solver import TurnCFRSolver
from experiments.turn_potential_validation.potential_analyzer import PotentialAnalyzer


class TurnExperimentRunner:
    """转牌实验运行器类。
    
    运行转牌阶段的Potential直方图验证实验，包括：
    - 计算Potential直方图
    - 运行Solver获取策略
    - 分析直方图与策略的相关性
    - 生成实验报告
    """
    
    def __init__(
        self,
        num_histogram_bins: int = 50,
        default_solver_iterations: int = 500,
        default_num_clusters: int = 5
    ):
        """初始化实验运行器。
        
        Args:
            num_histogram_bins: Potential直方图的区间数量
            default_solver_iterations: 默认的Solver迭代次数
            default_num_clusters: 默认的聚类数量
        """
        self.num_histogram_bins = num_histogram_bins
        self.default_solver_iterations = default_solver_iterations
        self.default_num_clusters = default_num_clusters
        
        self.histogram_calculator = PotentialHistogramCalculator(
            num_bins=num_histogram_bins
        )
        self.analyzer = PotentialAnalyzer(num_bins=num_histogram_bins)
    
    def run_experiment(
        self,
        scenario: TurnScenario,
        solver_iterations: Optional[int] = None,
        num_clusters: Optional[int] = None
    ) -> TurnExperimentResult:
        """运行单个实验场景。
        
        Args:
            scenario: 实验场景配置
            solver_iterations: Solver迭代次数（可选）
            num_clusters: 聚类数量（可选）
            
        Returns:
            TurnExperimentResult实例
        """
        start_time = time.time()
        
        if solver_iterations is None:
            solver_iterations = self.default_solver_iterations
        if num_clusters is None:
            num_clusters = self.default_num_clusters
        
        try:
            # 1. 计算Potential直方图
            potential_histograms = self._compute_potential_histograms(scenario)
            
            if not potential_histograms:
                return TurnExperimentResult(
                    scenario=scenario,
                    error="无法计算Potential直方图：范围为空或所有手牌与公共牌冲突",
                    execution_time=time.time() - start_time,
                )
            
            # 2. 运行Solver获取策略
            solver_result = self._run_solver(scenario, solver_iterations)
            
            if not solver_result.strategies:
                return TurnExperimentResult(
                    scenario=scenario,
                    potential_histograms=potential_histograms,
                    solver_result=solver_result,
                    error="Solver未返回有效策略",
                    execution_time=time.time() - start_time,
                )
            
            # 3. 提取OOP根节点策略
            root_strategy = solver_result.strategies.get('root', {})
            
            # 4. 分析相关性
            correlation_result = self.analyzer.analyze_histogram_strategy_correlation(
                potential_histograms, root_strategy
            )
            
            # 5. 聚类分析
            cluster_labels = self.analyzer.cluster_by_potential(
                potential_histograms, num_clusters=num_clusters
            )
            
            clustering_result = self.analyzer.compare_clustering_with_strategy(
                cluster_labels, root_strategy
            )
            
            # 6. 计算验证指标
            metrics = self._compute_metrics(
                potential_histograms, root_strategy, correlation_result, clustering_result
            )
            
            return TurnExperimentResult(
                scenario=scenario,
                potential_histograms=potential_histograms,
                correlation_result=correlation_result,
                clustering_result=clustering_result,
                metrics=metrics,
                solver_result=solver_result,
                execution_time=time.time() - start_time,
            )
            
        except Exception as e:
            return TurnExperimentResult(
                scenario=scenario,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    
    def _compute_potential_histograms(
        self,
        scenario: TurnScenario
    ) -> Dict[str, np.ndarray]:
        """计算场景中OOP范围的Potential直方图。
        
        Args:
            scenario: 实验场景
            
        Returns:
            每个手牌的Potential直方图
        """
        return self.histogram_calculator.calculate_range_potential_histograms(
            my_range=scenario.oop_range,
            opponent_range=scenario.ip_range,
            turn_community=scenario.turn_community,
        )
    
    def _run_solver(
        self,
        scenario: TurnScenario,
        iterations: int
    ):
        """运行Solver获取策略。
        
        Args:
            scenario: 实验场景
            iterations: 迭代次数
            
        Returns:
            SolverResult实例
        """
        solver = TurnCFRSolver(scenario.solver_config)
        return solver.solve(
            turn_community=scenario.turn_community,
            oop_range=scenario.oop_range,
            ip_range=scenario.ip_range,
            iterations=iterations,
        )
    
    def _compute_metrics(
        self,
        potential_histograms: Dict[str, np.ndarray],
        solver_strategy: Dict[str, Dict[str, float]],
        correlation_result: CorrelationResult,
        clustering_result: ClusteringComparisonResult
    ) -> TurnValidationMetrics:
        """计算验证指标。
        
        Args:
            potential_histograms: Potential直方图
            solver_strategy: Solver策略
            correlation_result: 相关性结果
            clustering_result: 聚类结果
            
        Returns:
            TurnValidationMetrics实例
        """
        # 计算直方图特征
        entropies = []
        sparsities = []
        
        for histogram in potential_histograms.values():
            features = self.histogram_calculator.get_histogram_features(histogram)
            entropies.append(features['entropy'])
            sparsities.append(features['sparsity'])
        
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
        avg_sparsity = float(np.mean(sparsities)) if sparsities else 0.0
        
        # 计算动作一致率
        action_agreement = self._compute_action_agreement(
            potential_histograms, solver_strategy
        )
        
        # 计算平均EMD距离
        avg_intra_emd = 0.0
        avg_inter_emd = 0.0
        
        if correlation_result.intra_action_emd:
            avg_intra_emd = float(np.mean(list(correlation_result.intra_action_emd.values())))
        
        if correlation_result.inter_action_emd:
            avg_inter_emd = float(np.mean(list(correlation_result.inter_action_emd.values())))
        
        return TurnValidationMetrics(
            avg_histogram_entropy=avg_entropy,
            histogram_sparsity=avg_sparsity,
            strategy_correlation=correlation_result.mean_equity_correlation,
            ev_correlation=0.0,  # 需要EV数据
            clustering_purity=clustering_result.purity,
            silhouette_score=clustering_result.silhouette_score,
            avg_intra_action_emd=avg_intra_emd,
            avg_inter_action_emd=avg_inter_emd,
            action_agreement_rate=action_agreement,
        )
    
    def _compute_action_agreement(
        self,
        potential_histograms: Dict[str, np.ndarray],
        solver_strategy: Dict[str, Dict[str, float]]
    ) -> float:
        """计算基于Potential的动作预测与Solver策略的一致率。
        
        使用简单的启发式：高均值Equity预测bet，低均值Equity预测check。
        
        Args:
            potential_histograms: Potential直方图
            solver_strategy: Solver策略
            
        Returns:
            动作一致率 [0, 1]
        """
        common_hands = set(potential_histograms.keys()) & set(solver_strategy.keys())
        
        if not common_hands:
            return 0.0
        
        # 计算每个手牌的均值Equity
        mean_equities = {}
        for hand in common_hands:
            features = self.histogram_calculator.get_histogram_features(
                potential_histograms[hand]
            )
            mean_equities[hand] = features['mean_equity']
        
        # 计算中位数作为阈值
        median_equity = np.median(list(mean_equities.values()))
        
        # 预测动作并比较
        correct = 0
        for hand in common_hands:
            # 预测：高于中位数预测bet，否则预测check
            predicted_action = 'bet' if mean_equities[hand] > median_equity else 'check'
            
            # 获取Solver的主要动作
            strategy = solver_strategy[hand]
            actual_action = max(strategy.items(), key=lambda x: x[1])[0]
            
            # 简化比较：bet/raise视为激进，check/fold/call视为被动
            predicted_aggressive = 'bet' in predicted_action or 'raise' in predicted_action
            actual_aggressive = 'bet' in actual_action or 'raise' in actual_action
            
            if predicted_aggressive == actual_aggressive:
                correct += 1
        
        return correct / len(common_hands)
    
    def run_batch_experiments(
        self,
        scenarios: List[TurnScenario],
        solver_iterations: Optional[int] = None,
        num_clusters: Optional[int] = None,
        verbose: bool = True
    ) -> TurnBatchExperimentResult:
        """运行批量实验。
        
        Args:
            scenarios: 实验场景列表
            solver_iterations: Solver迭代次数
            num_clusters: 聚类数量
            verbose: 是否打印进度
            
        Returns:
            TurnBatchExperimentResult实例
        """
        start_time = time.time()
        results = []
        
        for i, scenario in enumerate(scenarios):
            if verbose:
                print(f"运行实验 {i+1}/{len(scenarios)}: {scenario.name}")
            
            result = self.run_experiment(
                scenario,
                solver_iterations=solver_iterations,
                num_clusters=num_clusters,
            )
            results.append(result)
            
            if verbose:
                status = "成功" if result.success else f"失败: {result.error}"
                print(f"  状态: {status}, 耗时: {result.execution_time:.2f}s")
        
        batch_result = TurnBatchExperimentResult(
            results=results,
            total_execution_time=time.time() - start_time,
        )
        
        # 计算汇总指标
        batch_result.compute_summary()
        
        return batch_result
    
    def load_scenarios_from_file(self, filepath: str) -> List[TurnScenario]:
        """从JSON文件加载场景列表。
        
        Args:
            filepath: JSON文件路径
            
        Returns:
            场景列表
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenarios = []
        for scenario_data in data.get('scenarios', []):
            scenarios.append(TurnScenario.from_dict(scenario_data))
        
        return scenarios
    
    def save_results_to_file(
        self,
        result: TurnBatchExperimentResult,
        filepath: str
    ) -> None:
        """保存批量实验结果到文件。
        
        Args:
            result: 批量实验结果
            filepath: 输出文件路径
        """
        result.save_to_file(filepath)
    
    def generate_summary_report(
        self,
        result: TurnBatchExperimentResult
    ) -> str:
        """生成实验汇总报告。
        
        Args:
            result: 批量实验结果
            
        Returns:
            报告文本
        """
        lines = [
            "=" * 60,
            "转牌阶段Potential直方图验证实验报告",
            "=" * 60,
            "",
            f"实验数量: {len(result.results)}",
            f"成功数量: {result.success_count}",
            f"失败数量: {result.failure_count}",
            f"总耗时: {result.total_execution_time:.2f}秒",
            "",
            "-" * 60,
            "汇总指标",
            "-" * 60,
        ]
        
        for metric, value in result.summary_metrics.items():
            lines.append(f"  {metric}: {value:.4f}")
        
        lines.extend([
            "",
            "-" * 60,
            "各场景详情",
            "-" * 60,
        ])
        
        for exp_result in result.results:
            status = "✓" if exp_result.success else "✗"
            lines.append(f"\n{status} {exp_result.scenario.name}")
            lines.append(f"  公共牌: {exp_result.scenario.get_turn_community_str()}")
            
            if exp_result.success and exp_result.metrics:
                lines.append(f"  策略相关性: {exp_result.metrics.strategy_correlation:.4f}")
                lines.append(f"  聚类纯度: {exp_result.metrics.clustering_purity:.4f}")
                lines.append(f"  动作一致率: {exp_result.metrics.action_agreement_rate:.4f}")
            elif exp_result.error:
                lines.append(f"  错误: {exp_result.error}")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def create_default_scenarios() -> List[TurnScenario]:
    """创建默认的实验场景列表。
    
    Returns:
        预定义的实验场景列表
    """
    default_config = SolverConfig(
        pot_size=100.0,
        effective_stack=200.0,
        oop_bet_sizes=[0.5, 1.0],
        ip_bet_sizes=[0.5, 1.0],
    )
    
    # 标准范围
    standard_oop_range = {
        "AA": 1.0, "KK": 1.0, "QQ": 1.0, "JJ": 1.0, "TT": 1.0,
        "AKs": 1.0, "AQs": 1.0, "AJs": 1.0,
        "AKo": 1.0, "AQo": 1.0,
    }
    
    standard_ip_range = {
        "AA": 1.0, "KK": 1.0, "QQ": 1.0, "JJ": 1.0, "TT": 1.0, "99": 1.0,
        "AKs": 1.0, "AQs": 1.0, "AJs": 1.0, "ATs": 1.0,
        "KQs": 1.0, "KJs": 1.0,
        "AKo": 1.0, "AQo": 1.0, "AJo": 1.0,
    }
    
    scenarios = [
        # 干燥牌面
        TurnScenario(
            name="dry_board_1",
            description="干燥牌面：K高无连接",
            turn_community=[
                Card(rank=13, suit='s'),  # Ks
                Card(rank=7, suit='d'),   # 7d
                Card(rank=2, suit='c'),   # 2c
                Card(rank=4, suit='h'),   # 4h
            ],
            oop_range=standard_oop_range,
            ip_range=standard_ip_range,
            solver_config=default_config,
            tags=["dry_board"],
        ),
        
        # 湿润牌面
        TurnScenario(
            name="wet_board_1",
            description="湿润牌面：连接性高",
            turn_community=[
                Card(rank=11, suit='s'),  # Js
                Card(rank=10, suit='d'),  # Td
                Card(rank=9, suit='c'),   # 9c
                Card(rank=8, suit='h'),   # 8h
            ],
            oop_range=standard_oop_range,
            ip_range=standard_ip_range,
            solver_config=default_config,
            tags=["wet_board"],
        ),
        
        # 同花听牌牌面
        TurnScenario(
            name="flush_draw_board",
            description="同花听牌牌面：三张同花",
            turn_community=[
                Card(rank=14, suit='s'),  # As
                Card(rank=10, suit='s'),  # Ts
                Card(rank=7, suit='s'),   # 7s
                Card(rank=3, suit='d'),   # 3d
            ],
            oop_range=standard_oop_range,
            ip_range=standard_ip_range,
            solver_config=default_config,
            tags=["draw_board", "flush_draw"],
        ),
        
        # 配对牌面
        TurnScenario(
            name="paired_board",
            description="配对牌面：公共牌有对子",
            turn_community=[
                Card(rank=13, suit='s'),  # Ks
                Card(rank=13, suit='d'),  # Kd
                Card(rank=7, suit='c'),   # 7c
                Card(rank=3, suit='h'),   # 3h
            ],
            oop_range=standard_oop_range,
            ip_range=standard_ip_range,
            solver_config=default_config,
            tags=["paired_board"],
        ),
    ]
    
    return scenarios
