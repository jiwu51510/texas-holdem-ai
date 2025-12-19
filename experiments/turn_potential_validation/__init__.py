"""转牌阶段Potential直方图验证实验模块。

本模块实现了转牌阶段的Potential-Aware抽象验证，包括：
- Potential直方图计算
- EMD距离计算
- 与Solver策略的对比分析
"""

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
from experiments.turn_potential_validation.river_enumerator import (
    RiverCardEnumerator,
)
from experiments.turn_potential_validation.turn_solver import (
    TurnCFRSolver,
)
from experiments.turn_potential_validation.potential_analyzer import (
    PotentialAnalyzer,
)
from experiments.turn_potential_validation.experiment_runner import (
    TurnExperimentRunner,
    create_default_scenarios,
)
from experiments.turn_potential_validation.visualizer import (
    TurnVisualizer,
)
from experiments.turn_potential_validation.histogram_validator import (
    HistogramValidator,
    ManualHistogramCalculator,
    ValidationResult,
    BatchValidationResult,
)
from experiments.turn_potential_validation.scenarios import (
    get_all_turn_scenarios,
    get_turn_scenarios_by_tag,
    get_dry_board_scenarios,
    get_wet_board_scenarios,
    get_draw_board_scenarios,
    get_paired_board_scenarios,
    get_special_scenarios,
    get_scenario_by_name,
    list_all_scenario_names,
    list_all_tags,
    get_standard_wide_range,
    get_standard_tight_range,
    get_polarized_range,
    get_condensed_range,
    get_flush_draw_range,
    get_straight_draw_range,
    get_made_hand_range,
    get_mixed_range,
    create_default_turn_solver_config,
    create_deep_stack_solver_config,
)
from experiments.turn_potential_validation.report_generator import (
    TurnReportGenerator,
)

__all__ = [
    # 数据模型
    'TurnScenario',
    'CorrelationResult',
    'ClusteringComparisonResult',
    'TurnValidationMetrics',
    'TurnExperimentResult',
    'TurnBatchExperimentResult',
    # 核心组件
    'PotentialHistogramCalculator',
    'RiverCardEnumerator',
    'TurnCFRSolver',
    'PotentialAnalyzer',
    'TurnExperimentRunner',
    'create_default_scenarios',
    'TurnVisualizer',
    # 验证器
    'HistogramValidator',
    'ManualHistogramCalculator',
    'ValidationResult',
    'BatchValidationResult',
    # 场景配置
    'get_all_turn_scenarios',
    'get_turn_scenarios_by_tag',
    'get_dry_board_scenarios',
    'get_wet_board_scenarios',
    'get_draw_board_scenarios',
    'get_paired_board_scenarios',
    'get_special_scenarios',
    'get_scenario_by_name',
    'list_all_scenario_names',
    'list_all_tags',
    # 范围配置
    'get_standard_wide_range',
    'get_standard_tight_range',
    'get_polarized_range',
    'get_condensed_range',
    'get_flush_draw_range',
    'get_straight_draw_range',
    'get_made_hand_range',
    'get_mixed_range',
    # Solver配置
    'create_default_turn_solver_config',
    'create_deep_stack_solver_config',
    # 报告生成
    'TurnReportGenerator',
]
