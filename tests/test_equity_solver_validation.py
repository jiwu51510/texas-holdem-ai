"""胜率-Solver验证实验的属性测试。

本模块包含数据模型的属性测试，验证序列化往返、数据验证等功能。
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
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


# ============================================================================
# 策略生成器
# ============================================================================

@st.composite
def valid_solver_config(draw):
    """生成有效的SolverConfig。"""
    pot_size = draw(st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    effective_stack = draw(st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    
    # 生成下注尺寸列表
    bet_sizes = draw(st.lists(
        st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
        min_size=1, max_size=4
    ))
    
    return SolverConfig(
        pot_size=pot_size,
        effective_stack=effective_stack,
        oop_bet_sizes=bet_sizes,
        ip_bet_sizes=bet_sizes,
        oop_raise_sizes=bet_sizes,
        ip_raise_sizes=bet_sizes,
        all_in_threshold=draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)),
        max_iterations=draw(st.integers(min_value=1, max_value=10000)),
        target_exploitability=draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def valid_solver_result(draw):
    """生成有效的SolverResult。"""
    return SolverResult(
        exploitability=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        iterations=draw(st.integers(min_value=0, max_value=10000)),
        root_ev=(
            draw(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
            draw(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        ),
        converged=draw(st.booleans()),
    )


@st.composite
def valid_validation_metrics(draw):
    """生成有效的ValidationMetrics。"""
    return ValidationMetrics(
        total_variation_distance=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        kl_divergence=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        action_agreement_rate=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        ev_correlation=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        ev_rmse=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        equity_strategy_correlation=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def valid_card(draw):
    """生成有效的Card。"""
    rank = draw(st.integers(min_value=2, max_value=14))
    suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
    return Card(rank=rank, suit=suit)


@st.composite
def valid_community_cards(draw):
    """生成5张不重复的公共牌。"""
    cards = []
    used = set()
    
    while len(cards) < 5:
        rank = draw(st.integers(min_value=2, max_value=14))
        suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
        key = (rank, suit)
        
        if key not in used:
            used.add(key)
            cards.append(Card(rank=rank, suit=suit))
    
    return cards


@st.composite
def valid_range(draw):
    """生成有效的手牌范围。"""
    # 生成一些手牌和权重
    hands = ['AA', 'KK', 'QQ', 'AKs', 'AKo', 'AQs', 'JJ', 'TT']
    selected = draw(st.lists(st.sampled_from(hands), min_size=1, max_size=len(hands), unique=True))
    
    range_dict = {}
    for hand in selected:
        weight = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        range_dict[hand] = weight
    
    return range_dict


@st.composite
def valid_experiment_scenario(draw):
    """生成有效的ExperimentScenario。"""
    name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))))
    assume(len(name.strip()) > 0)
    
    return ExperimentScenario(
        name=name,
        description=draw(st.text(max_size=200)),
        community_cards=draw(valid_community_cards()),
        oop_range=draw(valid_range()),
        ip_range=draw(valid_range()),
        solver_config=draw(valid_solver_config()),
        tags=draw(st.lists(st.text(min_size=1, max_size=20), max_size=5)),
    )


# ============================================================================
# Property 1: 数据模型序列化往返
# ============================================================================

class TestDataModelSerialization:
    """数据模型序列化往返测试。
    
    **Feature: equity-solver-validation, Property: 数据模型序列化往返**
    **Validates: Requirements 5.1**
    """
    
    @given(config=valid_solver_config())
    @settings(max_examples=50)
    def test_solver_config_round_trip(self, config: SolverConfig):
        """SolverConfig序列化往返测试。"""
        # 序列化
        data = config.to_dict()
        
        # 反序列化
        restored = SolverConfig.from_dict(data)
        
        # 验证
        assert restored.pot_size == config.pot_size
        assert restored.effective_stack == config.effective_stack
        assert restored.oop_bet_sizes == config.oop_bet_sizes
        assert restored.ip_bet_sizes == config.ip_bet_sizes
        assert restored.all_in_threshold == config.all_in_threshold
        assert restored.max_iterations == config.max_iterations
    
    @given(result=valid_solver_result())
    @settings(max_examples=50)
    def test_solver_result_round_trip(self, result: SolverResult):
        """SolverResult序列化往返测试。"""
        data = result.to_dict()
        restored = SolverResult.from_dict(data)
        
        assert restored.exploitability == result.exploitability
        assert restored.iterations == result.iterations
        assert restored.root_ev == result.root_ev
        assert restored.converged == result.converged
    
    @given(metrics=valid_validation_metrics())
    @settings(max_examples=50)
    def test_validation_metrics_round_trip(self, metrics: ValidationMetrics):
        """ValidationMetrics序列化往返测试。"""
        data = metrics.to_dict()
        restored = ValidationMetrics.from_dict(data)
        
        assert abs(restored.total_variation_distance - metrics.total_variation_distance) < 1e-10
        assert abs(restored.kl_divergence - metrics.kl_divergence) < 1e-10
        assert abs(restored.action_agreement_rate - metrics.action_agreement_rate) < 1e-10
        assert abs(restored.ev_correlation - metrics.ev_correlation) < 1e-10
        assert abs(restored.ev_rmse - metrics.ev_rmse) < 1e-10
    
    @given(metrics=valid_validation_metrics())
    @settings(max_examples=50)
    def test_comparison_result_round_trip(self, metrics: ValidationMetrics):
        """ComparisonResult序列化往返测试。"""
        comparison = ComparisonResult(
            metrics=metrics,
            per_hand_diff={'AA': 0.1, 'KK': 0.2},
            action_distribution={'fold': (0.3, 0.4), 'call': (0.7, 0.6)},
            equity_vector={'AA': 0.85, 'KK': 0.82},
        )
        
        data = comparison.to_dict()
        restored = ComparisonResult.from_dict(data)
        
        assert restored.per_hand_diff == comparison.per_hand_diff
        assert restored.action_distribution == comparison.action_distribution
        assert restored.equity_vector == comparison.equity_vector
    
    @given(scenario=valid_experiment_scenario())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.large_base_example])
    def test_experiment_scenario_round_trip(self, scenario: ExperimentScenario):
        """ExperimentScenario序列化往返测试。"""
        data = scenario.to_dict()
        restored = ExperimentScenario.from_dict(data)
        
        assert restored.name == scenario.name
        assert restored.description == scenario.description
        assert len(restored.community_cards) == len(scenario.community_cards)
        assert restored.oop_range == scenario.oop_range
        assert restored.ip_range == scenario.ip_range
        assert restored.tags == scenario.tags
        
        # 验证公共牌
        for orig, rest in zip(scenario.community_cards, restored.community_cards):
            assert orig.rank == rest.rank
            assert orig.suit == rest.suit


# ============================================================================
# 数据验证测试
# ============================================================================

class TestDataValidation:
    """数据验证测试。"""
    
    def test_solver_config_invalid_pot_size(self):
        """测试无效的底池大小。"""
        with pytest.raises(ValueError, match="底池大小必须为正数"):
            SolverConfig(pot_size=0, effective_stack=100)
        
        with pytest.raises(ValueError, match="底池大小必须为正数"):
            SolverConfig(pot_size=-10, effective_stack=100)
    
    def test_solver_config_invalid_stack(self):
        """测试无效的筹码量。"""
        with pytest.raises(ValueError, match="有效筹码不能为负数"):
            SolverConfig(pot_size=100, effective_stack=-10)
    
    def test_solver_config_invalid_all_in_threshold(self):
        """测试无效的全押阈值。"""
        with pytest.raises(ValueError, match="全押阈值必须在"):
            SolverConfig(pot_size=100, effective_stack=100, all_in_threshold=0)
        
        with pytest.raises(ValueError, match="全押阈值必须在"):
            SolverConfig(pot_size=100, effective_stack=100, all_in_threshold=1.5)
    
    def test_validation_metrics_invalid_tvd(self):
        """测试无效的总变差距离。"""
        with pytest.raises(ValueError, match="总变差距离必须在"):
            ValidationMetrics(total_variation_distance=-0.1)
        
        with pytest.raises(ValueError, match="总变差距离必须在"):
            ValidationMetrics(total_variation_distance=1.5)
    
    def test_validation_metrics_invalid_kl(self):
        """测试无效的KL散度。"""
        with pytest.raises(ValueError, match="KL散度不能为负数"):
            ValidationMetrics(kl_divergence=-0.1)
    
    def test_validation_metrics_invalid_correlation(self):
        """测试无效的相关系数。"""
        with pytest.raises(ValueError, match="EV相关系数必须在"):
            ValidationMetrics(ev_correlation=-1.5)
        
        with pytest.raises(ValueError, match="EV相关系数必须在"):
            ValidationMetrics(ev_correlation=1.5)
    
    def test_experiment_scenario_empty_name(self):
        """测试空场景名称。"""
        cards = [Card(rank=r, suit='h') for r in [14, 13, 12, 11, 10]]
        
        with pytest.raises(ValueError, match="场景名称不能为空"):
            ExperimentScenario(
                name='',
                description='test',
                community_cards=cards,
                oop_range={'AA': 1.0},
                ip_range={'KK': 1.0},
                solver_config=SolverConfig(pot_size=100, effective_stack=100),
            )
    
    def test_experiment_scenario_wrong_card_count(self):
        """测试错误的公共牌数量。"""
        cards = [Card(rank=r, suit='h') for r in [14, 13, 12]]  # 只有3张
        
        with pytest.raises(ValueError, match="河牌阶段必须有5张公共牌"):
            ExperimentScenario(
                name='test',
                description='test',
                community_cards=cards,
                oop_range={'AA': 1.0},
                ip_range={'KK': 1.0},
                solver_config=SolverConfig(pot_size=100, effective_stack=100),
            )
    
    def test_experiment_scenario_duplicate_cards(self):
        """测试重复的公共牌。"""
        cards = [
            Card(rank=14, suit='h'),
            Card(rank=14, suit='h'),  # 重复
            Card(rank=12, suit='h'),
            Card(rank=11, suit='h'),
            Card(rank=10, suit='h'),
        ]
        
        with pytest.raises(ValueError, match="公共牌中有重复的牌"):
            ExperimentScenario(
                name='test',
                description='test',
                community_cards=cards,
                oop_range={'AA': 1.0},
                ip_range={'KK': 1.0},
                solver_config=SolverConfig(pot_size=100, effective_stack=100),
            )
    
    def test_experiment_scenario_empty_range(self):
        """测试空范围。"""
        cards = [Card(rank=r, suit='h') for r in [14, 13, 12, 11, 10]]
        
        with pytest.raises(ValueError, match="OOP范围不能为空"):
            ExperimentScenario(
                name='test',
                description='test',
                community_cards=cards,
                oop_range={},
                ip_range={'KK': 1.0},
                solver_config=SolverConfig(pot_size=100, effective_stack=100),
            )


# ============================================================================
# 功能测试
# ============================================================================

class TestFunctionality:
    """功能测试。"""
    
    def test_comparison_result_get_worst_hands(self):
        """测试获取最差手牌功能。"""
        comparison = ComparisonResult(
            metrics=ValidationMetrics(),
            per_hand_diff={'AA': 0.1, 'KK': 0.5, 'QQ': 0.3, 'JJ': 0.2},
        )
        
        worst = comparison.get_worst_hands(n=2)
        
        assert len(worst) == 2
        assert worst[0] == ('KK', 0.5)
        assert worst[1] == ('QQ', 0.3)
    
    def test_experiment_result_success_property(self):
        """测试实验结果成功属性。"""
        cards = [Card(rank=r, suit='h') for r in [14, 13, 12, 11, 10]]
        scenario = ExperimentScenario(
            name='test',
            description='test',
            community_cards=cards,
            oop_range={'AA': 1.0},
            ip_range={'KK': 1.0},
            solver_config=SolverConfig(pot_size=100, effective_stack=100),
        )
        
        # 成功的结果
        success_result = ExperimentResult(
            scenario=scenario,
            comparison=ComparisonResult(metrics=ValidationMetrics()),
        )
        assert success_result.success is True
        
        # 失败的结果（有错误）
        error_result = ExperimentResult(
            scenario=scenario,
            error="测试错误",
        )
        assert error_result.success is False
        
        # 失败的结果（无对比结果）
        no_comparison_result = ExperimentResult(scenario=scenario)
        assert no_comparison_result.success is False
    
    def test_batch_experiment_result_summary(self):
        """测试批量实验结果汇总。"""
        cards = [Card(rank=r, suit='h') for r in [14, 13, 12, 11, 10]]
        scenario = ExperimentScenario(
            name='test',
            description='test',
            community_cards=cards,
            oop_range={'AA': 1.0},
            ip_range={'KK': 1.0},
            solver_config=SolverConfig(pot_size=100, effective_stack=100),
        )
        
        results = [
            ExperimentResult(
                scenario=scenario,
                comparison=ComparisonResult(
                    metrics=ValidationMetrics(
                        total_variation_distance=0.1,
                        action_agreement_rate=0.9,
                        ev_correlation=0.95,
                    )
                ),
            ),
            ExperimentResult(
                scenario=scenario,
                comparison=ComparisonResult(
                    metrics=ValidationMetrics(
                        total_variation_distance=0.2,
                        action_agreement_rate=0.8,
                        ev_correlation=0.85,
                    )
                ),
            ),
            ExperimentResult(
                scenario=scenario,
                error="测试错误",
            ),
        ]
        
        batch = BatchExperimentResult(results=results)
        batch.compute_summary()
        
        assert batch.success_count == 2
        assert batch.failure_count == 1
        assert abs(batch.summary_metrics['avg_total_variation_distance'] - 0.15) < 1e-10
        assert abs(batch.summary_metrics['avg_action_agreement_rate'] - 0.85) < 1e-10
        assert abs(batch.summary_metrics['success_rate'] - 2/3) < 1e-10
