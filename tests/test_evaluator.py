"""评估器模块的单元测试和属性测试。

测试内容：
- 评估运行指定局数
- 指标计算正确性
- 不同对手策略
- 多模型比较
- 结果保存和加载
- 属性测试：评估对局数量、指标计算、策略应用等
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np
from hypothesis import given, settings, strategies as st, assume

from analysis.evaluator import (
    Evaluator,
    EvaluationResult,
    ComparisonResult,
    OpponentStrategy,
    RandomStrategy,
    FixedStrategy,
    CallOnlyStrategy,
    AlwaysFoldStrategy,
    NeuralNetworkStrategy
)
from models.networks import PolicyNetwork
from models.core import GameState, Action, ActionType, GameStage, Card
from environment.state_encoder import StateEncoder


# ============================================================================
# 辅助函数和固定数据
# ============================================================================

def create_test_model() -> PolicyNetwork:
    """创建用于测试的策略网络模型。"""
    return PolicyNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=6)


def create_simple_game_state() -> GameState:
    """创建简单的游戏状态用于测试。"""
    return GameState(
        player_hands=[
            (Card(14, 'h'), Card(13, 'h')),  # 玩家0: AK同花
            (Card(2, 's'), Card(7, 'd'))     # 玩家1: 27杂色
        ],
        community_cards=[],
        pot=15,
        player_stacks=[995, 990],
        current_bets=[5, 10],
        button_position=0,
        stage=GameStage.PREFLOP,
        action_history=[],
        current_player=0
    )


# ============================================================================
# 对手策略单元测试
# ============================================================================

class TestRandomStrategy:
    """随机策略单元测试。"""
    
    def test_name(self):
        """测试策略名称。"""
        strategy = RandomStrategy()
        assert strategy.name == "RandomStrategy"
    
    def test_select_action_returns_legal_action(self):
        """测试选择的行动在合法行动列表中。"""
        strategy = RandomStrategy(seed=42)
        state = create_simple_game_state()
        legal_actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
            Action(ActionType.RAISE, 20)
        ]
        
        action = strategy.select_action(state, legal_actions)
        assert action in legal_actions
    
    def test_reproducibility_with_seed(self):
        """测试使用相同种子时结果可重现。"""
        state = create_simple_game_state()
        legal_actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
            Action(ActionType.RAISE, 20)
        ]
        
        strategy1 = RandomStrategy(seed=42)
        strategy2 = RandomStrategy(seed=42)
        
        actions1 = [strategy1.select_action(state, legal_actions) for _ in range(10)]
        actions2 = [strategy2.select_action(state, legal_actions) for _ in range(10)]
        
        assert actions1 == actions2


class TestFixedStrategy:
    """固定策略单元测试。"""
    
    def test_name_passive(self):
        """测试被动策略名称。"""
        strategy = FixedStrategy(prefer_aggressive=False)
        assert "passive" in strategy.name
    
    def test_name_aggressive(self):
        """测试激进策略名称。"""
        strategy = FixedStrategy(prefer_aggressive=True)
        assert "aggressive" in strategy.name
    
    def test_passive_prefers_check(self):
        """测试被动策略优先过牌。"""
        strategy = FixedStrategy(prefer_aggressive=False)
        state = create_simple_game_state()
        legal_actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CHECK),
            Action(ActionType.RAISE, 20)
        ]
        
        action = strategy.select_action(state, legal_actions)
        assert action.action_type == ActionType.CHECK
    
    def test_passive_prefers_call_over_fold(self):
        """测试被动策略优先跟注而非弃牌。"""
        strategy = FixedStrategy(prefer_aggressive=False)
        state = create_simple_game_state()
        legal_actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
            Action(ActionType.RAISE, 20)
        ]
        
        action = strategy.select_action(state, legal_actions)
        assert action.action_type == ActionType.CALL
    
    def test_aggressive_prefers_raise(self):
        """测试激进策略优先加注。"""
        strategy = FixedStrategy(prefer_aggressive=True)
        state = create_simple_game_state()
        legal_actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
            Action(ActionType.RAISE, 20),
            Action(ActionType.RAISE, 50)
        ]
        
        action = strategy.select_action(state, legal_actions)
        assert action.action_type == ActionType.RAISE
        # 应该选择最小加注
        assert action.amount == 20


class TestCallOnlyStrategy:
    """只跟注策略单元测试。"""
    
    def test_name(self):
        """测试策略名称。"""
        strategy = CallOnlyStrategy()
        assert strategy.name == "CallOnlyStrategy"
    
    def test_prefers_check(self):
        """测试优先过牌。"""
        strategy = CallOnlyStrategy()
        state = create_simple_game_state()
        legal_actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CHECK),
            Action(ActionType.RAISE, 20)
        ]
        
        action = strategy.select_action(state, legal_actions)
        assert action.action_type == ActionType.CHECK
    
    def test_calls_when_no_check(self):
        """测试无法过牌时跟注。"""
        strategy = CallOnlyStrategy()
        state = create_simple_game_state()
        legal_actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
            Action(ActionType.RAISE, 20)
        ]
        
        action = strategy.select_action(state, legal_actions)
        assert action.action_type == ActionType.CALL


class TestAlwaysFoldStrategy:
    """总是弃牌策略单元测试。"""
    
    def test_name(self):
        """测试策略名称。"""
        strategy = AlwaysFoldStrategy()
        assert strategy.name == "AlwaysFoldStrategy"
    
    def test_checks_when_possible(self):
        """测试可以过牌时过牌。"""
        strategy = AlwaysFoldStrategy()
        state = create_simple_game_state()
        legal_actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CHECK),
            Action(ActionType.RAISE, 20)
        ]
        
        action = strategy.select_action(state, legal_actions)
        assert action.action_type == ActionType.CHECK
    
    def test_folds_when_no_check(self):
        """测试无法过牌时弃牌。"""
        strategy = AlwaysFoldStrategy()
        state = create_simple_game_state()
        legal_actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
            Action(ActionType.RAISE, 20)
        ]
        
        action = strategy.select_action(state, legal_actions)
        assert action.action_type == ActionType.FOLD


# ============================================================================
# 评估器单元测试
# ============================================================================

class TestEvaluator:
    """评估器单元测试。"""
    
    def test_initialization(self):
        """测试评估器初始化。"""
        evaluator = Evaluator(
            initial_stack=1000,
            small_blind=5,
            big_blind=10
        )
        assert evaluator.initial_stack == 1000
        assert evaluator.small_blind == 5
        assert evaluator.big_blind == 10
    
    def test_evaluate_runs_specified_games(self):
        """测试评估运行指定局数。"""
        evaluator = Evaluator()
        model = create_test_model()
        opponent = RandomStrategy(seed=42)
        
        num_games = 10
        result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=num_games,
            model_name="test_model"
        )
        
        assert result.num_games == num_games
        assert len(result.profits) == num_games
        assert evaluator.games_played == num_games
    
    def test_evaluate_calculates_win_rate(self):
        """测试胜率计算正确性。"""
        evaluator = Evaluator()
        model = create_test_model()
        opponent = AlwaysFoldStrategy()  # 对手总是弃牌，模型应该总是赢
        
        num_games = 10
        result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=num_games,
            model_name="test_model"
        )
        
        # 验证胜率计算
        expected_win_rate = result.wins / num_games
        assert abs(result.win_rate - expected_win_rate) < 1e-6
    
    def test_evaluate_calculates_statistics(self):
        """测试统计指标计算。"""
        evaluator = Evaluator()
        model = create_test_model()
        opponent = RandomStrategy(seed=42)
        
        num_games = 20
        result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=num_games,
            model_name="test_model"
        )
        
        # 验证平均盈利计算
        expected_avg = np.mean(result.profits)
        assert abs(result.avg_profit - expected_avg) < 1e-6
        
        # 验证标准差计算
        expected_std = np.std(result.profits)
        assert abs(result.std_profit - expected_std) < 1e-6
        
        # 验证总盈利计算
        expected_total = sum(result.profits)
        assert abs(result.total_profit - expected_total) < 1e-6
    
    def test_evaluate_with_different_opponents(self):
        """测试不同对手策略。"""
        evaluator = Evaluator()
        model = create_test_model()
        
        opponents = [
            RandomStrategy(seed=42),
            FixedStrategy(prefer_aggressive=False),
            FixedStrategy(prefer_aggressive=True),
            CallOnlyStrategy(),
            AlwaysFoldStrategy()
        ]
        
        for opponent in opponents:
            result = evaluator.evaluate(
                model=model,
                opponent=opponent,
                num_games=5,
                model_name="test_model"
            )
            
            assert result.opponent_name == opponent.name
            assert result.num_games == 5
    
    def test_evaluate_wins_losses_ties_sum(self):
        """测试胜负平局数之和等于总局数。"""
        evaluator = Evaluator()
        model = create_test_model()
        opponent = RandomStrategy(seed=42)
        
        num_games = 20
        result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=num_games,
            model_name="test_model"
        )
        
        assert result.wins + result.losses + result.ties == num_games


class TestEvaluatorCompareModels:
    """多模型比较测试。"""
    
    def test_compare_models_basic(self):
        """测试基本的多模型比较。"""
        evaluator = Evaluator()
        models = {
            "model_a": create_test_model(),
            "model_b": create_test_model()
        }
        opponent = RandomStrategy(seed=42)
        
        result = evaluator.compare_models(
            models=models,
            opponent=opponent,
            num_games=5
        )
        
        assert isinstance(result, ComparisonResult)
        assert set(result.models) == {"model_a", "model_b"}
        assert result.opponent_name == opponent.name
        assert len(result.results) == 2
    
    def test_compare_models_contains_all_results(self):
        """测试比较结果包含所有模型的数据。"""
        evaluator = Evaluator()
        models = {
            "model_1": create_test_model(),
            "model_2": create_test_model(),
            "model_3": create_test_model()
        }
        opponent = RandomStrategy(seed=42)
        
        result = evaluator.compare_models(
            models=models,
            opponent=opponent,
            num_games=5
        )
        
        for model_name in models.keys():
            assert model_name in result.results
            assert result.results[model_name].model_name == model_name
            assert result.results[model_name].num_games == 5


class TestEvaluatorSaveLoad:
    """评估结果保存和加载测试。"""
    
    def test_save_evaluation_result(self):
        """测试保存单个评估结果。"""
        evaluator = Evaluator()
        model = create_test_model()
        opponent = RandomStrategy(seed=42)
        
        result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=5,
            model_name="test_model"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "result.json"
            saved_path = evaluator.save_results(result, filepath)
            
            assert Path(saved_path).exists()
            
            # 验证文件内容
            with open(saved_path, 'r') as f:
                data = json.load(f)
            
            assert data['model_name'] == "test_model"
            assert data['num_games'] == 5
    
    def test_load_evaluation_result(self):
        """测试加载单个评估结果。"""
        evaluator = Evaluator()
        model = create_test_model()
        opponent = RandomStrategy(seed=42)
        
        original_result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=5,
            model_name="test_model"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "result.json"
            evaluator.save_results(original_result, filepath)
            
            loaded_result = evaluator.load_results(filepath)
            
            assert isinstance(loaded_result, EvaluationResult)
            assert loaded_result.model_name == original_result.model_name
            assert loaded_result.num_games == original_result.num_games
            assert loaded_result.win_rate == original_result.win_rate
            assert loaded_result.profits == original_result.profits
    
    def test_save_comparison_result(self):
        """测试保存比较结果。"""
        evaluator = Evaluator()
        models = {
            "model_a": create_test_model(),
            "model_b": create_test_model()
        }
        opponent = RandomStrategy(seed=42)
        
        result = evaluator.compare_models(
            models=models,
            opponent=opponent,
            num_games=5
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "comparison.json"
            saved_path = evaluator.save_results(result, filepath)
            
            assert Path(saved_path).exists()
            
            # 验证文件内容
            with open(saved_path, 'r') as f:
                data = json.load(f)
            
            assert 'models' in data
            assert 'results' in data
            assert set(data['models']) == {"model_a", "model_b"}
    
    def test_load_comparison_result(self):
        """测试加载比较结果。"""
        evaluator = Evaluator()
        models = {
            "model_a": create_test_model(),
            "model_b": create_test_model()
        }
        opponent = RandomStrategy(seed=42)
        
        original_result = evaluator.compare_models(
            models=models,
            opponent=opponent,
            num_games=5
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "comparison.json"
            evaluator.save_results(original_result, filepath)
            
            loaded_result = evaluator.load_results(filepath)
            
            assert isinstance(loaded_result, ComparisonResult)
            assert set(loaded_result.models) == set(original_result.models)
            assert loaded_result.opponent_name == original_result.opponent_name


class TestEvaluationResult:
    """评估结果数据类测试。"""
    
    def test_to_dict(self):
        """测试转换为字典。"""
        result = EvaluationResult(
            model_name="test",
            opponent_name="random",
            num_games=10,
            wins=5,
            losses=4,
            ties=1,
            win_rate=0.5,
            avg_profit=10.0,
            std_profit=5.0,
            total_profit=100.0,
            profits=[10.0] * 10,
            timestamp="2024-01-01T00:00:00"
        )
        
        d = result.to_dict()
        
        assert d['model_name'] == "test"
        assert d['num_games'] == 10
        assert d['win_rate'] == 0.5


class TestComparisonResult:
    """比较结果数据类测试。"""
    
    def test_to_dict(self):
        """测试转换为字典。"""
        eval_result = EvaluationResult(
            model_name="test",
            opponent_name="random",
            num_games=10,
            wins=5,
            losses=4,
            ties=1,
            win_rate=0.5,
            avg_profit=10.0,
            std_profit=5.0,
            total_profit=100.0,
            profits=[10.0] * 10,
            timestamp="2024-01-01T00:00:00"
        )
        
        result = ComparisonResult(
            models=["test"],
            opponent_name="random",
            results={"test": eval_result},
            timestamp="2024-01-01T00:00:00"
        )
        
        d = result.to_dict()
        
        assert d['models'] == ["test"]
        assert 'test' in d['results']



# ============================================================================
# 属性测试（Property-Based Testing）
# ============================================================================

class TestEvaluatorProperties:
    """评估器属性测试。
    
    使用Hypothesis进行基于属性的测试，验证评估器的正确性。
    """
    
    @given(
        num_games=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100)
    def test_property_evaluation_game_count(self, num_games: int):
        """属性测试：评估对局数量正确性。
        
        # Feature: texas-holdem-ai-training, Property 21: 评估对局数量正确性
        # *对于任何*指定的评估对局数N，评估会话应该恰好执行N局游戏
        # **验证需求：6.1**
        
        指定N局，验证恰好执行N局。
        """
        evaluator = Evaluator()
        model = create_test_model()
        opponent = RandomStrategy()
        
        result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=num_games,
            model_name="test_model"
        )
        
        # 验证对局数量
        assert result.num_games == num_games, \
            f"期望执行{num_games}局，实际记录{result.num_games}局"
        assert len(result.profits) == num_games, \
            f"期望{num_games}个盈利记录，实际{len(result.profits)}个"
        assert evaluator.games_played == num_games, \
            f"期望执行{num_games}局，实际执行{evaluator.games_played}局"
    
    @given(
        num_games=st.integers(min_value=1, max_value=30)
    )
    @settings(max_examples=100)
    def test_property_win_rate_calculation(self, num_games: int):
        """属性测试：评估指标计算正确性。
        
        # Feature: texas-holdem-ai-training, Property 22: 评估指标计算正确性
        # *对于任何*评估会话，返回的胜率应该等于（胜局数 / 总局数），
        # 且报告应该包含平均盈利和标准差
        # **验证需求：6.2**
        
        验证胜率 = 胜局数 / 总局数。
        """
        evaluator = Evaluator()
        model = create_test_model()
        opponent = RandomStrategy()
        
        result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=num_games,
            model_name="test_model"
        )
        
        # 验证胜率计算
        expected_win_rate = result.wins / num_games
        assert abs(result.win_rate - expected_win_rate) < 1e-6, \
            f"胜率计算错误：期望{expected_win_rate}，实际{result.win_rate}"
        
        # 验证胜负平局数之和
        assert result.wins + result.losses + result.ties == num_games, \
            f"胜负平局数之和应等于总局数{num_games}"
        
        # 验证平均盈利计算
        expected_avg = np.mean(result.profits)
        assert abs(result.avg_profit - expected_avg) < 1e-6, \
            f"平均盈利计算错误：期望{expected_avg}，实际{result.avg_profit}"
        
        # 验证标准差计算（当局数>1时）
        if num_games > 1:
            expected_std = np.std(result.profits)
            assert abs(result.std_profit - expected_std) < 1e-6, \
                f"标准差计算错误：期望{expected_std}，实际{result.std_profit}"
    
    @given(
        strategy_type=st.sampled_from(['random', 'passive', 'aggressive', 'call_only', 'always_fold'])
    )
    @settings(max_examples=100)
    def test_property_opponent_strategy_applied(self, strategy_type: str):
        """属性测试：基准策略应用正确性。
        
        # Feature: texas-holdem-ai-training, Property 23: 基准策略应用正确性
        # *对于任何*指定的基准策略，评估会话中的对手应该使用该策略做决策
        # **验证需求：6.3**
        
        验证对手使用指定策略做决策。
        """
        evaluator = Evaluator()
        model = create_test_model()
        
        # 根据类型创建策略
        if strategy_type == 'random':
            opponent = RandomStrategy()
        elif strategy_type == 'passive':
            opponent = FixedStrategy(prefer_aggressive=False)
        elif strategy_type == 'aggressive':
            opponent = FixedStrategy(prefer_aggressive=True)
        elif strategy_type == 'call_only':
            opponent = CallOnlyStrategy()
        else:
            opponent = AlwaysFoldStrategy()
        
        result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=5,
            model_name="test_model"
        )
        
        # 验证对手策略名称被正确记录
        assert result.opponent_name == opponent.name, \
            f"对手策略名称错误：期望{opponent.name}，实际{result.opponent_name}"
        
        # 对于AlwaysFoldStrategy，模型应该总是赢（除非可以过牌）
        if strategy_type == 'always_fold':
            # 由于对手总是弃牌，模型应该赢得大部分对局
            assert result.wins >= result.losses, \
                f"对手总是弃牌时，模型应该赢得更多：胜{result.wins}，负{result.losses}"
    
    @given(
        num_models=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100)
    def test_property_multi_model_comparison_completeness(self, num_models: int):
        """属性测试：多模型评估报告完整性。
        
        # Feature: texas-holdem-ai-training, Property 24: 多模型评估报告完整性
        # *对于任何*多模型评估请求，生成的报告应该包含所有指定模型的性能数据
        # **验证需求：6.4**
        
        验证报告包含所有模型的性能数据。
        """
        evaluator = Evaluator()
        
        # 创建多个模型
        models = {f"model_{i}": create_test_model() for i in range(num_models)}
        opponent = RandomStrategy()
        
        result = evaluator.compare_models(
            models=models,
            opponent=opponent,
            num_games=3
        )
        
        # 验证所有模型都在结果中
        assert len(result.models) == num_models, \
            f"期望{num_models}个模型，实际{len(result.models)}个"
        assert len(result.results) == num_models, \
            f"期望{num_models}个结果，实际{len(result.results)}个"
        
        # 验证每个模型都有完整的评估数据
        for model_name in models.keys():
            assert model_name in result.results, \
                f"模型{model_name}不在结果中"
            model_result = result.results[model_name]
            assert model_result.model_name == model_name
            assert model_result.num_games == 3
            assert hasattr(model_result, 'win_rate')
            assert hasattr(model_result, 'avg_profit')
            assert hasattr(model_result, 'std_profit')
    
    @given(
        num_games=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    def test_property_results_persistence(self, num_games: int):
        """属性测试：评估结果持久化正确性。
        
        # Feature: texas-holdem-ai-training, Property 25: 评估结果持久化正确性
        # *对于任何*评估会话，结果应该被保存到文件中，且文件内容包含所有评估指标
        # **验证需求：6.5**
        
        验证保存的文件包含所有评估指标。
        """
        evaluator = Evaluator()
        model = create_test_model()
        opponent = RandomStrategy()
        
        result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=num_games,
            model_name="test_model"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "result.json"
            evaluator.save_results(result, filepath)
            
            # 验证文件存在
            assert filepath.exists(), "结果文件应该存在"
            
            # 验证文件内容
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # 验证所有必需字段都存在
            required_fields = [
                'model_name', 'opponent_name', 'num_games',
                'wins', 'losses', 'ties', 'win_rate',
                'avg_profit', 'std_profit', 'total_profit',
                'profits', 'timestamp'
            ]
            
            for field in required_fields:
                assert field in data, f"缺少必需字段：{field}"
            
            # 验证数据一致性
            assert data['num_games'] == num_games
            assert data['win_rate'] == result.win_rate
            assert data['avg_profit'] == result.avg_profit
            assert len(data['profits']) == num_games
            
            # 验证可以正确加载
            loaded_result = evaluator.load_results(filepath)
            assert loaded_result.num_games == result.num_games
            assert loaded_result.win_rate == result.win_rate


class TestOpponentStrategyProperties:
    """对手策略属性测试。"""
    
    @given(
        num_actions=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_property_random_strategy_selects_legal_action(self, num_actions: int):
        """属性测试：随机策略总是选择合法行动。"""
        strategy = RandomStrategy()
        state = create_simple_game_state()
        
        # 创建合法行动列表
        legal_actions = [Action(ActionType.FOLD)]
        if num_actions > 1:
            legal_actions.append(Action(ActionType.CALL))
        if num_actions > 2:
            for i in range(num_actions - 2):
                legal_actions.append(Action(ActionType.RAISE, 10 * (i + 1)))
        
        # 多次选择，验证都是合法行动
        for _ in range(20):
            action = strategy.select_action(state, legal_actions)
            assert action in legal_actions, \
                f"选择的行动{action}不在合法行动列表中"
    
    @given(
        has_check=st.booleans(),
        has_call=st.booleans()
    )
    @settings(max_examples=100)
    def test_property_fixed_passive_strategy_behavior(self, has_check: bool, has_call: bool):
        """属性测试：被动固定策略行为一致性。"""
        strategy = FixedStrategy(prefer_aggressive=False)
        state = create_simple_game_state()
        
        legal_actions = [Action(ActionType.FOLD)]
        if has_check:
            legal_actions.append(Action(ActionType.CHECK))
        if has_call:
            legal_actions.append(Action(ActionType.CALL))
        legal_actions.append(Action(ActionType.RAISE, 20))
        
        action = strategy.select_action(state, legal_actions)
        
        # 被动策略应该优先CHECK > CALL > FOLD
        if has_check:
            assert action.action_type == ActionType.CHECK, \
                "被动策略应该优先过牌"
        elif has_call:
            assert action.action_type == ActionType.CALL, \
                "被动策略无法过牌时应该跟注"
        else:
            assert action.action_type == ActionType.FOLD, \
                "被动策略无法过牌或跟注时应该弃牌"
    
    @given(
        has_raise=st.booleans()
    )
    @settings(max_examples=100)
    def test_property_fixed_aggressive_strategy_behavior(self, has_raise: bool):
        """属性测试：激进固定策略行为一致性。"""
        strategy = FixedStrategy(prefer_aggressive=True)
        state = create_simple_game_state()
        
        legal_actions = [Action(ActionType.FOLD), Action(ActionType.CALL)]
        if has_raise:
            legal_actions.append(Action(ActionType.RAISE, 20))
            legal_actions.append(Action(ActionType.RAISE, 50))
        
        action = strategy.select_action(state, legal_actions)
        
        # 激进策略应该优先RAISE
        if has_raise:
            assert action.action_type == ActionType.RAISE, \
                "激进策略应该优先加注"
            # 应该选择最小加注
            assert action.amount == 20, \
                "激进策略应该选择最小加注"
        else:
            assert action.action_type == ActionType.CALL, \
                "激进策略无法加注时应该跟注"
