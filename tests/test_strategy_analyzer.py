"""策略分析器的单元测试和属性测试。

测试覆盖：
- 模型加载（新旧两种检查点格式）
- 状态分析返回有效概率分布
- 策略热图生成
- 决策解释包含期望价值
- 多模型比较
- Deep CFR检查点格式兼容性
"""

import os
import time
import tempfile
from datetime import datetime
from pathlib import Path
import pytest
import numpy as np
import torch

from hypothesis import given, settings, strategies as st, HealthCheck

from models.core import GameState, GameStage, Card, Action, ActionType
from models.networks import PolicyNetwork, RegretNetwork
from environment.state_encoder import StateEncoder
from analysis.strategy_analyzer import (
    StrategyAnalyzer,
    ActionProbability,
    DecisionExplanation,
    StrategyComparison
)
from utils.checkpoint_manager import CheckpointManager


# ============================================================================
# 测试夹具
# ============================================================================

@pytest.fixture
def temp_checkpoint_dir():
    """创建临时检查点目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_policy_network():
    """创建示例策略网络。"""
    return PolicyNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=6)


@pytest.fixture
def sample_regret_network():
    """创建示例遗憾网络。"""
    return RegretNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=6)


@pytest.fixture
def saved_checkpoint(temp_checkpoint_dir, sample_policy_network):
    """保存一个旧格式检查点并返回路径（兼容性测试）。"""
    checkpoint_manager = CheckpointManager(temp_checkpoint_dir)
    
    # 保存检查点（旧格式）
    checkpoint_path = checkpoint_manager.save(
        model=sample_policy_network,
        optimizer=None,
        metadata={
            'episode_number': 100,
            'win_rate': 0.55,
            'avg_reward': 10.5
        }
    )
    
    return checkpoint_path


@pytest.fixture
def saved_deep_cfr_checkpoint(temp_checkpoint_dir, sample_policy_network, sample_regret_network):
    """保存一个Deep CFR格式检查点并返回路径。"""
    timestamp = int(time.time() * 1000000)
    filename = f"checkpoint_{timestamp}_200.pt"
    filepath = Path(temp_checkpoint_dir) / filename
    
    checkpoint_data = {
        # 网络参数
        'regret_network_state_dict': sample_regret_network.state_dict(),
        'policy_network_state_dict': sample_policy_network.state_dict(),
        # 优化器参数（可选）
        'regret_optimizer_state_dict': None,
        'policy_optimizer_state_dict': None,
        # 元数据
        'episode_number': 200,
        'timestamp': datetime.now().isoformat(),
        'win_rate': 0.60,
        'avg_reward': 15.0,
        'checkpoint_format': 'deep_cfr_v1',
        # Deep CFR特有元数据
        'cfr_iterations': 1000,
        'regret_buffer_size': 50000,
        'strategy_buffer_size': 50000
    }
    
    torch.save(checkpoint_data, filepath)
    return str(filepath)


@pytest.fixture
def sample_game_state():
    """创建示例游戏状态。"""
    return GameState(
        player_hands=[
            (Card(14, 'h'), Card(13, 'h')),  # AK同花
            (Card(7, 'd'), Card(2, 'c'))     # 72杂色
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


@pytest.fixture
def sample_game_state_with_community():
    """创建带公共牌的游戏状态。"""
    return GameState(
        player_hands=[
            (Card(14, 'h'), Card(13, 'h')),
            (Card(7, 'd'), Card(2, 'c'))
        ],
        community_cards=[Card(12, 'h'), Card(11, 'h'), Card(10, 's')],
        pot=100,
        player_stacks=[900, 900],
        current_bets=[50, 50],
        button_position=0,
        stage=GameStage.FLOP,
        action_history=[],
        current_player=0
    )


@pytest.fixture
def strategy_analyzer(temp_checkpoint_dir):
    """创建策略分析器实例。"""
    return StrategyAnalyzer(checkpoint_dir=temp_checkpoint_dir)


# ============================================================================
# 单元测试：模型加载
# ============================================================================

class TestModelLoading:
    """测试模型加载功能。"""
    
    def test_load_model_success(
        self, strategy_analyzer, saved_checkpoint
    ):
        """测试成功加载模型。"""
        # 加载模型
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        # 验证模型已加载
        assert strategy_analyzer.is_model_loaded
        assert strategy_analyzer._policy_network is not None
    
    def test_load_model_file_not_found(self, strategy_analyzer):
        """测试加载不存在的检查点文件。"""
        with pytest.raises(FileNotFoundError):
            strategy_analyzer.load_model(
                checkpoint_path="nonexistent_checkpoint.pt"
            )
    
    def test_load_model_metadata(
        self, strategy_analyzer, saved_checkpoint
    ):
        """测试加载模型后元数据正确。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        metadata = strategy_analyzer.model_metadata
        assert 'episode_number' in metadata
        assert metadata['episode_number'] == 100


# ============================================================================
# 单元测试：Deep CFR 检查点格式
# ============================================================================

class TestDeepCFRCheckpointFormat:
    """测试 Deep CFR 检查点格式的加载功能。"""
    
    def test_load_deep_cfr_checkpoint_success(
        self, strategy_analyzer, saved_deep_cfr_checkpoint
    ):
        """测试成功加载 Deep CFR 格式检查点。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_deep_cfr_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        # 验证模型已加载
        assert strategy_analyzer.is_model_loaded
        assert strategy_analyzer._policy_network is not None
        assert strategy_analyzer._regret_network is not None
        assert strategy_analyzer.checkpoint_format == 'deep_cfr_v1'
        assert strategy_analyzer.has_regret_network
    
    def test_load_deep_cfr_checkpoint_metadata(
        self, strategy_analyzer, saved_deep_cfr_checkpoint
    ):
        """测试 Deep CFR 检查点元数据正确。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_deep_cfr_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        metadata = strategy_analyzer.model_metadata
        
        # 验证基本元数据
        assert metadata['episode_number'] == 200
        assert metadata['win_rate'] == 0.60
        assert metadata['avg_reward'] == 15.0
        assert metadata['checkpoint_format'] == 'deep_cfr_v1'
        
        # 验证 Deep CFR 特有元数据
        assert metadata['cfr_iterations'] == 1000
        assert metadata['regret_buffer_size'] == 50000
        assert metadata['strategy_buffer_size'] == 50000
    
    def test_load_legacy_checkpoint_compatibility(
        self, strategy_analyzer, saved_checkpoint
    ):
        """测试加载旧格式检查点的兼容性。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        # 验证模型已加载
        assert strategy_analyzer.is_model_loaded
        assert strategy_analyzer._policy_network is not None
        # 旧格式没有遗憾网络
        assert strategy_analyzer._regret_network is None
        assert strategy_analyzer.checkpoint_format == 'legacy'
        assert not strategy_analyzer.has_regret_network
    
    def test_analyze_state_with_deep_cfr_checkpoint(
        self, strategy_analyzer, saved_deep_cfr_checkpoint, sample_game_state
    ):
        """测试使用 Deep CFR 检查点进行状态分析。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_deep_cfr_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        # 分析状态
        probs = strategy_analyzer.analyze_state(sample_game_state, player_id=0)
        
        # 验证返回有效概率分布
        assert isinstance(probs, dict)
        # 动态动作配置：返回的动作数量取决于模型的动作维度
        # 6维模型返回5个显示动作（CHECK和CALL合并为CHECK/CALL）
        assert len(probs) >= 4, f"应至少有4个动作，实际有{len(probs)}个"
        
        # 验证所有概率非负
        for action_name, prob in probs.items():
            assert prob >= 0, f"概率不应为负: {action_name}={prob}"
        
        # 验证概率和为1
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 1e-6, f"概率和应为1，实际为{total_prob}"
    
    def test_analyze_state_with_legacy_checkpoint(
        self, strategy_analyzer, saved_checkpoint, sample_game_state
    ):
        """测试使用旧格式检查点进行状态分析。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        # 分析状态
        probs = strategy_analyzer.analyze_state(sample_game_state, player_id=0)
        
        # 验证返回有效概率分布
        assert isinstance(probs, dict)
        # 动态动作配置：返回的动作数量取决于模型的动作维度
        # 6维模型返回5个显示动作（CHECK和CALL合并为CHECK/CALL）
        assert len(probs) >= 4, f"应至少有4个动作，实际有{len(probs)}个"
        
        # 验证所有概率非负
        for action_name, prob in probs.items():
            assert prob >= 0, f"概率不应为负: {action_name}={prob}"
        
        # 验证概率和为1
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 1e-6, f"概率和应为1，实际为{total_prob}"
    
    def test_explain_decision_with_deep_cfr_checkpoint(
        self, strategy_analyzer, saved_deep_cfr_checkpoint, sample_game_state
    ):
        """测试使用 Deep CFR 检查点进行决策解释。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_deep_cfr_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        # 获取决策解释
        explanation = strategy_analyzer.explain_decision(sample_game_state, player_id=0)
        
        # 验证返回类型
        assert isinstance(explanation, DecisionExplanation)
        
        # 验证包含期望价值
        assert hasattr(explanation, 'expected_value')
        assert isinstance(explanation.expected_value, float)
        
        # 验证包含行动概率
        assert len(explanation.action_probabilities) > 0
        
        # 验证包含推荐行动
        assert explanation.recommended_action is not None


# ============================================================================
# 单元测试：状态分析
# ============================================================================

class TestStateAnalysis:
    """测试状态分析功能。"""
    
    def test_analyze_state_returns_valid_distribution(
        self, strategy_analyzer, saved_checkpoint, sample_game_state
    ):
        """测试状态分析返回有效概率分布。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        probs = strategy_analyzer.analyze_state(sample_game_state, player_id=0)
        
        # 验证返回字典
        assert isinstance(probs, dict)
        # 动态动作配置：返回的动作数量取决于模型的动作维度
        # 6维模型返回5个显示动作（CHECK和CALL合并为CHECK/CALL）
        assert len(probs) >= 4, f"应至少有4个动作，实际有{len(probs)}个"
        
        # 验证所有概率非负
        for action_name, prob in probs.items():
            assert prob >= 0, f"概率不应为负: {action_name}={prob}"
        
        # 验证概率和为1
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 1e-6, f"概率和应为1，实际为{total_prob}"
    
    def test_analyze_state_without_model_raises_error(
        self, strategy_analyzer, sample_game_state
    ):
        """测试未加载模型时分析状态抛出错误。"""
        with pytest.raises(RuntimeError, match="模型未加载"):
            strategy_analyzer.analyze_state(sample_game_state, player_id=0)
    
    def test_analyze_state_different_players(
        self, strategy_analyzer, saved_checkpoint, sample_game_state
    ):
        """测试分析不同玩家的状态。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        probs_p0 = strategy_analyzer.analyze_state(sample_game_state, player_id=0)
        probs_p1 = strategy_analyzer.analyze_state(sample_game_state, player_id=1)
        
        # 两个玩家的概率分布应该不同（因为手牌不同）
        # 但都应该是有效的概率分布
        assert abs(sum(probs_p0.values()) - 1.0) < 1e-6
        assert abs(sum(probs_p1.values()) - 1.0) < 1e-6


# ============================================================================
# 单元测试：策略热图
# ============================================================================

class TestStrategyHeatmap:
    """测试策略热图生成功能。"""
    
    def test_generate_heatmap_basic(
        self, strategy_analyzer, saved_checkpoint
    ):
        """测试基本热图生成。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        # 创建手牌范围
        hand_range = [
            (Card(14, 'h'), Card(14, 'd')),  # AA
            (Card(13, 'h'), Card(13, 'd')),  # KK
            (Card(14, 'h'), Card(13, 'h')),  # AKs
        ]
        
        heatmap = strategy_analyzer.generate_strategy_heatmap(
            hand_range=hand_range,
            stage=GameStage.PREFLOP
        )
        
        # 验证热图形状
        assert heatmap.shape == (3, 6)  # 3个手牌，6个行动（FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN）
        
        # 验证每行概率和为1
        for i in range(heatmap.shape[0]):
            row_sum = np.sum(heatmap[i])
            assert abs(row_sum - 1.0) < 1e-5, f"行{i}概率和应为1，实际为{row_sum}"
    
    def test_generate_heatmap_with_community_cards(
        self, strategy_analyzer, saved_checkpoint
    ):
        """测试带公共牌的热图生成。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        hand_range = [
            (Card(14, 'h'), Card(13, 'h')),
            (Card(10, 'd'), Card(9, 'd')),
        ]
        
        community_cards = [Card(12, 'h'), Card(11, 'h'), Card(2, 's')]
        
        heatmap = strategy_analyzer.generate_strategy_heatmap(
            hand_range=hand_range,
            community_cards=community_cards,
            stage=GameStage.FLOP
        )
        
        assert heatmap.shape == (2, 6)  # 2个手牌，6个行动
    
    def test_generate_heatmap_without_model_raises_error(
        self, strategy_analyzer
    ):
        """测试未加载模型时生成热图抛出错误。"""
        hand_range = [(Card(14, 'h'), Card(14, 'd'))]
        
        with pytest.raises(RuntimeError, match="模型未加载"):
            strategy_analyzer.generate_strategy_heatmap(hand_range=hand_range)


# ============================================================================
# 单元测试：决策解释
# ============================================================================

class TestDecisionExplanation:
    """测试决策解释功能。"""
    
    def test_explain_decision_contains_expected_value(
        self, strategy_analyzer, saved_checkpoint, sample_game_state
    ):
        """测试决策解释包含期望价值。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        explanation = strategy_analyzer.explain_decision(
            sample_game_state, player_id=0
        )
        
        # 验证返回类型
        assert isinstance(explanation, DecisionExplanation)
        
        # 验证包含期望价值
        assert hasattr(explanation, 'expected_value')
        assert isinstance(explanation.expected_value, float)
    
    def test_explain_decision_contains_action_probabilities(
        self, strategy_analyzer, saved_checkpoint, sample_game_state
    ):
        """测试决策解释包含行动概率。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        explanation = strategy_analyzer.explain_decision(
            sample_game_state, player_id=0
        )
        
        # 验证包含行动概率
        assert len(explanation.action_probabilities) > 0
        
        # 验证每个行动概率对象
        for ap in explanation.action_probabilities:
            assert isinstance(ap, ActionProbability)
            assert ap.probability >= 0
    
    def test_explain_decision_contains_recommended_action(
        self, strategy_analyzer, saved_checkpoint, sample_game_state
    ):
        """测试决策解释包含推荐行动。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        explanation = strategy_analyzer.explain_decision(
            sample_game_state, player_id=0
        )
        
        # 验证包含推荐行动
        assert explanation.recommended_action is not None
        assert len(explanation.recommended_action) > 0
    
    def test_explain_decision_contains_reasoning(
        self, strategy_analyzer, saved_checkpoint, sample_game_state
    ):
        """测试决策解释包含决策理由。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        explanation = strategy_analyzer.explain_decision(
            sample_game_state, player_id=0
        )
        
        # 验证包含决策理由
        assert explanation.reasoning is not None
        assert len(explanation.reasoning) > 0
        assert "期望价值" in explanation.reasoning
    
    def test_explain_decision_to_dict(
        self, strategy_analyzer, saved_checkpoint, sample_game_state
    ):
        """测试决策解释转换为字典。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        explanation = strategy_analyzer.explain_decision(
            sample_game_state, player_id=0
        )
        
        result_dict = explanation.to_dict()
        
        assert 'expected_value' in result_dict
        assert 'action_probabilities' in result_dict
        assert 'recommended_action' in result_dict
        assert 'reasoning' in result_dict


# ============================================================================
# 单元测试：多模型比较
# ============================================================================

class TestStrategyComparison:
    """测试多模型比较功能。"""
    
    def test_compare_strategies_basic(
        self, temp_checkpoint_dir, sample_game_state
    ):
        """测试基本的策略比较。"""
        # 创建两个不同的模型并保存
        checkpoint_manager = CheckpointManager(temp_checkpoint_dir)
        
        model1 = PolicyNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=6)
        model2 = PolicyNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=6)
        
        # 修改model2的参数使其不同
        with torch.no_grad():
            for param in model2.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        path1 = checkpoint_manager.save(
            model=model1,
            optimizer=None,
            metadata={'episode_number': 100}
        )
        
        path2 = checkpoint_manager.save(
            model=model2,
            optimizer=None,
            metadata={'episode_number': 200}
        )
        
        # 比较策略
        analyzer = StrategyAnalyzer(checkpoint_dir=temp_checkpoint_dir)
        
        comparison = analyzer.compare_strategies(
            checkpoint_paths={
                'model_1': path1,
                'model_2': path2
            },
            state=sample_game_state,
            player_id=0,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        # 验证返回类型
        assert isinstance(comparison, StrategyComparison)
        
        # 验证包含所有模型
        assert 'model_1' in comparison.strategies
        assert 'model_2' in comparison.strategies
        assert len(comparison.models) == 2
    
    def test_compare_strategies_contains_all_models(
        self, temp_checkpoint_dir, sample_game_state
    ):
        """测试比较结果包含所有模型的策略信息。"""
        checkpoint_manager = CheckpointManager(temp_checkpoint_dir)
        
        # 创建3个模型
        models = {}
        paths = {}
        for i in range(3):
            model = PolicyNetwork(input_dim=370, hidden_dims=[32], action_dim=6)
            path = checkpoint_manager.save(
                model=model,
                optimizer=None,
                metadata={'episode_number': i * 100}
            )
            models[f'model_{i}'] = model
            paths[f'model_{i}'] = path
        
        analyzer = StrategyAnalyzer(checkpoint_dir=temp_checkpoint_dir)
        
        comparison = analyzer.compare_strategies(
            checkpoint_paths=paths,
            state=sample_game_state,
            player_id=0,
            input_dim=370,
            hidden_dims=[32],
            action_dim=6
        )
        
        # 验证所有模型都在结果中
        for model_name in paths.keys():
            assert model_name in comparison.strategies
            # 动态动作配置：返回的动作数量取决于模型的动作维度
            # 6维模型返回5个显示动作（CHECK和CALL合并为CHECK/CALL）
            assert len(comparison.strategies[model_name]) >= 4, f"应至少有4个动作"
    
    def test_compare_strategies_to_dict(
        self, temp_checkpoint_dir, sample_game_state
    ):
        """测试比较结果转换为字典。"""
        checkpoint_manager = CheckpointManager(temp_checkpoint_dir)
        
        model = PolicyNetwork(input_dim=370, hidden_dims=[32], action_dim=6)
        path = checkpoint_manager.save(
            model=model,
            optimizer=None,
            metadata={'episode_number': 100}
        )
        
        analyzer = StrategyAnalyzer(checkpoint_dir=temp_checkpoint_dir)
        
        comparison = analyzer.compare_strategies(
            checkpoint_paths={'test_model': path},
            state=sample_game_state,
            player_id=0,
            input_dim=370,
            hidden_dims=[32],
            action_dim=6
        )
        
        result_dict = comparison.to_dict()
        
        assert 'models' in result_dict
        assert 'strategies' in result_dict
        assert 'timestamp' in result_dict


# ============================================================================
# 单元测试：结果保存
# ============================================================================

class TestResultSaving:
    """测试结果保存功能。"""
    
    def test_save_decision_explanation(
        self, strategy_analyzer, saved_checkpoint, sample_game_state, temp_checkpoint_dir
    ):
        """测试保存决策解释。"""
        strategy_analyzer.load_model(
            checkpoint_path=saved_checkpoint,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        explanation = strategy_analyzer.explain_decision(
            sample_game_state, player_id=0
        )
        
        save_path = Path(temp_checkpoint_dir) / "explanation.json"
        result_path = strategy_analyzer.save_analysis(explanation, save_path)
        
        assert Path(result_path).exists()
    
    def test_save_strategy_comparison(
        self, temp_checkpoint_dir, sample_game_state
    ):
        """测试保存策略比较结果。"""
        checkpoint_manager = CheckpointManager(temp_checkpoint_dir)
        
        model = PolicyNetwork(input_dim=370, hidden_dims=[32], action_dim=6)
        path = checkpoint_manager.save(
            model=model,
            optimizer=None,
            metadata={'episode_number': 100}
        )
        
        analyzer = StrategyAnalyzer(checkpoint_dir=temp_checkpoint_dir)
        
        comparison = analyzer.compare_strategies(
            checkpoint_paths={'test_model': path},
            state=sample_game_state,
            player_id=0,
            input_dim=370,
            hidden_dims=[32],
            action_dim=6
        )
        
        save_path = Path(temp_checkpoint_dir) / "comparison.json"
        result_path = analyzer.save_analysis(comparison, save_path)
        
        assert Path(result_path).exists()



# ============================================================================
# 属性测试
# ============================================================================

# Hypothesis策略：生成有效的扑克牌
@st.composite
def card_strategy(draw):
    """生成有效的扑克牌。"""
    rank = draw(st.integers(min_value=2, max_value=14))
    suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
    return Card(rank, suit)


@st.composite
def unique_hand_strategy(draw):
    """生成两张不同的手牌。"""
    cards = []
    used = set()
    
    while len(cards) < 2:
        rank = draw(st.integers(min_value=2, max_value=14))
        suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
        key = (rank, suit)
        if key not in used:
            used.add(key)
            cards.append(Card(rank, suit))
    
    return (cards[0], cards[1])


@st.composite
def game_state_strategy(draw):
    """生成有效的游戏状态。"""
    # 生成不重复的牌
    used_cards = set()
    
    def get_unique_card():
        while True:
            rank = draw(st.integers(min_value=2, max_value=14))
            suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
            key = (rank, suit)
            if key not in used_cards:
                used_cards.add(key)
                return Card(rank, suit)
    
    # 生成两个玩家的手牌
    hand1 = (get_unique_card(), get_unique_card())
    hand2 = (get_unique_card(), get_unique_card())
    
    # 生成公共牌（0-5张）
    num_community = draw(st.sampled_from([0, 3, 4, 5]))
    community_cards = [get_unique_card() for _ in range(num_community)]
    
    # 确定游戏阶段
    stage_map = {0: GameStage.PREFLOP, 3: GameStage.FLOP, 4: GameStage.TURN, 5: GameStage.RIVER}
    stage = stage_map[num_community]
    
    # 生成筹码和下注
    initial_stack = 1000
    pot = draw(st.integers(min_value=10, max_value=500))
    stack1 = draw(st.integers(min_value=100, max_value=initial_stack))
    stack2 = draw(st.integers(min_value=100, max_value=initial_stack))
    bet1 = draw(st.integers(min_value=0, max_value=min(100, stack1)))
    bet2 = draw(st.integers(min_value=0, max_value=min(100, stack2)))
    
    button_position = draw(st.integers(min_value=0, max_value=1))
    current_player = draw(st.integers(min_value=0, max_value=1))
    
    return GameState(
        player_hands=[hand1, hand2],
        community_cards=community_cards,
        pot=pot,
        player_stacks=[stack1, stack2],
        current_bets=[bet1, bet2],
        button_position=button_position,
        stage=stage,
        action_history=[],
        current_player=current_player
    )


class TestPropertyBasedTests:
    """属性测试类。"""
    
    @pytest.fixture(autouse=True)
    def setup_checkpoint(self, temp_checkpoint_dir):
        """为属性测试设置检查点。"""
        self.temp_dir = temp_checkpoint_dir
        self.checkpoint_manager = CheckpointManager(temp_checkpoint_dir)
        
        # 创建并保存模型
        self.model = PolicyNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=6)
        self.checkpoint_path = self.checkpoint_manager.save(
            model=self.model,
            optimizer=None,
            metadata={'episode_number': 100, 'win_rate': 0.5, 'avg_reward': 0.0}
        )
        
        self.analyzer = StrategyAnalyzer(checkpoint_dir=temp_checkpoint_dir)
    
    # ========================================================================
    # 属性12：模型加载成功性
    # Feature: texas-holdem-ai-training, Property 12: 模型加载成功性
    # 验证加载已保存的检查点不抛出异常
    # **验证需求：4.1**
    # ========================================================================
    @settings(max_examples=100, deadline=None)
    @given(st.integers(min_value=1, max_value=100))
    def test_property_12_model_loading_success(self, episode_number):
        """
        属性12：模型加载成功性
        
        **Feature: texas-holdem-ai-training, Property 12: 模型加载成功性**
        **验证需求：4.1**
        
        对于任何已保存的模型检查点，策略查看器应该能够成功加载模型参数而不出现错误。
        """
        # 创建新模型并保存
        model = PolicyNetwork(input_dim=370, hidden_dims=[64, 32], action_dim=6)
        checkpoint_path = self.checkpoint_manager.save(
            model=model,
            optimizer=None,
            metadata={'episode_number': episode_number}
        )
        
        # 加载模型不应抛出异常
        analyzer = StrategyAnalyzer(checkpoint_dir=self.temp_dir)
        analyzer.load_model(
            checkpoint_path=checkpoint_path,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        # 验证模型已加载
        assert analyzer.is_model_loaded
        assert analyzer._policy_network is not None
    
    # ========================================================================
    # 属性14：策略热图数据完整性
    # Feature: texas-holdem-ai-training, Property 14: 策略热图数据完整性
    # 验证热图覆盖所有请求的手牌组合
    # **验证需求：4.3**
    # ========================================================================
    @settings(max_examples=100, deadline=None)
    @given(st.integers(min_value=1, max_value=10))
    def test_property_14_heatmap_data_completeness(self, num_hands):
        """
        属性14：策略热图数据完整性
        
        **Feature: texas-holdem-ai-training, Property 14: 策略热图数据完整性**
        **验证需求：4.3**
        
        对于任何手牌范围请求，生成的策略热图数据应该覆盖所有请求的手牌组合，
        且每个组合都有对应的行动概率。
        """
        # 加载模型
        self.analyzer.load_model(
            checkpoint_path=self.checkpoint_path,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        # 生成手牌范围（确保不重复）
        hand_range = []
        used_cards = set()
        
        for i in range(num_hands):
            # 生成两张不同的牌
            rank1 = 2 + (i * 2) % 13
            suit1 = ['h', 'd', 'c', 's'][i % 4]
            rank2 = 2 + (i * 2 + 1) % 13
            suit2 = ['h', 'd', 'c', 's'][(i + 1) % 4]
            
            # 确保两张牌不同
            if rank1 == rank2 and suit1 == suit2:
                rank2 = (rank2 % 13) + 2
            
            hand_range.append((Card(rank1, suit1), Card(rank2, suit2)))
        
        # 生成热图
        heatmap = self.analyzer.generate_strategy_heatmap(
            hand_range=hand_range,
            stage=GameStage.PREFLOP
        )
        
        # 验证热图形状
        assert heatmap.shape[0] == num_hands, f"热图行数应为{num_hands}，实际为{heatmap.shape[0]}"
        assert heatmap.shape[1] == 6, f"热图列数应为6，实际为{heatmap.shape[1]}"
        
        # 验证每个手牌组合都有概率数据
        for i in range(num_hands):
            row_sum = np.sum(heatmap[i])
            assert abs(row_sum - 1.0) < 1e-5, f"手牌{i}的概率和应为1，实际为{row_sum}"
            
            # 验证所有概率非负
            for j in range(6):
                assert heatmap[i, j] >= 0, f"概率不应为负: heatmap[{i},{j}]={heatmap[i,j]}"
    
    # ========================================================================
    # 属性15：决策解释完整性
    # Feature: texas-holdem-ai-training, Property 15: 决策解释完整性
    # 验证解释包含期望价值
    # **验证需求：4.4**
    # ========================================================================
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    @given(game_state_strategy())
    def test_property_15_decision_explanation_completeness(self, state):
        """
        属性15：决策解释完整性
        
        **Feature: texas-holdem-ai-training, Property 15: 决策解释完整性**
        **验证需求：4.4**
        
        对于任何游戏状态和行动查询，返回的决策解释应该包含期望价值计算结果。
        """
        # 加载模型
        self.analyzer.load_model(
            checkpoint_path=self.checkpoint_path,
            input_dim=370,
            hidden_dims=[64, 32],
            action_dim=6
        )
        
        # 获取决策解释
        explanation = self.analyzer.explain_decision(state, player_id=0)
        
        # 验证包含期望价值
        assert hasattr(explanation, 'expected_value'), "决策解释应包含expected_value字段"
        assert isinstance(explanation.expected_value, float), "期望价值应为浮点数"
        
        # 验证期望价值在合理范围内
        # 期望价值可能受当前下注影响，所以范围需要考虑下注金额
        max_bet = max(state.current_bets)
        max_ev = state.pot * 2 + max_bet  # 合理的上限
        min_ev = -(state.pot * 2 + max_bet)  # 合理的下限
        assert min_ev <= explanation.expected_value <= max_ev, \
            f"期望价值{explanation.expected_value}超出合理范围[{min_ev}, {max_ev}]"
        
        # 验证包含其他必要字段
        assert explanation.state_description is not None
        assert len(explanation.action_probabilities) > 0
        assert explanation.recommended_action is not None
        assert explanation.reasoning is not None
        assert "期望价值" in explanation.reasoning
    
    # ========================================================================
    # 属性16：多模型比较数据完整性
    # Feature: texas-holdem-ai-training, Property 16: 多模型比较数据完整性
    # 验证比较结果包含所有模型的策略信息
    # **验证需求：4.5**
    # ========================================================================
    @settings(max_examples=100, deadline=None)
    @given(st.integers(min_value=1, max_value=5))
    def test_property_16_multi_model_comparison_completeness(self, num_models):
        """
        属性16：多模型比较数据完整性
        
        **Feature: texas-holdem-ai-training, Property 16: 多模型比较数据完整性**
        **验证需求：4.5**
        
        对于任何多模型比较请求，返回的数据应该包含所有指定模型在相同状态下的策略信息。
        """
        # 创建多个模型并保存
        checkpoint_paths = {}
        for i in range(num_models):
            model = PolicyNetwork(input_dim=370, hidden_dims=[32], action_dim=6)
            # 修改参数使模型不同
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.1 * (i + 1))
            
            path = self.checkpoint_manager.save(
                model=model,
                optimizer=None,
                metadata={'episode_number': i * 100}
            )
            checkpoint_paths[f'model_{i}'] = path
        
        # 创建测试状态
        state = GameState(
            player_hands=[
                (Card(14, 'h'), Card(13, 'h')),
                (Card(7, 'd'), Card(2, 'c'))
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
        
        # 比较策略
        analyzer = StrategyAnalyzer(checkpoint_dir=self.temp_dir)
        comparison = analyzer.compare_strategies(
            checkpoint_paths=checkpoint_paths,
            state=state,
            player_id=0,
            input_dim=370,
            hidden_dims=[32],
            action_dim=6
        )
        
        # 验证包含所有模型
        assert len(comparison.models) == num_models, \
            f"应包含{num_models}个模型，实际为{len(comparison.models)}"
        
        # 验证每个模型都有策略信息
        for model_name in checkpoint_paths.keys():
            assert model_name in comparison.strategies, \
                f"比较结果应包含模型{model_name}的策略"
            
            strategy = comparison.strategies[model_name]
            # 动态动作配置：返回的动作数量取决于模型的动作维度
            # 6维模型返回5个显示动作（CHECK和CALL合并为CHECK/CALL）
            assert len(strategy) >= 4, f"模型{model_name}应至少有4个行动的概率"
            
            # 验证概率分布有效
            total_prob = sum(strategy.values())
            assert abs(total_prob - 1.0) < 1e-5, \
                f"模型{model_name}的概率和应为1，实际为{total_prob}"
            
            for action_name, prob in strategy.items():
                assert prob >= 0, f"模型{model_name}的{action_name}概率不应为负"
