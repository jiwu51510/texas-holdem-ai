"""CFR训练器的单元测试。

测试CFR（反事实遗憾最小化）算法的核心功能：
- 遗憾值计算逻辑
- 策略更新（遗憾值增加时策略概率增加）
- 简单场景下的收敛性（如石头剪刀布游戏）
- 平均策略计算
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

from training.cfr_trainer import CFRTrainer, InfoSet
from models.core import GameState, Action, ActionType, GameStage, Card


class TestInfoSet:
    """测试信息集的创建和比较。"""
    
    def test_info_set_creation(self):
        """测试从游戏状态创建信息集。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        hand1 = (Card(14, 'h'), Card(13, 'h'))  # AK suited
        hand2 = (Card(2, 's'), Card(7, 'd'))    # 72 offsuit
        state = GameState(
            player_hands=[hand1, hand2],
            community_cards=[],
            pot=15,
            player_stacks=[995, 990],
            current_bets=[5, 10],
            button_position=0,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=0
        )
        
        info_set = trainer.get_info_set(state, 0)
        
        # 验证信息集属性
        assert info_set.stage == 'preflop'
        assert len(info_set.hand_key) == 2
        assert info_set.action_history_key == ()
    
    def test_info_set_equality(self):
        """测试相同状态产生相同的信息集。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        hand1 = (Card(14, 'h'), Card(13, 'h'))
        hand2 = (Card(2, 's'), Card(7, 'd'))
        
        state1 = GameState(
            player_hands=[hand1, hand2],
            community_cards=[],
            pot=15,
            player_stacks=[995, 990],
            current_bets=[5, 10],
            button_position=0,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=0
        )
        
        state2 = GameState(
            player_hands=[hand1, hand2],
            community_cards=[],
            pot=15,
            player_stacks=[995, 990],
            current_bets=[5, 10],
            button_position=0,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=0
        )
        
        info_set1 = trainer.get_info_set(state1, 0)
        info_set2 = trainer.get_info_set(state2, 0)
        
        assert info_set1 == info_set2
        assert hash(info_set1) == hash(info_set2)
    
    def test_info_set_different_hands(self):
        """测试不同手牌产生不同的信息集。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        hand1 = (Card(14, 'h'), Card(13, 'h'))
        hand2 = (Card(2, 's'), Card(7, 'd'))
        hand3 = (Card(10, 'c'), Card(10, 'd'))
        
        state1 = GameState(
            player_hands=[hand1, hand2],
            community_cards=[],
            pot=15,
            player_stacks=[995, 990],
            current_bets=[5, 10],
            button_position=0,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=0
        )
        
        state2 = GameState(
            player_hands=[hand3, hand2],
            community_cards=[],
            pot=15,
            player_stacks=[995, 990],
            current_bets=[5, 10],
            button_position=0,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=0
        )
        
        info_set1 = trainer.get_info_set(state1, 0)
        info_set2 = trainer.get_info_set(state2, 0)
        
        assert info_set1 != info_set2


class TestRegretMatching:
    """测试Regret Matching策略计算。"""
    
    def test_initial_strategy_uniform(self):
        """测试初始策略为均匀分布。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        # 创建一个新的信息集
        info_set = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        strategy = trainer.get_strategy(info_set)
        
        # 初始策略应该是均匀分布
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(strategy, expected)
    
    def test_strategy_sums_to_one(self):
        """测试策略概率和为1。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        info_set = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        strategy = trainer.get_strategy(info_set)
        
        assert abs(np.sum(strategy) - 1.0) < 1e-6
    
    def test_positive_regret_increases_probability(self):
        """测试正遗憾值增加对应行动的概率。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        info_set = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        # 设置遗憾值：行动3有最高正遗憾
        trainer.regret_sum[info_set] = np.array([-10.0, 0.0, 5.0, 15.0])
        
        strategy = trainer.get_strategy(info_set)
        
        # 行动3应该有最高概率
        assert strategy[3] > strategy[2]
        assert strategy[3] > strategy[1]
        assert strategy[3] > strategy[0]
        
        # 负遗憾值的行动概率应该为0
        assert strategy[0] == 0.0
        assert strategy[1] == 0.0
    
    def test_all_negative_regrets_uniform(self):
        """测试所有遗憾值为负时使用均匀分布。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        info_set = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        # 设置所有遗憾值为负
        trainer.regret_sum[info_set] = np.array([-10.0, -5.0, -3.0, -1.0])
        
        strategy = trainer.get_strategy(info_set)
        
        # 应该是均匀分布
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(strategy, expected)


class TestRegretComputation:
    """测试遗憾值计算。"""
    
    def test_compute_regrets_basic(self):
        """测试基本遗憾值计算。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        hand1 = (Card(14, 'h'), Card(13, 'h'))
        hand2 = (Card(2, 's'), Card(7, 'd'))
        state = GameState(
            player_hands=[hand1, hand2],
            community_cards=[],
            pot=15,
            player_stacks=[995, 990],
            current_bets=[5, 10],
            button_position=0,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=0
        )
        
        # 各行动的效用值
        action_utilities = {0: -10.0, 1: 0.0, 2: 5.0, 3: 15.0}
        
        regrets = trainer.compute_regrets(state, 0, (1.0, 1.0), action_utilities)
        
        # 验证返回了遗憾值
        assert len(regrets) == 1
        info_set = list(regrets.keys())[0]
        regret_values = regrets[info_set]
        
        # 期望效用 = 0.25 * (-10) + 0.25 * 0 + 0.25 * 5 + 0.25 * 15 = 2.5
        # 遗憾值 = 效用 - 期望效用
        expected_utility = 0.25 * (-10) + 0.25 * 0 + 0.25 * 5 + 0.25 * 15
        expected_regrets = np.array([
            -10.0 - expected_utility,
            0.0 - expected_utility,
            5.0 - expected_utility,
            15.0 - expected_utility
        ])
        
        np.testing.assert_array_almost_equal(regret_values, expected_regrets)
    
    def test_compute_regrets_with_opponent_weight(self):
        """测试带对手权重的遗憾值计算。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        hand1 = (Card(14, 'h'), Card(13, 'h'))
        hand2 = (Card(2, 's'), Card(7, 'd'))
        state = GameState(
            player_hands=[hand1, hand2],
            community_cards=[],
            pot=15,
            player_stacks=[995, 990],
            current_bets=[5, 10],
            button_position=0,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=0
        )
        
        action_utilities = {0: -10.0, 1: 0.0, 2: 5.0, 3: 15.0}
        
        # 对手到达概率为0.5
        regrets = trainer.compute_regrets(state, 0, (1.0, 0.5), action_utilities)
        
        info_set = list(regrets.keys())[0]
        regret_values = regrets[info_set]
        
        # 遗憾值应该乘以对手权重0.5
        expected_utility = 0.25 * (-10) + 0.25 * 0 + 0.25 * 5 + 0.25 * 15
        expected_regrets = np.array([
            (-10.0 - expected_utility) * 0.5,
            (0.0 - expected_utility) * 0.5,
            (5.0 - expected_utility) * 0.5,
            (15.0 - expected_utility) * 0.5
        ])
        
        np.testing.assert_array_almost_equal(regret_values, expected_regrets)


class TestStrategyUpdate:
    """测试策略更新。"""
    
    def test_update_strategy_accumulates_regrets(self):
        """测试策略更新累积遗憾值。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        info_set = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        # 第一次更新
        regrets1 = {info_set: np.array([1.0, 2.0, 3.0, 4.0])}
        trainer.update_strategy(regrets1)
        
        # 第二次更新
        regrets2 = {info_set: np.array([1.0, 1.0, 1.0, 1.0])}
        trainer.update_strategy(regrets2)
        
        # 遗憾值应该累积
        expected = np.array([2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(trainer.regret_sum[info_set], expected)
    
    def test_update_strategy_increments_iterations(self):
        """测试策略更新增加迭代计数。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        info_set = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        assert trainer.iterations == 0
        
        regrets = {info_set: np.array([1.0, 2.0, 3.0, 4.0])}
        trainer.update_strategy(regrets)
        
        assert trainer.iterations == 1
        
        trainer.update_strategy(regrets)
        
        assert trainer.iterations == 2
    
    def test_update_strategy_accumulates_strategy_sum(self):
        """测试策略更新累积策略权重。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        info_set = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        # 初始策略是均匀分布
        regrets = {info_set: np.array([0.0, 0.0, 0.0, 0.0])}
        trainer.update_strategy(regrets)
        
        # 策略累积应该是均匀分布
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(trainer.strategy_sum[info_set], expected)


class TestAverageStrategy:
    """测试平均策略计算。"""
    
    def test_average_strategy_single_iteration(self):
        """测试单次迭代后的平均策略。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        info_set = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        # 设置策略累积
        trainer.strategy_sum[info_set] = np.array([1.0, 2.0, 3.0, 4.0])
        
        avg_strategy = trainer.get_average_strategy(info_set)
        
        # 平均策略应该是归一化的策略累积
        expected = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_almost_equal(avg_strategy[info_set], expected)
    
    def test_average_strategy_unknown_info_set(self):
        """测试未知信息集返回均匀分布。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        info_set = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        avg_strategy = trainer.get_average_strategy(info_set)
        
        # 未知信息集应该返回均匀分布
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(avg_strategy[info_set], expected)
    
    def test_average_strategy_all_info_sets(self):
        """测试获取所有信息集的平均策略。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        info_set1 = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        info_set2 = InfoSet(
            hand_key=((10, 'c'), (10, 'd')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        trainer.strategy_sum[info_set1] = np.array([1.0, 2.0, 3.0, 4.0])
        trainer.strategy_sum[info_set2] = np.array([4.0, 3.0, 2.0, 1.0])
        
        avg_strategies = trainer.get_average_strategy()
        
        assert len(avg_strategies) == 2
        assert info_set1 in avg_strategies
        assert info_set2 in avg_strategies


class TestRockPaperScissorsConvergence:
    """测试CFR在石头剪刀布游戏中的收敛性。
    
    石头剪刀布是一个简单的零和博弈，纳什均衡是均匀分布（1/3, 1/3, 1/3）。
    CFR算法应该收敛到这个均衡。
    """
    
    def test_rps_convergence(self):
        """测试石头剪刀布游戏的收敛性。"""
        # 简化的CFR实现用于石头剪刀布
        num_actions = 3  # 石头、剪刀、布
        regret_sum = np.zeros(num_actions)
        strategy_sum = np.zeros(num_actions)
        
        # 收益矩阵：行是玩家1的行动，列是玩家2的行动
        # 0=石头, 1=剪刀, 2=布
        # 石头赢剪刀，剪刀赢布，布赢石头
        payoff_matrix = np.array([
            [0, 1, -1],   # 石头 vs 石头/剪刀/布
            [-1, 0, 1],   # 剪刀 vs 石头/剪刀/布
            [1, -1, 0]    # 布 vs 石头/剪刀/布
        ])
        
        def get_strategy(regrets):
            """使用Regret Matching获取策略。"""
            positive_regrets = np.maximum(regrets, 0)
            total = np.sum(positive_regrets)
            if total > 0:
                return positive_regrets / total
            else:
                return np.ones(num_actions) / num_actions
        
        # 运行CFR迭代
        num_iterations = 10000
        for _ in range(num_iterations):
            # 获取当前策略
            strategy = get_strategy(regret_sum)
            strategy_sum += strategy
            
            # 计算各行动的期望效用（假设对手也使用相同策略）
            action_utilities = np.zeros(num_actions)
            for a in range(num_actions):
                for opp_a in range(num_actions):
                    action_utilities[a] += strategy[opp_a] * payoff_matrix[a][opp_a]
            
            # 计算期望效用
            expected_utility = np.dot(strategy, action_utilities)
            
            # 计算遗憾值
            regrets = action_utilities - expected_utility
            regret_sum += regrets
        
        # 计算平均策略
        avg_strategy = strategy_sum / np.sum(strategy_sum)
        
        # 验证收敛到均匀分布（误差在5%以内）
        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(avg_strategy, expected, decimal=1)
    
    def test_cfr_trainer_strategy_update_convergence(self):
        """测试CFRTrainer的策略更新在简单场景下的行为。"""
        trainer = CFRTrainer(num_actions=3, initial_stack=1000)
        
        info_set = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        # 模拟多次迭代
        for i in range(100):
            # 获取当前策略
            strategy = trainer.get_strategy(info_set)
            
            # 模拟对称的遗憾值（类似石头剪刀布）
            # 每个行动的遗憾值取决于当前策略
            regrets = np.array([
                (1 - strategy[0]) - strategy[0],  # 如果选择行动0太多，遗憾值为负
                (1 - strategy[1]) - strategy[1],
                (1 - strategy[2]) - strategy[2]
            ])
            
            trainer.update_strategy({info_set: regrets})
        
        # 获取平均策略
        avg_strategy = trainer.get_average_strategy(info_set)
        
        # 验证策略和为1
        assert abs(np.sum(avg_strategy[info_set]) - 1.0) < 1e-6


class TestTrainerReset:
    """测试训练器重置功能。"""
    
    def test_reset_clears_all_data(self):
        """测试重置清除所有数据。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        info_set = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        # 添加一些数据
        trainer.regret_sum[info_set] = np.array([1.0, 2.0, 3.0, 4.0])
        trainer.strategy_sum[info_set] = np.array([1.0, 2.0, 3.0, 4.0])
        trainer.iterations = 10
        
        # 重置
        trainer.reset()
        
        # 验证所有数据被清除
        assert len(trainer.regret_sum) == 0
        assert len(trainer.strategy_sum) == 0
        assert trainer.iterations == 0


class TestActionConversion:
    """测试行动转换功能。"""
    
    def test_action_to_index(self):
        """测试行动到索引的转换。"""
        trainer = CFRTrainer(num_actions=5, initial_stack=1000)
        
        assert trainer._action_to_index(Action(ActionType.FOLD)) == 0
        assert trainer._action_to_index(Action(ActionType.CHECK)) == 1
        assert trainer._action_to_index(Action(ActionType.CALL)) == 2
        assert trainer._action_to_index(Action(ActionType.RAISE_SMALL, 50)) == 3
        assert trainer._action_to_index(Action(ActionType.RAISE_BIG, 100)) == 4
        # 向后兼容：RAISE 映射到 RAISE_SMALL 的索引
        assert trainer._action_to_index(Action(ActionType.RAISE, 100)) == 3
    
    def test_index_to_action_type(self):
        """测试索引到行动类型的转换。"""
        trainer = CFRTrainer(num_actions=5, initial_stack=1000)
        
        assert trainer._index_to_action_type(0) == ActionType.FOLD
        assert trainer._index_to_action_type(1) == ActionType.CHECK
        assert trainer._index_to_action_type(2) == ActionType.CALL
        assert trainer._index_to_action_type(3) == ActionType.RAISE_SMALL
        assert trainer._index_to_action_type(4) == ActionType.RAISE_BIG


class TestGetActionFromStrategy:
    """测试从策略中选择行动。"""
    
    def test_get_action_from_strategy(self):
        """测试从策略概率分布中选择行动。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        # 创建一个确定性策略（只选择RAISE）
        strategy = np.array([0.0, 0.0, 0.0, 1.0])
        
        legal_actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
            Action(ActionType.RAISE, 100)
        ]
        
        # 多次选择，应该总是选择RAISE
        for _ in range(10):
            action = trainer.get_action_from_strategy(strategy, legal_actions)
            assert action.action_type == ActionType.RAISE
    
    def test_get_action_normalizes_probabilities(self):
        """测试选择行动时正确归一化概率。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        # 策略中只有FOLD和RAISE有概率
        strategy = np.array([0.5, 0.0, 0.0, 0.5])
        
        # 但合法行动只有FOLD和CALL
        legal_actions = [
            Action(ActionType.FOLD),
            Action(ActionType.CALL)
        ]
        
        # 应该能正常选择（FOLD概率为1，因为CALL的策略概率为0）
        action = trainer.get_action_from_strategy(strategy, legal_actions)
        assert action.action_type == ActionType.FOLD


class TestNumInfoSets:
    """测试信息集数量统计。"""
    
    def test_get_num_info_sets(self):
        """测试获取信息集数量。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        assert trainer.get_num_info_sets() == 0
        
        # 添加一些信息集
        info_set1 = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        info_set2 = InfoSet(
            hand_key=((10, 'c'), (10, 'd')),
            community_key=(),
            stage='preflop',
            action_history_key=(),
            pot_ratio=0
        )
        
        trainer.regret_sum[info_set1] = np.zeros(4)
        trainer.regret_sum[info_set2] = np.zeros(4)
        
        assert trainer.get_num_info_sets() == 2


class TestCFRGuidedTarget:
    """测试CFR引导目标策略生成。"""
    
    def test_cfr_guided_target_uniform_initial(self):
        """测试初始状态下返回均匀分布。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        hand1 = (Card(14, 'h'), Card(13, 'h'))
        hand2 = (Card(2, 's'), Card(7, 'd'))
        state = GameState(
            player_hands=[hand1, hand2],
            community_cards=[],
            pot=30,
            player_stacks=[985, 985],
            current_bets=[15, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=0
        )
        
        legal_indices = [0, 2, 3]  # FOLD, CALL, RAISE
        target = trainer.get_cfr_guided_target(state, 0, legal_indices)
        
        # 应该是均匀分布（只在合法行动上）
        assert abs(np.sum(target) - 1.0) < 1e-6
        assert target[1] == 0  # CHECK不合法
        assert abs(target[0] - 1/3) < 1e-6
        assert abs(target[2] - 1/3) < 1e-6
        assert abs(target[3] - 1/3) < 1e-6
    
    def test_cfr_guided_target_sums_to_one(self):
        """测试目标策略概率和为1。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        hand1 = (Card(14, 'h'), Card(13, 'h'))
        hand2 = (Card(2, 's'), Card(7, 'd'))
        state = GameState(
            player_hands=[hand1, hand2],
            community_cards=[],
            pot=30,
            player_stacks=[985, 985],
            current_bets=[15, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=0
        )
        
        # 测试不同的合法行动组合
        for legal_indices in [[0, 2, 3], [0, 1], [1, 3], [0, 1, 2, 3]]:
            target = trainer.get_cfr_guided_target(state, 0, legal_indices)
            assert abs(np.sum(target) - 1.0) < 1e-6


class TestComputeAndUpdateRegrets:
    """测试反事实遗憾值计算和更新。"""
    
    def test_fold_regret_positive_when_losing(self):
        """测试输钱时FOLD的遗憾值为正。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        hand1 = (Card(7, 'h'), Card(2, 's'))  # 72o - 差牌
        hand2 = (Card(14, 's'), Card(14, 'd'))
        state = GameState(
            player_hands=[hand1, hand2],
            community_cards=[],
            pot=30,
            player_stacks=[985, 985],
            current_bets=[15, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=0
        )
        
        legal_indices = [0, 2, 3]  # FOLD, CALL, RAISE
        
        # 模拟输钱的情况
        trainer.compute_and_update_regrets(state, 0, 2, -100.0, legal_indices)
        
        info_set = trainer.get_info_set(state, 0)
        regrets = trainer.regret_sum[info_set]
        
        # FOLD的遗憾值应该为正（因为输钱了，应该弃牌）
        assert regrets[0] > 0, "FOLD的遗憾值应该为正"
    
    def test_fold_regret_negative_when_winning(self):
        """测试赢钱时FOLD的遗憾值为负。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        hand1 = (Card(14, 'h'), Card(14, 's'))  # AA - 好牌
        hand2 = (Card(2, 's'), Card(7, 'd'))
        state = GameState(
            player_hands=[hand1, hand2],
            community_cards=[],
            pot=30,
            player_stacks=[985, 985],
            current_bets=[15, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=0
        )
        
        legal_indices = [0, 2, 3]  # FOLD, CALL, RAISE
        
        # 模拟赢钱的情况
        trainer.compute_and_update_regrets(state, 0, 2, 100.0, legal_indices)
        
        info_set = trainer.get_info_set(state, 0)
        regrets = trainer.regret_sum[info_set]
        
        # FOLD的遗憾值应该为负（因为赢钱了，不应该弃牌）
        assert regrets[0] < 0, "FOLD的遗憾值应该为负"
    
    def test_cfr_updates_strategy_after_regrets(self):
        """测试遗憾值更新后策略会改变。"""
        trainer = CFRTrainer(num_actions=4, initial_stack=1000)
        
        hand1 = (Card(7, 'h'), Card(2, 's'))
        hand2 = (Card(14, 's'), Card(14, 'd'))
        state = GameState(
            player_hands=[hand1, hand2],
            community_cards=[],
            pot=30,
            player_stacks=[985, 985],
            current_bets=[15, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            action_history=[],
            current_player=0
        )
        
        legal_indices = [0, 2, 3]
        
        # 多次模拟输钱
        for _ in range(10):
            trainer.compute_and_update_regrets(state, 0, 2, -100.0, legal_indices)
        
        # 获取更新后的策略
        info_set = trainer.get_info_set(state, 0)
        strategy = trainer.get_strategy(info_set)
        
        # FOLD的概率应该增加
        assert strategy[0] > 0.5, "多次输钱后FOLD概率应该增加"
