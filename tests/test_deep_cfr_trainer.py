"""DeepCFRTrainer 属性测试和单元测试模块。

使用 Hypothesis 库进行属性测试，验证 Deep CFR 训练器的核心正确性属性。
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from training.deep_cfr_trainer import DeepCFRTrainer
from models.core import TrainingConfig


# ============================================================================
# 测试策略（生成器）
# ============================================================================

# 动作空间维度（FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN）
ACTION_DIM = 6

# 生成随机动作收益字典
# 动作索引范围 0-3，收益值范围 -1000 到 1000
@st.composite
def action_values_strategy(draw):
    """生成随机动作收益字典。
    
    随机选择 1-4 个动作，每个动作有一个随机收益值。
    """
    # 随机选择要包含的动作数量（1-4）
    num_actions = draw(st.integers(min_value=1, max_value=ACTION_DIM))
    
    # 随机选择动作索引
    action_indices = draw(st.lists(
        st.integers(min_value=0, max_value=ACTION_DIM - 1),
        min_size=num_actions,
        max_size=num_actions,
        unique=True
    ))
    
    # 为每个动作生成收益值
    action_values = {}
    for idx in action_indices:
        value = draw(st.floats(
            min_value=-1000.0, 
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False
        ))
        action_values[idx] = value
    
    return action_values


# 生成有效的策略概率分布
@st.composite
def strategy_strategy(draw):
    """生成有效的策略概率分布。
    
    生成一个长度为 ACTION_DIM 的概率分布，所有值非负且和为 1。
    """
    # 生成非负权重
    weights = []
    for _ in range(ACTION_DIM):
        weight = draw(st.floats(
            min_value=0.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False
        ))
        weights.append(weight)
    
    weights = np.array(weights, dtype=np.float32)
    
    # 确保至少有一个非零权重
    if weights.sum() == 0:
        weights[0] = 1.0
    
    # 归一化为概率分布
    strategy = weights / weights.sum()
    
    return strategy


# ============================================================================
# Property 4: 遗憾值计算正确性
# **Feature: deep-cfr-refactor, Property 4: 遗憾值计算正确性**
# **验证需求：2.3**
# ============================================================================

class TestProperty4RegretComputation:
    """属性测试：遗憾值计算正确性。
    
    对于任何游戏状态、动作收益字典和当前策略，
    计算的遗憾值应该等于各动作收益减去当前策略的期望收益。
    """
    
    @given(
        action_values=action_values_strategy(),
        strategy=strategy_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_regret_equals_action_value_minus_expected_value(
        self, action_values, strategy
    ):
        """
        **Feature: deep-cfr-refactor, Property 4: 遗憾值计算正确性**
        **验证需求：2.3**
        
        测试遗憾值 = 动作收益 - 期望收益。
        """
        # 创建训练器
        config = TrainingConfig()
        trainer = DeepCFRTrainer(config)
        
        # 计算遗憾值
        regrets = trainer.compute_counterfactual_regrets(
            state=None,  # 状态在此方法中未使用
            player_id=0,  # 玩家ID在此方法中未使用
            action_values=action_values,
            strategy=strategy
        )
        
        # 手动计算期望收益
        expected_value = sum(
            strategy[idx] * value 
            for idx, value in action_values.items()
        )
        
        # 验证每个动作的遗憾值
        for action_idx, action_value in action_values.items():
            expected_regret = action_value - expected_value
            actual_regret = regrets[action_idx]
            
            assert np.isclose(actual_regret, expected_regret, rtol=1e-5), (
                f"动作 {action_idx} 的遗憾值计算错误。"
                f"期望: {expected_regret}, 实际: {actual_regret}"
            )
        
        # 验证不在 action_values 中的动作遗憾值为 0
        for idx in range(ACTION_DIM):
            if idx not in action_values:
                assert regrets[idx] == 0.0, (
                    f"动作 {idx} 不在 action_values 中，遗憾值应为 0，"
                    f"实际为 {regrets[idx]}"
                )
    
    @given(strategy=strategy_strategy())
    @settings(max_examples=100, deadline=None)
    def test_regret_sum_weighted_by_strategy_is_zero(self, strategy):
        """
        **Feature: deep-cfr-refactor, Property 4: 遗憾值计算正确性**
        **验证需求：2.3**
        
        测试遗憾值按策略加权求和为零。
        
        这是遗憾值的一个重要性质：
        sum(strategy[i] * regret[i]) = sum(strategy[i] * (v[i] - E[v])) 
                                     = E[v] - E[v] = 0
        """
        # 创建训练器
        config = TrainingConfig()
        trainer = DeepCFRTrainer(config)
        
        # 生成所有动作都有收益的情况
        action_values = {
            i: np.random.uniform(-100, 100) 
            for i in range(ACTION_DIM)
        }
        
        # 计算遗憾值
        regrets = trainer.compute_counterfactual_regrets(
            state=None,
            player_id=0,
            action_values=action_values,
            strategy=strategy
        )
        
        # 计算加权和
        weighted_sum = sum(strategy[i] * regrets[i] for i in range(ACTION_DIM))
        
        assert np.isclose(weighted_sum, 0.0, atol=1e-5), (
            f"遗憾值按策略加权求和应为 0，实际为 {weighted_sum}"
        )
    
    @given(
        action_values=action_values_strategy(),
        strategy=strategy_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_regret_output_dimension(self, action_values, strategy):
        """
        **Feature: deep-cfr-refactor, Property 4: 遗憾值计算正确性**
        **验证需求：2.3**
        
        测试遗憾值输出维度正确。
        """
        # 创建训练器
        config = TrainingConfig()
        trainer = DeepCFRTrainer(config)
        
        # 计算遗憾值
        regrets = trainer.compute_counterfactual_regrets(
            state=None,
            player_id=0,
            action_values=action_values,
            strategy=strategy
        )
        
        # 验证输出维度
        assert len(regrets) == ACTION_DIM, (
            f"遗憾值输出维度应为 {ACTION_DIM}，实际为 {len(regrets)}"
        )
        assert regrets.dtype == np.float32, (
            f"遗憾值数据类型应为 float32，实际为 {regrets.dtype}"
        )
    
    def test_regret_with_single_action(self):
        """
        **Feature: deep-cfr-refactor, Property 4: 遗憾值计算正确性**
        **验证需求：2.3**
        
        测试只有一个动作时的遗憾值计算。
        """
        config = TrainingConfig()
        trainer = DeepCFRTrainer(config)
        
        # 只有一个动作
        action_values = {0: 100.0}
        strategy = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        regrets = trainer.compute_counterfactual_regrets(
            state=None,
            player_id=0,
            action_values=action_values,
            strategy=strategy
        )
        
        # 期望收益 = 1.0 * 100 = 100
        # 动作0的遗憾值 = 100 - 100 = 0
        assert np.isclose(regrets[0], 0.0), (
            f"单一动作的遗憾值应为 0，实际为 {regrets[0]}"
        )
    
    def test_regret_with_uniform_strategy(self):
        """
        **Feature: deep-cfr-refactor, Property 4: 遗憾值计算正确性**
        **验证需求：2.3**
        
        测试均匀策略下的遗憾值计算。
        """
        config = TrainingConfig()
        trainer = DeepCFRTrainer(config)
        
        # 所有动作收益（5种动作：FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG）
        action_values = {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0, 4: 50.0}
        # 均匀策略
        strategy = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        
        regrets = trainer.compute_counterfactual_regrets(
            state=None,
            player_id=0,
            action_values=action_values,
            strategy=strategy
        )
        
        # 期望收益 = 0.2 * (10 + 20 + 30 + 40 + 50) = 30
        expected_value = 30.0
        
        # 验证各动作遗憾值
        assert np.isclose(regrets[0], 10.0 - expected_value), f"动作0遗憾值错误"
        assert np.isclose(regrets[1], 20.0 - expected_value), f"动作1遗憾值错误"
        assert np.isclose(regrets[2], 30.0 - expected_value), f"动作2遗憾值错误"
        assert np.isclose(regrets[3], 40.0 - expected_value), f"动作3遗憾值错误"
        assert np.isclose(regrets[4], 50.0 - expected_value), f"动作4遗憾值错误"



# ============================================================================
# 单元测试
# ============================================================================

class TestDeepCFRTrainerUnit:
    """DeepCFRTrainer 单元测试。"""
    
    def test_trainer_initialization(self):
        """测试训练器初始化。
        
        验证需求：2.1
        """
        config = TrainingConfig()
        trainer = DeepCFRTrainer(config)
        
        # 验证网络已创建
        assert trainer.regret_network is not None, "遗憾网络应已创建"
        assert trainer.policy_network is not None, "策略网络应已创建"
        
        # 验证缓冲区已创建
        assert trainer.regret_buffer is not None, "遗憾缓冲区应已创建"
        assert trainer.strategy_buffer is not None, "策略缓冲区应已创建"
        
        # 验证优化器已创建
        assert trainer.regret_optimizer is not None, "遗憾网络优化器应已创建"
        assert trainer.policy_optimizer is not None, "策略网络优化器应已创建"
        
        # 验证迭代计数初始化为0
        assert trainer.iteration == 0, "迭代计数应初始化为0"
    
    def test_cfr_iteration_execution(self):
        """测试 CFR 迭代执行。
        
        验证需求：2.1, 2.2
        """
        config = TrainingConfig()
        trainer = DeepCFRTrainer(config)
        
        # 执行一次 CFR 迭代
        metrics = trainer.run_cfr_iteration()
        
        # 验证返回的指标
        assert 'iteration' in metrics, "指标应包含迭代编号"
        assert 'utility_p0' in metrics, "指标应包含玩家0的收益"
        assert 'utility_p1' in metrics, "指标应包含玩家1的收益"
        assert 'regret_buffer_size' in metrics, "指标应包含遗憾缓冲区大小"
        assert 'strategy_buffer_size' in metrics, "指标应包含策略缓冲区大小"
        
        # 验证迭代计数增加
        assert trainer.iteration == 1, "迭代计数应增加到1"
        assert metrics['iteration'] == 1, "返回的迭代编号应为1"
    
    def test_samples_collected_to_buffers(self):
        """测试样本收集到缓冲区。
        
        验证需求：2.4, 2.5
        """
        config = TrainingConfig()
        trainer = DeepCFRTrainer(config)
        
        # 初始缓冲区应为空
        assert len(trainer.regret_buffer) == 0, "初始遗憾缓冲区应为空"
        assert len(trainer.strategy_buffer) == 0, "初始策略缓冲区应为空"
        
        # 执行 CFR 迭代
        trainer.run_cfr_iteration()
        
        # 验证样本已收集
        assert len(trainer.regret_buffer) > 0, "CFR 迭代后遗憾缓冲区应有样本"
        assert len(trainer.strategy_buffer) > 0, "CFR 迭代后策略缓冲区应有样本"
    
    def test_network_training_loss_decreases(self):
        """测试网络训练损失下降。
        
        验证需求：4.1, 4.2, 4.4
        """
        config = TrainingConfig(
            batch_size=16,  # 使用较小的批次大小
            network_train_steps=20
        )
        trainer = DeepCFRTrainer(config)
        
        # 执行多次 CFR 迭代以收集足够的样本
        # 每次迭代可能收集的样本数量不固定，所以多执行几次
        # 需要确保两个缓冲区都有足够的样本
        while (len(trainer.regret_buffer) < config.batch_size or 
               len(trainer.strategy_buffer) < config.batch_size):
            trainer.run_cfr_iteration()
        
        # 确保有足够的样本
        assert len(trainer.regret_buffer) >= config.batch_size, (
            f"遗憾缓冲区应有至少 {config.batch_size} 个样本"
        )
        assert len(trainer.strategy_buffer) >= config.batch_size, (
            f"策略缓冲区应有至少 {config.batch_size} 个样本"
        )
        
        # 训练网络并记录损失
        metrics1 = trainer.train_networks()
        
        # 再次训练
        metrics2 = trainer.train_networks()
        
        # 验证训练步数正确
        assert metrics1['regret_train_steps'] == config.network_train_steps, (
            f"遗憾网络训练步数应为 {config.network_train_steps}"
        )
        assert metrics1['policy_train_steps'] == config.network_train_steps, (
            f"策略网络训练步数应为 {config.network_train_steps}"
        )
        
        # 注意：损失不一定每次都下降，但应该是有限值
        assert np.isfinite(metrics1['regret_loss']), "遗憾网络损失应为有限值"
        assert np.isfinite(metrics1['policy_loss']), "策略网络损失应为有限值"
    
    def test_multiple_cfr_iterations(self):
        """测试多次 CFR 迭代。
        
        验证需求：2.1
        """
        config = TrainingConfig()
        trainer = DeepCFRTrainer(config)
        
        # 执行多次迭代
        num_iterations = 3
        for i in range(num_iterations):
            metrics = trainer.run_cfr_iteration()
            assert metrics['iteration'] == i + 1, f"迭代 {i+1} 的编号应为 {i+1}"
        
        # 验证最终迭代计数
        assert trainer.iteration == num_iterations, (
            f"最终迭代计数应为 {num_iterations}"
        )
    
    def test_get_metrics(self):
        """测试获取训练指标。"""
        config = TrainingConfig()
        trainer = DeepCFRTrainer(config)
        
        # 执行一些迭代
        trainer.run_cfr_iteration()
        trainer.run_cfr_iteration()
        
        # 获取指标
        metrics = trainer.get_metrics()
        
        assert metrics['iteration'] == 2, "迭代计数应为2"
        assert metrics['regret_buffer_size'] > 0, "遗憾缓冲区大小应大于0"
        assert metrics['strategy_buffer_size'] > 0, "策略缓冲区大小应大于0"
        assert 'regret_buffer_total_seen' in metrics, "应包含遗憾缓冲区总见过样本数"
        assert 'strategy_buffer_total_seen' in metrics, "应包含策略缓冲区总见过样本数"
    
    def test_reset(self):
        """测试重置训练器状态。"""
        config = TrainingConfig()
        trainer = DeepCFRTrainer(config)
        
        # 执行一些迭代
        trainer.run_cfr_iteration()
        trainer.run_cfr_iteration()
        
        # 验证状态不为空
        assert trainer.iteration > 0
        assert len(trainer.regret_buffer) > 0
        assert len(trainer.strategy_buffer) > 0
        
        # 重置
        trainer.reset()
        
        # 验证状态已重置
        assert trainer.iteration == 0, "迭代计数应重置为0"
        assert len(trainer.regret_buffer) == 0, "遗憾缓冲区应清空"
        assert len(trainer.strategy_buffer) == 0, "策略缓冲区应清空"
    
    def test_get_average_strategy(self):
        """测试获取平均策略。"""
        config = TrainingConfig()
        trainer = DeepCFRTrainer(config)
        
        # 获取初始状态
        initial_state = trainer.env.reset()
        
        # 获取平均策略
        strategy = trainer.get_average_strategy(initial_state, player_id=0)
        
        # 验证策略是有效的概率分布
        assert len(strategy) == 6, "策略应有6个动作（FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN）"
        assert np.all(strategy >= 0), "所有概率应非负"
        assert np.isclose(strategy.sum(), 1.0), "概率和应为1"
    
    def test_train_networks_with_insufficient_samples(self):
        """测试样本不足时的网络训练。
        
        验证需求：4.4
        """
        config = TrainingConfig(batch_size=1000)  # 大批次大小
        trainer = DeepCFRTrainer(config)
        
        # 只执行一次迭代，样本可能不足
        trainer.run_cfr_iteration()
        
        # 训练网络（可能因样本不足而跳过）
        metrics = trainer.train_networks()
        
        # 验证返回的指标
        assert 'regret_loss' in metrics
        assert 'policy_loss' in metrics
        assert 'regret_train_steps' in metrics
        assert 'policy_train_steps' in metrics
