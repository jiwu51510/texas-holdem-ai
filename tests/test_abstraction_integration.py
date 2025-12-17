"""抽象模块集成测试。

测试卡牌抽象模块与训练系统的集成：
- 使用抽象的训练流程
- 抽象配置变化检测功能
- 属性54：抽象配置变化检测
"""

import tempfile
import os
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from models.core import TrainingConfig, GameState, Card, GameStage
from abstraction.data_classes import AbstractionConfig, AbstractionResult
from abstraction.card_abstraction import CardAbstraction
from environment.state_encoder import StateEncoder
from training.cfr_trainer import CFRTrainer, InfoSet


class TestAbstractionConfigIntegration:
    """测试抽象配置与TrainingConfig的集成。"""
    
    def test_training_config_with_abstraction_disabled(self):
        """测试禁用抽象的训练配置。"""
        config = TrainingConfig(use_abstraction=False)
        assert config.use_abstraction is False
        assert config.abstraction_path == ""
        assert config.abstraction_config == {}
    
    def test_training_config_with_abstraction_enabled(self):
        """测试启用抽象的训练配置。"""
        abstraction_config = {
            'preflop_buckets': 169,
            'flop_buckets': 1000,
            'turn_buckets': 1000,
            'river_buckets': 1000,
            'use_potential_aware': True
        }
        config = TrainingConfig(
            use_abstraction=True,
            abstraction_path="/path/to/abstraction",
            abstraction_config=abstraction_config
        )
        assert config.use_abstraction is True
        assert config.abstraction_path == "/path/to/abstraction"
        assert config.abstraction_config == abstraction_config
    
    def test_training_config_validates_abstraction_config(self):
        """测试训练配置验证抽象配置参数。"""
        # 无效的桶数量（负数）
        with pytest.raises(ValueError):
            TrainingConfig(
                use_abstraction=True,
                abstraction_config={'flop_buckets': -100}
            )
        
        # 无效的翻牌前桶数（超过169）
        with pytest.raises(ValueError):
            TrainingConfig(
                use_abstraction=True,
                abstraction_config={'preflop_buckets': 200}
            )
    
    def test_training_config_accepts_valid_abstraction_config(self):
        """测试训练配置接受有效的抽象配置。"""
        config = TrainingConfig(
            use_abstraction=True,
            abstraction_config={
                'preflop_buckets': 100,
                'flop_buckets': 500,
                'turn_buckets': 500,
                'river_buckets': 500,
                'equity_bins': 25,
                'kmeans_restarts': 10,
                'use_potential_aware': False,
                'random_seed': 123
            }
        )
        assert config.abstraction_config['preflop_buckets'] == 100
        assert config.abstraction_config['use_potential_aware'] is False


class TestStateEncoderWithAbstraction:
    """测试状态编码器与卡牌抽象的集成。"""
    
    def _create_mock_abstraction(self) -> CardAbstraction:
        """创建一个模拟的卡牌抽象对象。"""
        config = AbstractionConfig(
            preflop_buckets=169,
            flop_buckets=100,
            turn_buckets=100,
            river_buckets=100
        )
        abstraction = CardAbstraction(config)
        
        # 创建模拟的抽象结果
        abstraction.result = AbstractionResult(
            config=config,
            preflop_mapping=np.arange(169),
            flop_mapping=np.zeros(1000, dtype=np.int32),
            turn_mapping=np.zeros(1000, dtype=np.int32),
            river_mapping=np.zeros(1000, dtype=np.int32),
            wcss={'flop': 0.0, 'turn': 0.0, 'river': 0.0},
            generation_time=0.0
        )
        
        return abstraction
    
    def test_state_encoder_without_abstraction(self):
        """测试不使用抽象的状态编码器。"""
        encoder = StateEncoder(use_abstraction=True, card_abstraction=None)
        assert encoder.card_abstraction is None
        assert not encoder.has_card_abstraction()
    
    def test_state_encoder_with_abstraction(self):
        """测试使用抽象的状态编码器。"""
        abstraction = self._create_mock_abstraction()
        encoder = StateEncoder(use_abstraction=True, card_abstraction=abstraction)
        
        assert encoder.card_abstraction is not None
        assert encoder.has_card_abstraction()
    
    def test_set_card_abstraction(self):
        """测试设置卡牌抽象。"""
        encoder = StateEncoder()
        assert not encoder.has_card_abstraction()
        
        abstraction = self._create_mock_abstraction()
        encoder.set_card_abstraction(abstraction)
        
        assert encoder.has_card_abstraction()
    
    def test_encode_with_abstraction_requires_loaded_abstraction(self):
        """测试encode_with_abstraction需要已加载的抽象。"""
        encoder = StateEncoder()
        
        # 创建测试状态
        state = GameState(
            player_hands=[
                (Card(14, 'h'), Card(13, 'h')),
                (Card(12, 'd'), Card(11, 'd'))
            ],
            community_cards=[Card(10, 'c'), Card(9, 's'), Card(8, 'h')],
            pot=100,
            player_stacks=[900, 900],
            current_bets=[50, 50],
            button_position=0,
            stage=GameStage.FLOP,
            current_player=0
        )
        
        # 未设置抽象时应该抛出错误
        with pytest.raises(ValueError, match="卡牌抽象未设置"):
            encoder.encode_with_abstraction(state, 0)
    
    def test_encode_with_abstraction_returns_correct_dimension(self):
        """测试encode_with_abstraction返回正确的维度。"""
        abstraction = self._create_mock_abstraction()
        encoder = StateEncoder(card_abstraction=abstraction)
        
        # 创建测试状态
        state = GameState(
            player_hands=[
                (Card(14, 'h'), Card(13, 'h')),
                (Card(12, 'd'), Card(11, 'd'))
            ],
            community_cards=[Card(10, 'c'), Card(9, 's'), Card(8, 'h')],
            pot=100,
            player_stacks=[900, 900],
            current_bets=[50, 50],
            button_position=0,
            stage=GameStage.FLOP,
            current_player=0
        )
        
        encoding = encoder.encode_with_abstraction(state, 0)
        
        # 编码维度应该是370（与完整编码相同）
        assert encoding.shape == (370,)
        assert encoding.dtype == np.float32


class TestCFRTrainerWithAbstraction:
    """测试CFR训练器与卡牌抽象的集成。"""
    
    def _create_mock_abstraction(self) -> CardAbstraction:
        """创建一个模拟的卡牌抽象对象。"""
        config = AbstractionConfig(
            preflop_buckets=169,
            flop_buckets=100,
            turn_buckets=100,
            river_buckets=100
        )
        abstraction = CardAbstraction(config)
        
        # 创建模拟的抽象结果
        abstraction.result = AbstractionResult(
            config=config,
            preflop_mapping=np.arange(169),
            flop_mapping=np.zeros(1000, dtype=np.int32),
            turn_mapping=np.zeros(1000, dtype=np.int32),
            river_mapping=np.zeros(1000, dtype=np.int32),
            wcss={'flop': 0.0, 'turn': 0.0, 'river': 0.0},
            generation_time=0.0
        )
        
        return abstraction
    
    def test_cfr_trainer_without_abstraction(self):
        """测试不使用抽象的CFR训练器。"""
        trainer = CFRTrainer(num_actions=5)
        assert trainer.card_abstraction is None
        assert not trainer.use_abstraction
    
    def test_cfr_trainer_with_abstraction(self):
        """测试使用抽象的CFR训练器。"""
        abstraction = self._create_mock_abstraction()
        trainer = CFRTrainer(num_actions=5, card_abstraction=abstraction)
        
        assert trainer.card_abstraction is not None
        assert trainer.use_abstraction
    
    def test_set_card_abstraction(self):
        """测试设置卡牌抽象。"""
        trainer = CFRTrainer(num_actions=5)
        assert not trainer.use_abstraction
        
        abstraction = self._create_mock_abstraction()
        trainer.set_card_abstraction(abstraction)
        
        assert trainer.use_abstraction
    
    def test_enable_abstraction_requires_loaded_abstraction(self):
        """测试启用抽象需要已加载的抽象。"""
        trainer = CFRTrainer(num_actions=5)
        
        with pytest.raises(ValueError, match="卡牌抽象对象未设置"):
            trainer.enable_abstraction(True)
    
    def test_info_set_with_abstraction_uses_bucket_id(self):
        """测试使用抽象时信息集使用桶ID。"""
        abstraction = self._create_mock_abstraction()
        trainer = CFRTrainer(num_actions=5, card_abstraction=abstraction)
        
        # 创建测试状态
        state = GameState(
            player_hands=[
                (Card(14, 'h'), Card(13, 'h')),
                (Card(12, 'd'), Card(11, 'd'))
            ],
            community_cards=[Card(10, 'c'), Card(9, 's'), Card(8, 'h')],
            pot=100,
            player_stacks=[900, 900],
            current_bets=[50, 50],
            button_position=0,
            stage=GameStage.FLOP,
            current_player=0
        )
        
        info_set = trainer.get_info_set(state, 0)
        
        # 使用抽象时，信息集应该使用桶ID
        assert info_set.is_abstracted
        assert info_set.bucket_id >= 0
        assert info_set.hand_key == ()
        assert info_set.community_key == ()
    
    def test_info_set_without_abstraction_uses_cards(self):
        """测试不使用抽象时信息集使用具体牌。"""
        trainer = CFRTrainer(num_actions=5)
        
        # 创建测试状态
        state = GameState(
            player_hands=[
                (Card(14, 'h'), Card(13, 'h')),
                (Card(12, 'd'), Card(11, 'd'))
            ],
            community_cards=[Card(10, 'c'), Card(9, 's'), Card(8, 'h')],
            pot=100,
            player_stacks=[900, 900],
            current_bets=[50, 50],
            button_position=0,
            stage=GameStage.FLOP,
            current_player=0
        )
        
        info_set = trainer.get_info_set(state, 0)
        
        # 不使用抽象时，信息集应该使用具体牌
        assert not info_set.is_abstracted
        assert info_set.bucket_id == -1
        assert len(info_set.hand_key) == 2
        assert len(info_set.community_key) == 3
    
    def test_abstraction_stats(self):
        """测试获取抽象统计信息。"""
        abstraction = self._create_mock_abstraction()
        trainer = CFRTrainer(num_actions=5, card_abstraction=abstraction)
        
        stats = trainer.get_abstraction_stats()
        
        assert 'total_info_sets' in stats
        assert 'abstracted_info_sets' in stats
        assert 'raw_info_sets' in stats
        assert 'use_abstraction' in stats
        assert stats['use_abstraction'] is True


class TestAbstractionConfigChangeDetection:
    """测试抽象配置变化检测功能。
    
    **属性54：抽象配置变化检测**
    **验证需求：14.4**
    """
    
    def _create_abstraction_with_config(self, config: AbstractionConfig) -> CardAbstraction:
        """创建具有指定配置的抽象对象。"""
        abstraction = CardAbstraction(config)
        
        # 创建模拟的抽象结果
        abstraction.result = AbstractionResult(
            config=config,
            preflop_mapping=np.arange(config.preflop_buckets),
            flop_mapping=np.zeros(1000, dtype=np.int32),
            turn_mapping=np.zeros(1000, dtype=np.int32),
            river_mapping=np.zeros(1000, dtype=np.int32),
            wcss={'flop': 0.0, 'turn': 0.0, 'river': 0.0},
            generation_time=0.0
        )
        
        return abstraction
    
    def test_config_matches_same_config(self):
        """测试相同配置匹配。"""
        config = AbstractionConfig(
            preflop_buckets=169,
            flop_buckets=1000,
            turn_buckets=1000,
            river_buckets=1000
        )
        abstraction = self._create_abstraction_with_config(config)
        
        # 相同配置应该匹配
        assert abstraction.config_matches(config)
    
    def test_config_mismatch_different_buckets(self):
        """测试不同桶数量的配置不匹配。"""
        original_config = AbstractionConfig(
            preflop_buckets=169,
            flop_buckets=1000,
            turn_buckets=1000,
            river_buckets=1000
        )
        abstraction = self._create_abstraction_with_config(original_config)
        
        # 不同的翻牌桶数量
        different_config = AbstractionConfig(
            preflop_buckets=169,
            flop_buckets=2000,  # 不同
            turn_buckets=1000,
            river_buckets=1000
        )
        
        assert not abstraction.config_matches(different_config)
    
    def test_config_mismatch_different_potential_aware(self):
        """测试不同use_potential_aware的配置不匹配。"""
        original_config = AbstractionConfig(
            preflop_buckets=169,
            flop_buckets=1000,
            turn_buckets=1000,
            river_buckets=1000,
            use_potential_aware=True
        )
        abstraction = self._create_abstraction_with_config(original_config)
        
        # 不同的use_potential_aware
        different_config = AbstractionConfig(
            preflop_buckets=169,
            flop_buckets=1000,
            turn_buckets=1000,
            river_buckets=1000,
            use_potential_aware=False  # 不同
        )
        
        assert not abstraction.config_matches(different_config)
    
    def test_config_mismatch_different_random_seed(self):
        """测试不同随机种子的配置不匹配。"""
        original_config = AbstractionConfig(
            preflop_buckets=169,
            flop_buckets=1000,
            turn_buckets=1000,
            river_buckets=1000,
            random_seed=42
        )
        abstraction = self._create_abstraction_with_config(original_config)
        
        # 不同的随机种子
        different_config = AbstractionConfig(
            preflop_buckets=169,
            flop_buckets=1000,
            turn_buckets=1000,
            river_buckets=1000,
            random_seed=123  # 不同
        )
        
        assert not abstraction.config_matches(different_config)


class TestAbstractionConfigChangeDetectionProperty:
    """属性测试：抽象配置变化检测。
    
    **属性54：抽象配置变化检测**
    *对于任何*抽象配置参数的改变，系统应该能够检测到配置与已缓存抽象不匹配
    **验证需求：14.4**
    """
    
    @given(
        preflop_buckets=st.integers(min_value=1, max_value=169),
        flop_buckets=st.integers(min_value=1, max_value=5000),
        turn_buckets=st.integers(min_value=1, max_value=5000),
        river_buckets=st.integers(min_value=1, max_value=5000),
        use_potential_aware=st.booleans(),
        random_seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=50)
    def test_property_config_change_detection(
        self,
        preflop_buckets: int,
        flop_buckets: int,
        turn_buckets: int,
        river_buckets: int,
        use_potential_aware: bool,
        random_seed: int
    ):
        """属性测试：配置变化检测。
        
        **Feature: texas-holdem-ai-training, Property 54: 抽象配置变化检测**
        """
        # 创建原始配置
        original_config = AbstractionConfig(
            preflop_buckets=preflop_buckets,
            flop_buckets=flop_buckets,
            turn_buckets=turn_buckets,
            river_buckets=river_buckets,
            use_potential_aware=use_potential_aware,
            random_seed=random_seed
        )
        
        # 创建抽象对象
        abstraction = CardAbstraction(original_config)
        abstraction.result = AbstractionResult(
            config=original_config,
            preflop_mapping=np.arange(preflop_buckets),
            flop_mapping=np.zeros(1000, dtype=np.int32),
            turn_mapping=np.zeros(1000, dtype=np.int32),
            river_mapping=np.zeros(1000, dtype=np.int32),
            wcss={},
            generation_time=0.0
        )
        
        # 相同配置应该匹配
        assert abstraction.config_matches(original_config), \
            "相同配置应该匹配"
        
        # 修改任意一个参数，配置应该不匹配
        # 测试修改preflop_buckets
        if preflop_buckets > 1:
            modified_config = AbstractionConfig(
                preflop_buckets=preflop_buckets - 1,
                flop_buckets=flop_buckets,
                turn_buckets=turn_buckets,
                river_buckets=river_buckets,
                use_potential_aware=use_potential_aware,
                random_seed=random_seed
            )
            assert not abstraction.config_matches(modified_config), \
                "修改preflop_buckets后配置应该不匹配"
        
        # 测试修改use_potential_aware
        modified_config = AbstractionConfig(
            preflop_buckets=preflop_buckets,
            flop_buckets=flop_buckets,
            turn_buckets=turn_buckets,
            river_buckets=river_buckets,
            use_potential_aware=not use_potential_aware,
            random_seed=random_seed
        )
        assert not abstraction.config_matches(modified_config), \
            "修改use_potential_aware后配置应该不匹配"


class TestInfoSetEquality:
    """测试信息集的相等性比较。"""
    
    def test_abstracted_info_sets_equal_by_bucket_id(self):
        """测试抽象信息集按桶ID比较相等性。"""
        info_set1 = InfoSet(
            hand_key=(),
            community_key=(),
            stage='flop',
            action_history_key=('call',),
            pot_ratio=5,
            bucket_id=42
        )
        
        info_set2 = InfoSet(
            hand_key=(),
            community_key=(),
            stage='flop',
            action_history_key=('call',),
            pot_ratio=5,
            bucket_id=42
        )
        
        assert info_set1 == info_set2
        assert hash(info_set1) == hash(info_set2)
    
    def test_abstracted_info_sets_different_bucket_id(self):
        """测试不同桶ID的抽象信息集不相等。"""
        info_set1 = InfoSet(
            hand_key=(),
            community_key=(),
            stage='flop',
            action_history_key=('call',),
            pot_ratio=5,
            bucket_id=42
        )
        
        info_set2 = InfoSet(
            hand_key=(),
            community_key=(),
            stage='flop',
            action_history_key=('call',),
            pot_ratio=5,
            bucket_id=43  # 不同的桶ID
        )
        
        assert info_set1 != info_set2
    
    def test_raw_info_sets_equal_by_cards(self):
        """测试原始信息集按具体牌比较相等性。"""
        info_set1 = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=((10, 'c'), (9, 's'), (8, 'h')),
            stage='flop',
            action_history_key=('call',),
            pot_ratio=5,
            bucket_id=-1
        )
        
        info_set2 = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=((10, 'c'), (9, 's'), (8, 'h')),
            stage='flop',
            action_history_key=('call',),
            pot_ratio=5,
            bucket_id=-1
        )
        
        assert info_set1 == info_set2
        assert hash(info_set1) == hash(info_set2)
    
    def test_raw_info_sets_different_cards(self):
        """测试不同牌的原始信息集不相等。"""
        info_set1 = InfoSet(
            hand_key=((14, 'h'), (13, 'h')),
            community_key=((10, 'c'), (9, 's'), (8, 'h')),
            stage='flop',
            action_history_key=('call',),
            pot_ratio=5,
            bucket_id=-1
        )
        
        info_set2 = InfoSet(
            hand_key=((14, 's'), (13, 's')),  # 不同的牌
            community_key=((10, 'c'), (9, 's'), (8, 'h')),
            stage='flop',
            action_history_key=('call',),
            pot_ratio=5,
            bucket_id=-1
        )
        
        assert info_set1 != info_set2
