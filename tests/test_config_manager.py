"""配置管理器的单元测试和属性测试。"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from models.core import TrainingConfig
from utils.config_manager import ConfigManager, DEFAULT_CONFIG


class TestConfigManagerUnit:
    """ConfigManager的单元测试。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.manager = ConfigManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """每个测试方法后的清理。"""
        # 清理临时文件
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    # ==================== 保存和加载测试 ====================
    
    def test_save_and_load_valid_config(self):
        """测试保存和加载有效配置。"""
        # 创建配置
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=64,
            num_episodes=5000,
            discount_factor=0.95,
            network_architecture=[256, 128],
            checkpoint_interval=500,
            num_parallel_envs=4,
            initial_stack=2000,
            small_blind=10,
            big_blind=20
        )
        
        # 保存配置
        config_path = Path(self.temp_dir) / "test_config.json"
        self.manager.save_config(config, config_path)
        
        # 验证文件存在
        assert config_path.exists()
        
        # 加载配置
        loaded_config = self.manager.load_config(config_path)
        
        # 验证加载的配置与原配置相同
        assert loaded_config.learning_rate == config.learning_rate
        assert loaded_config.batch_size == config.batch_size
        assert loaded_config.num_episodes == config.num_episodes
        assert loaded_config.discount_factor == config.discount_factor
        assert loaded_config.network_architecture == config.network_architecture
        assert loaded_config.checkpoint_interval == config.checkpoint_interval
        assert loaded_config.num_parallel_envs == config.num_parallel_envs
        assert loaded_config.initial_stack == config.initial_stack
        assert loaded_config.small_blind == config.small_blind
        assert loaded_config.big_blind == config.big_blind
    
    def test_save_creates_parent_directories(self):
        """测试保存配置时自动创建父目录。"""
        config = self.manager.get_default_config()
        nested_path = Path(self.temp_dir) / "nested" / "dir" / "config.json"
        
        self.manager.save_config(config, nested_path)
        
        assert nested_path.exists()
    
    def test_load_nonexistent_file_raises_error(self):
        """测试加载不存在的文件时抛出错误。"""
        with pytest.raises(FileNotFoundError):
            self.manager.load_config("/nonexistent/path/config.json")
    
    def test_load_invalid_json_raises_error(self):
        """测试加载无效JSON文件时抛出错误。"""
        invalid_json_path = Path(self.temp_dir) / "invalid.json"
        with open(invalid_json_path, 'w') as f:
            f.write("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            self.manager.load_config(invalid_json_path)
    
    # ==================== 无效配置拒绝测试 ====================
    
    def test_reject_negative_learning_rate(self):
        """测试拒绝负数学习率。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['learning_rate'] = -0.01
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('learning_rate' in e for e in errors)
    
    def test_reject_zero_learning_rate(self):
        """测试拒绝零学习率。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['learning_rate'] = 0
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('learning_rate' in e for e in errors)
    
    def test_reject_learning_rate_greater_than_one(self):
        """测试拒绝大于1的学习率。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['learning_rate'] = 1.5
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('learning_rate' in e for e in errors)
    
    def test_reject_negative_batch_size(self):
        """测试拒绝负数批次大小。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['batch_size'] = -10
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('batch_size' in e for e in errors)
    
    def test_reject_invalid_discount_factor(self):
        """测试拒绝无效的折扣因子。"""
        # 测试大于1
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['discount_factor'] = 1.5
        
        errors = self.manager.validate_config(config_dict)
        assert len(errors) > 0
        assert any('discount_factor' in e for e in errors)
        
        # 测试小于0
        config_dict['discount_factor'] = -0.1
        errors = self.manager.validate_config(config_dict)
        assert len(errors) > 0
        assert any('discount_factor' in e for e in errors)
    
    def test_reject_invalid_blinds(self):
        """测试拒绝无效盲注（大盲注小于等于小盲注）。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['small_blind'] = 20
        config_dict['big_blind'] = 10  # 大盲注小于小盲注
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('big_blind' in e for e in errors)
    
    def test_reject_equal_blinds(self):
        """测试拒绝相等的盲注。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['small_blind'] = 10
        config_dict['big_blind'] = 10  # 大盲注等于小盲注
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('big_blind' in e for e in errors)
    
    def test_reject_negative_initial_stack(self):
        """测试拒绝负数初始筹码。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['initial_stack'] = -100
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('initial_stack' in e for e in errors)
    
    def test_reject_initial_stack_less_than_big_blind(self):
        """测试拒绝初始筹码小于大盲注。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['initial_stack'] = 5
        config_dict['big_blind'] = 10
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('initial_stack' in e for e in errors)
    
    def test_reject_empty_network_architecture(self):
        """测试拒绝空的网络架构。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['network_architecture'] = []
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('network_architecture' in e for e in errors)
    
    def test_reject_negative_network_layer_dimension(self):
        """测试拒绝负数的网络层维度。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['network_architecture'] = [256, -128, 64]
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('network_architecture' in e for e in errors)
    
    # ==================== 错误信息清晰性测试 ====================
    
    def test_error_message_contains_parameter_name(self):
        """测试错误信息包含参数名。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['learning_rate'] = -0.01
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        # 错误信息应该包含参数名
        assert 'learning_rate' in errors[0]
    
    def test_error_message_contains_reason(self):
        """测试错误信息包含原因。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['learning_rate'] = -0.01
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        # 错误信息应该包含原因（如"必须为正数"）
        assert '正数' in errors[0] or '正' in errors[0]
    
    def test_multiple_errors_reported(self):
        """测试多个错误都被报告。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['learning_rate'] = -0.01
        config_dict['batch_size'] = -10
        config_dict['discount_factor'] = 2.0
        
        errors = self.manager.validate_config(config_dict)
        
        # 应该报告多个错误
        assert len(errors) >= 3
        assert any('learning_rate' in e for e in errors)
        assert any('batch_size' in e for e in errors)
        assert any('discount_factor' in e for e in errors)
    
    # ==================== 默认值应用测试 ====================
    
    def test_apply_defaults_for_missing_optional_params(self):
        """测试为缺失的可选参数应用默认值。"""
        # 创建只有部分参数的配置文件
        partial_config = {
            'learning_rate': 0.005,
            'batch_size': 128
        }
        
        config_path = Path(self.temp_dir) / "partial_config.json"
        with open(config_path, 'w') as f:
            json.dump(partial_config, f)
        
        # 加载配置
        loaded_config = self.manager.load_config(config_path)
        
        # 验证指定的参数被正确加载
        assert loaded_config.learning_rate == 0.005
        assert loaded_config.batch_size == 128
        
        # 验证缺失的参数使用默认值
        assert loaded_config.num_episodes == DEFAULT_CONFIG['num_episodes']
        assert loaded_config.discount_factor == DEFAULT_CONFIG['discount_factor']
        assert loaded_config.network_architecture == DEFAULT_CONFIG['network_architecture']
        assert loaded_config.checkpoint_interval == DEFAULT_CONFIG['checkpoint_interval']
        assert loaded_config.num_parallel_envs == DEFAULT_CONFIG['num_parallel_envs']
        assert loaded_config.initial_stack == DEFAULT_CONFIG['initial_stack']
        assert loaded_config.small_blind == DEFAULT_CONFIG['small_blind']
        assert loaded_config.big_blind == DEFAULT_CONFIG['big_blind']
    
    def test_apply_defaults_for_empty_config(self):
        """测试为空配置应用所有默认值。"""
        empty_config = {}
        
        config_path = Path(self.temp_dir) / "empty_config.json"
        with open(config_path, 'w') as f:
            json.dump(empty_config, f)
        
        # 加载配置
        loaded_config = self.manager.load_config(config_path)
        
        # 验证所有参数都使用默认值
        assert loaded_config.learning_rate == DEFAULT_CONFIG['learning_rate']
        assert loaded_config.batch_size == DEFAULT_CONFIG['batch_size']
        assert loaded_config.num_episodes == DEFAULT_CONFIG['num_episodes']
        assert loaded_config.discount_factor == DEFAULT_CONFIG['discount_factor']
        assert loaded_config.network_architecture == DEFAULT_CONFIG['network_architecture']
        assert loaded_config.checkpoint_interval == DEFAULT_CONFIG['checkpoint_interval']
        assert loaded_config.num_parallel_envs == DEFAULT_CONFIG['num_parallel_envs']
        assert loaded_config.initial_stack == DEFAULT_CONFIG['initial_stack']
        assert loaded_config.small_blind == DEFAULT_CONFIG['small_blind']
        assert loaded_config.big_blind == DEFAULT_CONFIG['big_blind']
    
    def test_get_default_config(self):
        """测试获取默认配置。"""
        default_config = self.manager.get_default_config()
        
        assert isinstance(default_config, TrainingConfig)
        assert default_config.learning_rate == DEFAULT_CONFIG['learning_rate']
        assert default_config.batch_size == DEFAULT_CONFIG['batch_size']
    
    # ==================== 其他功能测试 ====================
    
    def test_validate_training_config_object(self):
        """测试验证TrainingConfig对象。"""
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=64,
            num_episodes=5000,
            discount_factor=0.95,
            network_architecture=[256, 128],
            checkpoint_interval=500,
            num_parallel_envs=4,
            initial_stack=2000,
            small_blind=10,
            big_blind=20
        )
        
        errors = self.manager.validate_config(config)
        
        assert len(errors) == 0
    
    def test_merge_configs(self):
        """测试合并配置。"""
        base = {'learning_rate': 0.001, 'batch_size': 32}
        override = {'batch_size': 64, 'num_episodes': 5000}
        
        merged = self.manager.merge_configs(base, override)
        
        assert merged['learning_rate'] == 0.001  # 来自base
        assert merged['batch_size'] == 64  # 被override覆盖
        assert merged['num_episodes'] == 5000  # 来自override
    
    def test_saved_json_is_readable(self):
        """测试保存的JSON文件可读。"""
        config = self.manager.get_default_config()
        config_path = Path(self.temp_dir) / "readable_config.json"
        
        self.manager.save_config(config, config_path)
        
        # 直接读取JSON文件
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'learning_rate' in data
        assert 'batch_size' in data
        assert data['learning_rate'] == config.learning_rate
    
    # ==================== Deep CFR 参数测试 ====================
    
    def test_deep_cfr_default_values(self):
        """测试 Deep CFR 参数的默认值。"""
        default_config = self.manager.get_default_config()
        
        # 验证 Deep CFR 参数的默认值
        assert default_config.regret_buffer_size == 2000000
        assert default_config.strategy_buffer_size == 2000000
        assert default_config.cfr_iterations_per_update == 1000
        assert default_config.network_train_steps == 4000
    
    def test_reject_negative_regret_buffer_size(self):
        """测试拒绝负数遗憾缓冲区大小。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['regret_buffer_size'] = -100
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('regret_buffer_size' in e for e in errors)
    
    def test_reject_zero_regret_buffer_size(self):
        """测试拒绝零遗憾缓冲区大小。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['regret_buffer_size'] = 0
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('regret_buffer_size' in e for e in errors)
    
    def test_reject_negative_strategy_buffer_size(self):
        """测试拒绝负数策略缓冲区大小。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['strategy_buffer_size'] = -100
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('strategy_buffer_size' in e for e in errors)
    
    def test_reject_zero_strategy_buffer_size(self):
        """测试拒绝零策略缓冲区大小。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['strategy_buffer_size'] = 0
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('strategy_buffer_size' in e for e in errors)
    
    def test_reject_negative_cfr_iterations_per_update(self):
        """测试拒绝负数 CFR 迭代次数。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['cfr_iterations_per_update'] = -100
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('cfr_iterations_per_update' in e for e in errors)
    
    def test_reject_zero_cfr_iterations_per_update(self):
        """测试拒绝零 CFR 迭代次数。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['cfr_iterations_per_update'] = 0
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('cfr_iterations_per_update' in e for e in errors)
    
    def test_reject_negative_network_train_steps(self):
        """测试拒绝负数网络训练步数。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['network_train_steps'] = -100
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('network_train_steps' in e for e in errors)
    
    def test_reject_zero_network_train_steps(self):
        """测试拒绝零网络训练步数。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['network_train_steps'] = 0
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) > 0
        assert any('network_train_steps' in e for e in errors)
    
    def test_valid_deep_cfr_config(self):
        """测试有效的 Deep CFR 配置。"""
        config_dict = dict(DEFAULT_CONFIG)
        config_dict['regret_buffer_size'] = 1000000
        config_dict['strategy_buffer_size'] = 500000
        config_dict['cfr_iterations_per_update'] = 500
        config_dict['network_train_steps'] = 2000
        
        errors = self.manager.validate_config(config_dict)
        
        assert len(errors) == 0
    
    # ==================== 旧配置兼容性测试 ====================
    
    def test_load_old_config_with_cfr_weight(self):
        """测试加载包含废弃 cfr_weight 参数的旧配置文件。"""
        # 创建包含旧参数的配置
        old_config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_episodes': 10000,
            'discount_factor': 0.99,
            'network_architecture': [512, 256, 128],
            'checkpoint_interval': 1000,
            'num_parallel_envs': 1,
            'initial_stack': 1000,
            'small_blind': 5,
            'big_blind': 10,
            'cfr_weight': 0.5  # 废弃的参数
        }
        
        config_path = Path(self.temp_dir) / "old_config.json"
        with open(config_path, 'w') as f:
            json.dump(old_config, f)
        
        # 加载配置应该成功，cfr_weight 被忽略
        loaded_config = self.manager.load_config(config_path)
        
        # 验证配置加载成功
        assert loaded_config.learning_rate == 0.001
        assert loaded_config.batch_size == 32
        
        # 验证新参数使用默认值
        assert loaded_config.regret_buffer_size == DEFAULT_CONFIG['regret_buffer_size']
        assert loaded_config.strategy_buffer_size == DEFAULT_CONFIG['strategy_buffer_size']
        assert loaded_config.cfr_iterations_per_update == DEFAULT_CONFIG['cfr_iterations_per_update']
        assert loaded_config.network_train_steps == DEFAULT_CONFIG['network_train_steps']
        
        # 验证 cfr_weight 不存在于加载的配置中
        assert not hasattr(loaded_config, 'cfr_weight')
    
    def test_save_and_load_deep_cfr_config(self):
        """测试保存和加载包含 Deep CFR 参数的配置。"""
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=2048,
            num_episodes=100000,
            discount_factor=1.0,
            network_architecture=[512, 256, 128],
            checkpoint_interval=1000,
            num_parallel_envs=1,
            initial_stack=1000,
            small_blind=5,
            big_blind=10,
            regret_buffer_size=1000000,
            strategy_buffer_size=500000,
            cfr_iterations_per_update=500,
            network_train_steps=2000
        )
        
        config_path = Path(self.temp_dir) / "deep_cfr_config.json"
        self.manager.save_config(config, config_path)
        
        loaded_config = self.manager.load_config(config_path)
        
        # 验证 Deep CFR 参数
        assert loaded_config.regret_buffer_size == 1000000
        assert loaded_config.strategy_buffer_size == 500000
        assert loaded_config.cfr_iterations_per_update == 500
        assert loaded_config.network_train_steps == 2000


# ==================== 属性测试 ====================

from hypothesis import given, strategies as st, settings, assume


# 配置生成策略
@st.composite
def valid_config_strategy(draw):
    """生成有效的配置字典。"""
    small_blind = draw(st.integers(min_value=1, max_value=100))
    big_blind = draw(st.integers(min_value=small_blind + 1, max_value=small_blind + 200))
    initial_stack = draw(st.integers(min_value=big_blind, max_value=100000))
    
    return {
        'learning_rate': draw(st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'batch_size': draw(st.integers(min_value=1, max_value=1024)),
        'num_episodes': draw(st.integers(min_value=1, max_value=1000000)),
        'discount_factor': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'network_architecture': draw(st.lists(st.integers(min_value=1, max_value=1024), min_size=1, max_size=10)),
        'checkpoint_interval': draw(st.integers(min_value=1, max_value=100000)),
        'num_parallel_envs': draw(st.integers(min_value=1, max_value=64)),
        'initial_stack': initial_stack,
        'small_blind': small_blind,
        'big_blind': big_blind,
        'entropy_coefficient': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'max_raises_per_street': draw(st.integers(min_value=0, max_value=10)),
        # Deep CFR 参数
        'regret_buffer_size': draw(st.integers(min_value=1, max_value=10000000)),
        'strategy_buffer_size': draw(st.integers(min_value=1, max_value=10000000)),
        'cfr_iterations_per_update': draw(st.integers(min_value=1, max_value=10000)),
        'network_train_steps': draw(st.integers(min_value=1, max_value=10000))
    }


@st.composite
def invalid_config_strategy(draw):
    """生成无效的配置字典（至少有一个无效参数）。"""
    # 选择一种无效类型
    invalid_type = draw(st.sampled_from([
        'negative_learning_rate',
        'zero_learning_rate',
        'learning_rate_too_high',
        'negative_batch_size',
        'invalid_discount_factor',
        'invalid_blinds',
        'negative_initial_stack',
        'empty_network_architecture',
        'negative_network_dim',
        # Deep CFR 参数无效情况
        'negative_regret_buffer_size',
        'zero_regret_buffer_size',
        'negative_strategy_buffer_size',
        'zero_strategy_buffer_size',
        'negative_cfr_iterations',
        'zero_cfr_iterations',
        'negative_network_train_steps',
        'zero_network_train_steps'
    ]))
    
    # 从有效配置开始
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_episodes': 10000,
        'discount_factor': 0.99,
        'network_architecture': [512, 256, 128],
        'checkpoint_interval': 1000,
        'num_parallel_envs': 1,
        'initial_stack': 1000,
        'small_blind': 5,
        'big_blind': 10,
        'entropy_coefficient': 0.01,
        'max_raises_per_street': 4,
        'regret_buffer_size': 2000000,
        'strategy_buffer_size': 2000000,
        'cfr_iterations_per_update': 1000,
        'network_train_steps': 4000
    }
    
    # 根据类型引入无效值
    if invalid_type == 'negative_learning_rate':
        config['learning_rate'] = draw(st.floats(max_value=-0.0001, allow_nan=False, allow_infinity=False))
    elif invalid_type == 'zero_learning_rate':
        config['learning_rate'] = 0
    elif invalid_type == 'learning_rate_too_high':
        config['learning_rate'] = draw(st.floats(min_value=1.001, max_value=100, allow_nan=False, allow_infinity=False))
    elif invalid_type == 'negative_batch_size':
        config['batch_size'] = draw(st.integers(max_value=0))
    elif invalid_type == 'invalid_discount_factor':
        # 选择小于0或大于1
        if draw(st.booleans()):
            config['discount_factor'] = draw(st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False))
        else:
            config['discount_factor'] = draw(st.floats(min_value=1.001, max_value=100, allow_nan=False, allow_infinity=False))
    elif invalid_type == 'invalid_blinds':
        # 大盲注小于等于小盲注
        sb = draw(st.integers(min_value=1, max_value=100))
        bb = draw(st.integers(min_value=1, max_value=sb))
        config['small_blind'] = sb
        config['big_blind'] = bb
    elif invalid_type == 'negative_initial_stack':
        config['initial_stack'] = draw(st.integers(max_value=0))
    elif invalid_type == 'empty_network_architecture':
        config['network_architecture'] = []
    elif invalid_type == 'negative_network_dim':
        config['network_architecture'] = [256, draw(st.integers(max_value=0)), 64]
    # Deep CFR 参数无效情况
    elif invalid_type == 'negative_regret_buffer_size':
        config['regret_buffer_size'] = draw(st.integers(max_value=-1))
    elif invalid_type == 'zero_regret_buffer_size':
        config['regret_buffer_size'] = 0
    elif invalid_type == 'negative_strategy_buffer_size':
        config['strategy_buffer_size'] = draw(st.integers(max_value=-1))
    elif invalid_type == 'zero_strategy_buffer_size':
        config['strategy_buffer_size'] = 0
    elif invalid_type == 'negative_cfr_iterations':
        config['cfr_iterations_per_update'] = draw(st.integers(max_value=-1))
    elif invalid_type == 'zero_cfr_iterations':
        config['cfr_iterations_per_update'] = 0
    elif invalid_type == 'negative_network_train_steps':
        config['network_train_steps'] = draw(st.integers(max_value=-1))
    elif invalid_type == 'zero_network_train_steps':
        config['network_train_steps'] = 0
    
    return config


@st.composite
def partial_config_strategy(draw):
    """生成部分配置字典（缺少一些可选参数）。
    
    注意：生成的部分配置在与默认值合并后必须是有效的。
    这意味着需要考虑参数之间的依赖关系：
    - 如果只提供initial_stack，它必须 >= 默认的big_blind（10）
    - 如果只提供small_blind，它必须 < 默认的big_blind（10）
    - 如果只提供big_blind，它必须 > 默认的small_blind（5）
    """
    # 所有可能的参数
    all_params = [
        'learning_rate', 'batch_size', 'num_episodes', 'discount_factor',
        'network_architecture', 'checkpoint_interval', 'num_parallel_envs',
        'initial_stack', 'small_blind', 'big_blind', 'entropy_coefficient',
        'max_raises_per_street', 'regret_buffer_size', 'strategy_buffer_size',
        'cfr_iterations_per_update', 'network_train_steps'
    ]
    
    # 随机选择要包含的参数（至少0个，最多全部）
    num_params = draw(st.integers(min_value=0, max_value=len(all_params)))
    included_params = set(draw(st.permutations(all_params))[:num_params])
    
    # 默认值（用于计算依赖关系）
    default_small_blind = DEFAULT_CONFIG['small_blind']  # 5
    default_big_blind = DEFAULT_CONFIG['big_blind']  # 10
    
    # 确定实际使用的盲注值（考虑是否在部分配置中）
    if 'small_blind' in included_params and 'big_blind' in included_params:
        # 两者都提供，生成有效的组合
        small_blind = draw(st.integers(min_value=1, max_value=100))
        big_blind = draw(st.integers(min_value=small_blind + 1, max_value=small_blind + 200))
    elif 'small_blind' in included_params:
        # 只提供small_blind，必须 < 默认的big_blind
        small_blind = draw(st.integers(min_value=1, max_value=default_big_blind - 1))
        big_blind = default_big_blind
    elif 'big_blind' in included_params:
        # 只提供big_blind，必须 > 默认的small_blind
        big_blind = draw(st.integers(min_value=default_small_blind + 1, max_value=200))
        small_blind = default_small_blind
    else:
        # 都不提供，使用默认值
        small_blind = default_small_blind
        big_blind = default_big_blind
    
    # 确定实际使用的big_blind（用于initial_stack约束）
    effective_big_blind = big_blind if 'big_blind' in included_params else default_big_blind
    
    # 生成initial_stack（必须 >= effective_big_blind）
    if 'initial_stack' in included_params:
        initial_stack = draw(st.integers(min_value=effective_big_blind, max_value=100000))
    else:
        initial_stack = DEFAULT_CONFIG['initial_stack']
    
    valid_values = {
        'learning_rate': draw(st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'batch_size': draw(st.integers(min_value=1, max_value=1024)),
        'num_episodes': draw(st.integers(min_value=1, max_value=1000000)),
        'discount_factor': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'network_architecture': draw(st.lists(st.integers(min_value=1, max_value=1024), min_size=1, max_size=10)),
        'checkpoint_interval': draw(st.integers(min_value=1, max_value=100000)),
        'num_parallel_envs': draw(st.integers(min_value=1, max_value=64)),
        'initial_stack': initial_stack,
        'small_blind': small_blind,
        'big_blind': big_blind,
        'entropy_coefficient': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'max_raises_per_street': draw(st.integers(min_value=0, max_value=10)),
        # Deep CFR 参数
        'regret_buffer_size': draw(st.integers(min_value=1, max_value=10000000)),
        'strategy_buffer_size': draw(st.integers(min_value=1, max_value=10000000)),
        'cfr_iterations_per_update': draw(st.integers(min_value=1, max_value=10000)),
        'network_train_steps': draw(st.integers(min_value=1, max_value=10000))
    }
    
    # 构建部分配置
    partial_config = {}
    for param in included_params:
        partial_config[param] = valid_values[param]
    
    return partial_config


class TestConfigManagerProperties:
    """ConfigManager的属性测试。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        self.manager = ConfigManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """每个测试方法后的清理。"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @given(config=valid_config_strategy())
    @settings(max_examples=100)
    def test_property_26_valid_config_accepted(self, config):
        """
        属性26：配置验证正确性 - 有效配置被接受
        
        **Feature: texas-holdem-ai-training, Property 26: 配置验证正确性**
        *对于任何*有效的训练配置，validate_config应该返回空的错误列表
        **验证需求：7.1**
        """
        errors = self.manager.validate_config(config)
        assert len(errors) == 0, f"有效配置被拒绝: {errors}"
    
    @given(config=invalid_config_strategy())
    @settings(max_examples=100)
    def test_property_26_invalid_config_rejected(self, config):
        """
        属性26：配置验证正确性 - 无效配置被拒绝
        
        **Feature: texas-holdem-ai-training, Property 26: 配置验证正确性**
        *对于任何*无效的训练配置，validate_config应该返回非空的错误列表
        **验证需求：7.1**
        """
        errors = self.manager.validate_config(config)
        assert len(errors) > 0, f"无效配置被接受: {config}"
    
    @given(config=invalid_config_strategy())
    @settings(max_examples=100)
    def test_property_27_error_message_clarity(self, config):
        """
        属性27：配置错误信息清晰性
        
        **Feature: texas-holdem-ai-training, Property 27: 配置错误信息清晰性**
        *对于任何*无效的配置参数，错误信息应该包含参数名和原因
        **验证需求：7.2**
        """
        errors = self.manager.validate_config(config)
        
        assert len(errors) > 0, "无效配置应该产生错误"
        
        for error in errors:
            # 错误信息应该包含参数名（以冒号分隔）
            assert ':' in error, f"错误信息应该包含参数名和原因，格式为'参数名: 原因'，实际为: {error}"
            
            # 提取参数名
            param_name = error.split(':')[0].strip()
            
            # 参数名应该是已知的配置参数之一
            known_params = [
                'learning_rate', 'batch_size', 'num_episodes', 'discount_factor',
                'network_architecture', 'checkpoint_interval', 'num_parallel_envs',
                'initial_stack', 'small_blind', 'big_blind', 'entropy_coefficient',
                'max_raises_per_street', 'regret_buffer_size', 'strategy_buffer_size',
                'cfr_iterations_per_update', 'network_train_steps'
            ]
            # 也可能是带索引的参数名，如 network_architecture[1]
            base_param = param_name.split('[')[0]
            assert base_param in known_params, f"未知的参数名: {param_name}"
    
    @given(config=valid_config_strategy())
    @settings(max_examples=100)
    def test_property_28_config_round_trip(self, config):
        """
        属性28：配置往返一致性
        
        **Feature: texas-holdem-ai-training, Property 28: 配置往返一致性**
        *对于任何*有效的训练配置，保存为JSON文件后再加载，应该恢复出等价的配置对象
        **验证需求：7.3, 7.4**
        """
        # 创建TrainingConfig对象
        original_config = TrainingConfig(**config)
        
        # 保存配置
        config_path = Path(self.temp_dir) / f"round_trip_config_{id(config)}.json"
        self.manager.save_config(original_config, config_path)
        
        # 加载配置
        loaded_config = self.manager.load_config(config_path)
        
        # 验证所有字段相等
        assert loaded_config.learning_rate == original_config.learning_rate
        assert loaded_config.batch_size == original_config.batch_size
        assert loaded_config.num_episodes == original_config.num_episodes
        assert loaded_config.discount_factor == original_config.discount_factor
        assert loaded_config.network_architecture == original_config.network_architecture
        assert loaded_config.checkpoint_interval == original_config.checkpoint_interval
        assert loaded_config.num_parallel_envs == original_config.num_parallel_envs
        assert loaded_config.initial_stack == original_config.initial_stack
        assert loaded_config.small_blind == original_config.small_blind
        assert loaded_config.big_blind == original_config.big_blind
        assert loaded_config.entropy_coefficient == original_config.entropy_coefficient
        assert loaded_config.max_raises_per_street == original_config.max_raises_per_street
        # Deep CFR 参数
        assert loaded_config.regret_buffer_size == original_config.regret_buffer_size
        assert loaded_config.strategy_buffer_size == original_config.strategy_buffer_size
        assert loaded_config.cfr_iterations_per_update == original_config.cfr_iterations_per_update
        assert loaded_config.network_train_steps == original_config.network_train_steps
    
    @given(partial_config=partial_config_strategy())
    @settings(max_examples=100)
    def test_property_29_default_values_applied(self, partial_config):
        """
        属性29：配置默认值应用正确性
        
        **Feature: texas-holdem-ai-training, Property 29: 配置默认值应用正确性**
        *对于任何*缺少可选参数的配置文件，加载后的配置对象应该包含所有可选参数的默认值
        **验证需求：7.5**
        """
        # 保存部分配置
        config_path = Path(self.temp_dir) / f"partial_config_{id(partial_config)}.json"
        with open(config_path, 'w') as f:
            json.dump(partial_config, f)
        
        # 加载配置
        loaded_config = self.manager.load_config(config_path)
        
        # 验证所有参数都有值
        assert loaded_config.learning_rate is not None
        assert loaded_config.batch_size is not None
        assert loaded_config.num_episodes is not None
        assert loaded_config.discount_factor is not None
        assert loaded_config.network_architecture is not None
        assert loaded_config.checkpoint_interval is not None
        assert loaded_config.num_parallel_envs is not None
        assert loaded_config.initial_stack is not None
        assert loaded_config.small_blind is not None
        assert loaded_config.big_blind is not None
        
        # 验证指定的参数使用指定值，缺失的参数使用默认值
        for param, value in partial_config.items():
            actual_value = getattr(loaded_config, param)
            assert actual_value == value, f"参数 {param} 应该为 {value}，实际为 {actual_value}"
        
        # 验证缺失的参数使用默认值
        for param, default_value in DEFAULT_CONFIG.items():
            if param not in partial_config:
                actual_value = getattr(loaded_config, param)
                assert actual_value == default_value, f"缺失参数 {param} 应该使用默认值 {default_value}，实际为 {actual_value}"
