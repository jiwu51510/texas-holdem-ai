"""错误处理和日志系统的单元测试。

测试内容：
- 配置错误处理（需求7.2）
- 运行时错误处理
- 数据错误处理（需求9.4）
"""

import pytest
import os
import tempfile
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

# 导入异常类
from utils.exceptions import (
    # 基础异常
    PokerAIError,
    # 配置错误
    ConfigurationError,
    InvalidParameterError,
    MissingParameterError,
    IncompatibleParametersError,
    ConfigValidationError,
    # 运行时错误
    ResourceError,
    DiskSpaceError,
    GPUError,
    TrainingInterruptedError,
    # 数据错误
    DataError,
    CheckpointError,
    CheckpointCorruptedError,
    CheckpointNotFoundError,
    ConfigFileError,
    LogFileError,
    DataSerializationError,
    # 游戏逻辑错误
    GameError,
    IllegalActionError,
    InvalidGameStateError,
    GameNotStartedError,
    GameAlreadyEndedError,
    # 并行训练错误
    ParallelTrainingError,
    WorkerProcessError,
    InterProcessCommunicationError,
    ParameterSyncError,
    # 模型错误
    ModelError,
    ModelLoadError,
    ModelSaveError,
    # 评估错误
    EvaluationError,
    InvalidOpponentError,
)

# 导入日志功能
from utils.logger import (
    LoggerConfig,
    setup_logger,
    get_logger,
    set_log_level,
    clear_loggers,
    LoggerMixin,
    configure_logging,
    get_global_config,
)

# 导入错误恢复功能
from utils.error_recovery import (
    RetryConfig,
    retry,
    check_disk_space,
    check_gpu_availability,
    get_device,
    RecoveryManager,
    graceful_shutdown,
    safe_execute,
)


class TestConfigurationErrors:
    """配置错误测试类 - 验证需求7.2"""
    
    def test_invalid_parameter_error_message(self):
        """测试无效参数错误信息包含参数名和原因"""
        error = InvalidParameterError(
            parameter_name='learning_rate',
            parameter_value=-0.1,
            reason='必须为正数'
        )
        
        # 验证错误信息包含参数名
        assert 'learning_rate' in str(error)
        # 验证错误信息包含原因
        assert '必须为正数' in str(error)
        # 验证错误信息包含当前值
        assert '-0.1' in str(error)
        # 验证详情字典
        assert error.details['parameter_name'] == 'learning_rate'
        assert error.details['parameter_value'] == -0.1
        assert error.details['reason'] == '必须为正数'
    
    def test_missing_parameter_error_message(self):
        """测试缺失参数错误信息包含参数名"""
        error = MissingParameterError(parameter_name='batch_size')
        
        # 验证错误信息包含参数名
        assert 'batch_size' in str(error)
        assert '缺失' in str(error)
        assert error.details['parameter_name'] == 'batch_size'
    
    def test_incompatible_parameters_error_message(self):
        """测试参数不兼容错误信息包含所有相关参数"""
        error = IncompatibleParametersError(
            parameters=['small_blind', 'big_blind'],
            reason='大盲注必须大于小盲注'
        )
        
        # 验证错误信息包含所有参数名
        assert 'small_blind' in str(error)
        assert 'big_blind' in str(error)
        # 验证错误信息包含原因
        assert '大盲注必须大于小盲注' in str(error)
    
    def test_config_validation_error_with_multiple_errors(self):
        """测试配置验证错误包含所有验证错误"""
        errors = [
            'learning_rate: 必须为正数',
            'batch_size: 必须为正整数',
            'big_blind: 必须大于small_blind'
        ]
        error = ConfigValidationError(errors=errors)
        
        # 验证所有错误都在消息中
        for err in errors:
            assert err in str(error)
        
        # 验证详情包含错误列表
        assert error.details['errors'] == errors
    
    def test_configuration_error_inheritance(self):
        """测试配置错误的继承关系"""
        # 所有配置错误都应该继承自ConfigurationError
        assert issubclass(InvalidParameterError, ConfigurationError)
        assert issubclass(MissingParameterError, ConfigurationError)
        assert issubclass(IncompatibleParametersError, ConfigurationError)
        assert issubclass(ConfigValidationError, ConfigurationError)
        
        # ConfigurationError应该继承自PokerAIError
        assert issubclass(ConfigurationError, PokerAIError)


class TestRuntimeErrors:
    """运行时错误测试类"""
    
    def test_disk_space_error(self):
        """测试磁盘空间不足错误"""
        error = DiskSpaceError(
            path='/tmp',
            message='磁盘空间不足',
            required_space=1024 * 1024 * 100,  # 100MB
            available_space=1024 * 1024 * 10   # 10MB
        )
        
        assert '/tmp' in str(error) or error.path == '/tmp'
        assert error.required_space == 1024 * 1024 * 100
        assert error.available_space == 1024 * 1024 * 10
    
    def test_gpu_error(self):
        """测试GPU错误"""
        error = GPUError(message='CUDA内存不足', gpu_id=0)
        
        assert 'CUDA内存不足' in str(error)
        assert error.gpu_id == 0
    
    def test_training_interrupted_error(self):
        """测试训练中断错误"""
        error = TrainingInterruptedError(
            message='用户手动停止训练',
            episode_number=5000,
            checkpoint_saved=True
        )
        
        assert error.episode_number == 5000
        assert error.checkpoint_saved is True
    
    def test_resource_error_inheritance(self):
        """测试资源错误的继承关系"""
        assert issubclass(DiskSpaceError, ResourceError)
        assert issubclass(GPUError, ResourceError)


class TestDataErrors:
    """数据错误测试类 - 验证需求9.4"""
    
    def test_checkpoint_corrupted_error(self):
        """测试检查点文件损坏错误"""
        error = CheckpointCorruptedError(
            checkpoint_path='/path/to/checkpoint.pt',
            message='文件格式无效'
        )
        
        assert '/path/to/checkpoint.pt' in str(error) or error.checkpoint_path == '/path/to/checkpoint.pt'
        assert error.checkpoint_path == '/path/to/checkpoint.pt'
    
    def test_checkpoint_not_found_error(self):
        """测试检查点文件不存在错误"""
        error = CheckpointNotFoundError(checkpoint_path='/path/to/missing.pt')
        
        assert '/path/to/missing.pt' in str(error)
        assert error.checkpoint_path == '/path/to/missing.pt'
    
    def test_config_file_error(self):
        """测试配置文件错误"""
        error = ConfigFileError(
            file_path='/path/to/config.json',
            message='JSON解析失败'
        )
        
        assert '/path/to/config.json' in str(error) or error.file_path == '/path/to/config.json'
    
    def test_log_file_error(self):
        """测试日志文件错误"""
        error = LogFileError(
            file_path='/path/to/log.jsonl',
            operation='write',
            message='磁盘空间不足'
        )
        
        assert error.file_path == '/path/to/log.jsonl'
        assert error.operation == 'write'
    
    def test_data_serialization_error(self):
        """测试数据序列化错误"""
        error = DataSerializationError(
            data_type='Episode',
            operation='serialize'
        )
        
        assert 'Episode' in str(error)
        assert error.data_type == 'Episode'
        assert error.operation == 'serialize'
    
    def test_data_error_inheritance(self):
        """测试数据错误的继承关系"""
        assert issubclass(CheckpointError, DataError)
        assert issubclass(CheckpointCorruptedError, CheckpointError)
        assert issubclass(CheckpointNotFoundError, CheckpointError)
        assert issubclass(ConfigFileError, DataError)
        assert issubclass(LogFileError, DataError)
        assert issubclass(DataSerializationError, DataError)


class TestGameErrors:
    """游戏逻辑错误测试类"""
    
    def test_illegal_action_error(self):
        """测试非法行动错误"""
        error = IllegalActionError(
            action='RAISE 100',
            reason='加注金额超过筹码'
        )
        
        assert '加注金额超过筹码' in str(error)
        assert error.reason == '加注金额超过筹码'
    
    def test_invalid_game_state_error(self):
        """测试无效游戏状态错误"""
        error = InvalidGameStateError(
            state_description='底池为负数',
            reason='底池不能为负数'
        )
        
        assert '底池不能为负数' in str(error)
    
    def test_game_not_started_error(self):
        """测试游戏未开始错误"""
        error = GameNotStartedError()
        
        assert '游戏尚未开始' in str(error)
    
    def test_game_already_ended_error(self):
        """测试游戏已结束错误"""
        error = GameAlreadyEndedError()
        
        assert '游戏已结束' in str(error)


class TestParallelTrainingErrors:
    """并行训练错误测试类"""
    
    def test_worker_process_error(self):
        """测试工作进程错误"""
        original_error = RuntimeError('进程崩溃')
        error = WorkerProcessError(
            worker_id=3,
            original_error=original_error
        )
        
        assert '3' in str(error)
        assert '进程崩溃' in str(error)
        assert error.worker_id == 3
    
    def test_inter_process_communication_error(self):
        """测试进程间通信错误"""
        error = InterProcessCommunicationError(
            message='消息队列超时',
            source_process=0,
            target_process=1
        )
        
        assert '消息队列超时' in str(error)
        assert error.source_process == 0
        assert error.target_process == 1
    
    def test_parameter_sync_error(self):
        """测试参数同步错误"""
        error = ParameterSyncError(message='参数广播失败')
        
        assert '参数广播失败' in str(error)


class TestModelErrors:
    """模型错误测试类"""
    
    def test_model_load_error(self):
        """测试模型加载错误"""
        error = ModelLoadError(model_path='/path/to/model.pt')
        
        assert '/path/to/model.pt' in str(error)
        assert error.model_path == '/path/to/model.pt'
    
    def test_model_save_error(self):
        """测试模型保存错误"""
        error = ModelSaveError(
            model_path='/path/to/model.pt',
            message='权限不足'
        )
        
        assert '权限不足' in str(error)


class TestLoggerSystem:
    """日志系统测试类"""
    
    def setup_method(self):
        """每个测试前清除日志器缓存"""
        clear_loggers()
    
    def teardown_method(self):
        """每个测试后清除日志器缓存"""
        clear_loggers()
    
    def test_logger_config_validation(self):
        """测试日志配置验证"""
        # 有效配置
        config = LoggerConfig(level='DEBUG')
        assert config.level == 'DEBUG'
        
        # 无效日志级别应该抛出异常
        with pytest.raises(ValueError):
            LoggerConfig(level='INVALID')
    
    def test_setup_logger(self):
        """测试日志器设置"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoggerConfig(
                level='DEBUG',
                log_dir=tmpdir,
                log_file='test.log',
                console_output=False,
                file_output=True
            )
            
            logger = setup_logger('test_logger', config)
            
            assert logger is not None
            assert logger.level == logging.DEBUG
    
    def test_get_logger_creates_new(self):
        """测试获取日志器会创建新实例"""
        logger1 = get_logger('new_logger')
        logger2 = get_logger('new_logger')
        
        # 应该返回同一个实例
        assert logger1 is logger2
    
    def test_set_log_level(self):
        """测试设置日志级别"""
        logger = get_logger('level_test')
        
        set_log_level('level_test', 'ERROR')
        assert logger.level == logging.ERROR
        
        set_log_level('level_test', 'DEBUG')
        assert logger.level == logging.DEBUG
    
    def test_set_invalid_log_level(self):
        """测试设置无效日志级别"""
        get_logger('invalid_level_test')
        
        with pytest.raises(ValueError):
            set_log_level('invalid_level_test', 'INVALID')
    
    def test_logger_mixin(self):
        """测试日志器混入类"""
        class TestClass(LoggerMixin):
            def log_something(self):
                self.logger.info('测试消息')
                return self.logger
        
        obj = TestClass()
        logger = obj.log_something()
        
        assert logger is not None
        assert logger.name == 'TestClass'
    
    def test_logger_writes_to_file(self):
        """测试日志写入文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoggerConfig(
                level='INFO',
                log_dir=tmpdir,
                log_file='test.log',
                console_output=False,
                file_output=True
            )
            
            logger = setup_logger('file_test', config)
            logger.info('测试日志消息')
            
            # 强制刷新处理器
            for handler in logger.handlers:
                handler.flush()
            
            log_path = Path(tmpdir) / 'test.log'
            assert log_path.exists()
            
            content = log_path.read_text(encoding='utf-8')
            assert '测试日志消息' in content


class TestErrorRecovery:
    """错误恢复功能测试类"""
    
    def test_retry_decorator_success(self):
        """测试重试装饰器成功情况"""
        call_count = 0
        
        @retry(RetryConfig(max_retries=3, initial_delay=0.01))
        def successful_func():
            nonlocal call_count
            call_count += 1
            return 'success'
        
        result = successful_func()
        
        assert result == 'success'
        assert call_count == 1
    
    def test_retry_decorator_eventual_success(self):
        """测试重试装饰器最终成功"""
        call_count = 0
        
        @retry(RetryConfig(max_retries=3, initial_delay=0.01))
        def eventually_successful_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise IOError('临时错误')
            return 'success'
        
        result = eventually_successful_func()
        
        assert result == 'success'
        assert call_count == 3
    
    def test_retry_decorator_failure(self):
        """测试重试装饰器最终失败"""
        call_count = 0
        
        @retry(RetryConfig(max_retries=2, initial_delay=0.01))
        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise IOError('持续错误')
        
        with pytest.raises(IOError):
            always_failing_func()
        
        assert call_count == 3  # 初始尝试 + 2次重试
    
    def test_retry_decorator_non_retryable_exception(self):
        """测试重试装饰器不重试非指定异常"""
        call_count = 0
        
        @retry(RetryConfig(max_retries=3, initial_delay=0.01, retry_exceptions=[IOError]))
        def value_error_func():
            nonlocal call_count
            call_count += 1
            raise ValueError('不应重试')
        
        with pytest.raises(ValueError):
            value_error_func()
        
        assert call_count == 1  # 只调用一次，不重试
    
    def test_safe_execute_success(self):
        """测试安全执行成功情况"""
        def successful_func():
            return 42
        
        result = safe_execute(successful_func)
        
        assert result == 42
    
    def test_safe_execute_failure(self):
        """测试安全执行失败情况"""
        def failing_func():
            raise RuntimeError('错误')
        
        result = safe_execute(failing_func, default='default', log_error=False)
        
        assert result == 'default'
    
    def test_check_disk_space_sufficient(self):
        """测试磁盘空间检查（空间充足）"""
        # 检查当前目录，应该有足够空间
        result = check_disk_space('.', 1024)  # 1KB
        assert result is True
    
    def test_check_disk_space_insufficient(self):
        """测试磁盘空间检查（空间不足）"""
        # 请求一个非常大的空间
        with pytest.raises(DiskSpaceError):
            check_disk_space('.', 10 ** 18)  # 1EB
    
    def test_check_gpu_availability(self):
        """测试GPU可用性检查"""
        available, info = check_gpu_availability()
        
        # 无论GPU是否可用，都应该返回有效结果
        assert isinstance(available, bool)
        assert isinstance(info, str)
    
    def test_get_device_cpu(self):
        """测试获取设备（CPU）"""
        device = get_device(prefer_gpu=False)
        assert device == 'cpu'
    
    def test_recovery_manager_initialization(self):
        """测试恢复管理器初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RecoveryManager(checkpoint_dir=tmpdir)
            
            assert manager.checkpoint_dir == Path(tmpdir)
            assert manager.auto_save is True
    
    def test_recovery_manager_find_latest_checkpoint(self):
        """测试查找最新检查点"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RecoveryManager(checkpoint_dir=tmpdir)
            
            # 没有检查点时应返回None
            assert manager.find_latest_checkpoint() is None
            
            # 创建一些假的检查点文件
            import time
            for i in range(3):
                checkpoint_path = Path(tmpdir) / f'checkpoint_{i}.pt'
                checkpoint_path.write_text('fake checkpoint')
                time.sleep(0.01)  # 确保时间戳不同
            
            latest = manager.find_latest_checkpoint()
            assert latest is not None
            assert 'checkpoint_2.pt' in latest
    
    def test_recovery_manager_record_error(self):
        """测试记录错误"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RecoveryManager(checkpoint_dir=tmpdir)
            
            error = InvalidParameterError('test', 'value', 'reason')
            # 应该不抛出异常
            manager.record_error(error, context={'episode': 100})


class TestExceptionHierarchy:
    """异常层次结构测试类"""
    
    def test_all_exceptions_inherit_from_poker_ai_error(self):
        """测试所有自定义异常都继承自PokerAIError"""
        exception_classes = [
            ConfigurationError,
            InvalidParameterError,
            MissingParameterError,
            IncompatibleParametersError,
            ConfigValidationError,
            ResourceError,
            DiskSpaceError,
            GPUError,
            TrainingInterruptedError,
            DataError,
            CheckpointError,
            CheckpointCorruptedError,
            CheckpointNotFoundError,
            ConfigFileError,
            LogFileError,
            DataSerializationError,
            GameError,
            IllegalActionError,
            InvalidGameStateError,
            GameNotStartedError,
            GameAlreadyEndedError,
            ParallelTrainingError,
            WorkerProcessError,
            InterProcessCommunicationError,
            ParameterSyncError,
            ModelError,
            ModelLoadError,
            ModelSaveError,
            EvaluationError,
            InvalidOpponentError,
        ]
        
        for exc_class in exception_classes:
            assert issubclass(exc_class, PokerAIError), \
                f"{exc_class.__name__} 应该继承自 PokerAIError"
    
    def test_poker_ai_error_with_details(self):
        """测试PokerAIError的详情功能"""
        error = PokerAIError('测试错误', details={'key': 'value'})
        
        assert error.message == '测试错误'
        assert error.details == {'key': 'value'}
        assert '详情' in str(error)
    
    def test_poker_ai_error_without_details(self):
        """测试PokerAIError无详情"""
        error = PokerAIError('测试错误')
        
        assert error.message == '测试错误'
        assert error.details is None
        assert '详情' not in str(error)


class TestIntegration:
    """集成测试类"""
    
    def test_config_manager_with_custom_exceptions(self):
        """测试配置管理器使用自定义异常"""
        from utils.config_manager import ConfigManager
        
        manager = ConfigManager()
        
        # 测试无效配置
        invalid_config = {
            'learning_rate': -0.1,  # 无效
            'batch_size': 32
        }
        
        errors = manager.validate_config(invalid_config)
        assert len(errors) > 0
        assert any('learning_rate' in err for err in errors)
    
    def test_data_logger_io_error_handling(self):
        """测试数据日志器IO错误处理"""
        from utils.data_logger import DataLogger
        from models.core import Episode, GameState, Action, ActionType, GameStage, Card
        
        # 创建一个只读目录来模拟IO错误
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(log_dir=tmpdir)
            
            # 创建一个简单的Episode
            state = GameState(
                player_hands=[
                    (Card(14, 'h'), Card(13, 'h')),
                    (Card(2, 's'), Card(3, 's'))
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
            
            episode = Episode(
                states=[state, state],
                actions=[Action(ActionType.CALL, 0)],
                rewards=[0.0],
                player_id=0,
                final_reward=10.0
            )
            
            # 正常写入应该成功
            logger.write_episode(episode, 1)
            
            # 验证数据被写入
            records = logger.read_episodes()
            assert len(records) == 1
