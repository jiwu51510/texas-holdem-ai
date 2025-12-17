"""CLI命令行界面的单元测试和属性测试。

本模块测试CLI的以下功能：
- 命令行参数解析
- 各个子命令的调用
- 无效参数的错误处理
- 帮助信息显示
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from hypothesis import given, strategies as st, settings

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import (
    create_parser, main, cmd_train, cmd_evaluate, cmd_analyze, cmd_list,
    cmd_generate_abstraction, cmd_abstraction_info, _format_time,
    _create_sample_state, _parse_game_state, _generate_sample_hand_range
)
from models.core import GameState, GameStage, Card, TrainingConfig


# ============================================================================
# 参数解析测试
# ============================================================================

class TestArgumentParsing:
    """测试命令行参数解析。"""
    
    def test_parser_creation(self):
        """测试解析器创建成功。"""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == 'poker-ai'
    
    def test_train_command_parsing(self):
        """测试train命令参数解析。"""
        parser = create_parser()
        
        # 基本参数
        args = parser.parse_args(['train'])
        assert args.command == 'train'
        assert args.checkpoint_dir == 'checkpoints'
        
        # 带配置文件
        args = parser.parse_args(['train', '--config', 'config.json'])
        assert args.config == 'config.json'
        
        # 带回合数
        args = parser.parse_args(['train', '--episodes', '1000'])
        assert args.episodes == 1000
        
        # 带学习率
        args = parser.parse_args(['train', '-lr', '0.001'])
        assert args.learning_rate == 0.001
        
        # 带批次大小
        args = parser.parse_args(['train', '-b', '64'])
        assert args.batch_size == 64
        
        # 带恢复检查点
        args = parser.parse_args(['train', '--resume', 'checkpoint.pt'])
        assert args.resume == 'checkpoint.pt'
    
    def test_evaluate_command_parsing(self):
        """测试evaluate命令参数解析。"""
        parser = create_parser()
        
        # 必需参数
        args = parser.parse_args(['evaluate', '--model', 'model.pt'])
        assert args.command == 'evaluate'
        assert args.model == 'model.pt'
        assert args.opponent == 'random'  # 默认值
        assert args.games == 100  # 默认值
        
        # 带对手策略
        args = parser.parse_args(['evaluate', '-m', 'model.pt', '-o', 'aggressive'])
        assert args.opponent == 'aggressive'
        
        # 带对局数
        args = parser.parse_args(['evaluate', '-m', 'model.pt', '-g', '500'])
        assert args.games == 500
        
        # 带输出文件
        args = parser.parse_args(['evaluate', '-m', 'model.pt', '-out', 'result.json'])
        assert args.output == 'result.json'
    
    def test_analyze_command_parsing(self):
        """测试analyze命令参数解析。"""
        parser = create_parser()
        
        # 必需参数
        args = parser.parse_args(['analyze', '--model', 'model.pt'])
        assert args.command == 'analyze'
        assert args.model == 'model.pt'
        
        # 带状态
        args = parser.parse_args(['analyze', '-m', 'model.pt', '-s', '{"pot": 100}'])
        assert args.state == '{"pot": 100}'
        
        # 带热图标志
        args = parser.parse_args(['analyze', '-m', 'model.pt', '--heatmap'])
        assert args.heatmap is True
        
        # 带比较模型
        args = parser.parse_args(['analyze', '-m', 'model.pt', '--compare', 'model2.pt', 'model3.pt'])
        assert args.compare == ['model2.pt', 'model3.pt']
    
    def test_list_command_parsing(self):
        """测试list命令参数解析。"""
        parser = create_parser()
        
        # 默认参数
        args = parser.parse_args(['list'])
        assert args.command == 'list'
        assert args.checkpoint_dir == 'checkpoints'
        assert args.verbose is False
        
        # 带目录
        args = parser.parse_args(['list', '-d', 'my_checkpoints'])
        assert args.checkpoint_dir == 'my_checkpoints'
        
        # 带详细标志
        args = parser.parse_args(['list', '-v'])
        assert args.verbose is True
    
    def test_no_command_returns_none(self):
        """测试没有命令时返回None。"""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None
    
    def test_generate_abstraction_command_parsing(self):
        """测试generate-abstraction命令参数解析。"""
        parser = create_parser()
        
        # 必需参数
        args = parser.parse_args(['generate-abstraction', '--output', 'test_output'])
        assert args.command == 'generate-abstraction'
        assert args.output == 'test_output'
        assert args.flop_buckets == 5000  # 默认值
        assert args.turn_buckets == 5000  # 默认值
        assert args.river_buckets == 5000  # 默认值
        assert args.preflop_buckets == 169  # 默认值
        assert args.seed == 42  # 默认值
        assert args.potential_aware is True  # 默认值
        
        # 带自定义桶数量
        args = parser.parse_args([
            'generate-abstraction', '-o', 'output',
            '--flop-buckets', '1000',
            '--turn-buckets', '2000',
            '--river-buckets', '3000'
        ])
        assert args.flop_buckets == 1000
        assert args.turn_buckets == 2000
        assert args.river_buckets == 3000
        
        # 带随机种子
        args = parser.parse_args([
            'generate-abstraction', '-o', 'output',
            '--seed', '123'
        ])
        assert args.seed == 123
        
        # 禁用Potential-Aware
        args = parser.parse_args([
            'generate-abstraction', '-o', 'output',
            '--no-potential-aware'
        ])
        assert args.no_potential_aware is True
    
    def test_abstraction_info_command_parsing(self):
        """测试abstraction-info命令参数解析。"""
        parser = create_parser()
        
        # 必需参数
        args = parser.parse_args(['abstraction-info', '--path', 'test_path'])
        assert args.command == 'abstraction-info'
        assert args.path == 'test_path'
        assert args.verbose is False  # 默认值
        assert args.json is False  # 默认值
        
        # 带详细标志
        args = parser.parse_args(['abstraction-info', '-p', 'test_path', '-v'])
        assert args.verbose is True
        
        # 带JSON标志
        args = parser.parse_args(['abstraction-info', '-p', 'test_path', '--json'])
        assert args.json is True


# ============================================================================
# 子命令调用测试
# ============================================================================

class TestCommandExecution:
    """测试各个子命令的调用。"""
    
    def test_main_no_command_shows_help(self, capsys):
        """测试没有命令时显示帮助。"""
        result = main([])
        assert result == 0
        captured = capsys.readouterr()
        assert '德州扑克AI训练系统' in captured.out
        assert 'train' in captured.out
        assert 'evaluate' in captured.out
    
    def test_list_command_empty_directory(self, capsys):
        """测试list命令在空目录时的行为。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = main(['list', '-d', tmpdir])
            assert result == 0
            captured = capsys.readouterr()
            assert '没有找到检查点' in captured.out
    
    def test_list_command_nonexistent_directory(self, capsys):
        """测试list命令在目录不存在时的行为。"""
        result = main(['list', '-d', '/nonexistent/path'])
        assert result == 0
        captured = capsys.readouterr()
        assert '不存在' in captured.out
    
    def test_evaluate_command_missing_model(self, capsys):
        """测试evaluate命令缺少模型文件时的行为。"""
        result = main(['evaluate', '-m', '/nonexistent/model.pt'])
        assert result == 1
        captured = capsys.readouterr()
        assert '不存在' in captured.out
    
    def test_analyze_command_missing_model(self, capsys):
        """测试analyze命令缺少模型文件时的行为。"""
        result = main(['analyze', '-m', '/nonexistent/model.pt'])
        assert result == 1
        captured = capsys.readouterr()
        assert '不存在' in captured.out
    
    def test_train_command_invalid_config(self, capsys):
        """测试train命令使用无效配置文件时的行为。"""
        result = main(['train', '-c', '/nonexistent/config.json'])
        assert result == 1
        captured = capsys.readouterr()
        assert '不存在' in captured.out
    
    def test_generate_abstraction_invalid_buckets(self, capsys):
        """测试generate-abstraction命令使用无效桶数量时的行为。"""
        result = main(['generate-abstraction', '-o', 'test', '--flop-buckets', '-1'])
        assert result == 1
        captured = capsys.readouterr()
        assert '必须为正数' in captured.out
    
    def test_abstraction_info_nonexistent_path(self, capsys):
        """测试abstraction-info命令使用不存在路径时的行为。"""
        result = main(['abstraction-info', '-p', '/nonexistent/path'])
        assert result == 1
        captured = capsys.readouterr()
        assert '不存在' in captured.out


# ============================================================================
# 错误处理测试
# ============================================================================

class TestErrorHandling:
    """测试无效参数的错误处理。"""
    
    def test_evaluate_invalid_opponent(self):
        """测试evaluate命令使用无效对手策略时的行为。"""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['evaluate', '-m', 'model.pt', '-o', 'invalid_strategy'])
    
    def test_train_invalid_episodes(self):
        """测试train命令使用无效回合数时的行为。"""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['train', '-e', 'not_a_number'])
    
    def test_train_invalid_learning_rate(self):
        """测试train命令使用无效学习率时的行为。"""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['train', '-lr', 'not_a_float'])
    
    def test_evaluate_missing_required_model(self):
        """测试evaluate命令缺少必需的model参数时的行为。"""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['evaluate'])
    
    def test_analyze_missing_required_model(self):
        """测试analyze命令缺少必需的model参数时的行为。"""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['analyze'])
    
    def test_generate_abstraction_missing_required_output(self):
        """测试generate-abstraction命令缺少必需的output参数时的行为。"""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['generate-abstraction'])
    
    def test_abstraction_info_missing_required_path(self):
        """测试abstraction-info命令缺少必需的path参数时的行为。"""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['abstraction-info'])


# ============================================================================
# 帮助信息测试
# ============================================================================

class TestHelpInformation:
    """测试帮助信息显示。"""
    
    def test_main_help(self, capsys):
        """测试主帮助信息。"""
        with pytest.raises(SystemExit) as exc_info:
            main(['--help'])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert '德州扑克AI训练系统' in captured.out
        assert 'train' in captured.out
        assert 'evaluate' in captured.out
        assert 'analyze' in captured.out
        assert 'list' in captured.out
    
    def test_train_help(self, capsys):
        """测试train命令帮助信息。"""
        with pytest.raises(SystemExit) as exc_info:
            main(['train', '--help'])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert '--config' in captured.out
        assert '--episodes' in captured.out
        assert '--checkpoint-dir' in captured.out
        assert '--resume' in captured.out
    
    def test_evaluate_help(self, capsys):
        """测试evaluate命令帮助信息。"""
        with pytest.raises(SystemExit) as exc_info:
            main(['evaluate', '--help'])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert '--model' in captured.out
        assert '--opponent' in captured.out
        assert '--games' in captured.out
    
    def test_analyze_help(self, capsys):
        """测试analyze命令帮助信息。"""
        with pytest.raises(SystemExit) as exc_info:
            main(['analyze', '--help'])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert '--model' in captured.out
        assert '--state' in captured.out
        assert '--heatmap' in captured.out
    
    def test_list_help(self, capsys):
        """测试list命令帮助信息。"""
        with pytest.raises(SystemExit) as exc_info:
            main(['list', '--help'])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert '--checkpoint-dir' in captured.out
        assert '--verbose' in captured.out
    
    def test_generate_abstraction_help(self, capsys):
        """测试generate-abstraction命令帮助信息。"""
        with pytest.raises(SystemExit) as exc_info:
            main(['generate-abstraction', '--help'])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert '--output' in captured.out
        assert '--flop-buckets' in captured.out
        assert '--turn-buckets' in captured.out
        assert '--river-buckets' in captured.out
        assert '--seed' in captured.out
        assert '--potential-aware' in captured.out
    
    def test_abstraction_info_help(self, capsys):
        """测试abstraction-info命令帮助信息。"""
        with pytest.raises(SystemExit) as exc_info:
            main(['abstraction-info', '--help'])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert '--path' in captured.out
        assert '--verbose' in captured.out
        assert '--json' in captured.out


# ============================================================================
# 辅助函数测试
# ============================================================================

class TestHelperFunctions:
    """测试辅助函数。"""
    
    def test_create_sample_state(self):
        """测试创建示例游戏状态。"""
        state = _create_sample_state()
        assert isinstance(state, GameState)
        assert len(state.player_hands) == 2
        assert len(state.community_cards) == 3  # 翻牌
        assert state.stage == GameStage.FLOP
        assert state.pot == 100
    
    def test_parse_game_state(self):
        """测试解析游戏状态。"""
        state_dict = {
            'player_hands': [
                [{'rank': 14, 'suit': 'h'}, {'rank': 13, 'suit': 'h'}],
                [{'rank': 7, 'suit': 'd'}, {'rank': 2, 'suit': 'c'}]
            ],
            'community_cards': [
                {'rank': 12, 'suit': 'h'},
                {'rank': 11, 'suit': 'h'},
                {'rank': 5, 'suit': 's'}
            ],
            'pot': 200,
            'player_stacks': [900, 900],
            'current_bets': [50, 50],
            'button_position': 0,
            'stage': 'flop',
            'current_player': 0
        }
        
        state = _parse_game_state(state_dict)
        assert isinstance(state, GameState)
        assert state.pot == 200
        assert state.stage == GameStage.FLOP
        assert len(state.community_cards) == 3
    
    def test_generate_sample_hand_range(self):
        """测试生成示例手牌范围。"""
        hands = _generate_sample_hand_range()
        assert len(hands) > 0
        assert len(hands) <= 20
        for hand in hands:
            assert len(hand) == 2
            assert isinstance(hand[0], Card)
            assert isinstance(hand[1], Card)
    
    def test_format_time_seconds(self):
        """测试时间格式化（秒）。"""
        result = _format_time(30.5)
        assert '30.5秒' in result
    
    def test_format_time_minutes(self):
        """测试时间格式化（分钟）。"""
        result = _format_time(125.5)
        assert '2分' in result
        assert '5.5秒' in result
    
    def test_format_time_hours(self):
        """测试时间格式化（小时）。"""
        result = _format_time(3725.5)
        assert '1小时' in result
        assert '2分' in result


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试。"""
    
    def test_train_with_valid_config_file(self, capsys):
        """测试使用有效配置文件进行训练。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建配置文件
            config_path = Path(tmpdir) / 'config.json'
            config = {
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_episodes': 10,  # 很少的回合数用于测试
                'discount_factor': 0.99,
                'network_architecture': [64, 32],  # 小网络用于测试
                'checkpoint_interval': 5,
                'num_parallel_envs': 1,
                'initial_stack': 1000,
                'small_blind': 5,
                'big_blind': 10
            }
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            checkpoint_dir = Path(tmpdir) / 'checkpoints'
            
            # 运行训练（只运行很少的回合）
            result = main([
                'train',
                '-c', str(config_path),
                '-e', '2',  # 只运行2个回合
                '-d', str(checkpoint_dir)
            ])
            
            # 训练应该成功完成
            assert result == 0
            captured = capsys.readouterr()
            assert '训练完成' in captured.out or '已初始化' in captured.out
    
    def test_list_with_checkpoints(self, capsys):
        """测试列出检查点。"""
        import torch
        from models.networks import PolicyNetwork
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            
            # 创建一个模拟检查点
            model = PolicyNetwork(input_dim=370, hidden_dims=[64], action_dim=6)
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': None,
                'episode_number': 100,
                'timestamp': '2024-01-01T00:00:00',
                'win_rate': 0.55,
                'avg_reward': 10.5
            }
            checkpoint_path = checkpoint_dir / 'checkpoint_123456_100.pt'
            torch.save(checkpoint_data, checkpoint_path)
            
            # 列出检查点
            result = main(['list', '-d', str(checkpoint_dir)])
            assert result == 0
            captured = capsys.readouterr()
            assert '1 个检查点' in captured.out
            assert 'checkpoint_123456_100.pt' in captured.out


# ============================================================================
# 属性测试
# ============================================================================

class TestPropertyBasedTests:
    """基于属性的测试。"""
    
    # Feature: texas-holdem-ai-training, Property 36: 命令行训练启动正确性
    # 生成随机有效参数，验证train命令成功启动
    # **验证需求：10.1**
    @given(
        episodes=st.integers(min_value=1, max_value=100),
        learning_rate=st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False),
        batch_size=st.integers(min_value=1, max_value=128)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_36_train_command_valid_params(self, episodes, learning_rate, batch_size):
        """属性36：命令行训练启动正确性。
        
        对于任何有效的训练命令行参数，系统应该成功解析参数。
        """
        parser = create_parser()
        args = parser.parse_args([
            'train',
            '-e', str(episodes),
            '-lr', str(learning_rate),
            '-b', str(batch_size)
        ])
        
        assert args.command == 'train'
        assert args.episodes == episodes
        assert abs(args.learning_rate - learning_rate) < 1e-6
        assert args.batch_size == batch_size
    
    # Feature: texas-holdem-ai-training, Property 37: 命令行评估启动正确性
    # 生成随机有效参数，验证evaluate命令成功启动
    # **验证需求：10.2**
    @given(
        games=st.integers(min_value=1, max_value=1000),
        opponent=st.sampled_from(['random', 'fixed', 'call-only', 'always-fold', 'aggressive'])
    )
    @settings(max_examples=100, deadline=None)
    def test_property_37_evaluate_command_valid_params(self, games, opponent):
        """属性37：命令行评估启动正确性。
        
        对于任何有效的评估命令行参数，系统应该成功解析参数。
        """
        parser = create_parser()
        args = parser.parse_args([
            'evaluate',
            '-m', 'model.pt',
            '-g', str(games),
            '-o', opponent
        ])
        
        assert args.command == 'evaluate'
        assert args.model == 'model.pt'
        assert args.games == games
        assert args.opponent == opponent
    
    # Feature: texas-holdem-ai-training, Property 38: 命令行策略查看启动正确性
    # 生成随机有效参数，验证analyze命令成功启动
    # **验证需求：10.3**
    @given(
        heatmap=st.booleans(),
        output_file=st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
            min_size=1,
            max_size=20
        ).filter(lambda x: x.strip() != '' and not x.startswith('-'))
    )
    @settings(max_examples=100, deadline=None)
    def test_property_38_analyze_command_valid_params(self, heatmap, output_file):
        """属性38：命令行策略查看启动正确性。
        
        对于任何有效的策略查看命令行参数，系统应该成功解析参数。
        """
        parser = create_parser()
        cmd_args = ['analyze', '-m', 'model.pt']
        
        if heatmap:
            cmd_args.append('--heatmap')
        
        # 确保文件名以字母开头，避免被误认为参数
        output_path = f'output_{output_file}.json'
        cmd_args.extend(['-out', output_path])
        
        args = parser.parse_args(cmd_args)
        
        assert args.command == 'analyze'
        assert args.model == 'model.pt'
        assert args.heatmap == heatmap
        assert args.output == output_path
    
    # Feature: texas-holdem-ai-training, Property 39: 命令行错误处理正确性
    # 生成随机无效参数，验证显示帮助信息而不崩溃
    # **验证需求：10.4**
    @given(
        invalid_command=st.text(
            alphabet=st.characters(whitelist_categories=('L',)),
            min_size=1,
            max_size=10
        ).filter(lambda x: x not in ['train', 'evaluate', 'analyze', 'list', 'help', '-h', '--help'])
    )
    @settings(max_examples=100, deadline=None)
    def test_property_39_invalid_command_handling(self, invalid_command):
        """属性39：命令行错误处理正确性。
        
        对于任何无效的命令，系统应该优雅地处理而不崩溃。
        """
        # 无效命令应该导致解析错误或返回帮助
        try:
            result = main([invalid_command])
            # 如果没有抛出异常，应该返回0或1
            assert result in [0, 1]
        except SystemExit as e:
            # argparse可能会调用sys.exit
            assert e.code in [0, 1, 2]
    
    # Feature: texas-holdem-ai-training, Property 40: 命令行帮助信息完整性
    # 验证帮助信息列出所有命令及说明
    # **验证需求：10.5**
    @given(st.just(None))  # 不需要随机输入
    @settings(max_examples=1, deadline=None)
    def test_property_40_help_completeness(self, _):
        """属性40：命令行帮助信息完整性。
        
        帮助信息应该列出所有可用命令及其说明。
        """
        import io
        import contextlib
        
        # 捕获stdout
        f = io.StringIO()
        try:
            with contextlib.redirect_stdout(f):
                main(['--help'])
        except SystemExit:
            pass
        
        help_text = f.getvalue()
        
        # 如果stdout为空，尝试从parser获取帮助
        if not help_text:
            parser = create_parser()
            help_text = parser.format_help()
        
        # 验证所有命令都在帮助信息中
        required_commands = ['train', 'evaluate', 'analyze', 'list', 'generate-abstraction', 'abstraction-info']
        for cmd in required_commands:
            assert cmd in help_text, f"帮助信息中缺少命令: {cmd}"
        
        # 验证有描述信息
        assert '训练' in help_text or 'train' in help_text
        assert '评估' in help_text or 'evaluate' in help_text
        assert '分析' in help_text or 'analyze' in help_text
        assert '检查点' in help_text or 'checkpoint' in help_text
        assert '抽象' in help_text or 'abstraction' in help_text


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
