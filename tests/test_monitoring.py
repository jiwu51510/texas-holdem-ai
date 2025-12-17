"""监控系统测试模块。

测试MetricsCollector和TrainingMonitor的功能。
"""

import json
import tempfile
import time
from pathlib import Path
from typing import List
from datetime import datetime

import pytest
from hypothesis import given, strategies as st, settings

from models.core import (
    Card, Action, ActionType, GameState, GameStage, Episode
)
from monitoring.metrics_collector import MetricsCollector, EpisodeRecord
from monitoring.training_monitor import TrainingMonitor, LogEntry


# ============== 辅助函数 ==============

def create_simple_game_state() -> GameState:
    """创建一个简单的游戏状态用于测试。"""
    return GameState(
        player_hands=[
            (Card(14, 'h'), Card(13, 'h')),  # 玩家0: AK
            (Card(10, 's'), Card(10, 'd'))   # 玩家1: TT
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


def create_test_episode(player_id: int = 0, final_reward: float = 100.0) -> Episode:
    """创建一个测试用的Episode。
    
    Args:
        player_id: 玩家ID
        final_reward: 最终奖励
        
    Returns:
        测试用的Episode对象
    """
    state1 = create_simple_game_state()
    state2 = GameState(
        player_hands=state1.player_hands,
        community_cards=[Card(2, 'c'), Card(7, 'd'), Card(9, 's')],
        pot=30,
        player_stacks=[985, 985],
        current_bets=[0, 0],
        button_position=0,
        stage=GameStage.FLOP,
        action_history=[Action(ActionType.CALL, 0)],
        current_player=1
    )
    
    return Episode(
        states=[state1, state2],
        actions=[Action(ActionType.CALL, 0)],
        rewards=[0.0],
        player_id=player_id,
        final_reward=final_reward
    )


# ============== MetricsCollector 单元测试 ==============

class TestMetricsCollector:
    """MetricsCollector单元测试类。"""
    
    def test_init_default_window_size(self):
        """测试默认窗口大小初始化。"""
        collector = MetricsCollector()
        assert collector.window_size == 100
        assert collector.episodes_completed == 0
    
    def test_init_custom_window_size(self):
        """测试自定义窗口大小初始化。"""
        collector = MetricsCollector(window_size=50)
        assert collector.window_size == 50
    
    def test_init_invalid_window_size(self):
        """测试无效窗口大小抛出异常。"""
        with pytest.raises(ValueError):
            MetricsCollector(window_size=0)
        with pytest.raises(ValueError):
            MetricsCollector(window_size=-10)
    
    def test_record_episode_increments_count(self):
        """测试记录回合增加计数。"""
        collector = MetricsCollector()
        episode = create_test_episode()
        
        collector.record_episode(episode)
        assert collector.episodes_completed == 1
        
        collector.record_episode(episode)
        assert collector.episodes_completed == 2
    
    def test_record_episode_updates_win_rate(self):
        """测试记录回合更新胜率。"""
        collector = MetricsCollector(window_size=10)
        
        # 记录5个获胜回合
        for _ in range(5):
            collector.record_episode(create_test_episode(final_reward=100.0))
        
        # 记录5个失败回合
        for _ in range(5):
            collector.record_episode(create_test_episode(final_reward=-100.0))
        
        metrics = collector.get_current_metrics()
        assert metrics['win_rate'] == 0.5  # 5胜5负
    
    def test_record_loss(self):
        """测试记录损失值。"""
        collector = MetricsCollector()
        
        collector.record_loss(0.5)
        collector.record_loss(0.3)
        
        metrics = collector.get_current_metrics()
        assert metrics['loss'] == 0.3  # 最近的损失值
    
    def test_get_current_metrics_contains_required_fields(self):
        """测试当前指标包含所有必需字段。"""
        collector = MetricsCollector()
        collector.record_episode(create_test_episode())
        
        metrics = collector.get_current_metrics()
        
        assert 'win_rate' in metrics
        assert 'avg_reward' in metrics
        assert 'loss' in metrics
        assert 'episodes_completed' in metrics
    
    def test_get_metric_history(self):
        """测试获取指标历史。"""
        collector = MetricsCollector()
        
        for i in range(5):
            collector.record_episode(create_test_episode(final_reward=float(i * 10)))
        
        history = collector.get_metric_history('episodes_completed')
        assert len(history) == 5
        assert history == [1.0, 2.0, 3.0, 4.0, 5.0]
    
    def test_get_metric_history_invalid_name(self):
        """测试获取无效指标名称抛出异常。"""
        collector = MetricsCollector()
        
        with pytest.raises(ValueError):
            collector.get_metric_history('invalid_metric')
    
    def test_sliding_window_behavior(self):
        """测试滑动窗口行为。"""
        collector = MetricsCollector(window_size=5)
        
        # 记录5个获胜回合
        for _ in range(5):
            collector.record_episode(create_test_episode(final_reward=100.0))
        
        assert collector.get_current_metrics()['win_rate'] == 1.0
        
        # 再记录5个失败回合，窗口应该只包含最近5个
        for _ in range(5):
            collector.record_episode(create_test_episode(final_reward=-100.0))
        
        assert collector.get_current_metrics()['win_rate'] == 0.0
    
    def test_get_global_stats(self):
        """测试获取全局统计。"""
        collector = MetricsCollector(window_size=5)
        
        # 记录10个回合：5胜5负
        for _ in range(5):
            collector.record_episode(create_test_episode(final_reward=100.0))
        for _ in range(5):
            collector.record_episode(create_test_episode(final_reward=-100.0))
        
        stats = collector.get_global_stats()
        assert stats['total_episodes'] == 10.0
        assert stats['total_wins'] == 5.0
        assert stats['global_win_rate'] == 0.5
    
    def test_reset(self):
        """测试重置功能。"""
        collector = MetricsCollector()
        
        for _ in range(5):
            collector.record_episode(create_test_episode())
        collector.record_loss(0.5)
        
        collector.reset()
        
        assert collector.episodes_completed == 0
        metrics = collector.get_current_metrics()
        assert metrics['win_rate'] == 0.0
        assert metrics['avg_reward'] == 0.0
        assert metrics['loss'] == 0.0


# ============== TrainingMonitor 单元测试 ==============

class TestTrainingMonitor:
    """TrainingMonitor单元测试类。"""
    
    def test_init_default_values(self):
        """测试默认值初始化。"""
        collector = MetricsCollector()
        monitor = TrainingMonitor(collector)
        
        assert monitor.metrics_collector is collector
        assert monitor.update_interval == 1.0
        assert monitor.log_file_path is None
        assert not monitor.is_running
    
    def test_init_custom_values(self):
        """测试自定义值初始化。"""
        collector = MetricsCollector()
        monitor = TrainingMonitor(
            collector,
            update_interval=0.5,
            log_file_path='/tmp/test.log'
        )
        
        assert monitor.update_interval == 0.5
        assert monitor.log_file_path == Path('/tmp/test.log')
    
    def test_init_invalid_update_interval(self):
        """测试无效更新间隔抛出异常。"""
        collector = MetricsCollector()
        
        with pytest.raises(ValueError):
            TrainingMonitor(collector, update_interval=0)
        with pytest.raises(ValueError):
            TrainingMonitor(collector, update_interval=-1)
    
    def test_update_returns_metrics(self):
        """测试更新返回指标。"""
        collector = MetricsCollector()
        collector.record_episode(create_test_episode())
        
        monitor = TrainingMonitor(collector)
        metrics = monitor.update()
        
        assert 'win_rate' in metrics
        assert 'avg_reward' in metrics
        assert 'loss' in metrics
        assert 'episodes_completed' in metrics
    
    def test_update_with_custom_metrics(self):
        """测试使用自定义指标更新。"""
        collector = MetricsCollector()
        monitor = TrainingMonitor(collector)
        
        custom_metrics = {
            'win_rate': 0.75,
            'avg_reward': 50.0,
            'loss': 0.1,
            'episodes_completed': 100.0
        }
        
        result = monitor.update(custom_metrics)
        assert result == custom_metrics
        assert monitor.get_latest_metrics() == custom_metrics
    
    def test_start_and_stop(self):
        """测试启动和停止监控。"""
        collector = MetricsCollector()
        monitor = TrainingMonitor(collector, update_interval=0.1)
        
        assert not monitor.is_running
        
        monitor.start()
        assert monitor.is_running
        
        time.sleep(0.2)  # 等待至少一次更新
        
        monitor.stop()
        assert not monitor.is_running
    
    def test_log_file_writing(self):
        """测试日志文件写入。"""
        collector = MetricsCollector()
        collector.record_episode(create_test_episode())
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            log_path = f.name
        
        try:
            monitor = TrainingMonitor(collector, log_file_path=log_path)
            monitor.update()
            
            # 读取日志文件
            with open(log_path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert 'timestamp' in entry
            assert 'episode_number' in entry
            assert 'metrics' in entry
        finally:
            Path(log_path).unlink(missing_ok=True)
    
    def test_read_log_file(self):
        """测试读取日志文件。"""
        collector = MetricsCollector()
        collector.record_episode(create_test_episode())
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            log_path = f.name
        
        try:
            monitor = TrainingMonitor(collector, log_file_path=log_path)
            
            # 写入多个条目
            for _ in range(3):
                monitor.update()
            
            # 读取日志
            entries = monitor.read_log_file()
            assert len(entries) == 3
            
            for entry in entries:
                assert isinstance(entry, LogEntry)
                assert entry.timestamp
                assert 'win_rate' in entry.metrics
        finally:
            Path(log_path).unlink(missing_ok=True)
    
    def test_read_log_file_no_path(self):
        """测试未设置日志路径时读取抛出异常。"""
        collector = MetricsCollector()
        monitor = TrainingMonitor(collector)
        
        with pytest.raises(ValueError):
            monitor.read_log_file()
    
    def test_check_anomalies_normal(self):
        """测试正常情况下无异常。"""
        collector = MetricsCollector()
        monitor = TrainingMonitor(collector)
        
        metrics = {'loss': 0.5, 'win_rate': 0.5, 'avg_reward': 10.0}
        anomalies = monitor.check_anomalies(metrics)
        
        assert len(anomalies) == 0
    
    def test_check_anomalies_high_loss(self):
        """测试高损失值检测。"""
        collector = MetricsCollector()
        monitor = TrainingMonitor(collector, loss_threshold=10.0)
        
        metrics = {'loss': 100.0, 'win_rate': 0.5, 'avg_reward': 10.0}
        anomalies = monitor.check_anomalies(metrics)
        
        assert len(anomalies) > 0
        assert any('超过阈值' in a for a in anomalies)
    
    def test_check_anomalies_nan_loss(self):
        """测试NaN损失值检测。"""
        collector = MetricsCollector()
        monitor = TrainingMonitor(collector)
        
        metrics = {'loss': float('nan'), 'win_rate': 0.5, 'avg_reward': 10.0}
        anomalies = monitor.check_anomalies(metrics)
        
        assert len(anomalies) > 0
        assert any('NaN' in a for a in anomalies)
    
    def test_check_anomalies_inf_loss(self):
        """测试无穷大损失值检测。"""
        collector = MetricsCollector()
        monitor = TrainingMonitor(collector)
        
        metrics = {'loss': float('inf'), 'win_rate': 0.5, 'avg_reward': 10.0}
        anomalies = monitor.check_anomalies(metrics)
        
        assert len(anomalies) > 0
        assert any('无穷大' in a for a in anomalies)
    
    def test_check_anomalies_diverging_loss(self):
        """测试损失值发散检测。"""
        collector = MetricsCollector()
        monitor = TrainingMonitor(collector, loss_increase_threshold=2.0)
        
        # 模拟损失值持续增长
        for loss in [1.0, 2.0, 4.0, 8.0, 16.0]:
            metrics = {'loss': loss, 'win_rate': 0.5, 'avg_reward': 10.0}
            anomalies = monitor.check_anomalies(metrics)
        
        # 最后一次应该检测到发散
        assert len(anomalies) > 0
        assert any('发散' in a for a in anomalies)
    
    def test_on_update_callback(self):
        """测试更新回调函数。"""
        collector = MetricsCollector()
        collector.record_episode(create_test_episode())
        monitor = TrainingMonitor(collector)
        
        callback_called = []
        
        def callback(metrics):
            callback_called.append(metrics)
        
        monitor.set_on_update_callback(callback)
        monitor.update()
        
        assert len(callback_called) == 1
        assert 'win_rate' in callback_called[0]
    
    def test_on_anomaly_callback(self):
        """测试异常回调函数。"""
        collector = MetricsCollector()
        monitor = TrainingMonitor(collector, loss_threshold=10.0)
        
        anomaly_list = []
        
        def callback(anomalies):
            anomaly_list.extend(anomalies)
        
        monitor.set_on_anomaly_callback(callback)
        monitor.update({'loss': 100.0, 'win_rate': 0.5, 'avg_reward': 10.0, 'episodes_completed': 1.0})
        
        assert len(anomaly_list) > 0
    
    def test_last_update_time(self):
        """测试上次更新时间。"""
        collector = MetricsCollector()
        monitor = TrainingMonitor(collector)
        
        assert monitor.last_update_time is None
        
        before = datetime.now()
        monitor.update()
        after = datetime.now()
        
        assert monitor.last_update_time is not None
        assert before <= monitor.last_update_time <= after


# ============== 属性测试 ==============

# Hypothesis策略：生成有效的Episode
@st.composite
def episode_strategy(draw):
    """生成随机的Episode用于测试。"""
    player_id = draw(st.integers(min_value=0, max_value=1))
    final_reward = draw(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
    return create_test_episode(player_id=player_id, final_reward=final_reward)


class TestMonitoringProperties:
    """监控系统属性测试类。"""
    
    @given(st.floats(min_value=0.05, max_value=0.2))
    @settings(max_examples=100, deadline=None)
    def test_property_9_update_frequency(self, update_interval: float):
        """属性9：监控指标更新频率。
        
        Feature: texas-holdem-ai-training, Property 9: 监控指标更新频率
        验证在指定时间间隔内至少更新一次。
        验证需求：3.1
        """
        collector = MetricsCollector()
        collector.record_episode(create_test_episode())
        
        monitor = TrainingMonitor(collector, update_interval=update_interval)
        
        monitor.start()
        try:
            # 等待足够的时间让监控器至少更新一次
            time.sleep(update_interval * 2.5)
            
            # 验证已经更新过
            assert monitor.last_update_time is not None
        finally:
            monitor.stop()
    
    @given(episode_strategy())
    @settings(max_examples=100)
    def test_property_10_metrics_completeness(self, episode: Episode):
        """属性10：监控指标完整性。
        
        Feature: texas-holdem-ai-training, Property 10: 监控指标完整性
        验证返回的指标包含胜率、平均奖励、损失值、回合数。
        验证需求：3.2
        """
        collector = MetricsCollector()
        collector.record_episode(episode)
        
        metrics = collector.get_current_metrics()
        
        # 验证所有必需字段都存在
        required_fields = ['win_rate', 'avg_reward', 'loss', 'episodes_completed']
        for field in required_fields:
            assert field in metrics, f"缺少必需字段: {field}"
        
        # 验证字段类型
        assert isinstance(metrics['win_rate'], float)
        assert isinstance(metrics['avg_reward'], float)
        assert isinstance(metrics['loss'], float)
        assert isinstance(metrics['episodes_completed'], float)
        
        # 验证胜率在有效范围内
        assert 0.0 <= metrics['win_rate'] <= 1.0
    
    @given(st.lists(episode_strategy(), min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_property_11_persistence_consistency(self, episodes: List[Episode]):
        """属性11：指标持久化一致性。
        
        Feature: texas-holdem-ai-training, Property 11: 指标持久化一致性
        验证日志文件中的数据与内存中的指标一致。
        验证需求：3.3
        """
        collector = MetricsCollector()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            log_path = f.name
        
        try:
            monitor = TrainingMonitor(collector, log_file_path=log_path)
            
            # 记录所有回合并更新监控器
            for episode in episodes:
                collector.record_episode(episode)
                monitor.update()
            
            # 获取内存中的最新指标
            memory_metrics = collector.get_current_metrics()
            
            # 读取日志文件中的最后一条记录
            log_entries = monitor.read_log_file()
            assert len(log_entries) == len(episodes)
            
            last_entry = log_entries[-1]
            
            # 验证日志中的指标与内存中的一致
            for key in ['win_rate', 'avg_reward', 'loss', 'episodes_completed']:
                assert abs(last_entry.metrics[key] - memory_metrics[key]) < 1e-6, \
                    f"指标 {key} 不一致: 日志={last_entry.metrics[key]}, 内存={memory_metrics[key]}"
        finally:
            Path(log_path).unlink(missing_ok=True)
