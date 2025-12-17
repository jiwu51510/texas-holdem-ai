"""DataLogger数据日志器的单元测试和属性测试。

测试覆盖：
- 写入和读取回合数据
- 数据查询和过滤
- CSV和JSON导出
- 磁盘空间不足错误处理
- 属性测试：数据完整性、往返一致性、索引正确性、导出格式正确性
"""

import pytest
import os
import json
import csv
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, mock_open
from hypothesis import given, strategies as st, settings

from utils.data_logger import DataLogger
from models.core import (
    Episode, GameState, Action, ActionType, GameStage, Card
)


# ============== 测试辅助函数 ==============

def create_test_card(rank: int = 14, suit: str = 'h') -> Card:
    """创建测试用的Card对象。"""
    return Card(rank=rank, suit=suit)


def create_test_action(action_type: ActionType = ActionType.CALL, amount: int = 0) -> Action:
    """创建测试用的Action对象。"""
    if action_type == ActionType.RAISE:
        return Action(action_type=action_type, amount=amount if amount > 0 else 10)
    return Action(action_type=action_type, amount=0)


def create_test_game_state(
    pot: int = 100,
    stage: GameStage = GameStage.PREFLOP
) -> GameState:
    """创建测试用的GameState对象。"""
    return GameState(
        player_hands=[
            (Card(14, 'h'), Card(13, 'h')),  # 玩家0: A♥ K♥
            (Card(12, 's'), Card(11, 's'))   # 玩家1: Q♠ J♠
        ],
        community_cards=[],
        pot=pot,
        player_stacks=[900, 900],
        current_bets=[50, 50],
        button_position=0,
        stage=stage,
        action_history=[],
        current_player=0
    )


def create_test_episode(
    player_id: int = 0,
    final_reward: float = 100.0,
    num_actions: int = 2
) -> Episode:
    """创建测试用的Episode对象。"""
    states = [create_test_game_state(pot=100 + i * 50) for i in range(num_actions + 1)]
    actions = [create_test_action(ActionType.CALL) for _ in range(num_actions)]
    rewards = [0.0] * (num_actions - 1) + [final_reward] if num_actions > 0 else []
    
    return Episode(
        states=states,
        actions=actions,
        rewards=rewards,
        player_id=player_id,
        final_reward=final_reward
    )


# ============== 单元测试 ==============

class TestDataLoggerBasic:
    """DataLogger基本功能测试。"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试。"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def logger(self, temp_dir):
        """创建DataLogger实例。"""
        return DataLogger(log_dir=temp_dir)
    
    def test_init_creates_directory(self, temp_dir):
        """测试初始化时创建日志目录。"""
        new_dir = os.path.join(temp_dir, "new_logs")
        logger = DataLogger(log_dir=new_dir)
        assert os.path.exists(new_dir)
    
    def test_write_and_read_episode(self, logger):
        """测试写入和读取回合数据。"""
        episode = create_test_episode(player_id=0, final_reward=50.0)
        
        # 写入数据
        logger.write_episode(episode, episode_number=1)
        
        # 读取数据
        records = logger.read_episodes()
        
        assert len(records) == 1
        assert records[0]["episode_number"] == 1
        assert "timestamp" in records[0]
        
        # 验证Episode数据
        read_episode = records[0]["episode"]
        assert read_episode.player_id == episode.player_id
        assert read_episode.final_reward == episode.final_reward
        assert len(read_episode.states) == len(episode.states)
        assert len(read_episode.actions) == len(episode.actions)

    def test_write_multiple_episodes(self, logger):
        """测试写入多个回合数据。"""
        for i in range(5):
            episode = create_test_episode(player_id=i % 2, final_reward=float(i * 10))
            logger.write_episode(episode, episode_number=i)
        
        records = logger.read_episodes()
        assert len(records) == 5
        
        # 验证回合编号顺序
        for i, record in enumerate(records):
            assert record["episode_number"] == i
    
    def test_read_empty_log(self, logger):
        """测试读取空日志文件。"""
        records = logger.read_episodes()
        assert records == []
    
    def test_get_episode_count(self, logger):
        """测试获取回合数量。"""
        assert logger.get_episode_count() == 0
        
        for i in range(3):
            episode = create_test_episode()
            logger.write_episode(episode, episode_number=i)
        
        assert logger.get_episode_count() == 3
    
    def test_clear_log(self, logger):
        """测试清空日志。"""
        episode = create_test_episode()
        logger.write_episode(episode, episode_number=0)
        
        assert logger.get_episode_count() == 1
        
        logger.clear_log()
        
        assert logger.get_episode_count() == 0


class TestDataLoggerQuery:
    """DataLogger查询功能测试。"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试。"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def logger_with_data(self, temp_dir):
        """创建带有测试数据的DataLogger。"""
        logger = DataLogger(log_dir=temp_dir)
        
        for i in range(10):
            episode = create_test_episode(
                player_id=i % 2,
                final_reward=float(i * 10)
            )
            logger.write_episode(episode, episode_number=i)
        
        return logger
    
    def test_query_by_episode_range(self, logger_with_data):
        """测试按回合编号范围查询。"""
        records = logger_with_data.query_episodes(min_episode=3, max_episode=7)
        
        assert len(records) == 5
        episode_numbers = [r["episode_number"] for r in records]
        assert episode_numbers == [3, 4, 5, 6, 7]
    
    def test_query_by_min_episode(self, logger_with_data):
        """测试按最小回合编号查询。"""
        records = logger_with_data.query_episodes(min_episode=8)
        
        assert len(records) == 2
        assert all(r["episode_number"] >= 8 for r in records)
    
    def test_query_by_max_episode(self, logger_with_data):
        """测试按最大回合编号查询。"""
        records = logger_with_data.query_episodes(max_episode=2)
        
        assert len(records) == 3
        assert all(r["episode_number"] <= 2 for r in records)
    
    def test_query_with_custom_filter(self, logger_with_data):
        """测试使用自定义过滤函数查询。"""
        # 过滤出player_id为0的回合
        def filter_player_0(record):
            return record["episode"].player_id == 0
        
        records = logger_with_data.query_episodes(filter_func=filter_player_0)
        
        assert len(records) == 5
        assert all(r["episode"].player_id == 0 for r in records)
    
    def test_query_combined_filters(self, logger_with_data):
        """测试组合多个过滤条件。"""
        def filter_high_reward(record):
            return record["episode"].final_reward >= 50.0
        
        records = logger_with_data.query_episodes(
            min_episode=3,
            max_episode=8,
            filter_func=filter_high_reward
        )
        
        # 回合3-8中，final_reward >= 50的有：5, 6, 7, 8
        assert len(records) == 4


class TestDataLoggerExport:
    """DataLogger导出功能测试。"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试。"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def logger_with_data(self, temp_dir):
        """创建带有测试数据的DataLogger。"""
        logger = DataLogger(log_dir=temp_dir)
        
        for i in range(3):
            episode = create_test_episode(
                player_id=i % 2,
                final_reward=float(i * 10)
            )
            logger.write_episode(episode, episode_number=i)
        
        return logger, temp_dir
    
    def test_export_to_csv(self, logger_with_data):
        """测试导出为CSV格式。"""
        logger, temp_dir = logger_with_data
        csv_path = os.path.join(temp_dir, "export.csv")
        
        logger.export_to_csv(csv_path)
        
        assert os.path.exists(csv_path)
        
        # 验证CSV内容
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 3
        assert 'timestamp' in rows[0]
        assert 'episode_number' in rows[0]
        assert 'player_id' in rows[0]
        assert 'final_reward' in rows[0]
    
    def test_export_to_csv_empty(self, temp_dir):
        """测试导出空日志为CSV。"""
        logger = DataLogger(log_dir=temp_dir)
        csv_path = os.path.join(temp_dir, "empty.csv")
        
        logger.export_to_csv(csv_path)
        
        assert os.path.exists(csv_path)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 0
    
    def test_export_to_json(self, logger_with_data):
        """测试导出为JSON格式。"""
        logger, temp_dir = logger_with_data
        json_path = os.path.join(temp_dir, "export.json")
        
        logger.export_to_json(json_path)
        
        assert os.path.exists(json_path)
        
        # 验证JSON内容
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert len(data) == 3
        assert 'timestamp' in data[0]
        assert 'episode_number' in data[0]
        assert 'data' in data[0]
    
    def test_export_to_json_empty(self, temp_dir):
        """测试导出空日志为JSON。"""
        logger = DataLogger(log_dir=temp_dir)
        json_path = os.path.join(temp_dir, "empty.json")
        
        logger.export_to_json(json_path)
        
        assert os.path.exists(json_path)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data == []
    
    def test_export_to_json_not_pretty(self, logger_with_data):
        """测试导出为非格式化JSON。"""
        logger, temp_dir = logger_with_data
        json_path = os.path.join(temp_dir, "compact.json")
        
        logger.export_to_json(json_path, pretty=False)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 非格式化JSON应该只有一行
        assert '\n' not in content.strip()


class TestDataLoggerErrorHandling:
    """DataLogger错误处理测试。"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试。"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_write_io_error(self, temp_dir):
        """测试写入时的IO错误处理。"""
        logger = DataLogger(log_dir=temp_dir)
        episode = create_test_episode()
        
        # 模拟IO错误
        with patch('builtins.open', side_effect=OSError("磁盘空间不足")):
            with pytest.raises(IOError) as exc_info:
                logger.write_episode(episode, episode_number=0)
            
            assert "写入日志文件失败" in str(exc_info.value)
    
    def test_export_csv_io_error(self, temp_dir):
        """测试CSV导出时的IO错误处理。"""
        logger = DataLogger(log_dir=temp_dir)
        episode = create_test_episode()
        logger.write_episode(episode, episode_number=0)
        
        # 使用只读目录模拟写入失败
        readonly_path = "/nonexistent/path/export.csv"
        
        with pytest.raises(IOError) as exc_info:
            logger.export_to_csv(readonly_path)
        
        assert "导出CSV文件失败" in str(exc_info.value)
    
    def test_export_json_io_error(self, temp_dir):
        """测试JSON导出时的IO错误处理。"""
        logger = DataLogger(log_dir=temp_dir)
        episode = create_test_episode()
        logger.write_episode(episode, episode_number=0)
        
        readonly_path = "/nonexistent/path/export.json"
        
        with pytest.raises(IOError) as exc_info:
            logger.export_to_json(readonly_path)
        
        assert "导出JSON文件失败" in str(exc_info.value)


# ============== 属性测试 ==============

# Hypothesis策略：生成有效的Card
@st.composite
def card_strategy(draw):
    """生成有效的Card对象。"""
    rank = draw(st.integers(min_value=2, max_value=14))
    suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
    return Card(rank=rank, suit=suit)


# Hypothesis策略：生成有效的Action
@st.composite
def action_strategy(draw):
    """生成有效的Action对象。"""
    action_type = draw(st.sampled_from([ActionType.FOLD, ActionType.CHECK, ActionType.CALL, ActionType.RAISE]))
    if action_type == ActionType.RAISE:
        amount = draw(st.integers(min_value=1, max_value=1000))
        return Action(action_type=action_type, amount=amount)
    return Action(action_type=action_type, amount=0)


# Hypothesis策略：生成不重复的牌组
@st.composite
def unique_cards_strategy(draw, count: int):
    """生成指定数量的不重复Card对象。"""
    all_cards = [(r, s) for r in range(2, 15) for s in ['h', 'd', 'c', 's']]
    selected = draw(st.lists(
        st.sampled_from(all_cards),
        min_size=count,
        max_size=count,
        unique=True
    ))
    return [Card(rank=r, suit=s) for r, s in selected]


# Hypothesis策略：生成有效的GameState
@st.composite
def game_state_strategy(draw):
    """生成有效的GameState对象。"""
    # 生成9张不重复的牌（4张手牌 + 最多5张公共牌）
    cards = draw(unique_cards_strategy(9))
    
    player_hands = [
        (cards[0], cards[1]),
        (cards[2], cards[3])
    ]
    
    # 随机选择公共牌数量（0-5）
    num_community = draw(st.integers(min_value=0, max_value=5))
    community_cards = cards[4:4+num_community]
    
    # 根据公共牌数量确定游戏阶段
    if num_community == 0:
        stage = GameStage.PREFLOP
    elif num_community == 3:
        stage = GameStage.FLOP
    elif num_community == 4:
        stage = GameStage.TURN
    else:
        stage = GameStage.RIVER
    
    pot = draw(st.integers(min_value=0, max_value=10000))
    stack1 = draw(st.integers(min_value=0, max_value=10000))
    stack2 = draw(st.integers(min_value=0, max_value=10000))
    bet1 = draw(st.integers(min_value=0, max_value=min(1000, stack1)))
    bet2 = draw(st.integers(min_value=0, max_value=min(1000, stack2)))
    
    return GameState(
        player_hands=player_hands,
        community_cards=community_cards,
        pot=pot,
        player_stacks=[stack1, stack2],
        current_bets=[bet1, bet2],
        button_position=draw(st.integers(min_value=0, max_value=1)),
        stage=stage,
        action_history=[],
        current_player=draw(st.integers(min_value=0, max_value=1))
    )


# Hypothesis策略：生成有效的Episode
@st.composite
def episode_strategy(draw):
    """生成有效的Episode对象。"""
    num_actions = draw(st.integers(min_value=1, max_value=5))
    
    states = [draw(game_state_strategy()) for _ in range(num_actions + 1)]
    actions = [draw(action_strategy()) for _ in range(num_actions)]
    rewards = [draw(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)) 
               for _ in range(num_actions)]
    
    player_id = draw(st.integers(min_value=0, max_value=1))
    final_reward = draw(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
    
    return Episode(
        states=states,
        actions=actions,
        rewards=rewards,
        player_id=player_id,
        final_reward=final_reward
    )


class TestDataLoggerProperties:
    """DataLogger属性测试。"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试。"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @settings(max_examples=100)
    @given(episode=episode_strategy())
    def test_property_32_log_completeness(self, episode):
        """
        属性32：训练日志记录完整性
        
        Feature: texas-holdem-ai-training, Property 32: 训练日志记录完整性
        
        *对于任何*完成的训练回合，训练日志应该包含该回合的状态、行动和奖励数据。
        
        **验证需求：9.1**
        """
        temp_dir = tempfile.mkdtemp()
        try:
            logger = DataLogger(log_dir=temp_dir)
            episode_number = 42
            
            # 写入Episode
            logger.write_episode(episode, episode_number=episode_number)
            
            # 读取并验证
            records = logger.read_episodes()
            
            assert len(records) == 1
            record = records[0]
            
            # 验证所有字段都被记录
            assert record["episode_number"] == episode_number
            assert "timestamp" in record
            
            read_episode = record["episode"]
            
            # 验证状态数据完整
            assert len(read_episode.states) == len(episode.states)
            
            # 验证行动数据完整
            assert len(read_episode.actions) == len(episode.actions)
            for orig_action, read_action in zip(episode.actions, read_episode.actions):
                assert orig_action.action_type == read_action.action_type
                assert orig_action.amount == read_action.amount
            
            # 验证奖励数据完整
            assert len(read_episode.rewards) == len(episode.rewards)
            for orig_reward, read_reward in zip(episode.rewards, read_episode.rewards):
                assert abs(orig_reward - read_reward) < 1e-6
            
            # 验证其他字段
            assert read_episode.player_id == episode.player_id
            assert abs(read_episode.final_reward - episode.final_reward) < 1e-6
            
        finally:
            shutil.rmtree(temp_dir)

    @settings(max_examples=100)
    @given(episode=episode_strategy())
    def test_property_33_round_trip_consistency(self, episode):
        """
        属性33：训练数据往返一致性
        
        Feature: texas-holdem-ai-training, Property 33: 训练数据往返一致性
        
        *对于任何*写入训练日志的数据，从持久化存储中读取应该返回相同的数据。
        
        **验证需求：9.2**
        """
        temp_dir = tempfile.mkdtemp()
        try:
            logger = DataLogger(log_dir=temp_dir)
            episode_number = 123
            
            # 写入Episode
            logger.write_episode(episode, episode_number=episode_number)
            
            # 读取Episode
            records = logger.read_episodes()
            read_episode = records[0]["episode"]
            
            # 验证往返一致性
            assert read_episode.player_id == episode.player_id
            assert abs(read_episode.final_reward - episode.final_reward) < 1e-6
            
            # 验证状态数量一致
            assert len(read_episode.states) == len(episode.states)
            
            # 验证每个状态的关键字段
            for orig_state, read_state in zip(episode.states, read_episode.states):
                assert read_state.pot == orig_state.pot
                assert read_state.player_stacks == orig_state.player_stacks
                assert read_state.current_bets == orig_state.current_bets
                assert read_state.button_position == orig_state.button_position
                assert read_state.stage == orig_state.stage
                assert read_state.current_player == orig_state.current_player
                
                # 验证手牌
                for i in range(2):
                    assert read_state.player_hands[i][0].rank == orig_state.player_hands[i][0].rank
                    assert read_state.player_hands[i][0].suit == orig_state.player_hands[i][0].suit
                    assert read_state.player_hands[i][1].rank == orig_state.player_hands[i][1].rank
                    assert read_state.player_hands[i][1].suit == orig_state.player_hands[i][1].suit
                
                # 验证公共牌
                assert len(read_state.community_cards) == len(orig_state.community_cards)
                for orig_card, read_card in zip(orig_state.community_cards, read_state.community_cards):
                    assert read_card.rank == orig_card.rank
                    assert read_card.suit == orig_card.suit
            
            # 验证行动一致
            assert len(read_episode.actions) == len(episode.actions)
            for orig_action, read_action in zip(episode.actions, read_episode.actions):
                assert read_action.action_type == orig_action.action_type
                assert read_action.amount == orig_action.amount
            
            # 验证奖励一致
            assert len(read_episode.rewards) == len(episode.rewards)
            for orig_reward, read_reward in zip(episode.rewards, read_episode.rewards):
                assert abs(read_reward - orig_reward) < 1e-6
                
        finally:
            shutil.rmtree(temp_dir)
    
    @settings(max_examples=100)
    @given(episode=episode_strategy(), episode_number=st.integers(min_value=0, max_value=100000))
    def test_property_34_index_correctness(self, episode, episode_number):
        """
        属性34：训练指标索引正确性
        
        Feature: texas-holdem-ai-training, Property 34: 训练指标索引正确性
        
        *对于任何*存储的训练指标，数据记录应该包含时间戳和回合编号字段。
        
        **验证需求：9.3**
        """
        temp_dir = tempfile.mkdtemp()
        try:
            logger = DataLogger(log_dir=temp_dir)
            
            # 记录写入前的时间
            before_write = datetime.now().isoformat()
            
            # 写入Episode
            logger.write_episode(episode, episode_number=episode_number)
            
            # 记录写入后的时间
            after_write = datetime.now().isoformat()
            
            # 读取记录
            records = logger.read_episodes()
            
            assert len(records) == 1
            record = records[0]
            
            # 验证包含timestamp字段
            assert "timestamp" in record
            timestamp = record["timestamp"]
            
            # 验证timestamp在合理范围内
            assert before_write <= timestamp <= after_write
            
            # 验证包含episode_number字段
            assert "episode_number" in record
            assert record["episode_number"] == episode_number
            
        finally:
            shutil.rmtree(temp_dir)
    
    @settings(max_examples=100)
    @given(episodes=st.lists(episode_strategy(), min_size=1, max_size=5))
    def test_property_35_export_format_correctness(self, episodes):
        """
        属性35：数据导出格式正确性
        
        Feature: texas-holdem-ai-training, Property 35: 数据导出格式正确性
        
        *对于任何*训练数据导出请求，生成的文件应该是有效的CSV或JSON格式，
        且可以被标准解析器解析。
        
        **验证需求：9.5**
        """
        temp_dir = tempfile.mkdtemp()
        try:
            logger = DataLogger(log_dir=temp_dir)
            
            # 写入多个Episode
            for i, episode in enumerate(episodes):
                logger.write_episode(episode, episode_number=i)
            
            # 测试CSV导出
            csv_path = os.path.join(temp_dir, "export.csv")
            logger.export_to_csv(csv_path)
            
            # 验证CSV可被标准解析器解析
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)
            
            assert len(csv_rows) == len(episodes)
            
            # 验证CSV包含必需的列
            required_columns = ['timestamp', 'episode_number', 'player_id', 'final_reward']
            for col in required_columns:
                assert col in csv_rows[0], f"CSV缺少必需列: {col}"
            
            # 测试JSON导出
            json_path = os.path.join(temp_dir, "export.json")
            logger.export_to_json(json_path)
            
            # 验证JSON可被标准解析器解析
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            assert len(json_data) == len(episodes)
            
            # 验证JSON包含必需的字段
            for record in json_data:
                assert "timestamp" in record
                assert "episode_number" in record
                assert "data" in record
                
        finally:
            shutil.rmtree(temp_dir)
