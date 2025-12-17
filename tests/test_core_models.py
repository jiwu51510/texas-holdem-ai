"""Tests for core data models."""

import pytest
from datetime import datetime
from models import (
    Card, Action, GameState, Episode, TrainingConfig, CheckpointInfo,
    ActionType, GameStage, HandRank
)


class TestCard:
    """Tests for Card class."""
    
    def test_valid_card_creation(self):
        """Test creating valid cards."""
        card = Card(rank=14, suit='h')
        assert card.rank == 14
        assert card.suit == 'h'
        
    def test_card_string_representation(self):
        """Test card string representation."""
        assert str(Card(14, 'h')) == 'A♥'
        assert str(Card(13, 's')) == 'K♠'
        assert str(Card(2, 'd')) == '2♦'
        
    def test_invalid_rank(self):
        """Test that invalid ranks raise ValueError."""
        with pytest.raises(ValueError, match="Invalid rank"):
            Card(rank=1, suit='h')
        with pytest.raises(ValueError, match="Invalid rank"):
            Card(rank=15, suit='h')
            
    def test_invalid_suit(self):
        """Test that invalid suits raise ValueError."""
        with pytest.raises(ValueError, match="Invalid suit"):
            Card(rank=10, suit='x')
            
    def test_card_equality(self):
        """Test card equality."""
        card1 = Card(10, 'h')
        card2 = Card(10, 'h')
        card3 = Card(10, 'd')
        assert card1 == card2
        assert card1 != card3
        
    def test_card_hash(self):
        """Test that cards can be hashed."""
        card_set = {Card(10, 'h'), Card(10, 'h'), Card(10, 'd')}
        assert len(card_set) == 2


class TestAction:
    """Tests for Action class."""
    
    def test_fold_action(self):
        """Test creating a fold action."""
        action = Action(ActionType.FOLD)
        assert action.action_type == ActionType.FOLD
        assert action.amount == 0
        
    def test_raise_action(self):
        """Test creating a raise action."""
        action = Action(ActionType.RAISE, amount=50)
        assert action.action_type == ActionType.RAISE
        assert action.amount == 50
        
    def test_raise_without_amount(self):
        """测试加注动作没有金额时抛出 ValueError。"""
        with pytest.raises(ValueError, match="动作必须有正的金额"):
            Action(ActionType.RAISE, amount=0)
        with pytest.raises(ValueError, match="动作必须有正的金额"):
            Action(ActionType.RAISE_SMALL, amount=0)
        with pytest.raises(ValueError, match="动作必须有正的金额"):
            Action(ActionType.RAISE_BIG, amount=0)
            
    def test_non_raise_with_amount(self):
        """测试非加注动作带有金额时抛出 ValueError。"""
        with pytest.raises(ValueError, match="动作的金额应为0"):
            Action(ActionType.FOLD, amount=10)


class TestGameState:
    """Tests for GameState class."""
    
    def test_valid_game_state(self):
        """Test creating a valid game state."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(10, 'd'), Card(9, 'd'))],
            community_cards=[],
            pot=15,
            player_stacks=[995, 990],
            current_bets=[5, 10],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=1
        )
        assert len(state.player_hands) == 2
        assert state.pot == 15
        
    def test_invalid_player_count(self):
        """Test that invalid player count raises ValueError."""
        with pytest.raises(ValueError, match="Must have exactly 2 players"):
            GameState(
                player_hands=[(Card(14, 'h'), Card(13, 'h'))],
                community_cards=[],
                pot=0,
                player_stacks=[1000],
                current_bets=[0],
                button_position=0,
                stage=GameStage.PREFLOP
            )
            
    def test_too_many_community_cards(self):
        """Test that too many community cards raises ValueError."""
        with pytest.raises(ValueError, match="Cannot have more than 5 community cards"):
            GameState(
                player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(10, 'd'), Card(9, 'd'))],
                community_cards=[Card(2, 'h'), Card(3, 'h'), Card(4, 'h'), 
                               Card(5, 'h'), Card(6, 'h'), Card(7, 'h')],
                pot=0,
                player_stacks=[1000, 1000],
                current_bets=[0, 0],
                button_position=0,
                stage=GameStage.RIVER
            )


class TestEpisode:
    """Tests for Episode class."""
    
    def test_valid_episode(self):
        """Test creating a valid episode."""
        state1 = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(10, 'd'), Card(9, 'd'))],
            community_cards=[],
            pot=15,
            player_stacks=[995, 990],
            current_bets=[5, 10],
            button_position=0,
            stage=GameStage.PREFLOP
        )
        state2 = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(10, 'd'), Card(9, 'd'))],
            community_cards=[],
            pot=25,
            player_stacks=[985, 990],
            current_bets=[10, 10],
            button_position=0,
            stage=GameStage.PREFLOP
        )
        
        episode = Episode(
            states=[state1, state2],
            actions=[Action(ActionType.CALL)],
            rewards=[0.0],
            player_id=0,
            final_reward=10.0
        )
        assert len(episode.states) == 2
        assert len(episode.actions) == 1
        
    def test_invalid_player_id(self):
        """Test that invalid player ID raises ValueError."""
        with pytest.raises(ValueError, match="Player ID must be 0 or 1"):
            Episode(
                states=[],
                actions=[],
                rewards=[],
                player_id=2,
                final_reward=0.0
            )


class TestTrainingConfig:
    """Tests for TrainingConfig class."""
    
    def test_default_config(self):
        """Test creating config with default values."""
        config = TrainingConfig()
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.num_episodes == 10000
        
    def test_custom_config(self):
        """Test creating config with custom values."""
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=64,
            num_episodes=5000
        )
        assert config.learning_rate == 0.01
        assert config.batch_size == 64
        
    def test_invalid_learning_rate(self):
        """Test that invalid learning rate raises ValueError."""
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            TrainingConfig(learning_rate=-0.001)
            
    def test_invalid_blind_configuration(self):
        """Test that big blind must be greater than small blind."""
        with pytest.raises(ValueError, match="Big blind .* must be greater than small blind"):
            TrainingConfig(small_blind=10, big_blind=5)


class TestCheckpointInfo:
    """Tests for CheckpointInfo class."""
    
    def test_valid_checkpoint_info(self):
        """Test creating valid checkpoint info."""
        info = CheckpointInfo(
            path="/path/to/checkpoint.pt",
            episode_number=1000,
            timestamp=datetime.now(),
            win_rate=0.55,
            avg_reward=10.5
        )
        assert info.episode_number == 1000
        assert info.win_rate == 0.55
        
    def test_invalid_win_rate(self):
        """Test that invalid win rate raises ValueError."""
        with pytest.raises(ValueError, match="Win rate must be in"):
            CheckpointInfo(
                path="/path/to/checkpoint.pt",
                episode_number=1000,
                timestamp=datetime.now(),
                win_rate=1.5,
                avg_reward=10.5
            )
