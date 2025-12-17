"""Unit tests for StateEncoder."""

import pytest
import numpy as np
from models.core import Card, GameState, GameStage, Action, ActionType
from environment.state_encoder import StateEncoder


class TestStateEncoder:
    """Test suite for StateEncoder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = StateEncoder()
    
    def create_test_state(self, stage=GameStage.PREFLOP, num_community_cards=0):
        """Helper to create a test game state.
        
        Args:
            stage: Game stage
            num_community_cards: Number of community cards to include
            
        Returns:
            A GameState object
        """
        # Create player hands
        player_hands = [
            (Card(14, 'h'), Card(13, 'h')),  # Player 0: A♥ K♥
            (Card(10, 's'), Card(10, 'd'))   # Player 1: 10♠ 10♦
        ]
        
        # Create community cards based on stage
        community_cards = []
        if num_community_cards >= 3:
            community_cards = [Card(12, 'h'), Card(11, 'h'), Card(2, 'c')]
        if num_community_cards >= 4:
            community_cards.append(Card(7, 'd'))
        if num_community_cards >= 5:
            community_cards.append(Card(3, 's'))
        
        return GameState(
            player_hands=player_hands,
            community_cards=community_cards,
            pot=100,
            player_stacks=[950, 950],
            current_bets=[50, 50],
            button_position=0,
            stage=stage,
            action_history=[],
            current_player=1
        )
    
    def test_encoding_dimension(self):
        """Test that encoding output is always 370 dimensions."""
        state = self.create_test_state()
        
        # Test for player 0
        encoding_p0 = self.encoder.encode(state, 0)
        assert encoding_p0.shape == (370,), f"Expected shape (370,), got {encoding_p0.shape}"
        
        # Test for player 1
        encoding_p1 = self.encoder.encode(state, 1)
        assert encoding_p1.shape == (370,), f"Expected shape (370,), got {encoding_p1.shape}"
    
    def test_encoding_dimension_different_stages(self):
        """Test encoding dimension is 370 for all game stages."""
        stages_and_cards = [
            (GameStage.PREFLOP, 0),
            (GameStage.FLOP, 3),
            (GameStage.TURN, 4),
            (GameStage.RIVER, 5)
        ]
        
        for stage, num_cards in stages_and_cards:
            state = self.create_test_state(stage=stage, num_community_cards=num_cards)
            encoding = self.encoder.encode(state, 0)
            assert encoding.shape == (370,), \
                f"Expected shape (370,) for {stage.value}, got {encoding.shape}"
    
    def test_card_one_hot_encoding(self):
        """Test that each card is encoded as one-hot vector with exactly one 1."""
        # Test encoding a single card
        card = Card(14, 'h')  # Ace of hearts
        encoding = self.encoder.encode_cards([card])
        
        # Should be 52 dimensions
        assert encoding.shape == (52,), f"Expected shape (52,), got {encoding.shape}"
        
        # Should have exactly one 1
        assert np.sum(encoding) == 1.0, f"Expected sum of 1.0, got {np.sum(encoding)}"
        
        # All values should be 0 or 1
        assert np.all((encoding == 0) | (encoding == 1)), "Encoding should only contain 0s and 1s"
        
        # Check the correct position is set to 1
        # Ace (rank 14) -> index 12, hearts (h) -> suit 0
        # Position = 12 * 4 + 0 = 48
        expected_position = 48
        assert encoding[expected_position] == 1.0, \
            f"Expected position {expected_position} to be 1.0"
    
    def test_card_encoding_different_cards(self):
        """Test that different cards produce different encodings."""
        cards = [
            Card(2, 'h'),   # 2 of hearts
            Card(14, 's'),  # Ace of spades
            Card(7, 'd'),   # 7 of diamonds
            Card(13, 'c')   # King of clubs
        ]
        
        encodings = [self.encoder.encode_cards([card]) for card in cards]
        
        # All encodings should be different
        for i in range(len(encodings)):
            for j in range(i + 1, len(encodings)):
                assert not np.array_equal(encodings[i], encodings[j]), \
                    f"Cards {cards[i]} and {cards[j]} produced identical encodings"
    
    def test_card_encoding_positions(self):
        """Test that card encoding positions are calculated correctly."""
        # Test a few specific cards
        test_cases = [
            (Card(2, 'h'), 0),      # rank 2, suit h -> (2-2)*4 + 0 = 0
            (Card(2, 's'), 3),      # rank 2, suit s -> (2-2)*4 + 3 = 3
            (Card(14, 'h'), 48),    # rank 14, suit h -> (14-2)*4 + 0 = 48
            (Card(14, 's'), 51),    # rank 14, suit s -> (14-2)*4 + 3 = 51
            (Card(7, 'd'), 21),     # rank 7, suit d -> (7-2)*4 + 1 = 21
        ]
        
        for card, expected_pos in test_cases:
            encoding = self.encoder.encode_cards([card])
            assert encoding[expected_pos] == 1.0, \
                f"Card {card} should have 1.0 at position {expected_pos}"
            assert np.sum(encoding) == 1.0, \
                f"Card {card} encoding should have exactly one 1"
    
    def test_multiple_cards_encoding(self):
        """Test encoding multiple cards."""
        cards = [Card(14, 'h'), Card(13, 'h')]
        encoding = self.encoder.encode_cards(cards)
        
        # Should be 104 dimensions (2 cards × 52)
        assert encoding.shape == (104,), f"Expected shape (104,), got {encoding.shape}"
        
        # Should have exactly two 1s
        assert np.sum(encoding) == 2.0, f"Expected sum of 2.0, got {np.sum(encoding)}"
    
    def test_community_cards_padding(self):
        """Test that community cards are padded to 5 cards."""
        # Test with 0 cards
        encoding_0 = self.encoder.encode_cards([], max_cards=5)
        assert encoding_0.shape == (260,), "Should be 260 dimensions for 5 cards"
        assert np.sum(encoding_0) == 0.0, "Empty cards should produce all zeros"
        
        # Test with 3 cards (flop)
        cards_3 = [Card(12, 'h'), Card(11, 'h'), Card(2, 'c')]
        encoding_3 = self.encoder.encode_cards(cards_3, max_cards=5)
        assert encoding_3.shape == (260,), "Should be 260 dimensions"
        assert np.sum(encoding_3) == 3.0, "Should have exactly 3 ones"
        
        # Test with 5 cards (river)
        cards_5 = [Card(12, 'h'), Card(11, 'h'), Card(2, 'c'), Card(7, 'd'), Card(3, 's')]
        encoding_5 = self.encoder.encode_cards(cards_5, max_cards=5)
        assert encoding_5.shape == (260,), "Should be 260 dimensions"
        assert np.sum(encoding_5) == 5.0, "Should have exactly 5 ones"
    
    def test_same_state_same_encoding(self):
        """Test that encoding the same state produces identical results."""
        state = self.create_test_state(GameStage.FLOP, 3)
        
        encoding1 = self.encoder.encode(state, 0)
        encoding2 = self.encoder.encode(state, 0)
        
        assert np.array_equal(encoding1, encoding2), \
            "Same state should produce identical encodings"
    
    def test_different_players_different_encoding(self):
        """Test that different players get different encodings of the same state."""
        state = self.create_test_state()
        
        encoding_p0 = self.encoder.encode(state, 0)
        encoding_p1 = self.encoder.encode(state, 1)
        
        # Encodings should be different (different hands, different perspectives)
        assert not np.array_equal(encoding_p0, encoding_p1), \
            "Different players should have different encodings"
    
    def test_preflop_encoding(self):
        """Test encoding for preflop stage (no community cards)."""
        state = self.create_test_state(GameStage.PREFLOP, 0)
        encoding = self.encoder.encode(state, 0)
        
        # 检查维度
        assert encoding.shape == (370,)
        
        # 翻牌前使用抽象编码：前169维是手牌等价类的one-hot编码
        # 应该只有1个位置为1
        hand_abstraction_section = encoding[0:169]
        assert np.sum(hand_abstraction_section) == 1.0, \
            "翻牌前手牌抽象编码应该只有1个位置为1"
        
        # 169-364之间应该全为0（填充区域）
        padding_section = encoding[169:364]
        assert np.sum(padding_section) == 0.0, \
            "翻牌前填充区域应该全为0"
    
    def test_flop_encoding(self):
        """Test encoding for flop stage (3 community cards)."""
        state = self.create_test_state(GameStage.FLOP, 3)
        encoding = self.encoder.encode(state, 0)
        
        # Community cards section should have exactly 3 ones
        community_section = encoding[104:364]
        assert np.sum(community_section) == 3.0, \
            "Community cards should have exactly 3 ones in flop"
    
    def test_turn_encoding(self):
        """Test encoding for turn stage (4 community cards)."""
        state = self.create_test_state(GameStage.TURN, 4)
        encoding = self.encoder.encode(state, 0)
        
        # Community cards section should have exactly 4 ones
        community_section = encoding[104:364]
        assert np.sum(community_section) == 4.0, \
            "Community cards should have exactly 4 ones in turn"
    
    def test_river_encoding(self):
        """Test encoding for river stage (5 community cards)."""
        state = self.create_test_state(GameStage.RIVER, 5)
        encoding = self.encoder.encode(state, 0)
        
        # Community cards section should have exactly 5 ones
        community_section = encoding[104:364]
        assert np.sum(community_section) == 5.0, \
            "Community cards should have exactly 5 ones in river"
    
    def test_chip_information_encoding(self):
        """Test that chip information is encoded correctly."""
        state = self.create_test_state()
        encoding = self.encoder.encode(state, 0)
        
        # Chip information is at positions 364-368
        chip_section = encoding[364:368]
        
        # All chip values should be non-negative
        assert np.all(chip_section >= 0), "Chip values should be non-negative"
        
        # Values should be normalized (typically between 0 and 1)
        assert np.all(chip_section <= 1.0), "Chip values should be normalized"
    
    def test_position_information_encoding(self):
        """Test that position information is encoded correctly."""
        state = self.create_test_state()
        
        # Test for player 0 (button)
        encoding_p0 = self.encoder.encode(state, 0)
        position_section_p0 = encoding_p0[368:370]
        
        # Player 0 is button, so first value should be 1.0
        assert position_section_p0[0] == 1.0, "Player 0 should be button"
        # Player 0 is not current player (current_player=1), so second value should be 0.0
        assert position_section_p0[1] == 0.0, "Player 0 should not be current player"
        
        # Test for player 1 (not button, is current player)
        encoding_p1 = self.encoder.encode(state, 1)
        position_section_p1 = encoding_p1[368:370]
        
        # Player 1 is not button, so first value should be 0.0
        assert position_section_p1[0] == 0.0, "Player 1 should not be button"
        # Player 1 is current player, so second value should be 1.0
        assert position_section_p1[1] == 1.0, "Player 1 should be current player"
    
    def test_invalid_player_id(self):
        """测试无效的玩家ID应该抛出错误。"""
        state = self.create_test_state()
        
        with pytest.raises(ValueError, match="玩家ID必须是0或1"):
            self.encoder.encode(state, 2)
        
        with pytest.raises(ValueError, match="玩家ID必须是0或1"):
            self.encoder.encode(state, -1)
    
    def test_encoding_dtype(self):
        """Test that encoding returns float32 dtype."""
        state = self.create_test_state()
        encoding = self.encoder.encode(state, 0)
        
        assert encoding.dtype == np.float32, \
            f"Expected dtype float32, got {encoding.dtype}"
