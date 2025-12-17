"""Property-based tests for PokerEnvironment."""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from models.core import GameState, Action, ActionType, GameStage, Card
from environment.poker_environment import PokerEnvironment


class TestGameInitialization:
    """Property-based tests for game initialization."""
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=100)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_reset_deals_correct_number_of_cards(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 5: 游戏初始化规则符合性
        
        Test that reset() deals exactly 2 cards to each player.
        **Validates: Requirements 2.1**
        """
        assume(big_blind > small_blind)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        # Each player should have exactly 2 cards
        assert len(state.player_hands) == 2
        assert len(state.player_hands[0]) == 2
        assert len(state.player_hands[1]) == 2
        
        # All cards should be unique
        all_cards = list(state.player_hands[0]) + list(state.player_hands[1])
        assert len(all_cards) == len(set(all_cards)), "Duplicate cards dealt"
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=100)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_reset_sets_blinds_correctly(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 5: 游戏初始化规则符合性
        
        Test that reset() sets blinds correctly.
        **Validates: Requirements 2.1**
        """
        assume(big_blind > small_blind)
        assume(initial_stack > big_blind)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        # Button (player 0) posts small blind
        button = state.button_position
        
        # Check that blinds are posted correctly
        assert state.current_bets[button] == small_blind
        assert state.current_bets[1 - button] == big_blind
        
        # Check that pot is correct
        assert state.pot == small_blind + big_blind
        
        # Check that stacks are reduced correctly
        assert state.player_stacks[button] == initial_stack - small_blind
        assert state.player_stacks[1 - button] == initial_stack - big_blind
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=100)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_reset_starts_at_preflop(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 5: 游戏初始化规则符合性
        
        Test that reset() starts the game at PREFLOP stage.
        **Validates: Requirements 2.1**
        """
        assume(big_blind > small_blind)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        # Should start at PREFLOP
        assert state.stage == GameStage.PREFLOP
        
        # No community cards yet
        assert len(state.community_cards) == 0
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=100)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_reset_button_acts_first_preflop(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 5: 游戏初始化规则符合性
        
        Test that button (small blind) acts first in PREFLOP.
        **Validates: Requirements 2.1**
        """
        assume(big_blind > small_blind)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        # In heads-up preflop, button acts first
        assert state.current_player == state.button_position


class TestStageTransitions:
    """Property-based tests for game stage transitions."""
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=50)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_flop_deals_three_cards(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 8: 游戏阶段转换正确性
        
        Test that FLOP stage deals exactly 3 community cards.
        **Validates: Requirements 2.4**
        """
        assume(big_blind > small_blind)
        assume(initial_stack > big_blind * 2)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        # Button calls to match big blind
        call_action = Action(ActionType.CALL)
        state, _, done = env.step(call_action)
        
        if done:
            return  # Hand ended early
        
        # Big blind checks to advance to FLOP
        check_action = Action(ActionType.CHECK)
        state, _, done = env.step(check_action)
        
        if done:
            return  # Hand ended early
        
        # Should now be at FLOP with 3 community cards
        assert state.stage == GameStage.FLOP
        assert len(state.community_cards) == 3
        
        # All community cards should be unique
        assert len(state.community_cards) == len(set(state.community_cards))
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=50)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_turn_deals_one_card(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 8: 游戏阶段转换正确性
        
        Test that TURN stage deals exactly 1 additional community card.
        **Validates: Requirements 2.4**
        """
        assume(big_blind > small_blind)
        assume(initial_stack > big_blind * 2)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        # Get to FLOP
        state, _, done = env.step(Action(ActionType.CALL))
        if done:
            return
        state, _, done = env.step(Action(ActionType.CHECK))
        if done:
            return
        
        # Should be at FLOP with 3 cards
        assert state.stage == GameStage.FLOP
        assert len(state.community_cards) == 3
        
        # Both players check to advance to TURN
        state, _, done = env.step(Action(ActionType.CHECK))
        if done:
            return
        state, _, done = env.step(Action(ActionType.CHECK))
        if done:
            return
        
        # Should now be at TURN with 4 community cards
        assert state.stage == GameStage.TURN
        assert len(state.community_cards) == 4
        
        # All community cards should be unique
        assert len(state.community_cards) == len(set(state.community_cards))
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=50)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_river_deals_one_card(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 8: 游戏阶段转换正确性
        
        Test that RIVER stage deals exactly 1 additional community card.
        **Validates: Requirements 2.4**
        """
        assume(big_blind > small_blind)
        assume(initial_stack > big_blind * 2)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        # Get to TURN
        state, _, done = env.step(Action(ActionType.CALL))
        if done:
            return
        state, _, done = env.step(Action(ActionType.CHECK))
        if done:
            return
        state, _, done = env.step(Action(ActionType.CHECK))
        if done:
            return
        state, _, done = env.step(Action(ActionType.CHECK))
        if done:
            return
        
        # Should be at TURN with 4 cards
        assert state.stage == GameStage.TURN
        assert len(state.community_cards) == 4
        
        # Both players check to advance to RIVER
        state, _, done = env.step(Action(ActionType.CHECK))
        if done:
            return
        state, _, done = env.step(Action(ActionType.CHECK))
        if done:
            return
        
        # Should now be at RIVER with 5 community cards
        assert state.stage == GameStage.RIVER
        assert len(state.community_cards) == 5
        
        # All community cards should be unique
        assert len(state.community_cards) == len(set(state.community_cards))
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=50)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_total_five_community_cards(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 8: 游戏阶段转换正确性
        
        Test that exactly 5 community cards are dealt by RIVER.
        **Validates: Requirements 2.4**
        """
        assume(big_blind > small_blind)
        assume(initial_stack > big_blind * 2)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        # Play through all stages
        actions = [
            Action(ActionType.CALL),   # Button calls
            Action(ActionType.CHECK),  # BB checks -> FLOP
            Action(ActionType.CHECK),  # BB checks
            Action(ActionType.CHECK),  # Button checks -> TURN
            Action(ActionType.CHECK),  # BB checks
            Action(ActionType.CHECK),  # Button checks -> RIVER
        ]
        
        for action in actions:
            state, _, done = env.step(action)
            if done:
                return  # Hand ended early
        
        # Should be at RIVER with exactly 5 community cards
        assert state.stage == GameStage.RIVER
        assert len(state.community_cards) == 5
        
        # All cards (hole + community) should be unique
        all_cards = (list(state.player_hands[0]) + 
                    list(state.player_hands[1]) + 
                    state.community_cards)
        assert len(all_cards) == len(set(all_cards)), "Duplicate cards in game"


class TestActionOrder:
    """Property-based tests for action order."""
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=50)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_big_blind_acts_first_postflop(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 8: 游戏阶段转换正确性
        
        Test that big blind acts first after FLOP.
        **Validates: Requirements 2.4**
        """
        assume(big_blind > small_blind)
        assume(initial_stack > big_blind * 2)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        button = state.button_position
        
        # Get to FLOP
        state, _, done = env.step(Action(ActionType.CALL))
        if done:
            return
        state, _, done = env.step(Action(ActionType.CHECK))
        if done:
            return
        
        # At FLOP, big blind (player after button) should act first
        assert state.stage == GameStage.FLOP
        assert state.current_player == 1 - button


class TestGameCompletion:
    """Property-based tests for game completion."""
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=50)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_fold_ends_game_immediately(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 8: 游戏阶段转换正确性
        
        Test that FOLD action ends the game immediately.
        **Validates: Requirements 2.4**
        """
        assume(big_blind > small_blind)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        # Fold immediately
        state, reward, done = env.step(Action(ActionType.FOLD))
        
        # Game should be done
        assert done is True
        
        # Reward should be non-positive (at best 0, typically negative)
        # When button folds preflop, they don't gain anything
        assert reward <= 0
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=50)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_showdown_ends_game(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 8: 游戏阶段转换正确性
        
        Test that game ends at showdown after both players check on RIVER.
        **Validates: Requirements 2.4**
        """
        assume(big_blind > small_blind)
        assume(initial_stack > big_blind * 2)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        # Play through to RIVER with all checks
        actions = [
            Action(ActionType.CALL),   # Button calls
            Action(ActionType.CHECK),  # BB checks -> FLOP
            Action(ActionType.CHECK),  # BB checks
            Action(ActionType.CHECK),  # Button checks -> TURN
            Action(ActionType.CHECK),  # BB checks
            Action(ActionType.CHECK),  # Button checks -> RIVER
            Action(ActionType.CHECK),  # BB checks
        ]
        
        done = False
        for action in actions:
            state, reward, done = env.step(action)
            if done:
                break
        
        # After last check on RIVER, game should end
        if not done:
            state, reward, done = env.step(Action(ActionType.CHECK))
        
        assert done is True


class TestChipConservation:
    """Property-based tests for chip conservation."""
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=50)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_chips_conserved_after_reset(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 5: 游戏初始化规则符合性
        
        Test that total chips are conserved after reset.
        **Validates: Requirements 2.1**
        """
        assume(big_blind > small_blind)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        # Total chips should equal 2 * initial_stack
        total_chips = state.player_stacks[0] + state.player_stacks[1] + state.pot
        assert total_chips == 2 * initial_stack
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=50)
    @given(
        st.integers(min_value=100, max_value=10000),
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=2, max_value=100)
    )
    def test_chips_conserved_throughout_game(self, initial_stack, small_blind, big_blind):
        """Feature: texas-holdem-ai-training, Property 5: 游戏初始化规则符合性
        
        Test that total chips are conserved throughout the game.
        **Validates: Requirements 2.1**
        """
        assume(big_blind > small_blind)
        assume(initial_stack > big_blind * 2)
        
        env = PokerEnvironment(initial_stack=initial_stack, small_blind=small_blind, big_blind=big_blind)
        state = env.reset()
        
        initial_total = 2 * initial_stack
        
        # Play through several actions
        actions = [
            Action(ActionType.CALL),
            Action(ActionType.CHECK),
            Action(ActionType.CHECK),
        ]
        
        for action in actions:
            state, _, done = env.step(action)
            
            # Check chip conservation
            total_chips = state.player_stacks[0] + state.player_stacks[1] + state.pot
            assert total_chips == initial_total, f"Chips not conserved: {total_chips} != {initial_total}"
            
            if done:
                break
