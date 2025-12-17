"""Unit tests for RuleEngine."""

import pytest
from copy import deepcopy
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from models.core import GameState, Action, ActionType, GameStage, Card
from environment.rule_engine import RuleEngine


class TestActionLegality:
    """Test action legality validation."""
    
    def test_fold_always_legal(self):
        """Test that FOLD is always legal."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=30,
            player_stacks=[990, 985],
            current_bets=[10, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=0
        )
        
        action = Action(ActionType.FOLD)
        assert RuleEngine.is_action_legal(state, action) is True
    
    def test_check_legal_when_bets_equal(self):
        """Test that CHECK is legal when bets are equal."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[Card(10, 'h'), Card(9, 'h'), Card(8, 'h')],
            pot=40,
            player_stacks=[980, 980],
            current_bets=[20, 20],
            button_position=0,
            stage=GameStage.FLOP,
            current_player=1
        )
        
        action = Action(ActionType.CHECK)
        assert RuleEngine.is_action_legal(state, action) is True
    
    def test_check_illegal_when_bets_unequal(self):
        """Test that CHECK is illegal when opponent has bet more."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=30,
            player_stacks=[990, 985],
            current_bets=[10, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=0
        )
        
        action = Action(ActionType.CHECK)
        assert RuleEngine.is_action_legal(state, action) is False
    
    def test_call_legal_when_opponent_bet_more(self):
        """Test that CALL is legal when opponent has bet more."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=30,
            player_stacks=[990, 985],
            current_bets=[10, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=0
        )
        
        action = Action(ActionType.CALL)
        assert RuleEngine.is_action_legal(state, action) is True
    
    def test_call_illegal_when_bets_equal(self):
        """Test that CALL is illegal when bets are already equal."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=40,
            player_stacks=[980, 980],
            current_bets=[20, 20],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=0
        )
        
        action = Action(ActionType.CALL)
        assert RuleEngine.is_action_legal(state, action) is False
    
    def test_raise_legal_with_valid_amount(self):
        """Test that RAISE is legal with valid amount."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=30,
            player_stacks=[990, 985],
            current_bets=[10, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=0
        )
        
        # Raise by 20 (total bet becomes 30, which is 15 more than opponent's 15)
        action = Action(ActionType.RAISE, amount=20)
        assert RuleEngine.is_action_legal(state, action) is True
    
    def test_raise_illegal_insufficient_amount(self):
        """Test that RAISE is illegal with insufficient amount."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=30,
            player_stacks=[990, 985],
            current_bets=[10, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=0
        )
        
        # Raise by only 3 (total bet becomes 13, less than opponent's 15)
        action = Action(ActionType.RAISE, amount=3)
        assert RuleEngine.is_action_legal(state, action) is False
    
    def test_raise_illegal_exceeds_stack(self):
        """Test that RAISE is illegal when amount exceeds stack."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=30,
            player_stacks=[50, 985],
            current_bets=[10, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=0
        )
        
        # Try to raise by 100 when only have 50 in stack
        action = Action(ActionType.RAISE, amount=100)
        assert RuleEngine.is_action_legal(state, action) is False


class TestApplyAction:
    """Test applying actions to game state."""
    
    def test_apply_fold(self):
        """Test applying FOLD action."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=30,
            player_stacks=[990, 985],
            current_bets=[10, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=0
        )
        
        action = Action(ActionType.FOLD)
        new_state = RuleEngine.apply_action(state, action)
        
        # Action should be in history
        assert len(new_state.action_history) == 1
        assert new_state.action_history[0].action_type == ActionType.FOLD
    
    def test_apply_check_advances_stage(self):
        """Test that CHECK advances stage when both players have checked."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[Card(10, 'h'), Card(9, 'h'), Card(8, 'h')],
            pot=40,
            player_stacks=[980, 980],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.FLOP,
            current_player=1
        )
        
        # First player checks
        action = Action(ActionType.CHECK)
        new_state = RuleEngine.apply_action(state, action)
        
        # Should switch to other player, not advance stage yet
        assert new_state.stage == GameStage.FLOP
        assert new_state.current_player == 0
        
        # Second player also checks
        action2 = Action(ActionType.CHECK)
        new_state2 = RuleEngine.apply_action(new_state, action2)
        
        # Now should advance to TURN
        assert new_state2.stage == GameStage.TURN
        # Bets should be reset
        assert new_state2.current_bets == [0, 0]
    
    def test_apply_call_updates_pot_and_stack(self):
        """Test that CALL updates pot and player stack correctly."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=30,
            player_stacks=[990, 985],
            current_bets=[10, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=0
        )
        
        action = Action(ActionType.CALL)
        new_state = RuleEngine.apply_action(state, action)
        
        # Player 0 should have called 5 chips
        assert new_state.player_stacks[0] == 985
        # After call, bets are equal and player switches
        assert new_state.current_bets[0] == 15
        assert new_state.current_bets[1] == 15
        assert new_state.pot == 35
        # Stage should stay at PREFLOP (waiting for other player to check)
        assert new_state.stage == GameStage.PREFLOP
        # Current player should switch to player 1
        assert new_state.current_player == 1
    
    def test_apply_raise_updates_state(self):
        """Test that RAISE updates game state correctly."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=30,
            player_stacks=[990, 985],
            current_bets=[10, 15],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=0
        )
        
        action = Action(ActionType.RAISE, amount=20)
        new_state = RuleEngine.apply_action(state, action)
        
        # Player 0 should have raised by 20
        assert new_state.player_stacks[0] == 970
        assert new_state.current_bets[0] == 30
        assert new_state.pot == 50
        # Should switch to other player
        assert new_state.current_player == 1
    
    def test_apply_illegal_action_raises_error(self):
        """Test that applying illegal action raises ValueError."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=40,
            player_stacks=[980, 980],
            current_bets=[20, 20],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=0
        )
        
        # Try to call when bets are equal (illegal)
        action = Action(ActionType.CALL)
        with pytest.raises(ValueError):
            RuleEngine.apply_action(state, action)


class TestStageTransition:
    """Test game stage transitions."""
    
    def test_preflop_to_flop(self):
        """Test transition from PREFLOP to FLOP."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[Card(10, 'h'), Card(9, 'h'), Card(8, 'h')],
            pot=30,
            player_stacks=[985, 980],
            current_bets=[15, 20],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=0
        )
        
        # Player 0 calls to match player 1's bet
        action = Action(ActionType.CALL)
        new_state = RuleEngine.apply_action(state, action)
        
        # Should stay at PREFLOP, switch to player 1
        assert new_state.stage == GameStage.PREFLOP
        assert new_state.current_player == 1
        
        # Player 1 checks, should advance to FLOP
        action2 = Action(ActionType.CHECK)
        new_state2 = RuleEngine.apply_action(new_state, action2)
        
        assert new_state2.stage == GameStage.FLOP
    
    def test_flop_to_turn(self):
        """Test transition from FLOP to TURN."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[Card(10, 'h'), Card(9, 'h'), Card(8, 'h'), Card(7, 'h')],
            pot=60,
            player_stacks=[970, 970],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.FLOP,
            current_player=1
        )
        
        # First player checks
        action = Action(ActionType.CHECK)
        new_state = RuleEngine.apply_action(state, action)
        assert new_state.stage == GameStage.FLOP
        
        # Second player checks, should advance
        action2 = Action(ActionType.CHECK)
        new_state2 = RuleEngine.apply_action(new_state, action2)
        assert new_state2.stage == GameStage.TURN
    
    def test_turn_to_river(self):
        """Test transition from TURN to RIVER."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[Card(10, 'h'), Card(9, 'h'), Card(8, 'h'), Card(7, 'h'), Card(6, 'h')],
            pot=80,
            player_stacks=[960, 960],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.TURN,
            current_player=1
        )
        
        # First player checks
        action = Action(ActionType.CHECK)
        new_state = RuleEngine.apply_action(state, action)
        assert new_state.stage == GameStage.TURN
        
        # Second player checks, should advance
        action2 = Action(ActionType.CHECK)
        new_state2 = RuleEngine.apply_action(new_state, action2)
        assert new_state2.stage == GameStage.RIVER


class TestWinnerDetermination:
    """Test winner determination."""
    
    def test_winner_by_fold(self):
        """Test that winner is determined correctly when opponent folds."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=50,
            player_stacks=[970, 980],
            current_bets=[30, 20],
            button_position=0,
            stage=GameStage.PREFLOP,
            current_player=1,
            action_history=[Action(ActionType.FOLD)]
        )
        
        winner = RuleEngine.determine_winner(state)
        # Player 1 folded, so player 0 wins
        assert winner == 0
    
    def test_winner_by_showdown(self):
        """Test winner determination at showdown."""
        # Player 0 has straight flush
        state = GameState(
            player_hands=[
                (Card(14, 's'), Card(13, 's')),  # A♠ K♠
                (Card(2, 'h'), Card(3, 'h'))     # 2♥ 3♥
            ],
            community_cards=[Card(12, 's'), Card(11, 's'), Card(10, 's'), Card(9, 'd'), Card(8, 'd')],
            pot=100,
            player_stacks=[950, 950],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.RIVER,
            current_player=0
        )
        
        winner = RuleEngine.determine_winner(state)
        # Player 0 has straight flush, should win
        assert winner == 0


class TestPotDistribution:
    """Test pot distribution."""
    
    def test_distribute_pot_to_winner(self):
        """Test pot distribution to single winner."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=100,
            player_stacks=[950, 950],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.RIVER,
            current_player=0
        )
        
        new_state = RuleEngine.distribute_pot(state, winner=0)
        
        # Player 0 should get the pot
        assert new_state.player_stacks[0] == 1050
        assert new_state.player_stacks[1] == 950
        assert new_state.pot == 0
    
    def test_distribute_pot_on_tie(self):
        """Test pot distribution on tie (split pot)."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=100,
            player_stacks=[950, 950],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.RIVER,
            current_player=0
        )
        
        new_state = RuleEngine.distribute_pot(state, winner=-1)
        
        # Pot should be split evenly
        assert new_state.player_stacks[0] == 1000
        assert new_state.player_stacks[1] == 1000
        assert new_state.pot == 0
    
    def test_distribute_odd_chip_on_tie(self):
        """Test that odd chip goes to correct player on tie."""
        state = GameState(
            player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(12, 'd'), Card(11, 'd'))],
            community_cards=[],
            pot=101,  # Odd pot
            player_stacks=[950, 950],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.RIVER,
            current_player=0
        )
        
        new_state = RuleEngine.distribute_pot(state, winner=-1)
        
        # Pot should be split, odd chip goes to player after button (player 1)
        assert new_state.player_stacks[0] == 1000
        assert new_state.player_stacks[1] == 1001
        assert new_state.pot == 0



# Property-based tests

@st.composite
def card_strategy(draw):
    """Generate a random valid card."""
    rank = draw(st.integers(min_value=2, max_value=14))
    suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
    return Card(rank, suit)


def unique_cards_strategy(n):
    """Generate n unique cards."""
    # Generate all possible cards
    all_cards = [(r, s) for r in range(2, 15) for s in ['h', 'd', 'c', 's']]
    # Sample n unique cards
    return st.lists(
        st.sampled_from(all_cards),
        min_size=n,
        max_size=n,
        unique=True
    ).map(lambda cards: [Card(r, s) for r, s in cards])


@st.composite
def game_state_strategy(draw):
    """Generate a random valid game state."""
    # Generate community cards (0-5)
    num_community = draw(st.integers(min_value=0, max_value=5))
    total_cards = 4 + num_community
    
    # Generate all unique cards at once
    all_cards = draw(unique_cards_strategy(total_cards))
    
    player_hands = [
        (all_cards[0], all_cards[1]),
        (all_cards[2], all_cards[3])
    ]
    community_cards = all_cards[4:] if num_community > 0 else []
    
    # Generate stacks and bets
    stack0 = draw(st.integers(min_value=0, max_value=2000))
    stack1 = draw(st.integers(min_value=0, max_value=2000))
    bet0 = draw(st.integers(min_value=0, max_value=min(500, stack0 + 500)))
    bet1 = draw(st.integers(min_value=0, max_value=min(500, stack1 + 500)))
    
    pot = draw(st.integers(min_value=0, max_value=1000))
    button = draw(st.integers(min_value=0, max_value=1))
    current_player = draw(st.integers(min_value=0, max_value=1))
    
    # Stage based on community cards
    if num_community == 0:
        stage = GameStage.PREFLOP
    elif num_community == 3:
        stage = GameStage.FLOP
    elif num_community == 4:
        stage = GameStage.TURN
    else:
        stage = GameStage.RIVER
    
    return GameState(
        player_hands=player_hands,
        community_cards=community_cards,
        pot=pot,
        player_stacks=[stack0, stack1],
        current_bets=[bet0, bet1],
        button_position=button,
        stage=stage,
        current_player=current_player
    )


@st.composite
def action_strategy(draw):
    """Generate a random action."""
    action_type = draw(st.sampled_from([ActionType.FOLD, ActionType.CHECK, ActionType.CALL, ActionType.RAISE]))
    
    if action_type == ActionType.RAISE:
        amount = draw(st.integers(min_value=1, max_value=1000))
        return Action(action_type, amount)
    else:
        return Action(action_type)


class TestPropertyActionLegality:
    """Property-based tests for action legality."""
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=100)
    @given(game_state_strategy())
    def test_fold_always_legal_property(self, state):
        """Feature: texas-holdem-ai-training, Property 6: 行动合法性验证
        
        Test that FOLD is always legal in any game state.
        **Validates: Requirements 2.2**
        """
        action = Action(ActionType.FOLD)
        assert RuleEngine.is_action_legal(state, action) is True
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=100)
    @given(game_state_strategy(), action_strategy())
    def test_legal_action_can_be_applied(self, state, action):
        """Feature: texas-holdem-ai-training, Property 6: 行动合法性验证
        
        Test that if an action is legal, it can be applied without error.
        **Validates: Requirements 2.2**
        """
        if RuleEngine.is_action_legal(state, action):
            # Should not raise an exception
            try:
                new_state = RuleEngine.apply_action(state, action)
                # Verify state is still valid
                assert len(new_state.player_hands) == 2
                assert len(new_state.player_stacks) == 2
                assert new_state.pot >= 0
            except Exception as e:
                # If it fails, it's a bug
                pytest.fail(f"Legal action failed to apply: {e}")
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=100)
    @given(game_state_strategy(), action_strategy())
    def test_illegal_action_raises_error(self, state, action):
        """Feature: texas-holdem-ai-training, Property 6: 行动合法性验证
        
        Test that if an action is illegal, applying it raises an error.
        **Validates: Requirements 2.2**
        """
        if not RuleEngine.is_action_legal(state, action):
            with pytest.raises(ValueError):
                RuleEngine.apply_action(state, action)
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=100)
    @given(game_state_strategy())
    def test_check_legality_matches_bet_equality(self, state):
        """Feature: texas-holdem-ai-training, Property 6: 行动合法性验证
        
        Test that CHECK is legal if and only if bets are equal.
        **Validates: Requirements 2.2**
        """
        action = Action(ActionType.CHECK)
        is_legal = RuleEngine.is_action_legal(state, action)
        bets_equal = state.current_bets[0] == state.current_bets[1]
        
        assert is_legal == bets_equal
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=100)
    @given(game_state_strategy())
    def test_call_legality_matches_bet_inequality(self, state):
        """Feature: texas-holdem-ai-training, Property 6: 行动合法性验证
        
        Test that CALL is legal if and only if opponent has bet more AND player has enough chips.
        **Validates: Requirements 2.2**
        """
        action = Action(ActionType.CALL)
        is_legal = RuleEngine.is_action_legal(state, action)
        
        current_player = state.current_player
        opponent = 1 - current_player
        opponent_bet_more = state.current_bets[opponent] > state.current_bets[current_player]
        
        # CALL 只有在对手下注更多且玩家有足够筹码跟注时才合法
        bet_to_call = state.current_bets[opponent] - state.current_bets[current_player]
        has_enough_chips = state.player_stacks[current_player] >= bet_to_call
        
        expected_legal = opponent_bet_more and has_enough_chips
        assert is_legal == expected_legal



class TestPropertyChipConservation:
    """Property-based tests for chip conservation."""
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=100)
    @given(game_state_strategy())
    def test_chip_conservation_after_pot_distribution(self, state):
        """Feature: texas-holdem-ai-training, Property 7: 胜负判定正确性
        
        Test that total chips are conserved after pot distribution.
        **Validates: Requirements 2.3**
        """
        # Calculate initial total chips
        initial_total = state.player_stacks[0] + state.player_stacks[1] + state.pot
        
        # Distribute pot to a winner (test all cases: player 0, player 1, tie)
        for winner in [0, 1, -1]:
            test_state = deepcopy(state)
            new_state = RuleEngine.distribute_pot(test_state, winner)
            
            # Calculate final total chips
            final_total = new_state.player_stacks[0] + new_state.player_stacks[1] + new_state.pot
            
            # Total chips should be conserved
            assert final_total == initial_total, f"Chips not conserved: {initial_total} -> {final_total}"
    
    @settings(suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow], max_examples=100)
    @given(game_state_strategy(), action_strategy())
    def test_chip_conservation_after_action(self, state, action):
        """Feature: texas-holdem-ai-training, Property 7: 胜负判定正确性
        
        Test that total chips are conserved after applying an action.
        **Validates: Requirements 2.3**
        """
        if not RuleEngine.is_action_legal(state, action):
            return  # Skip illegal actions
        
        # Calculate initial total chips
        initial_total = state.player_stacks[0] + state.player_stacks[1] + state.pot
        
        # Apply action
        new_state = RuleEngine.apply_action(state, action)
        
        # Calculate final total chips
        final_total = new_state.player_stacks[0] + new_state.player_stacks[1] + new_state.pot
        
        # Total chips should be conserved
        assert final_total == initial_total, f"Chips not conserved after {action.action_type}: {initial_total} -> {final_total}"
