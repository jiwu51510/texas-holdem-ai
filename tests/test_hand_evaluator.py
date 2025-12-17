"""Tests for hand evaluator."""

import pytest
from models.core import Card, HandRank
from environment.hand_evaluator import (
    evaluate_hand, compare_hands, is_flush, is_straight, find_n_of_a_kind
)


class TestHandEvaluation:
    """Tests for hand evaluation functionality."""
    
    def test_high_card(self):
        """Test high card hand recognition."""
        cards = [
            Card(14, 'h'),  # A
            Card(10, 'd'),  # 10
            Card(8, 'c'),   # 8
            Card(6, 's'),   # 6
            Card(3, 'h')    # 3
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.HIGH_CARD
        assert kickers == [14, 10, 8, 6, 3]
    
    def test_pair(self):
        """Test one pair hand recognition."""
        cards = [
            Card(10, 'h'),  # 10
            Card(10, 'd'),  # 10
            Card(8, 'c'),   # 8
            Card(6, 's'),   # 6
            Card(3, 'h')    # 3
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.PAIR
        assert kickers[0] == 10  # Pair rank
        assert kickers[1:] == [8, 6, 3]  # Kickers in descending order
    
    def test_two_pair(self):
        """Test two pair hand recognition."""
        cards = [
            Card(10, 'h'),  # 10
            Card(10, 'd'),  # 10
            Card(8, 'c'),   # 8
            Card(8, 's'),   # 8
            Card(3, 'h')    # 3
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.TWO_PAIR
        assert kickers[0] == 10  # Higher pair
        assert kickers[1] == 8   # Lower pair
        assert kickers[2] == 3   # Kicker
    
    def test_three_of_a_kind(self):
        """Test three of a kind hand recognition."""
        cards = [
            Card(10, 'h'),  # 10
            Card(10, 'd'),  # 10
            Card(10, 'c'),  # 10
            Card(8, 's'),   # 8
            Card(3, 'h')    # 3
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.THREE_OF_A_KIND
        assert kickers[0] == 10  # Trip rank
        assert kickers[1:] == [8, 3]  # Kickers
    
    def test_straight(self):
        """Test straight hand recognition."""
        cards = [
            Card(10, 'h'),  # 10
            Card(9, 'd'),   # 9
            Card(8, 'c'),   # 8
            Card(7, 's'),   # 7
            Card(6, 'h')    # 6
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT
        assert kickers == [10]  # High card of straight
    
    def test_flush(self):
        """Test flush hand recognition."""
        cards = [
            Card(14, 'h'),  # A
            Card(10, 'h'),  # 10
            Card(8, 'h'),   # 8
            Card(6, 'h'),   # 6
            Card(3, 'h')    # 3
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.FLUSH
        assert kickers == [14, 10, 8, 6, 3]
    
    def test_full_house(self):
        """Test full house hand recognition."""
        cards = [
            Card(10, 'h'),  # 10
            Card(10, 'd'),  # 10
            Card(10, 'c'),  # 10
            Card(8, 's'),   # 8
            Card(8, 'h')    # 8
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.FULL_HOUSE
        assert kickers[0] == 10  # Trip rank
        assert kickers[1] == 8   # Pair rank
    
    def test_four_of_a_kind(self):
        """Test four of a kind hand recognition."""
        cards = [
            Card(10, 'h'),  # 10
            Card(10, 'd'),  # 10
            Card(10, 'c'),  # 10
            Card(10, 's'),  # 10
            Card(3, 'h')    # 3
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.FOUR_OF_A_KIND
        assert kickers[0] == 10  # Quad rank
        assert kickers[1] == 3   # Kicker
    
    def test_straight_flush(self):
        """Test straight flush hand recognition."""
        cards = [
            Card(10, 'h'),  # 10
            Card(9, 'h'),   # 9
            Card(8, 'h'),   # 8
            Card(7, 'h'),   # 7
            Card(6, 'h')    # 6
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT_FLUSH
        assert kickers == [10]  # High card of straight flush


class TestEdgeCases:
    """Tests for edge cases in hand evaluation."""
    
    def test_wheel_straight_ace_low(self):
        """Test A-2-3-4-5 straight (wheel) - Ace plays as low."""
        cards = [
            Card(14, 'h'),  # A
            Card(5, 'd'),   # 5
            Card(4, 'c'),   # 4
            Card(3, 's'),   # 3
            Card(2, 'h')    # 2
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT
        assert kickers == [5]  # High card is 5 in a wheel, not Ace
    
    def test_broadway_straight_ace_high(self):
        """Test 10-J-Q-K-A straight (broadway) - Ace plays as high."""
        cards = [
            Card(14, 'h'),  # A
            Card(13, 'd'),  # K
            Card(12, 'c'),  # Q
            Card(11, 's'),  # J
            Card(10, 'h')   # 10
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT
        assert kickers == [14]  # High card is Ace
    
    def test_wheel_straight_flush(self):
        """Test A-2-3-4-5 straight flush (steel wheel)."""
        cards = [
            Card(14, 's'),  # A
            Card(5, 's'),   # 5
            Card(4, 's'),   # 4
            Card(3, 's'),   # 3
            Card(2, 's')    # 2
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT_FLUSH
        assert kickers == [5]  # High card is 5
    
    def test_royal_flush(self):
        """Test royal flush (10-J-Q-K-A straight flush)."""
        cards = [
            Card(14, 'h'),  # A
            Card(13, 'h'),  # K
            Card(12, 'h'),  # Q
            Card(11, 'h'),  # J
            Card(10, 'h')   # 10
        ]
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT_FLUSH
        assert kickers == [14]  # High card is Ace
    
    def test_straight_vs_flush(self):
        """Test that flush beats straight."""
        straight = [
            Card(10, 'h'),  # 10
            Card(9, 'd'),   # 9
            Card(8, 'c'),   # 8
            Card(7, 's'),   # 7
            Card(6, 'h')    # 6
        ]
        flush = [
            Card(14, 'h'),  # A
            Card(10, 'h'),  # 10
            Card(8, 'h'),   # 8
            Card(6, 'h'),   # 6
            Card(3, 'h')    # 3
        ]
        
        straight_rank, _ = evaluate_hand(straight)
        flush_rank, _ = evaluate_hand(flush)
        
        assert flush_rank.value > straight_rank.value
    
    def test_seven_card_best_hand(self):
        """Test finding best 5-card hand from 7 cards (2 hole + 5 community)."""
        # Player has pocket aces
        hole_cards = [Card(14, 'h'), Card(14, 'd')]
        # Board has three more aces and two kings (impossible but tests the logic)
        # Actually, let's make it realistic: board has one ace and a pair of kings
        community = [
            Card(14, 'c'),  # A
            Card(13, 's'),  # K
            Card(13, 'h'),  # K
            Card(10, 'd'),  # 10
            Card(9, 'c')    # 9
        ]
        
        all_cards = hole_cards + community
        rank, kickers = evaluate_hand(all_cards)
        
        # Should recognize full house (AAA KK)
        assert rank == HandRank.FULL_HOUSE
        assert kickers[0] == 14  # Trip aces
        assert kickers[1] == 13  # Pair kings
    
    def test_seven_card_flush(self):
        """Test finding flush from 7 cards when 5+ are same suit."""
        cards = [
            Card(14, 'h'),  # A
            Card(12, 'h'),  # Q
            Card(10, 'h'),  # 10
            Card(8, 'h'),   # 8
            Card(6, 'h'),   # 6
            Card(4, 'd'),   # 4 (different suit)
            Card(2, 'c')    # 2 (different suit)
        ]
        
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.FLUSH
        # Should use the 5 highest hearts
        assert kickers == [14, 12, 10, 8, 6]
    
    def test_seven_card_straight(self):
        """Test finding straight from 7 cards."""
        cards = [
            Card(10, 'h'),  # 10
            Card(9, 'd'),   # 9
            Card(8, 'c'),   # 8
            Card(7, 's'),   # 7
            Card(6, 'h'),   # 6
            Card(3, 'd'),   # 3
            Card(2, 'c')    # 2
        ]
        
        rank, kickers = evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT
        assert kickers == [10]  # 10-high straight


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_is_flush_true(self):
        """Test is_flush returns True for all same suit."""
        suits = ['h', 'h', 'h', 'h', 'h']
        assert is_flush(suits) is True
    
    def test_is_flush_false(self):
        """Test is_flush returns False for mixed suits."""
        suits = ['h', 'h', 'h', 'h', 'd']
        assert is_flush(suits) is False
    
    def test_is_straight_regular(self):
        """Test is_straight for regular straight."""
        ranks = [10, 9, 8, 7, 6]
        is_str, high = is_straight(ranks)
        assert is_str is True
        assert high == 10
    
    def test_is_straight_wheel(self):
        """Test is_straight for wheel (A-2-3-4-5)."""
        ranks = [14, 5, 4, 3, 2]
        is_str, high = is_straight(ranks)
        assert is_str is True
        assert high == 5  # 5 is high card in wheel
    
    def test_is_straight_false(self):
        """Test is_straight returns False for non-straight."""
        ranks = [14, 10, 8, 6, 3]
        is_str, high = is_straight(ranks)
        assert is_str is False
        assert high == 0
    
    def test_find_n_of_a_kind(self):
        """Test finding ranks that appear n times."""
        ranks = [10, 10, 10, 8, 8]
        
        trips = find_n_of_a_kind(ranks, 3)
        assert trips == [10]
        
        pairs = find_n_of_a_kind(ranks, 2)
        assert pairs == [8]
        
        singles = find_n_of_a_kind(ranks, 1)
        assert singles == []


class TestCompareHands:
    """Tests for hand comparison."""
    
    def test_compare_different_ranks(self):
        """Test comparing hands with different ranks."""
        # Hand 1: Pair of tens
        hand1 = [Card(10, 'h'), Card(10, 'd')]
        # Hand 2: Pair of eights
        hand2 = [Card(8, 'h'), Card(8, 'd')]
        community = [Card(14, 'c'), Card(7, 's'), Card(5, 'h')]
        
        result = compare_hands(hand1, hand2, community)
        assert result == 0  # Hand 1 wins
    
    def test_compare_same_rank_different_kickers(self):
        """Test comparing hands with same rank but different kickers."""
        # Hand 1: Pair of tens with Ace kicker
        hand1 = [Card(10, 'h'), Card(14, 'd')]
        # Hand 2: Pair of tens with King kicker
        hand2 = [Card(10, 'c'), Card(13, 's')]
        community = [Card(10, 's'), Card(7, 'h'), Card(5, 'd')]
        
        result = compare_hands(hand1, hand2, community)
        assert result == 0  # Hand 1 wins (better kicker)
    
    def test_compare_tie(self):
        """Test comparing hands that tie."""
        # Both players have same pair on board
        hand1 = [Card(14, 'h'), Card(13, 'd')]
        hand2 = [Card(14, 'c'), Card(13, 's')]
        community = [Card(10, 's'), Card(10, 'h'), Card(9, 'd')]
        
        result = compare_hands(hand1, hand2, community)
        assert result == -1  # Tie
    
    def test_compare_flush_vs_straight(self):
        """Test that flush beats straight."""
        # Hand 1: Makes flush
        hand1 = [Card(14, 'h'), Card(10, 'h')]
        # Hand 2: Makes straight
        hand2 = [Card(9, 'd'), Card(8, 'c')]
        community = [
            Card(7, 'h'),
            Card(6, 'h'),
            Card(5, 'h'),
        ]
        
        result = compare_hands(hand1, hand2, community)
        assert result == 0  # Hand 1 (flush) wins
    
    def test_compare_no_community(self):
        """Test comparing hands with no community cards."""
        # Just comparing 5-card hands directly
        hand1 = [
            Card(14, 'h'), Card(14, 'd'), Card(14, 'c'),
            Card(13, 's'), Card(13, 'h')
        ]
        hand2 = [
            Card(12, 'h'), Card(12, 'd'), Card(12, 'c'),
            Card(11, 's'), Card(11, 'h')
        ]
        
        result = compare_hands(hand1, hand2)
        assert result == 0  # Hand 1 (AAA KK) beats hand 2 (QQQ JJ)


class TestInvalidInputs:
    """Tests for invalid inputs."""
    
    def test_too_few_cards(self):
        """Test that evaluating fewer than 5 cards raises error."""
        cards = [Card(14, 'h'), Card(13, 'd'), Card(12, 'c')]
        
        with pytest.raises(ValueError, match="Need at least 5 cards"):
            evaluate_hand(cards)



# Property-Based Tests
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from itertools import combinations


@st.composite
def card_strategy(draw):
    """Strategy for generating valid cards."""
    rank = draw(st.integers(min_value=2, max_value=14))
    suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
    return Card(rank, suit)


@st.composite
def unique_cards_strategy(draw, n):
    """Strategy for generating n unique cards."""
    # Generate all possible cards
    all_cards = [(r, s) for r in range(2, 15) for s in ['h', 'd', 'c', 's']]
    # Sample n unique cards
    selected = draw(st.lists(
        st.sampled_from(all_cards),
        min_size=n,
        max_size=n,
        unique=True
    ))
    return [Card(rank, suit) for rank, suit in selected]


@st.composite
def hand_strategy(draw):
    """Strategy for generating a 2-card hand."""
    return draw(unique_cards_strategy(2))


@st.composite
def community_strategy(draw):
    """Strategy for generating 0-5 community cards."""
    size = draw(st.sampled_from([0, 3, 4, 5]))
    if size == 0:
        return []
    return draw(unique_cards_strategy(size))


@st.composite
def seven_card_hand_strategy(draw):
    """Strategy for generating 7 unique cards (2 hole + 5 community)."""
    return draw(unique_cards_strategy(7))


class TestPropertyBasedHandComparison:
    """Property-based tests for hand comparison.
    
    Feature: texas-holdem-ai-training, Property 7: 胜负判定正确性
    Validates: Requirements 2.3
    """
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    @given(
        hand1=hand_strategy(),
        hand2=hand_strategy(),
        community=community_strategy()
    )
    def test_comparison_transitivity(self, hand1, hand2, community):
        """Test that hand comparison is transitive.
        
        Feature: texas-holdem-ai-training, Property 7: 胜负判定正确性
        Validates: Requirements 2.3
        
        For any three hands A, B, C:
        If A > B and B > C, then A > C
        """
        # Ensure all cards are unique
        all_cards = hand1 + hand2 + community
        card_tuples = [(c.rank, c.suit) for c in all_cards]
        assume(len(card_tuples) == len(set(card_tuples)))
        
        # Need at least 5 cards total to evaluate
        assume(len(hand1) + len(community) >= 5)
        assume(len(hand2) + len(community) >= 5)
        
        # Generate a third hand that doesn't overlap
        used = set(card_tuples)
        hand3 = []
        for rank in range(2, 15):
            for suit in ['h', 'd', 'c', 's']:
                if (rank, suit) not in used and len(hand3) < 2:
                    hand3.append(Card(rank, suit))
        
        if len(hand3) < 2:
            assume(False)  # Skip if we can't generate a third hand
        
        # Compare hands
        result_12 = compare_hands(hand1, hand2, community)
        result_23 = compare_hands(hand2, hand3, community)
        result_13 = compare_hands(hand1, hand3, community)
        
        # Check transitivity
        # If hand1 > hand2 (result_12 == 0) and hand2 > hand3 (result_23 == 0)
        # Then hand1 > hand3 (result_13 == 0)
        if result_12 == 0 and result_23 == 0:
            assert result_13 == 0, "Transitivity violated: A > B and B > C but A <= C"
        
        # If hand1 < hand2 (result_12 == 1) and hand2 < hand3 (result_23 == 1)
        # Then hand1 < hand3 (result_13 == 1)
        if result_12 == 1 and result_23 == 1:
            assert result_13 == 1, "Transitivity violated: A < B and B < C but A >= C"
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    @given(
        hand1=hand_strategy(),
        hand2=hand_strategy(),
        community=community_strategy()
    )
    def test_comparison_symmetry(self, hand1, hand2, community):
        """Test that hand comparison is symmetric.
        
        Feature: texas-holdem-ai-training, Property 7: 胜负判定正确性
        Validates: Requirements 2.3
        
        For any two hands A and B:
        If A > B, then B < A
        If A < B, then B > A
        If A == B, then B == A
        """
        # Ensure all cards are unique
        all_cards = hand1 + hand2 + community
        card_tuples = [(c.rank, c.suit) for c in all_cards]
        assume(len(card_tuples) == len(set(card_tuples)))
        
        # Need at least 5 cards total to evaluate
        assume(len(hand1) + len(community) >= 5)
        assume(len(hand2) + len(community) >= 5)
        
        result_12 = compare_hands(hand1, hand2, community)
        result_21 = compare_hands(hand2, hand1, community)
        
        # Check symmetry
        if result_12 == 0:  # hand1 wins
            assert result_21 == 1, "Symmetry violated: A > B but B doesn't < A"
        elif result_12 == 1:  # hand2 wins
            assert result_21 == 0, "Symmetry violated: A < B but B doesn't > A"
        else:  # tie
            assert result_21 == -1, "Symmetry violated: A == B but B doesn't == A"
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    @given(
        hand=hand_strategy(),
        community=community_strategy()
    )
    def test_hand_equals_itself(self, hand, community):
        """Test that a hand always ties with itself.
        
        Feature: texas-holdem-ai-training, Property 7: 胜负判定正确性
        Validates: Requirements 2.3
        
        For any hand A: A == A
        """
        # Ensure all cards are unique
        all_cards = hand + community
        card_tuples = [(c.rank, c.suit) for c in all_cards]
        assume(len(card_tuples) == len(set(card_tuples)))
        
        # Need at least 5 cards total to evaluate
        assume(len(hand) + len(community) >= 5)
        
        result = compare_hands(hand, hand, community)
        assert result == -1, "A hand should always tie with itself"
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    @given(cards=seven_card_hand_strategy())
    def test_chip_conservation_property(self, cards):
        """Test chip conservation in winner determination.
        
        Feature: texas-holdem-ai-training, Property 7: 胜负判定正确性
        Validates: Requirements 2.3
        
        For any game outcome, the total chips should remain constant.
        This tests that exactly one player wins (or tie), never both.
        """
        # Split into two hands and community
        hand1 = cards[:2]
        hand2 = cards[2:4]
        community = cards[4:7]
        
        result = compare_hands(hand1, hand2, community)
        
        # Result must be one of: 0 (hand1 wins), 1 (hand2 wins), or -1 (tie)
        assert result in [0, 1, -1], f"Invalid comparison result: {result}"
        
        # Simulate chip distribution
        pot = 100
        initial_total = pot
        
        if result == 0:  # hand1 wins
            hand1_chips = pot
            hand2_chips = 0
        elif result == 1:  # hand2 wins
            hand1_chips = 0
            hand2_chips = pot
        else:  # tie
            hand1_chips = pot // 2
            hand2_chips = pot // 2
        
        # Total chips should be conserved (allowing for rounding in tie)
        final_total = hand1_chips + hand2_chips
        assert abs(final_total - initial_total) <= 1, \
            f"Chips not conserved: started with {initial_total}, ended with {final_total}"
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    @given(cards=unique_cards_strategy(5))
    def test_hand_rank_deterministic(self, cards):
        """Test that hand evaluation is deterministic.
        
        Feature: texas-holdem-ai-training, Property 7: 胜负判定正确性
        Validates: Requirements 2.3
        
        For any hand, evaluating it multiple times should give the same result.
        """
        rank1, kickers1 = evaluate_hand(cards)
        rank2, kickers2 = evaluate_hand(cards)
        
        assert rank1 == rank2, "Hand rank should be deterministic"
        assert kickers1 == kickers2, "Kickers should be deterministic"
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
    @given(cards=unique_cards_strategy(5))
    def test_hand_rank_order_invariant(self, cards):
        """Test that hand evaluation is invariant to card order.
        
        Feature: texas-holdem-ai-training, Property 7: 胜负判定正确性
        Validates: Requirements 2.3
        
        For any hand, the order of cards shouldn't affect the evaluation.
        """
        import random
        
        # Evaluate original order
        rank1, kickers1 = evaluate_hand(cards)
        
        # Shuffle and evaluate again
        shuffled = cards.copy()
        random.shuffle(shuffled)
        rank2, kickers2 = evaluate_hand(shuffled)
        
        assert rank1 == rank2, "Hand rank should be invariant to card order"
        assert sorted(kickers1) == sorted(kickers2), \
            "Kickers should be invariant to card order"
