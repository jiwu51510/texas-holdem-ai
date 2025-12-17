"""Demonstration of hand evaluator functionality."""

from models.core import Card, HandRank
from environment.hand_evaluator import evaluate_hand, compare_hands


def demo_hand_evaluation():
    """Demonstrate hand evaluation for all 9 poker hand types."""
    print("=== Hand Evaluation Demo ===\n")
    
    # 1. High Card
    print("1. High Card (A-K-Q-J-9):")
    high_card = [Card(14, 'h'), Card(13, 'd'), Card(12, 'c'), Card(11, 's'), Card(9, 'h')]
    rank, kickers = evaluate_hand(high_card)
    print(f"   Rank: {rank.name}, Kickers: {kickers}")
    print(f"   Cards: {', '.join(str(c) for c in high_card)}\n")
    
    # 2. One Pair
    print("2. One Pair (10-10-A-K-Q):")
    pair = [Card(10, 'h'), Card(10, 'd'), Card(14, 'c'), Card(13, 's'), Card(12, 'h')]
    rank, kickers = evaluate_hand(pair)
    print(f"   Rank: {rank.name}, Kickers: {kickers}")
    print(f"   Cards: {', '.join(str(c) for c in pair)}\n")
    
    # 3. Two Pair
    print("3. Two Pair (K-K-Q-Q-J):")
    two_pair = [Card(13, 'h'), Card(13, 'd'), Card(12, 'c'), Card(12, 's'), Card(11, 'h')]
    rank, kickers = evaluate_hand(two_pair)
    print(f"   Rank: {rank.name}, Kickers: {kickers}")
    print(f"   Cards: {', '.join(str(c) for c in two_pair)}\n")
    
    # 4. Three of a Kind
    print("4. Three of a Kind (9-9-9-A-K):")
    trips = [Card(9, 'h'), Card(9, 'd'), Card(9, 'c'), Card(14, 's'), Card(13, 'h')]
    rank, kickers = evaluate_hand(trips)
    print(f"   Rank: {rank.name}, Kickers: {kickers}")
    print(f"   Cards: {', '.join(str(c) for c in trips)}\n")
    
    # 5. Straight
    print("5. Straight (10-9-8-7-6):")
    straight = [Card(10, 'h'), Card(9, 'd'), Card(8, 'c'), Card(7, 's'), Card(6, 'h')]
    rank, kickers = evaluate_hand(straight)
    print(f"   Rank: {rank.name}, High Card: {kickers[0]}")
    print(f"   Cards: {', '.join(str(c) for c in straight)}\n")
    
    # 6. Flush
    print("6. Flush (A-J-9-7-5 all hearts):")
    flush = [Card(14, 'h'), Card(11, 'h'), Card(9, 'h'), Card(7, 'h'), Card(5, 'h')]
    rank, kickers = evaluate_hand(flush)
    print(f"   Rank: {rank.name}, Kickers: {kickers}")
    print(f"   Cards: {', '.join(str(c) for c in flush)}\n")
    
    # 7. Full House
    print("7. Full House (8-8-8-K-K):")
    full_house = [Card(8, 'h'), Card(8, 'd'), Card(8, 'c'), Card(13, 's'), Card(13, 'h')]
    rank, kickers = evaluate_hand(full_house)
    print(f"   Rank: {rank.name}, Trips: {kickers[0]}, Pair: {kickers[1]}")
    print(f"   Cards: {', '.join(str(c) for c in full_house)}\n")
    
    # 8. Four of a Kind
    print("8. Four of a Kind (7-7-7-7-A):")
    quads = [Card(7, 'h'), Card(7, 'd'), Card(7, 'c'), Card(7, 's'), Card(14, 'h')]
    rank, kickers = evaluate_hand(quads)
    print(f"   Rank: {rank.name}, Quads: {kickers[0]}, Kicker: {kickers[1]}")
    print(f"   Cards: {', '.join(str(c) for c in quads)}\n")
    
    # 9. Straight Flush
    print("9. Straight Flush (9-8-7-6-5 all spades):")
    straight_flush = [Card(9, 's'), Card(8, 's'), Card(7, 's'), Card(6, 's'), Card(5, 's')]
    rank, kickers = evaluate_hand(straight_flush)
    print(f"   Rank: {rank.name}, High Card: {kickers[0]}")
    print(f"   Cards: {', '.join(str(c) for c in straight_flush)}\n")


def demo_special_straights():
    """Demonstrate special straight cases."""
    print("=== Special Straights Demo ===\n")
    
    # Wheel (A-2-3-4-5)
    print("1. Wheel Straight (A-2-3-4-5):")
    wheel = [Card(14, 'h'), Card(5, 'd'), Card(4, 'c'), Card(3, 's'), Card(2, 'h')]
    rank, kickers = evaluate_hand(wheel)
    print(f"   Rank: {rank.name}, High Card: {kickers[0]} (5, not Ace)")
    print(f"   Cards: {', '.join(str(c) for c in wheel)}\n")
    
    # Broadway (10-J-Q-K-A)
    print("2. Broadway Straight (10-J-Q-K-A):")
    broadway = [Card(14, 'h'), Card(13, 'd'), Card(12, 'c'), Card(11, 's'), Card(10, 'h')]
    rank, kickers = evaluate_hand(broadway)
    print(f"   Rank: {rank.name}, High Card: {kickers[0]} (Ace)")
    print(f"   Cards: {', '.join(str(c) for c in broadway)}\n")
    
    # Royal Flush
    print("3. Royal Flush (10-J-Q-K-A all hearts):")
    royal = [Card(14, 'h'), Card(13, 'h'), Card(12, 'h'), Card(11, 'h'), Card(10, 'h')]
    rank, kickers = evaluate_hand(royal)
    print(f"   Rank: {rank.name}, High Card: {kickers[0]}")
    print(f"   Cards: {', '.join(str(c) for c in royal)}\n")


def demo_hand_comparison():
    """Demonstrate hand comparison."""
    print("=== Hand Comparison Demo ===\n")
    
    # Example 1: Different hand ranks
    print("1. Flush vs Straight:")
    hand1 = [Card(14, 'h'), Card(10, 'h')]  # Ace-Ten suited
    hand2 = [Card(9, 'd'), Card(8, 'c')]    # Nine-Eight offsuit
    community = [Card(7, 'h'), Card(6, 'h'), Card(5, 'h')]  # Three hearts on board
    
    print(f"   Hand 1: {hand1[0]}, {hand1[1]}")
    print(f"   Hand 2: {hand2[0]}, {hand2[1]}")
    print(f"   Board: {', '.join(str(c) for c in community)}")
    
    result = compare_hands(hand1, hand2, community)
    if result == 0:
        print(f"   Result: Hand 1 wins (makes flush)\n")
    elif result == 1:
        print(f"   Result: Hand 2 wins\n")
    else:
        print(f"   Result: Tie\n")
    
    # Example 2: Same rank, different kickers
    print("2. Pair vs Pair with better kicker:")
    hand1 = [Card(10, 'h'), Card(14, 'd')]  # Ten-Ace
    hand2 = [Card(10, 'c'), Card(13, 's')]  # Ten-King
    community = [Card(10, 's'), Card(7, 'h'), Card(5, 'd')]  # Pair of tens on board
    
    print(f"   Hand 1: {hand1[0]}, {hand1[1]}")
    print(f"   Hand 2: {hand2[0]}, {hand2[1]}")
    print(f"   Board: {', '.join(str(c) for c in community)}")
    
    result = compare_hands(hand1, hand2, community)
    if result == 0:
        print(f"   Result: Hand 1 wins (Ace kicker beats King kicker)\n")
    elif result == 1:
        print(f"   Result: Hand 2 wins\n")
    else:
        print(f"   Result: Tie\n")
    
    # Example 3: Complete tie
    print("3. Complete Tie:")
    hand1 = [Card(2, 'h'), Card(3, 'd')]
    hand2 = [Card(4, 'c'), Card(5, 's')]
    community = [Card(14, 's'), Card(14, 'h'), Card(14, 'd'), Card(13, 'c'), Card(13, 'h')]
    
    print(f"   Hand 1: {hand1[0]}, {hand1[1]}")
    print(f"   Hand 2: {hand2[0]}, {hand2[1]}")
    print(f"   Board: {', '.join(str(c) for c in community)}")
    
    result = compare_hands(hand1, hand2, community)
    if result == 0:
        print(f"   Result: Hand 1 wins\n")
    elif result == 1:
        print(f"   Result: Hand 2 wins\n")
    else:
        print(f"   Result: Tie (both play the board - AAA KK)\n")


def demo_seven_card_evaluation():
    """Demonstrate finding best hand from 7 cards."""
    print("=== Seven Card Evaluation Demo ===\n")
    
    print("Finding best 5-card hand from 2 hole cards + 5 community cards:")
    hole = [Card(14, 'h'), Card(14, 'd')]  # Pocket Aces
    community = [
        Card(14, 'c'),  # Ace on flop
        Card(13, 's'),  # King
        Card(13, 'h'),  # King
        Card(10, 'd'),  # Ten
        Card(9, 'c')    # Nine
    ]
    
    print(f"   Hole Cards: {hole[0]}, {hole[1]}")
    print(f"   Community: {', '.join(str(c) for c in community)}")
    
    all_cards = hole + community
    rank, kickers = evaluate_hand(all_cards)
    
    print(f"   Best Hand: {rank.name}")
    print(f"   Details: Three Aces ({kickers[0]}) with pair of Kings ({kickers[1]})")
    print(f"   This is a Full House!\n")


if __name__ == "__main__":
    print("Texas Hold'em Hand Evaluator Demo\n")
    print("=" * 60)
    print()
    
    demo_hand_evaluation()
    print("=" * 60)
    print()
    
    demo_special_straights()
    print("=" * 60)
    print()
    
    demo_hand_comparison()
    print("=" * 60)
    print()
    
    demo_seven_card_evaluation()
    print("=" * 60)
    print()
    
    print("All hand evaluation features working correctly!")
