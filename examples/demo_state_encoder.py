"""Demo script for StateEncoder functionality."""

from models.core import Card, GameState, GameStage
from environment.state_encoder import StateEncoder


def main():
    """Demonstrate StateEncoder usage."""
    print("=" * 60)
    print("StateEncoder Demo")
    print("=" * 60)
    
    # Create encoder
    encoder = StateEncoder()
    print(f"\nEncoder configuration:")
    print(f"  Total encoding dimension: {encoder.encoding_dim}")
    print(f"  Hand dimension: {encoder.hand_dim}")
    print(f"  Community cards dimension: {encoder.community_dim}")
    print(f"  Chip information dimension: {encoder.chip_dim}")
    print(f"  Position information dimension: {encoder.position_dim}")
    
    # Create a sample game state (flop stage)
    player_hands = [
        (Card(14, 'h'), Card(13, 'h')),  # Player 0: A♥ K♥
        (Card(10, 's'), Card(10, 'd'))   # Player 1: 10♠ 10♦
    ]
    
    community_cards = [
        Card(12, 'h'),  # Q♥
        Card(11, 'h'),  # J♥
        Card(2, 'c')    # 2♣
    ]
    
    state = GameState(
        player_hands=player_hands,
        community_cards=community_cards,
        pot=150,
        player_stacks=[925, 925],
        current_bets=[75, 75],
        button_position=0,
        stage=GameStage.FLOP,
        action_history=[],
        current_player=1
    )
    
    print("\n" + "=" * 60)
    print("Game State:")
    print("=" * 60)
    print(f"Stage: {state.stage.value}")
    print(f"Player 0 hand: {player_hands[0][0]} {player_hands[0][1]}")
    print(f"Player 1 hand: {player_hands[1][0]} {player_hands[1][1]}")
    print(f"Community cards: {' '.join(str(c) for c in community_cards)}")
    print(f"Pot: {state.pot}")
    print(f"Player stacks: {state.player_stacks}")
    print(f"Current bets: {state.current_bets}")
    print(f"Button position: {state.button_position}")
    print(f"Current player: {state.current_player}")
    
    # Encode state from player 0's perspective
    print("\n" + "=" * 60)
    print("Encoding from Player 0's perspective:")
    print("=" * 60)
    encoding_p0 = encoder.encode(state, 0)
    print(f"Encoding shape: {encoding_p0.shape}")
    print(f"Encoding dtype: {encoding_p0.dtype}")
    
    # Show breakdown of encoding
    hand_section = encoding_p0[0:104]
    community_section = encoding_p0[104:364]
    chip_section = encoding_p0[364:368]
    position_section = encoding_p0[368:370]
    
    print(f"\nHand section (0:104):")
    print(f"  Number of 1s: {int(hand_section.sum())} (should be 2 for 2 cards)")
    
    print(f"\nCommunity section (104:364):")
    print(f"  Number of 1s: {int(community_section.sum())} (should be 3 for flop)")
    
    print(f"\nChip section (364:368):")
    print(f"  Player stack (normalized): {chip_section[0]:.4f}")
    print(f"  Opponent stack (normalized): {chip_section[1]:.4f}")
    print(f"  Pot (normalized): {chip_section[2]:.4f}")
    print(f"  Current bet (normalized): {chip_section[3]:.4f}")
    
    print(f"\nPosition section (368:370):")
    print(f"  Is button: {position_section[0]:.1f}")
    print(f"  Is current player: {position_section[1]:.1f}")
    
    # Encode state from player 1's perspective
    print("\n" + "=" * 60)
    print("Encoding from Player 1's perspective:")
    print("=" * 60)
    encoding_p1 = encoder.encode(state, 1)
    
    hand_section_p1 = encoding_p1[0:104]
    position_section_p1 = encoding_p1[368:370]
    
    print(f"Hand section (0:104):")
    print(f"  Number of 1s: {int(hand_section_p1.sum())} (should be 2 for 2 cards)")
    
    print(f"\nPosition section (368:370):")
    print(f"  Is button: {position_section_p1[0]:.1f}")
    print(f"  Is current player: {position_section_p1[1]:.1f}")
    
    # Demonstrate card encoding
    print("\n" + "=" * 60)
    print("Card Encoding Examples:")
    print("=" * 60)
    
    test_cards = [
        Card(2, 'h'),   # 2♥
        Card(14, 's'),  # A♠
        Card(7, 'd'),   # 7♦
    ]
    
    for card in test_cards:
        card_encoding = encoder.encode_cards([card])
        position = int(card_encoding.argmax())
        print(f"{card}: one-hot position = {position}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
