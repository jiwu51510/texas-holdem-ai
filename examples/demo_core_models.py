"""Demonstration of core data models."""

from models import Card, Action, GameState, TrainingConfig, ActionType, GameStage


def demo_cards():
    """Demonstrate Card creation and usage."""
    print("=== Card Demo ===")
    ace_hearts = Card(14, 'h')
    king_spades = Card(13, 's')
    print(f"Created cards: {ace_hearts}, {king_spades}")
    print(f"Ace of hearts rank: {ace_hearts.rank}, suit: {ace_hearts.suit}")
    print()


def demo_actions():
    """Demonstrate Action creation."""
    print("=== Action Demo ===")
    fold = Action(ActionType.FOLD)
    call = Action(ActionType.CALL)
    raise_action = Action(ActionType.RAISE, amount=50)
    print(f"Fold action: {fold.action_type.value}")
    print(f"Call action: {call.action_type.value}")
    print(f"Raise action: {raise_action.action_type.value} by {raise_action.amount}")
    print()


def demo_game_state():
    """Demonstrate GameState creation."""
    print("=== GameState Demo ===")
    player1_hand = (Card(14, 'h'), Card(13, 'h'))
    player2_hand = (Card(10, 'd'), Card(9, 'd'))
    
    state = GameState(
        player_hands=[player1_hand, player2_hand],
        community_cards=[],
        pot=15,
        player_stacks=[995, 990],
        current_bets=[5, 10],
        button_position=0,
        stage=GameStage.PREFLOP,
        current_player=1
    )
    
    print(f"Player 1 hand: {player1_hand[0]}, {player1_hand[1]}")
    print(f"Player 2 hand: {player2_hand[0]}, {player2_hand[1]}")
    print(f"Pot: {state.pot}")
    print(f"Stage: {state.stage.value}")
    print(f"Current player: {state.current_player}")
    print()


def demo_training_config():
    """Demonstrate TrainingConfig creation."""
    print("=== TrainingConfig Demo ===")
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        num_episodes=10000,
        initial_stack=1000,
        small_blind=5,
        big_blind=10
    )
    
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Number of episodes: {config.num_episodes}")
    print(f"Network architecture: {config.network_architecture}")
    print(f"Initial stack: {config.initial_stack}")
    print(f"Blinds: {config.small_blind}/{config.big_blind}")
    print()


if __name__ == "__main__":
    print("Texas Hold'em AI Training System - Core Models Demo\n")
    demo_cards()
    demo_actions()
    demo_game_state()
    demo_training_config()
    print("All core models working correctly!")
