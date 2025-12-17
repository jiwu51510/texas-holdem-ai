"""Demo script for PokerEnvironment."""

from environment.poker_environment import PokerEnvironment
from models.core import Action, ActionType


def main():
    """Demonstrate the poker environment."""
    print("=== Texas Hold'em Poker Environment Demo ===\n")
    
    # Create environment
    env = PokerEnvironment(initial_stack=1000, small_blind=5, big_blind=10)
    
    # Start a new hand
    print("Starting a new hand...")
    state = env.reset()
    
    button = state.button_position
    big_blind = 1 - button
    
    print(f"\nInitial State:")
    print(f"  Button (Small Blind): Player {button}")
    print(f"  Big Blind: Player {big_blind}")
    print(f"  Player 0 Hand: {state.player_hands[0][0]} {state.player_hands[0][1]}")
    print(f"  Player 1 Hand: {state.player_hands[1][0]} {state.player_hands[1][1]}")
    print(f"  Community Cards: {[str(c) for c in state.community_cards]}")
    print(f"  Pot: {state.pot}")
    print(f"  Player Stacks: {state.player_stacks}")
    print(f"  Current Bets: {state.current_bets}")
    print(f"  Stage: {state.stage.value}")
    print(f"  Current Player: Player {state.current_player} (Button acts first preflop)")
    
    # Get legal actions
    legal_actions = env.get_legal_actions()
    raise_types = (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG)
    print(f"\nLegal Actions for Player {state.current_player}:")
    for i, action in enumerate(legal_actions):
        if action.action_type in raise_types:
            print(f"  {i+1}. {action.action_type.value.upper()} {action.amount}")
        else:
            print(f"  {i+1}. {action.action_type.value.upper()}")
    
    # Simulate a few actions
    print("\n=== Simulating Actions ===")
    
    # Player 0 (button/small blind) calls the big blind
    action = Action(ActionType.CALL)
    print(f"\nPlayer {state.current_player} action: CALL")
    state, reward, done = env.step(action)
    print(f"  Pot: {state.pot}, Stacks: {state.player_stacks}, Done: {done}")
    
    if not done:
        # Player 1 checks
        action = Action(ActionType.CHECK)
        print(f"\nPlayer {state.current_player} action: CHECK")
        state, reward, done = env.step(action)
        print(f"  Stage: {state.stage.value}")
        print(f"  Community Cards: {[str(c) for c in state.community_cards]}")
        print(f"  Pot: {state.pot}, Stacks: {state.player_stacks}, Done: {done}")
    
    if not done:
        # Player 1 (first to act post-flop) checks
        action = Action(ActionType.CHECK)
        print(f"\nPlayer {state.current_player} action: CHECK")
        state, reward, done = env.step(action)
        print(f"  Pot: {state.pot}, Done: {done}")
    
    if not done:
        # Player 0 bets
        legal_actions = env.get_legal_actions()
        # Find a raise action
        raise_types = (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG)
        raise_action = next((a for a in legal_actions if a.action_type in raise_types), None)
        if raise_action:
            print(f"\nPlayer {state.current_player} action: RAISE {raise_action.amount}")
            state, reward, done = env.step(raise_action)
            print(f"  Pot: {state.pot}, Stacks: {state.player_stacks}, Done: {done}")
    
    if not done:
        # Player 1 folds
        action = Action(ActionType.FOLD)
        print(f"\nPlayer {state.current_player} action: FOLD")
        state, reward, done = env.step(action)
        print(f"  Pot: {state.pot}, Stacks: {state.player_stacks}, Done: {done}")
        print(f"  Reward for Player {1 - state.current_player}: {-reward}")
    
    print("\n=== Hand Complete ===")
    print(f"Final Stacks: {state.player_stacks}")
    
    # Demonstrate a full hand to showdown
    print("\n\n=== Second Hand - Going to Showdown ===")
    state = env.reset()
    print(f"\nPlayer 0 Hand: {state.player_hands[0][0]} {state.player_hands[0][1]}")
    print(f"Player 1 Hand: {state.player_hands[1][0]} {state.player_hands[1][1]}")
    
    # Both players check through all streets
    actions_sequence = [
        (ActionType.CALL, 0),   # Player 0 calls
        (ActionType.CHECK, 0),  # Player 1 checks (flop)
        (ActionType.CHECK, 0),  # Player 1 checks
        (ActionType.CHECK, 0),  # Player 0 checks (turn)
        (ActionType.CHECK, 0),  # Player 1 checks
        (ActionType.CHECK, 0),  # Player 0 checks (river)
        (ActionType.CHECK, 0),  # Player 1 checks
        (ActionType.CHECK, 0),  # Player 0 checks (showdown)
    ]
    
    done = False
    for action_type, amount in actions_sequence:
        if done:
            break
        action = Action(action_type, amount)
        print(f"\nPlayer {state.current_player} action: {action_type.value.upper()}")
        state, reward, done = env.step(action)
        print(f"  Stage: {state.stage.value}, Community: {[str(c) for c in state.community_cards]} ({len(state.community_cards)} cards)")
        print(f"  Current player: {state.current_player}, Bets: {state.current_bets}")
        if done:
            print(f"  Hand complete! Final stacks: {state.player_stacks}")
            break


if __name__ == "__main__":
    main()
