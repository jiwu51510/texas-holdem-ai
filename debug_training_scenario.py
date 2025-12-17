#!/usr/bin/env python3
"""诊断训练场景。

模拟实际的训练场景，检查遗憾值计算是否正确。
"""

import torch
import numpy as np
from typing import List, Dict

from models.core import GameState, GameStage, Card, Action, ActionType, TrainingConfig
from models.networks import RegretNetwork, PolicyNetwork
from environment.state_encoder import StateEncoder
from environment.poker_environment import PokerEnvironment
from environment.hand_evaluator import compare_hands


def parse_card(card_str: str) -> Card:
    """解析牌的字符串表示。"""
    rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
    rank_str = card_str[:-1].upper()
    suit_str = card_str[-1].lower()
    if rank_str in rank_map:
        rank = rank_map[rank_str]
    else:
        rank = int(rank_str)
    return Card(rank, suit_str)


def card_to_str(card: Card) -> str:
    """将牌转换为字符串。"""
    rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
    rank_str = rank_map.get(card.rank, str(card.rank))
    return f"{rank_str}{card.suit}"


def print_separator(title: str):
    """打印分隔线。"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def create_river_state(
    player_hands: List[List[Card]],
    community_cards: List[Card],
    initial_stack: int = 30,
    big_blind: int = 10
) -> GameState:
    """创建河牌阶段的初始状态。"""
    hands_as_tuples = [
        (hand[0], hand[1]) for hand in player_hands
    ]
    
    pot = 2 * big_blind
    stacks = [
        initial_stack - big_blind,
        initial_stack - big_blind
    ]
    
    return GameState(
        player_hands=hands_as_tuples,
        community_cards=community_cards,
        pot=pot,
        player_stacks=stacks,
        current_bets=[0, 0],
        button_position=0,
        stage=GameStage.RIVER,
        action_history=[],
        current_player=1  # 大盲位先行动
    )


def get_terminal_utility(state: GameState, player_id: int, initial_stack: int = 30) -> float:
    """获取终止状态的收益。"""
    winner = -1
    
    for action in state.action_history:
        if action.action_type == ActionType.FOLD:
            folder = 1 - state.current_player
            winner = 1 - folder
            break
    
    if winner == -1:
        hand1 = list(state.player_hands[0])
        hand2 = list(state.player_hands[1])
        community = state.community_cards
        winner = compare_hands(hand1, hand2, community)
    
    current_stack = state.player_stacks[player_id]
    pot = state.pot
    
    if winner == -1:
        pot_share = pot / 2
    elif winner == player_id:
        pot_share = pot
    else:
        pot_share = 0
    
    final_stack = current_stack + pot_share
    return float(final_stack - initial_stack)


def simulate_game(
    p0_hand: List[Card],
    p1_hand: List[Card],
    board: List[Card],
    p0_action: ActionType,
    p1_action: ActionType,
    initial_stack: int = 30,
    big_blind: int = 10
) -> Dict:
    """模拟一局游戏。"""
    state = create_river_state(
        [p0_hand, p1_hand],
        board,
        initial_stack,
        big_blind
    )
    
    env = PokerEnvironment(
        initial_stack=initial_stack,
        small_blind=big_blind // 2,
        big_blind=big_blind,
        max_raises_per_street=4
    )
    
    # 玩家1（大盲位）先行动
    legal_actions = env.get_legal_actions(state)
    
    # 找到对应的动作
    p1_action_obj = None
    for action in legal_actions:
        if action.action_type == p1_action:
            p1_action_obj = action
            break
    
    if p1_action_obj is None:
        return {'error': f'P1 动作 {p1_action} 不合法'}
    
    # 应用 P1 的动作
    new_stacks = state.player_stacks.copy()
    new_bets = state.current_bets.copy()
    new_pot = state.pot
    
    if p1_action_obj.action_type == ActionType.CHECK:
        pass
    elif p1_action_obj.action_type in (ActionType.RAISE_SMALL, ActionType.RAISE_BIG, ActionType.ALL_IN):
        new_stacks[1] -= p1_action_obj.amount
        new_bets[1] += p1_action_obj.amount
        new_pot += p1_action_obj.amount
    
    state = GameState(
        player_hands=state.player_hands,
        community_cards=state.community_cards,
        pot=new_pot,
        player_stacks=new_stacks,
        current_bets=new_bets,
        button_position=state.button_position,
        stage=state.stage,
        action_history=state.action_history + [p1_action_obj],
        current_player=0
    )
    
    # 玩家0 行动
    legal_actions = env.get_legal_actions(state)
    
    p0_action_obj = None
    for action in legal_actions:
        if action.action_type == p0_action:
            p0_action_obj = action
            break
    
    if p0_action_obj is None:
        # 如果指定动作不合法，选择第一个合法动作
        p0_action_obj = legal_actions[0]
    
    # 应用 P0 的动作
    new_stacks = state.player_stacks.copy()
    new_bets = state.current_bets.copy()
    new_pot = state.pot
    
    if p0_action_obj.action_type == ActionType.FOLD:
        pass
    elif p0_action_obj.action_type == ActionType.CHECK:
        pass
    elif p0_action_obj.action_type == ActionType.CALL:
        call_amount = new_bets[1] - new_bets[0]
        new_stacks[0] -= call_amount
        new_bets[0] = new_bets[1]
        new_pot += call_amount
    elif p0_action_obj.action_type in (ActionType.RAISE_SMALL, ActionType.RAISE_BIG, ActionType.ALL_IN):
        new_stacks[0] -= p0_action_obj.amount
        new_bets[0] += p0_action_obj.amount
        new_pot += p0_action_obj.amount
    
    final_state = GameState(
        player_hands=state.player_hands,
        community_cards=state.community_cards,
        pot=new_pot,
        player_stacks=new_stacks,
        current_bets=new_bets,
        button_position=state.button_position,
        stage=state.stage,
        action_history=state.action_history + [p0_action_obj],
        current_player=1
    )
    
    # 计算收益
    p0_utility = get_terminal_utility(final_state, 0, initial_stack)
    p1_utility = get_terminal_utility(final_state, 1, initial_stack)
    
    return {
        'p0_action': p0_action_obj.action_type.name,
        'p1_action': p1_action_obj.action_type.name,
        'p0_utility': p0_utility,
        'p1_utility': p1_utility,
        'pot': new_pot,
        'final_state': final_state
    }


def main():
    """主函数。"""
    print("训练场景诊断")
    print("=" * 70)
    
    # 配置
    initial_stack = 30
    big_blind = 10
    
    print(f"初始筹码: {initial_stack}")
    print(f"大盲注: {big_blind}")
    print(f"河牌开始时筹码: {initial_stack - big_blind}")
    print(f"河牌开始时底池: {2 * big_blind}")
    print(f"SPR: {(initial_stack - big_blind) / (2 * big_blind):.2f}")
    
    # 固定公共牌
    board = [parse_card(c) for c in ['Ah', 'As', 'Qd', '7c', '2h']]
    print(f"公共牌: {' '.join(card_to_str(c) for c in board)}")
    
    print_separator("场景分析：坚果牌 vs 空气牌")
    
    # 坚果牌：AA（四条A）
    nuts_hand = [parse_card('Ac'), parse_card('Ad')]
    # 空气牌：34o
    air_hand = [parse_card('3h'), parse_card('4s')]
    
    print(f"玩家0（坚果牌）: {card_to_str(nuts_hand[0])}{card_to_str(nuts_hand[1])}")
    print(f"玩家1（空气牌）: {card_to_str(air_hand[0])}{card_to_str(air_hand[1])}")
    
    # 模拟不同的动作组合
    scenarios = [
        (ActionType.CHECK, ActionType.CHECK, "双方 CHECK"),
        (ActionType.CHECK, ActionType.RAISE_BIG, "P1 CHECK, P0 RAISE_BIG"),
        (ActionType.RAISE_BIG, ActionType.FOLD, "P1 RAISE_BIG, P0 FOLD"),
        (ActionType.RAISE_BIG, ActionType.CALL, "P1 RAISE_BIG, P0 CALL"),
    ]
    
    print("\n坚果牌（P0）vs 空气牌（P1）的收益：")
    for p1_action, p0_action, desc in scenarios:
        result = simulate_game(
            nuts_hand, air_hand, board,
            p0_action, p1_action,
            initial_stack, big_blind
        )
        if 'error' in result:
            print(f"  {desc}: {result['error']}")
        else:
            print(f"  {desc}: P0={result['p0_utility']:+.0f}, P1={result['p1_utility']:+.0f}")
    
    print_separator("场景分析：空气牌 vs 坚果牌")
    
    print(f"玩家0（空气牌）: {card_to_str(air_hand[0])}{card_to_str(air_hand[1])}")
    print(f"玩家1（坚果牌）: {card_to_str(nuts_hand[0])}{card_to_str(nuts_hand[1])}")
    
    print("\n空气牌（P0）vs 坚果牌（P1）的收益：")
    for p1_action, p0_action, desc in scenarios:
        result = simulate_game(
            air_hand, nuts_hand, board,
            p0_action, p1_action,
            initial_stack, big_blind
        )
        if 'error' in result:
            print(f"  {desc}: {result['error']}")
        else:
            print(f"  {desc}: P0={result['p0_utility']:+.0f}, P1={result['p1_utility']:+.0f}")
    
    print_separator("GTO 策略分析")
    
    print("在这个场景下（SPR=1，公共牌 AA Q 7 2）：")
    print()
    print("对于坚果牌（四条A）：")
    print("  - CHECK 然后 CALL/RAISE：可以获得价值")
    print("  - 直接 RAISE：可能吓跑对手")
    print()
    print("对于空气牌：")
    print("  - CHECK：免费摊牌，输掉底池")
    print("  - BLUFF RAISE：可能赢得底池（如果对手弃牌）")
    print()
    print("问题：当前网络对所有手牌都输出 100% CHECK")
    print("这说明网络没有学会区分不同手牌的价值")


if __name__ == '__main__':
    main()
