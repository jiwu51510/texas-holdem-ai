#!/usr/bin/env python3
"""河牌训练诊断脚本 - 第二部分。

检查更复杂的场景和训练过程。
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from copy import deepcopy

from models.core import TrainingConfig, GameState, Action, ActionType, GameStage, Card
from models.networks import RegretNetwork, PolicyNetwork
from environment.poker_environment import PokerEnvironment
from environment.state_encoder import StateEncoder
from environment.hand_evaluator import compare_hands


def create_test_config() -> TrainingConfig:
    """创建测试配置。"""
    return TrainingConfig(
        learning_rate=0.001,
        batch_size=256,
        network_architecture=[512, 256, 128],
        cfr_iterations_per_update=500,
        network_train_steps=300,
        regret_buffer_size=2000000,
        strategy_buffer_size=2000000,
        initial_stack=30,
        small_blind=5,
        big_blind=10,
        max_raises_per_street=4,
    )


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
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_raise_scenario():
    """检查加注场景的收益计算。"""
    print_separator("检查加注场景")
    
    config = create_test_config()
    env = PokerEnvironment(
        initial_stack=config.initial_stack,
        small_blind=config.small_blind,
        big_blind=config.big_blind,
        max_raises_per_street=config.max_raises_per_street
    )
    
    # 固定公共牌
    board = [parse_card(c) for c in ['Ah', 'As', 'Qd', '7c', '2h']]
    
    # 玩家0有坚果（三条A），玩家1有较弱牌
    player0_hand = [parse_card('Ad'), parse_card('Kh')]  # 三条A
    player1_hand = [parse_card('Kd'), parse_card('Ks')]  # 两对 AA KK
    
    print(f"公共牌: {' '.join(card_to_str(c) for c in board)}")
    print(f"玩家0: {card_to_str(player0_hand[0])} {card_to_str(player0_hand[1])} (三条A)")
    print(f"玩家1: {card_to_str(player1_hand[0])} {card_to_str(player1_hand[1])} (两对AA KK)")
    
    # 创建初始状态
    state = GameState(
        player_hands=[(player0_hand[0], player0_hand[1]), (player1_hand[0], player1_hand[1])],
        community_cards=board,
        pot=20,
        player_stacks=[20, 20],
        current_bets=[0, 0],
        button_position=0,
        stage=GameStage.RIVER,
        action_history=[],
        current_player=1  # 大盲位先行动
    )
    
    print(f"\n初始状态:")
    print(f"  底池: {state.pot}, 筹码: {state.player_stacks}")
    
    # 场景1：玩家1过牌，玩家0加注，玩家1跟注
    print(f"\n场景1：P1 CHECK -> P0 RAISE_BIG(20) -> P1 CALL")
    
    # P1 CHECK
    state1 = apply_action(state, Action(ActionType.CHECK))
    print(f"  P1 CHECK: 底池={state1.pot}, 筹码={state1.player_stacks}, 下注={state1.current_bets}")
    
    # P0 RAISE_BIG
    state2 = apply_action(state1, Action(ActionType.RAISE_BIG, amount=20))
    print(f"  P0 RAISE_BIG(20): 底池={state2.pot}, 筹码={state2.player_stacks}, 下注={state2.current_bets}")
    
    # P1 CALL
    state3 = apply_action(state2, Action(ActionType.CALL))
    print(f"  P1 CALL: 底池={state3.pot}, 筹码={state3.player_stacks}, 下注={state3.current_bets}")
    
    # 计算收益
    utility_p0 = get_terminal_utility(state3, 0, config)
    utility_p1 = get_terminal_utility(state3, 1, config)
    print(f"\n  玩家0收益: {utility_p0}")
    print(f"  玩家1收益: {utility_p1}")
    print(f"  收益和: {utility_p0 + utility_p1}")
    
    # 场景2：玩家1过牌，玩家0加注，玩家1弃牌
    print(f"\n场景2：P1 CHECK -> P0 RAISE_BIG(20) -> P1 FOLD")
    
    # P1 FOLD
    state4 = apply_action(state2, Action(ActionType.FOLD))
    print(f"  P1 FOLD: 底池={state4.pot}, 筹码={state4.player_stacks}")
    
    utility_p0 = get_terminal_utility(state4, 0, config)
    utility_p1 = get_terminal_utility(state4, 1, config)
    print(f"\n  玩家0收益: {utility_p0}")
    print(f"  玩家1收益: {utility_p1}")
    print(f"  收益和: {utility_p0 + utility_p1}")


def apply_action(state: GameState, action: Action) -> GameState:
    """应用行动到状态。"""
    new_stacks = state.player_stacks.copy()
    new_bets = state.current_bets.copy()
    new_pot = state.pot
    current_player = state.current_player
    
    if action.action_type == ActionType.FOLD:
        pass
    elif action.action_type == ActionType.CHECK:
        pass
    elif action.action_type == ActionType.CALL:
        call_amount = new_bets[1 - current_player] - new_bets[current_player]
        new_stacks[current_player] -= call_amount
        new_bets[current_player] = new_bets[1 - current_player]
        new_pot += call_amount
    elif action.action_type in (ActionType.RAISE_SMALL, ActionType.RAISE_BIG, ActionType.ALL_IN):
        new_stacks[current_player] -= action.amount
        new_bets[current_player] += action.amount
        new_pot += action.amount
    
    return GameState(
        player_hands=state.player_hands,
        community_cards=state.community_cards,
        pot=new_pot,
        player_stacks=new_stacks,
        current_bets=new_bets,
        button_position=state.button_position,
        stage=state.stage,
        action_history=state.action_history + [action],
        current_player=1 - current_player
    )


def get_terminal_utility(state: GameState, player_id: int, config: TrainingConfig) -> float:
    """计算终止状态收益。"""
    winner = -1
    
    for action in state.action_history:
        if action.action_type == ActionType.FOLD:
            folder = 1 - state.current_player
            winner = 1 - folder
            break
    
    if winner == -1:
        hand0 = list(state.player_hands[0])
        hand1 = list(state.player_hands[1])
        winner = compare_hands(hand0, hand1, state.community_cards)
    
    current_stack = state.player_stacks[player_id]
    pot = state.pot
    
    if winner == -1:
        pot_share = pot / 2
    elif winner == player_id:
        pot_share = pot
    else:
        pot_share = 0
    
    final_stack = current_stack + pot_share
    return float(final_stack - config.initial_stack)


def check_gto_intuition():
    """检查 GTO 直觉。"""
    print_separator("GTO 直觉检查")
    
    print("在河牌阶段，理论上的 GTO 策略应该是：")
    print()
    print("1. 坚果牌（nuts）：应该价值下注/加注")
    print("   - 如果对手弃牌，赢得底池")
    print("   - 如果对手跟注，赢得更大底池")
    print()
    print("2. 空气牌（bluff）：应该有一定比例的诈唬")
    print("   - 使对手无法总是弃牌")
    print("   - 诈唬频率取决于底池赔率")
    print()
    print("3. 中等牌（medium）：应该过牌/跟注")
    print("   - 不值得价值下注（可能被更好的牌跟注）")
    print("   - 不值得诈唬（有摊牌价值）")
    print()
    print("4. 弱牌面对下注：根据底池赔率决定跟注/弃牌")
    print("   - 如果对手下注底池，需要33%胜率才能跟注")
    print()
    
    # 计算底池赔率
    print("底池赔率计算（初始底池20，筹码各20）：")
    print("  - 对手下注10（半底池）：需要跟注10赢30，赔率 10/30 = 33%")
    print("  - 对手下注20（全底池）：需要跟注20赢40，赔率 20/40 = 50%")


def check_training_convergence():
    """检查训练收敛性。"""
    print_separator("训练收敛性检查")
    
    print("Deep CFR 收敛的关键因素：")
    print()
    print("1. 遗憾值累积")
    print("   - 遗憾值应该随迭代累积")
    print("   - 正遗憾值的行动概率应该增加")
    print()
    print("2. 策略平均")
    print("   - 最终策略是所有迭代策略的加权平均")
    print("   - 需要足够多的迭代才能收敛")
    print()
    print("3. 网络拟合")
    print("   - 遗憾网络需要准确拟合累积遗憾值")
    print("   - 策略网络需要准确拟合平均策略")
    print()
    print("4. 采样效率")
    print("   - 需要足够多的样本覆盖状态空间")
    print("   - 河牌阶段状态空间相对较小")
    print()
    
    # 估算状态空间大小
    print("河牌阶段状态空间估算（固定公共牌）：")
    print("  - 剩余牌数：52 - 5 = 47")
    print("  - 私牌组合：C(47,2) × C(45,2) = 1081 × 990 ≈ 107万")
    print("  - 行动序列：每个节点最多3-4个行动，深度约4-8")
    print("  - 总状态数：约 10^7 - 10^8")


def check_loaded_checkpoint():
    """检查已训练的检查点。"""
    print_separator("检查已训练的检查点")
    
    import os
    checkpoint_path = "checkpoints/river_only_fixed/checkpoint_15000.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"检查点不存在: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"检查点信息:")
    print(f"  迭代次数: {checkpoint.get('iteration', checkpoint.get('episode_number', 'N/A'))}")
    print(f"  格式: {checkpoint.get('checkpoint_format', 'legacy')}")
    
    if 'stats' in checkpoint:
        stats = checkpoint['stats']
        print(f"  统计信息:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
    
    # 加载网络并测试
    config = create_test_config()
    encoder = StateEncoder()
    
    regret_net = RegretNetwork(
        input_dim=encoder.encoding_dim,
        hidden_dims=[512, 256, 128],
        action_dim=6
    )
    regret_net.load_state_dict(checkpoint['regret_network_state_dict'])
    regret_net.eval()
    
    # 测试几个不同的手牌
    board = [parse_card(c) for c in ['Ah', 'As', 'Qd', '7c', '2h']]
    
    test_hands = [
        ([parse_card('Ad'), parse_card('Kh')], "三条A（坚果）"),
        ([parse_card('Kd'), parse_card('Ks')], "两对AA KK"),
        ([parse_card('3d'), parse_card('4s')], "空气牌"),
    ]
    
    print(f"\n不同手牌的策略输出:")
    print(f"公共牌: {' '.join(card_to_str(c) for c in board)}")
    print()
    
    for hand, desc in test_hands:
        state = GameState(
            player_hands=[(hand[0], hand[1]), (parse_card('5d'), parse_card('6d'))],
            community_cards=board,
            pot=20,
            player_stacks=[20, 20],
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.RIVER,
            action_history=[],
            current_player=1
        )
        
        encoding = encoder.encode(state, player_id=1)
        state_tensor = torch.FloatTensor(encoding).unsqueeze(0)
        
        with torch.no_grad():
            regrets = regret_net(state_tensor).squeeze().numpy()
            strategy = regret_net.get_strategy(state_tensor).squeeze().numpy()
        
        print(f"{desc}: {card_to_str(hand[0])} {card_to_str(hand[1])}")
        print(f"  遗憾值: FOLD={regrets[0]:.2f}, CHECK={regrets[1]:.2f}, CALL={regrets[2]:.2f}, "
              f"R_S={regrets[3]:.2f}, R_B={regrets[4]:.2f}, ALL_IN={regrets[5]:.2f}")
        print(f"  策略: FOLD={strategy[0]:.2%}, CHECK={strategy[1]:.2%}, CALL={strategy[2]:.2%}, "
              f"R_S={strategy[3]:.2%}, R_B={strategy[4]:.2%}, ALL_IN={strategy[5]:.2%}")
        print()


def main():
    """主函数。"""
    print("河牌训练诊断 - 第二部分")
    print("="*60)
    
    # 检查加注场景
    check_raise_scenario()
    
    # 检查 GTO 直觉
    check_gto_intuition()
    
    # 检查训练收敛性
    check_training_convergence()
    
    # 检查已训练的检查点
    check_loaded_checkpoint()
    
    print_separator("诊断完成")


if __name__ == '__main__':
    main()
