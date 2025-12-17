#!/usr/bin/env python3
"""河牌训练诊断脚本。

逐步检查训练过程中的每个环节，帮助定位问题。
"""

import numpy as np
import torch
from typing import List, Dict, Tuple

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


def create_river_state(
    player_hands: List[List[Card]],
    community_cards: List[Card],
    config: TrainingConfig
) -> GameState:
    """创建河牌阶段的初始状态。"""
    hands_as_tuples = [(hand[0], hand[1]) for hand in player_hands]
    
    pot = 2 * config.big_blind  # 20
    stacks = [
        config.initial_stack - config.big_blind,  # 20
        config.initial_stack - config.big_blind   # 20
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


def print_separator(title: str):
    """打印分隔线。"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def step1_check_state_creation():
    """步骤1：检查状态创建。"""
    print_separator("步骤1：检查状态创建")
    
    config = create_test_config()
    
    # 固定公共牌
    board = [parse_card(c) for c in ['Ah', 'As', 'Qd', '7c', '2h']]
    print(f"公共牌: {' '.join(card_to_str(c) for c in board)}")
    
    # 测试私牌
    player0_hand = [parse_card('Kh'), parse_card('Kd')]  # KK
    player1_hand = [parse_card('Jh'), parse_card('Jd')]  # JJ
    
    print(f"玩家0私牌: {card_to_str(player0_hand[0])} {card_to_str(player0_hand[1])}")
    print(f"玩家1私牌: {card_to_str(player1_hand[0])} {card_to_str(player1_hand[1])}")
    
    state = create_river_state([player0_hand, player1_hand], board, config)
    
    print(f"\n状态信息:")
    print(f"  底池: {state.pot}")
    print(f"  筹码: {state.player_stacks}")
    print(f"  当前下注: {state.current_bets}")
    print(f"  当前玩家: {state.current_player}")
    print(f"  阶段: {state.stage}")
    
    return state, config


def step2_check_legal_actions(state: GameState, config: TrainingConfig):
    """步骤2：检查合法行动。"""
    print_separator("步骤2：检查合法行动")
    
    env = PokerEnvironment(
        initial_stack=config.initial_stack,
        small_blind=config.small_blind,
        big_blind=config.big_blind,
        max_raises_per_street=config.max_raises_per_street
    )
    
    legal_actions = env.get_legal_actions(state)
    
    print(f"当前玩家: {state.current_player}")
    print(f"合法行动数量: {len(legal_actions)}")
    print(f"合法行动:")
    for action in legal_actions:
        if action.amount:
            print(f"  - {action.action_type.value}: {action.amount}")
        else:
            print(f"  - {action.action_type.value}")
    
    return env, legal_actions


def step3_check_state_encoding(state: GameState):
    """步骤3：检查状态编码。"""
    print_separator("步骤3：检查状态编码")
    
    encoder = StateEncoder()
    
    encoding_p0 = encoder.encode(state, player_id=0)
    encoding_p1 = encoder.encode(state, player_id=1)
    
    print(f"编码维度: {len(encoding_p0)}")
    print(f"玩家0编码统计:")
    print(f"  最小值: {encoding_p0.min():.4f}")
    print(f"  最大值: {encoding_p0.max():.4f}")
    print(f"  均值: {encoding_p0.mean():.4f}")
    print(f"  非零元素: {np.count_nonzero(encoding_p0)}")
    
    print(f"\n玩家1编码统计:")
    print(f"  最小值: {encoding_p1.min():.4f}")
    print(f"  最大值: {encoding_p1.max():.4f}")
    print(f"  均值: {encoding_p1.mean():.4f}")
    print(f"  非零元素: {np.count_nonzero(encoding_p1)}")
    
    # 检查编码是否不同（不同私牌应该有不同编码）
    diff = np.abs(encoding_p0 - encoding_p1).sum()
    print(f"\n两个玩家编码差异: {diff:.4f}")
    
    return encoder


def step4_check_network_output(state: GameState, encoder: StateEncoder):
    """步骤4：检查网络输出。"""
    print_separator("步骤4：检查网络输出")
    
    input_dim = encoder.encoding_dim
    action_dim = 6
    
    # 创建网络
    regret_net = RegretNetwork(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        action_dim=action_dim
    )
    
    # 编码状态
    encoding = encoder.encode(state, player_id=state.current_player)
    state_tensor = torch.FloatTensor(encoding).unsqueeze(0)
    
    # 获取遗憾值
    with torch.no_grad():
        regrets = regret_net(state_tensor)
        strategy = regret_net.get_strategy(state_tensor)
    
    print(f"遗憾网络输出（初始随机权重）:")
    print(f"  遗憾值: {regrets.squeeze().numpy()}")
    print(f"  策略: {strategy.squeeze().numpy()}")
    print(f"  策略和: {strategy.sum().item():.6f}")
    
    return regret_net


def step5_check_terminal_utility(state: GameState, config: TrainingConfig):
    """步骤5：检查终止状态收益计算。"""
    print_separator("步骤5：检查终止状态收益计算")
    
    # 模拟几种终止情况
    
    # 情况1：玩家1弃牌
    state1 = GameState(
        player_hands=state.player_hands,
        community_cards=state.community_cards,
        pot=state.pot,
        player_stacks=state.player_stacks.copy(),
        current_bets=[0, 0],
        button_position=0,
        stage=GameStage.RIVER,
        action_history=[Action(ActionType.FOLD)],
        current_player=0
    )
    
    print("情况1：玩家1弃牌")
    utility_p0 = get_terminal_utility(state1, 0, config)
    utility_p1 = get_terminal_utility(state1, 1, config)
    print(f"  玩家0收益: {utility_p0}")
    print(f"  玩家1收益: {utility_p1}")
    print(f"  收益和: {utility_p0 + utility_p1}")
    
    # 情况2：双方过牌摊牌
    state2 = GameState(
        player_hands=state.player_hands,
        community_cards=state.community_cards,
        pot=state.pot,
        player_stacks=state.player_stacks.copy(),
        current_bets=[0, 0],
        button_position=0,
        stage=GameStage.RIVER,
        action_history=[Action(ActionType.CHECK), Action(ActionType.CHECK)],
        current_player=1
    )
    
    print("\n情况2：双方过牌摊牌")
    print(f"  玩家0手牌: {card_to_str(state.player_hands[0][0])} {card_to_str(state.player_hands[0][1])}")
    print(f"  玩家1手牌: {card_to_str(state.player_hands[1][0])} {card_to_str(state.player_hands[1][1])}")
    
    # 比较手牌
    hand0 = list(state.player_hands[0])
    hand1 = list(state.player_hands[1])
    winner = compare_hands(hand0, hand1, state.community_cards)
    print(f"  赢家: {'玩家0' if winner == 0 else '玩家1' if winner == 1 else '平局'}")
    
    utility_p0 = get_terminal_utility(state2, 0, config)
    utility_p1 = get_terminal_utility(state2, 1, config)
    print(f"  玩家0收益: {utility_p0}")
    print(f"  玩家1收益: {utility_p1}")
    print(f"  收益和: {utility_p0 + utility_p1}")


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


def step6_check_regret_calculation():
    """步骤6：检查遗憾值计算逻辑。"""
    print_separator("步骤6：检查遗憾值计算逻辑")
    
    # 模拟一个简单的场景
    # 假设有3个行动：CHECK, RAISE_SMALL, RAISE_BIG
    # 各行动的期望收益
    action_values = {
        'CHECK': 5.0,
        'RAISE_SMALL': 8.0,
        'RAISE_BIG': 3.0,
    }
    
    # 当前策略（均匀）
    strategy = {'CHECK': 1/3, 'RAISE_SMALL': 1/3, 'RAISE_BIG': 1/3}
    
    # 期望收益
    expected_value = sum(strategy[a] * action_values[a] for a in action_values)
    print(f"各行动收益: {action_values}")
    print(f"当前策略: {strategy}")
    print(f"期望收益: {expected_value:.4f}")
    
    # 遗憾值 = 行动收益 - 期望收益
    regrets = {a: action_values[a] - expected_value for a in action_values}
    print(f"遗憾值: {regrets}")
    
    # 验证：遗憾值加权和应为0
    weighted_sum = sum(strategy[a] * regrets[a] for a in regrets)
    print(f"遗憾值加权和（应为0）: {weighted_sum:.6f}")
    
    # Regret Matching：正遗憾值归一化得到新策略
    positive_regrets = {a: max(0, r) for a, r in regrets.items()}
    total_positive = sum(positive_regrets.values())
    
    if total_positive > 0:
        new_strategy = {a: positive_regrets[a] / total_positive for a in positive_regrets}
    else:
        new_strategy = {a: 1/len(positive_regrets) for a in positive_regrets}
    
    print(f"正遗憾值: {positive_regrets}")
    print(f"新策略（Regret Matching）: {new_strategy}")


def step7_check_buffer_sampling():
    """步骤7：检查缓冲区采样。"""
    print_separator("步骤7：检查缓冲区采样")
    
    from training.reservoir_buffer import ReservoirBuffer
    
    buffer = ReservoirBuffer(capacity=10000)
    
    # 添加一些样本
    for i in range(1000):
        state = np.random.randn(370).astype(np.float32)
        target = np.random.randn(6).astype(np.float32)
        buffer.add(state, target, iteration=i)
    
    print(f"缓冲区大小: {len(buffer)}")
    
    # 采样
    states, targets, iterations = buffer.sample(batch_size=32)
    
    print(f"采样批次大小: {len(states)}")
    print(f"状态形状: {states.shape}")
    print(f"目标形状: {targets.shape}")
    print(f"迭代范围: {iterations.min()} - {iterations.max()}")


def main():
    """主函数。"""
    print("河牌训练诊断")
    print("="*60)
    
    # 步骤1：检查状态创建
    state, config = step1_check_state_creation()
    
    # 步骤2：检查合法行动
    env, legal_actions = step2_check_legal_actions(state, config)
    
    # 步骤3：检查状态编码
    encoder = step3_check_state_encoding(state)
    
    # 步骤4：检查网络输出
    regret_net = step4_check_network_output(state, encoder)
    
    # 步骤5：检查终止状态收益计算
    step5_check_terminal_utility(state, config)
    
    # 步骤6：检查遗憾值计算逻辑
    step6_check_regret_calculation()
    
    # 步骤7：检查缓冲区采样
    step7_check_buffer_sampling()
    
    print_separator("诊断完成")
    print("请检查以上输出，确认每个步骤是否正确。")


if __name__ == '__main__':
    main()
