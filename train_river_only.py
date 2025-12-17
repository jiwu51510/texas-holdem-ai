#!/usr/bin/env python3
"""河牌专项训练脚本。

这个脚本用于验证 Deep CFR 在河牌阶段是否能收敛到 GTO 策略。
通过固定翻牌前到转牌的所有动作，只在河牌阶段进行训练。

固定的行动序列：
- Preflop: 小盲位 CALL（跟注大盲）
- Preflop: 大盲位 CHECK
- Flop: 大盲位 CHECK，小盲位 CHECK
- Turn: 大盲位 CHECK，小盲位 CHECK
- River: 开始训练（从大盲位先行动）

支持两种公共牌模式：
1. 随机模式（默认）：每次迭代随机采样公共牌
2. 固定模式：使用指定的公共牌进行训练

这样可以：
1. 大幅减少游戏树的复杂度
2. 专注于河牌阶段的策略学习
3. 快速验证算法是否能收敛
"""

import argparse
import json
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.core import (
    TrainingConfig, GameState, Action, ActionType, GameStage, Card
)
from models.networks import RegretNetwork, PolicyNetwork
from environment.poker_environment import PokerEnvironment
from environment.state_encoder import StateEncoder
from environment.hand_evaluator import compare_hands
from training.reservoir_buffer import ReservoirBuffer
from training.cfr_sampler import CFRSampler
from training.regret_processor import RegretProcessor, RegretProcessorConfig
from training.network_trainer import NetworkTrainer, NetworkTrainerConfig
from training.convergence_monitor import ConvergenceMonitor, ConvergenceMonitorConfig


def parse_card(card_str: str) -> Card:
    """解析牌的字符串表示。
    
    支持格式：
    - "Ah" = A♥
    - "Ks" = K♠
    - "Qd" = Q♦
    - "Jc" = J♣
    - "Th" = 10♥
    - "9s" = 9♠
    
    Args:
        card_str: 牌的字符串表示（如 "Ah", "Ks", "9d"）
        
    Returns:
        Card 对象
    """
    card_str = card_str.strip()
    if len(card_str) < 2:
        raise ValueError(f"无效的牌格式: {card_str}")
    
    rank_str = card_str[:-1].upper()
    suit_str = card_str[-1].lower()
    
    # 解析点数
    rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
    if rank_str in rank_map:
        rank = rank_map[rank_str]
    elif rank_str.isdigit():
        rank = int(rank_str)
        if rank < 2 or rank > 10:
            raise ValueError(f"无效的点数: {rank_str}")
    else:
        raise ValueError(f"无效的点数: {rank_str}")
    
    # 解析花色
    if suit_str not in ['h', 'd', 'c', 's']:
        raise ValueError(f"无效的花色: {suit_str}，应为 h/d/c/s")
    
    return Card(rank, suit_str)


def parse_board(board_str: str) -> List[Card]:
    """解析公共牌字符串。
    
    支持格式：
    - "AhKsQdJcTh" = 5张牌连写
    - "Ah Ks Qd Jc Th" = 空格分隔
    - "Ah,Ks,Qd,Jc,Th" = 逗号分隔
    
    Args:
        board_str: 公共牌字符串
        
    Returns:
        5张公共牌的列表
    """
    board_str = board_str.strip()
    
    # 尝试用空格或逗号分隔
    if ' ' in board_str or ',' in board_str:
        parts = re.split(r'[,\s]+', board_str)
        cards = [parse_card(p) for p in parts if p]
    else:
        # 连写格式，每2-3个字符一张牌
        cards = []
        i = 0
        while i < len(board_str):
            # 检查是否是 10（T）或数字
            if i + 2 <= len(board_str):
                # 尝试解析2字符的牌（如 Ah, 9s）
                try:
                    card = parse_card(board_str[i:i+2])
                    cards.append(card)
                    i += 2
                    continue
                except ValueError:
                    pass
            raise ValueError(f"无法解析位置 {i} 处的牌: {board_str[i:]}")
    
    if len(cards) != 5:
        raise ValueError(f"公共牌必须是5张，得到 {len(cards)} 张")
    
    return cards


def card_to_str(card: Card) -> str:
    """将牌转换为字符串表示。"""
    rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
    rank_str = rank_map.get(card.rank, str(card.rank))
    return f"{rank_str}{card.suit}"


def board_to_str(cards: List[Card]) -> str:
    """将公共牌列表转换为字符串。"""
    return ' '.join(card_to_str(c) for c in cards)


class RiverOnlyTrainer:
    """河牌专项训练器。
    
    只在河牌阶段进行 CFR 训练，翻牌前到转牌的动作都是固定的。
    
    支持两种公共牌模式：
    - 随机模式：每次迭代随机采样公共牌
    - 固定模式：使用指定的公共牌进行训练
    """
    
    # 行动类型到索引的映射
    ACTION_TYPE_TO_IDX = {
        ActionType.FOLD: 0,
        ActionType.CHECK: 1,
        ActionType.CALL: 2,
        ActionType.RAISE_SMALL: 3,
        ActionType.RAISE_BIG: 4,
        ActionType.ALL_IN: 5,
    }
    
    IDX_TO_ACTION_TYPE = {
        0: ActionType.FOLD,
        1: ActionType.CHECK,
        2: ActionType.CALL,
        3: ActionType.RAISE_SMALL,
        4: ActionType.RAISE_BIG,
        5: ActionType.ALL_IN,
    }
    
    def __init__(self, config: TrainingConfig, fixed_board: Optional[List[Card]] = None):
        """初始化训练器。
        
        Args:
            config: 训练配置
            fixed_board: 固定的公共牌（5张），如果为 None 则使用随机采样
        """
        self.config = config
        self.fixed_board = fixed_board
        
        # 初始化状态编码器
        self.state_encoder = StateEncoder()
        input_dim = self.state_encoder.encoding_dim
        action_dim = 6  # FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN
        
        # 初始化神经网络
        self.regret_network = RegretNetwork(
            input_dim=input_dim,
            hidden_dims=config.network_architecture,
            action_dim=action_dim
        )
        
        self.policy_network = PolicyNetwork(
            input_dim=input_dim,
            hidden_dims=config.network_architecture,
            action_dim=action_dim
        )
        
        # 初始化优化器
        self.regret_optimizer = optim.Adam(
            self.regret_network.parameters(),
            lr=config.learning_rate
        )
        
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate
        )
        
        # 初始化缓冲区
        self.regret_buffer = ReservoirBuffer(config.regret_buffer_size)
        self.strategy_buffer = ReservoirBuffer(config.strategy_buffer_size)
        
        # 初始化环境
        self.env = PokerEnvironment(
            initial_stack=config.initial_stack,
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            max_raises_per_street=config.max_raises_per_street
        )
        
        # 初始化采样器
        self.cfr_sampler = CFRSampler(num_flop_buckets=30)
        
        # 初始化遗憾值处理器（需求: 1.1, 1.2, 1.3）
        regret_processor_config = RegretProcessorConfig(
            use_positive_truncation=True,  # CFR+ 正遗憾值截断
            decay_factor=0.99,  # 遗憾值衰减因子
            clip_threshold=100.0  # 遗憾值裁剪阈值
        )
        self.regret_processor = RegretProcessor(regret_processor_config)
        
        # 初始化网络训练器（需求: 1.4, 3.1, 3.3）
        network_trainer_config = NetworkTrainerConfig(
            use_huber_loss=True,  # 使用Huber损失减少异常值影响
            huber_delta=1.0,
            use_ema=True,  # 使用EMA更新目标网络
            ema_decay=0.995,
            gradient_clip_norm=1.0  # 梯度裁剪防止梯度爆炸
        )
        self.network_trainer = NetworkTrainer(network_trainer_config)
        
        # 初始化EMA目标网络（用于稳定训练）
        self.regret_network_ema = RegretNetwork(
            input_dim=input_dim,
            hidden_dims=config.network_architecture,
            action_dim=action_dim
        )
        self.policy_network_ema = PolicyNetwork(
            input_dim=input_dim,
            hidden_dims=config.network_architecture,
            action_dim=action_dim
        )
        # 初始化EMA网络参数为主网络参数
        self.regret_network_ema.load_state_dict(self.regret_network.state_dict())
        self.policy_network_ema.load_state_dict(self.policy_network.state_dict())
        
        # 初始化收敛监控器（需求: 4.1, 4.2, 4.3, 4.4）
        convergence_monitor_config = ConvergenceMonitorConfig(
            entropy_window=100,  # 熵值监控窗口大小
            oscillation_threshold=0.1,  # 震荡检测阈值
            kl_warning_threshold=0.5,  # KL散度警告阈值
            monitor_interval=500  # 监控间隔（迭代次数）
        )
        self.convergence_monitor = ConvergenceMonitor(convergence_monitor_config)
        
        # 迭代计数
        self.iteration = 0
        
        # 统计信息
        self.stats = {
            'total_samples': 0,
            'regret_updates': 0,
            'strategy_updates': 0,
        }
    
    def _create_river_state(
        self,
        player_hands: List[List[Card]],
        community_cards: List[Card]
    ) -> GameState:
        """创建河牌阶段的初始状态。
        
        模拟固定的行动序列后的状态：
        - Preflop: SB call, BB check -> 底池 = 2 * BB
        - Flop: BB check, SB check
        - Turn: BB check, SB check
        - River: 从 BB（玩家1）开始行动
        
        Args:
            player_hands: 两个玩家的私牌
            community_cards: 5张公共牌
            
        Returns:
            河牌阶段的游戏状态
        """
        # 将私牌列表转换为元组
        hands_as_tuples = [
            (hand[0], hand[1]) for hand in player_hands
        ]
        
        # 固定行动后的状态：
        # - 底池 = 2 * big_blind（双方各投入 big_blind）
        # - 筹码 = initial_stack - big_blind
        # - 当前下注 = [0, 0]（新的一轮下注）
        # - 当前玩家 = 1（大盲位先行动）
        pot = 2 * self.config.big_blind
        stacks = [
            self.config.initial_stack - self.config.big_blind,
            self.config.initial_stack - self.config.big_blind
        ]
        
        return GameState(
            player_hands=hands_as_tuples,
            community_cards=community_cards,
            pot=pot,
            player_stacks=stacks,
            current_bets=[0, 0],
            button_position=0,
            stage=GameStage.RIVER,
            action_history=[],  # 河牌阶段的行动历史从空开始
            current_player=1  # 大盲位先行动
        )
    
    def _sample_river_state(self) -> GameState:
        """采样一个河牌阶段的状态。
        
        如果设置了固定公共牌，则使用固定公共牌；否则随机采样。
        
        Returns:
            河牌阶段的游戏状态
        """
        if self.fixed_board is not None:
            # 使用固定公共牌
            return self._sample_river_state_with_fixed_board()
        else:
            # 随机采样公共牌
            return self._sample_river_state_random()
    
    def _sample_river_state_random(self) -> GameState:
        """随机采样河牌阶段的状态。"""
        # 采样私牌
        player_hands = self.cfr_sampler.sample_preflop()
        
        # 收集已使用的牌
        used_cards = set()
        for hand in player_hands:
            for card in hand:
                used_cards.add((card.rank, card.suit))
        
        # 采样翻牌
        flop = self.cfr_sampler.sample_flop(used_cards)
        for card in flop:
            used_cards.add((card.rank, card.suit))
        
        # 采样转牌
        turn = self.cfr_sampler.sample_turn(used_cards)
        used_cards.add((turn.rank, turn.suit))
        
        # 采样河牌
        river = self.cfr_sampler.sample_turn(used_cards)
        
        # 组合公共牌
        community_cards = flop + [turn, river]
        
        return self._create_river_state(player_hands, community_cards)
    
    def _sample_river_state_with_fixed_board(self) -> GameState:
        """使用固定公共牌采样河牌阶段的状态。
        
        公共牌固定，只随机采样私牌。
        """
        # 收集公共牌已使用的牌
        used_cards = set()
        for card in self.fixed_board:
            used_cards.add((card.rank, card.suit))
        
        # 从剩余牌中采样私牌
        all_cards = []
        for suit in ['h', 'd', 'c', 's']:
            for rank in range(2, 15):
                if (rank, suit) not in used_cards:
                    all_cards.append(Card(rank, suit))
        
        # 随机选择4张牌作为私牌
        random.shuffle(all_cards)
        player_hands = [
            [all_cards[0], all_cards[1]],
            [all_cards[2], all_cards[3]]
        ]
        
        return self._create_river_state(player_hands, self.fixed_board.copy())
    
    def _is_terminal(self, state: GameState) -> bool:
        """检查是否为终止状态。"""
        # 检查是否有弃牌
        for action in state.action_history:
            if action.action_type == ActionType.FOLD:
                return True
        
        # 检查是否摊牌（河牌阶段，下注相等，双方都已行动）
        if state.current_bets[0] == state.current_bets[1]:
            if len(state.action_history) >= 2:
                last_action = state.action_history[-1]
                if last_action.action_type in [ActionType.CALL, ActionType.CHECK]:
                    return True
        
        return False
    
    def _get_terminal_utility(self, state: GameState, player_id: int) -> float:
        """获取终止状态的收益。
        
        收益计算：
        - 初始状态：每人投入 big_blind，底池 = 2 * big_blind
        - 收益 = 最终筹码 - 初始筹码 = (当前筹码 + 底池份额) - 初始筹码
        """
        # 确定赢家
        winner = -1
        
        # 检查是否有人弃牌
        for action in state.action_history:
            if action.action_type == ActionType.FOLD:
                folder = 1 - state.current_player
                winner = 1 - folder
                break
        
        # 如果没有人弃牌，进行摊牌比较
        if winner == -1:
            hand1 = list(state.player_hands[0])
            hand2 = list(state.player_hands[1])
            community = state.community_cards
            winner = compare_hands(hand1, hand2, community)
        
        # 计算收益
        # 初始筹码 = initial_stack
        # 河牌开始时筹码 = initial_stack - big_blind
        # 当前筹码 = state.player_stacks[player_id]
        current_stack = state.player_stacks[player_id]
        
        pot = state.pot
        if winner == -1:
            pot_share = pot / 2
        elif winner == player_id:
            pot_share = pot
        else:
            pot_share = 0
        
        # 最终筹码 = 当前筹码 + 底池份额
        final_stack = current_stack + pot_share
        
        # 收益 = 最终筹码 - 初始筹码
        return float(final_stack - self.config.initial_stack)
    
    def _get_strategy(self, state: GameState, player_id: int) -> np.ndarray:
        """获取当前状态的策略。"""
        state_encoding = self.state_encoder.encode(state, player_id)
        state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
        
        with torch.no_grad():
            strategy = self.regret_network.get_strategy(state_tensor)
            return strategy.squeeze(0).numpy()
    
    def _filter_strategy_to_legal(
        self, 
        strategy: np.ndarray, 
        legal_actions: List[Action]
    ) -> np.ndarray:
        """将策略过滤到合法行动。"""
        filtered = np.zeros_like(strategy)
        
        for action in legal_actions:
            idx = self.ACTION_TYPE_TO_IDX.get(action.action_type, 0)
            filtered[idx] = strategy[idx]
        
        total = filtered.sum()
        if total > 0:
            filtered = filtered / total
        else:
            for action in legal_actions:
                idx = self.ACTION_TYPE_TO_IDX.get(action.action_type, 0)
                filtered[idx] = 1.0 / len(legal_actions)
        
        return filtered
    
    def _sample_action(
        self, 
        legal_actions: List[Action], 
        strategy: np.ndarray
    ) -> Action:
        """根据策略采样一个行动。"""
        probs = []
        for action in legal_actions:
            idx = self.ACTION_TYPE_TO_IDX.get(action.action_type, 0)
            probs.append(strategy[idx])
        
        probs = np.array(probs)
        total = probs.sum()
        
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(len(legal_actions)) / len(legal_actions)
        
        choice = np.random.choice(len(legal_actions), p=probs)
        return legal_actions[choice]
    
    def _apply_action(self, state: GameState, action: Action) -> GameState:
        """应用行动到状态。"""
        new_stacks = state.player_stacks.copy()
        new_bets = state.current_bets.copy()
        new_pot = state.pot
        current_player = state.current_player
        next_player = 1 - current_player
        
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
            current_player=next_player
        )
    
    def traverse_river(
        self,
        state: GameState,
        player_id: int,
        reach_probs: Tuple[float, float]
    ) -> float:
        """遍历河牌阶段的游戏树。"""
        if self._is_terminal(state):
            return self._get_terminal_utility(state, player_id)
        
        current_player = state.current_player
        legal_actions = self.env.get_legal_actions(state)
        
        if not legal_actions:
            return 0.0
        
        strategy = self._get_strategy(state, current_player)
        legal_strategy = self._filter_strategy_to_legal(strategy, legal_actions)
        
        if current_player == player_id:
            # 遍历所有行动
            action_values = {}
            
            for action in legal_actions:
                action_idx = self.ACTION_TYPE_TO_IDX.get(action.action_type, 0)
                next_state = self._apply_action(state, action)
                
                new_reach_probs = list(reach_probs)
                new_reach_probs[current_player] *= legal_strategy[action_idx]
                
                action_value = self.traverse_river(
                    next_state, player_id, tuple(new_reach_probs)
                )
                action_values[action_idx] = action_value
            
            # 计算期望收益
            expected_value = sum(
                legal_strategy[idx] * value 
                for idx, value in action_values.items()
            )
            
            # 计算即时遗憾值（只为合法行动计算，非法行动保持为0）
            instant_regrets = np.zeros(6, dtype=np.float32)  # 6 个行动类型
            for action_idx, action_value in action_values.items():
                instant_regrets[action_idx] = action_value - expected_value
            
            # 存储样本（使用累积遗憾值）
            state_encoding = self.state_encoder.encode(state, player_id)
            opponent_reach = reach_probs[1 - player_id]
            if opponent_reach > 0:
                # 获取网络预测的累积遗憾值
                state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
                with torch.no_grad():
                    predicted_regrets = self.regret_network(state_tensor).squeeze(0).numpy()
                
                # 累积遗憾值 = 预测值 + 即时遗憾值
                accumulated_regrets = predicted_regrets + instant_regrets
                
                # 使用遗憾值处理器处理遗憾值（需求: 1.1, 1.2, 1.3）
                # 1. CFR+ 正遗憾值截断：将负遗憾值截断为0
                accumulated_regrets = self.regret_processor.truncate_positive(accumulated_regrets)
                # 2. 遗憾值裁剪：防止数值爆炸
                accumulated_regrets = self.regret_processor.clip_regrets(accumulated_regrets)
                
                # 使用对手到达概率加权
                weighted_regrets = accumulated_regrets * opponent_reach
                self.regret_buffer.add(state_encoding, weighted_regrets, self.iteration)
                self.stats['regret_updates'] += 1
            
            player_reach = reach_probs[player_id]
            if player_reach > 0:
                self.strategy_buffer.add(state_encoding, legal_strategy, self.iteration)
                self.stats['strategy_updates'] += 1
                
                # 更新收敛监控器（需求: 4.1, 4.2, 4.3, 4.4）
                self.convergence_monitor.update(
                    iteration=self.iteration,
                    strategy=legal_strategy,
                    regrets=instant_regrets
                )
            
            return expected_value
        else:
            # 对手：采样一个行动
            action = self._sample_action(legal_actions, legal_strategy)
            action_idx = self.ACTION_TYPE_TO_IDX.get(action.action_type, 0)
            
            next_state = self._apply_action(state, action)
            
            new_reach_probs = list(reach_probs)
            new_reach_probs[current_player] *= legal_strategy[action_idx]
            
            return self.traverse_river(
                next_state, player_id, tuple(new_reach_probs)
            )
    
    def run_cfr_iteration(self, verbose: bool = False) -> Dict[str, float]:
        """执行一次 CFR 迭代。"""
        self.iteration += 1
        
        # 采样河牌状态
        state = self._sample_river_state()
        
        # 为两个玩家分别遍历
        utility_p0 = self.traverse_river(state, player_id=0, reach_probs=(1.0, 1.0))
        
        # 重新采样（或使用相同状态）
        state = self._sample_river_state()
        utility_p1 = self.traverse_river(state, player_id=1, reach_probs=(1.0, 1.0))
        
        self.stats['total_samples'] += 2
        
        if verbose and self.iteration % 100 == 0:
            print(f"[CFR #{self.iteration}] P0收益: {utility_p0:.2f}, P1收益: {utility_p1:.2f}")
        
        return {
            'iteration': self.iteration,
            'utility_p0': utility_p0,
            'utility_p1': utility_p1,
            'regret_buffer_size': len(self.regret_buffer),
            'strategy_buffer_size': len(self.strategy_buffer),
        }
    
    def train_networks(self, verbose: bool = True) -> Dict[str, float]:
        """训练神经网络。
        
        使用改进的训练方法（需求: 1.4, 3.1, 3.3）：
        - Huber损失替代MSE损失，减少异常值影响
        - 梯度裁剪防止梯度爆炸
        - EMA更新目标网络，平滑策略更新
        """
        regret_loss_total = 0.0
        policy_loss_total = 0.0
        regret_steps = 0
        policy_steps = 0
        regret_grad_norm_total = 0.0
        policy_grad_norm_total = 0.0
        
        total_steps = self.config.network_train_steps
        
        # 训练遗憾网络
        if len(self.regret_buffer) >= self.config.batch_size:
            if verbose:
                print(f"训练遗憾网络 ({total_steps} 步)...", end="", flush=True)
            
            for step in range(total_steps):
                states, targets, _ = self.regret_buffer.sample(self.config.batch_size)
                
                if len(states) == 0:
                    break
                
                states_tensor = torch.FloatTensor(states)
                targets_tensor = torch.FloatTensor(targets)
                
                self.regret_optimizer.zero_grad()
                predictions = self.regret_network(states_tensor)
                
                # 使用Huber损失替代MSE损失（需求: 1.4）
                if self.network_trainer.config.use_huber_loss:
                    loss = self.network_trainer.compute_huber_loss(predictions, targets_tensor)
                else:
                    loss = nn.MSELoss()(predictions, targets_tensor)
                
                loss.backward()
                
                # 梯度裁剪（需求: 3.3）
                grad_norm = self.network_trainer.clip_gradients(self.regret_network)
                regret_grad_norm_total += grad_norm
                
                self.regret_optimizer.step()
                
                regret_loss_total += loss.item()
                regret_steps += 1
            
            # EMA更新目标网络（需求: 3.1）
            if self.network_trainer.config.use_ema:
                self.network_trainer.update_ema(self.regret_network_ema, self.regret_network)
            
            if verbose:
                avg_loss = regret_loss_total / max(regret_steps, 1)
                avg_grad_norm = regret_grad_norm_total / max(regret_steps, 1)
                print(f" 完成 (损失: {avg_loss:.6f}, 梯度范数: {avg_grad_norm:.4f})")
        
        # 训练策略网络
        if len(self.strategy_buffer) >= self.config.batch_size:
            if verbose:
                print(f"训练策略网络 ({total_steps} 步)...", end="", flush=True)
            
            for step in range(total_steps):
                states, targets, _ = self.strategy_buffer.sample(self.config.batch_size)
                
                if len(states) == 0:
                    break
                
                states_tensor = torch.FloatTensor(states)
                targets_tensor = torch.FloatTensor(targets)
                
                self.policy_optimizer.zero_grad()
                logits = self.policy_network(states_tensor)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = -torch.mean(torch.sum(targets_tensor * log_probs, dim=-1))
                loss.backward()
                
                # 梯度裁剪（需求: 3.3）
                grad_norm = self.network_trainer.clip_gradients(self.policy_network)
                policy_grad_norm_total += grad_norm
                
                self.policy_optimizer.step()
                
                policy_loss_total += loss.item()
                policy_steps += 1
            
            # EMA更新目标网络（需求: 3.1）
            if self.network_trainer.config.use_ema:
                self.network_trainer.update_ema(self.policy_network_ema, self.policy_network)
            
            if verbose:
                avg_loss = policy_loss_total / max(policy_steps, 1)
                avg_grad_norm = policy_grad_norm_total / max(policy_steps, 1)
                print(f" 完成 (损失: {avg_loss:.6f}, 梯度范数: {avg_grad_norm:.4f})")
        
        # 获取收敛报告（需求: 4.1, 4.2, 4.3, 4.4）
        convergence_report = self.convergence_monitor.get_convergence_report()
        
        return {
            'regret_loss': regret_loss_total / max(regret_steps, 1),
            'policy_loss': policy_loss_total / max(policy_steps, 1),
            'regret_grad_norm': regret_grad_norm_total / max(regret_steps, 1),
            'policy_grad_norm': policy_grad_norm_total / max(policy_steps, 1),
            'convergence_report': convergence_report,
        }
    
    def evaluate_strategy(self, num_hands: int = 1000) -> Dict[str, float]:
        """评估当前策略。
        
        通过模拟对局来评估策略的收敛程度。
        """
        p0_wins = 0
        p1_wins = 0
        ties = 0
        total_utility_p0 = 0.0
        
        for _ in range(num_hands):
            state = self._sample_river_state()
            
            # 模拟对局
            while not self._is_terminal(state):
                current_player = state.current_player
                legal_actions = self.env.get_legal_actions(state)
                
                if not legal_actions:
                    break
                
                # 使用策略网络选择行动
                strategy = self._get_strategy(state, current_player)
                legal_strategy = self._filter_strategy_to_legal(strategy, legal_actions)
                action = self._sample_action(legal_actions, legal_strategy)
                
                state = self._apply_action(state, action)
            
            # 计算收益
            utility_p0 = self._get_terminal_utility(state, 0)
            total_utility_p0 += utility_p0
            
            if utility_p0 > 0:
                p0_wins += 1
            elif utility_p0 < 0:
                p1_wins += 1
            else:
                ties += 1
        
        return {
            'p0_win_rate': p0_wins / num_hands,
            'p1_win_rate': p1_wins / num_hands,
            'tie_rate': ties / num_hands,
            'avg_utility_p0': total_utility_p0 / num_hands,
        }
    
    def get_board_mode(self) -> str:
        """获取当前公共牌模式的描述。"""
        if self.fixed_board is not None:
            return f"固定公共牌: {board_to_str(self.fixed_board)}"
        else:
            return "随机公共牌"
    
    def save_checkpoint(self, path: str) -> None:
        """保存检查点。
        
        使用与 viewer 兼容的 Deep CFR 格式。
        包含EMA目标网络状态。
        """
        torch.save({
            # Deep CFR 格式标识
            'checkpoint_format': 'deep_cfr_v1',
            # 网络状态（使用 viewer 期望的键名）
            'regret_network_state_dict': self.regret_network.state_dict(),
            'policy_network_state_dict': self.policy_network.state_dict(),
            # EMA目标网络状态
            'regret_network_ema_state_dict': self.regret_network_ema.state_dict(),
            'policy_network_ema_state_dict': self.policy_network_ema.state_dict(),
            # 优化器状态
            'regret_optimizer_state_dict': self.regret_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            # 训练元数据
            'episode_number': self.iteration,
            'iteration': self.iteration,
            'stats': self.stats,
            # 固定公共牌信息
            'fixed_board': [card_to_str(c) for c in self.fixed_board] if self.fixed_board else None,
        }, path)
        print(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点。
        
        支持新旧两种格式，包含EMA目标网络状态。
        """
        checkpoint = torch.load(path)
        
        # 检测格式并加载
        if checkpoint.get('checkpoint_format') == 'deep_cfr_v1':
            # 新格式
            self.iteration = checkpoint.get('iteration', checkpoint.get('episode_number', 0))
            self.regret_network.load_state_dict(checkpoint['regret_network_state_dict'])
            self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            self.regret_optimizer.load_state_dict(checkpoint['regret_optimizer_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            
            # 加载EMA目标网络状态（如果存在）
            if 'regret_network_ema_state_dict' in checkpoint:
                self.regret_network_ema.load_state_dict(checkpoint['regret_network_ema_state_dict'])
            else:
                # 如果没有EMA状态，使用主网络状态初始化
                self.regret_network_ema.load_state_dict(self.regret_network.state_dict())
            
            if 'policy_network_ema_state_dict' in checkpoint:
                self.policy_network_ema.load_state_dict(checkpoint['policy_network_ema_state_dict'])
            else:
                self.policy_network_ema.load_state_dict(self.policy_network.state_dict())
        else:
            # 旧格式（向后兼容）
            self.iteration = checkpoint['iteration']
            self.regret_network.load_state_dict(checkpoint['regret_network'])
            self.policy_network.load_state_dict(checkpoint['policy_network'])
            self.regret_optimizer.load_state_dict(checkpoint['regret_optimizer'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            # 旧格式没有EMA网络，使用主网络状态初始化
            self.regret_network_ema.load_state_dict(self.regret_network.state_dict())
            self.policy_network_ema.load_state_dict(self.policy_network.state_dict())
        
        self.stats = checkpoint.get('stats', self.stats)
        print(f"检查点已加载: {path}")


def load_config_file(config_path: str) -> Dict[str, Any]:
    """加载配置文件。
    
    Args:
        config_path: 配置文件路径（JSON 格式）
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description='河牌专项训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
公共牌格式示例：
  --board "AhKsQdJcTh"      # 连写格式
  --board "Ah Ks Qd Jc Th"  # 空格分隔
  --board "Ah,Ks,Qd,Jc,Th"  # 逗号分隔

牌的表示：
  点数: A=Ace, K=King, Q=Queen, J=Jack, T=10, 9-2
  花色: h=红桃, d=方块, c=梅花, s=黑桃

示例：
  # 随机公共牌训练
  python train_river_only.py --iterations 10000
  
  # 使用配置文件
  python train_river_only.py --config configs/river_only_config.json
  
  # 固定公共牌训练（彩虹面）
  python train_river_only.py --board "AhKsQdJc2h" --iterations 10000
  
  # 固定公共牌训练（同花面）
  python train_river_only.py --board "AhKhQhJh2h" --iterations 10000
  
  # 配置文件 + 命令行参数（命令行参数优先）
  python train_river_only.py --config configs/river_only_config.json --lr 0.0001
"""
    )
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（JSON 格式），命令行参数会覆盖配置文件中的值')
    parser.add_argument('--iterations', type=int, default=None,
                        help='CFR 迭代次数')
    parser.add_argument('--cfr-per-update', type=int, default=None,
                        help='每次网络更新前的 CFR 迭代次数')
    parser.add_argument('--train-steps', type=int, default=None,
                        help='每次更新的训练步数')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='训练批次大小')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='检查点保存目录')
    parser.add_argument('--checkpoint-interval', type=int, default=None,
                        help='检查点保存间隔')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='评估间隔')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--board', type=str, default=None,
                        help='固定的公共牌（5张），如 "AhKsQdJcTh"。不指定则随机采样')
    
    args = parser.parse_args()
    
    # 默认配置
    config_dict = {
        'iterations': 100000,
        'cfr_per_update': 1000,
        'train_steps': 2000,
        'batch_size': 256,
        'learning_rate': 0.001,
        'checkpoint_dir': 'checkpoints/river_only',
        'checkpoint_interval': 10000,
        'eval_interval': 5000,
        'network_architecture': [512, 256, 128],
        'regret_buffer_size': 500000,
        'strategy_buffer_size': 500000,
        'initial_stack': 1000,
        'small_blind': 5,
        'big_blind': 10,
        'max_raises_per_street': 4,
        'board': None,
    }
    
    # 从配置文件加载（如果指定）
    if args.config:
        if not os.path.exists(args.config):
            print(f"错误: 配置文件不存在: {args.config}")
            return
        try:
            file_config = load_config_file(args.config)
            # 映射配置文件中的键名
            key_mapping = {
                'cfr_iterations_per_update': 'cfr_per_update',
                'network_train_steps': 'train_steps',
            }
            for file_key, value in file_config.items():
                mapped_key = key_mapping.get(file_key, file_key)
                if mapped_key in config_dict:
                    config_dict[mapped_key] = value
            print(f"已加载配置文件: {args.config}")
        except Exception as e:
            print(f"错误: 无法加载配置文件: {e}")
            return
    
    # 命令行参数覆盖配置文件
    if args.iterations is not None:
        config_dict['iterations'] = args.iterations
    if args.cfr_per_update is not None:
        config_dict['cfr_per_update'] = args.cfr_per_update
    if args.train_steps is not None:
        config_dict['train_steps'] = args.train_steps
    if args.batch_size is not None:
        config_dict['batch_size'] = args.batch_size
    if args.lr is not None:
        config_dict['learning_rate'] = args.lr
    if args.checkpoint_dir is not None:
        config_dict['checkpoint_dir'] = args.checkpoint_dir
    if args.checkpoint_interval is not None:
        config_dict['checkpoint_interval'] = args.checkpoint_interval
    if args.eval_interval is not None:
        config_dict['eval_interval'] = args.eval_interval
    if args.board is not None:
        config_dict['board'] = args.board
    
    # 解析固定公共牌
    fixed_board = None
    if config_dict['board']:
        try:
            fixed_board = parse_board(config_dict['board'])
            print(f"使用固定公共牌: {board_to_str(fixed_board)}")
        except ValueError as e:
            print(f"错误: {e}")
            return
    
    # 创建检查点目录
    os.makedirs(config_dict['checkpoint_dir'], exist_ok=True)
    
    # 创建配置
    config = TrainingConfig(
        learning_rate=config_dict['learning_rate'],
        batch_size=config_dict['batch_size'],
        network_architecture=config_dict['network_architecture'],
        cfr_iterations_per_update=config_dict['cfr_per_update'],
        network_train_steps=config_dict['train_steps'],
        regret_buffer_size=config_dict['regret_buffer_size'],
        strategy_buffer_size=config_dict['strategy_buffer_size'],
        initial_stack=config_dict['initial_stack'],
        small_blind=config_dict['small_blind'],
        big_blind=config_dict['big_blind'],
        max_raises_per_street=config_dict['max_raises_per_street'],
    )
    
    # 创建训练器
    trainer = RiverOnlyTrainer(config, fixed_board=fixed_board)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 提取常用配置值
    iterations = config_dict['iterations']
    cfr_per_update = config_dict['cfr_per_update']
    train_steps = config_dict['train_steps']
    batch_size = config_dict['batch_size']
    learning_rate = config_dict['learning_rate']
    checkpoint_dir = config_dict['checkpoint_dir']
    checkpoint_interval = config_dict['checkpoint_interval']
    eval_interval = config_dict['eval_interval']
    
    print("=" * 60)
    print("河牌专项训练")
    print("=" * 60)
    print(f"公共牌模式: {trainer.get_board_mode()}")
    print(f"总迭代次数: {iterations}")
    print(f"每次更新的 CFR 迭代: {cfr_per_update}")
    print(f"每次更新的训练步数: {train_steps}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    if args.config:
        print(f"配置文件: {args.config}")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # 训练循环
    update_count = 0
    while trainer.iteration < iterations:
        # CFR 迭代
        print(f"\n[更新 {update_count + 1}] 执行 {cfr_per_update} 次 CFR 迭代...")
        for i in range(cfr_per_update):
            trainer.run_cfr_iteration(verbose=(i % 500 == 0))
            
            # 检查是否需要评估
            if trainer.iteration % eval_interval == 0:
                print(f"\n[评估] 迭代 {trainer.iteration}")
                eval_results = trainer.evaluate_strategy(num_hands=500)
                print(f"  P0 胜率: {eval_results['p0_win_rate']:.2%}")
                print(f"  P1 胜率: {eval_results['p1_win_rate']:.2%}")
                print(f"  平局率: {eval_results['tie_rate']:.2%}")
                print(f"  P0 平均收益: {eval_results['avg_utility_p0']:.2f}")
            
            # 检查是否需要保存检查点
            if trainer.iteration % checkpoint_interval == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_{trainer.iteration}.pt"
                )
                trainer.save_checkpoint(ckpt_path)
        
        # 训练网络
        print(f"\n[训练] 更新神经网络...")
        train_results = trainer.train_networks(verbose=True)
        print(f"  遗憾网络损失: {train_results['regret_loss']:.6f}")
        print(f"  策略网络损失: {train_results['policy_loss']:.6f}")
        
        # 输出收敛监控信息（需求: 4.1, 4.2, 4.3, 4.4）
        convergence_report = train_results.get('convergence_report', {})
        if convergence_report:
            latest_metrics = convergence_report.get('latest_metrics', {})
            if latest_metrics:
                print(f"\n[收敛监控]")
                print(f"  平均策略熵: {latest_metrics.get('avg_entropy', 0):.4f}")
                print(f"  遗憾值均值: {latest_metrics.get('regret_mean', 0):.4f}")
                print(f"  遗憾值标准差: {latest_metrics.get('regret_std', 0):.4f}")
                print(f"  策略KL散度: {latest_metrics.get('policy_kl', 0):.4f}")
                if latest_metrics.get('is_oscillating', False):
                    print(f"  [警告] 检测到策略震荡！建议调整学习率或增加正则化")
            
            entropy_stats = convergence_report.get('entropy_stats', {})
            if entropy_stats:
                print(f"  熵值范围: [{entropy_stats.get('min', 0):.4f}, {entropy_stats.get('max', 0):.4f}]")
        
        update_count += 1
        
        # 打印进度
        elapsed = datetime.now() - start_time
        progress = trainer.iteration / iterations
        eta = elapsed / progress - elapsed if progress > 0 else elapsed
        print(f"\n进度: {trainer.iteration}/{iterations} ({progress:.1%})")
        print(f"已用时间: {elapsed}, 预计剩余: {eta}")
    
    # 最终评估
    print("\n" + "=" * 60)
    print("最终评估")
    print("=" * 60)
    final_eval = trainer.evaluate_strategy(num_hands=2000)
    print(f"P0 胜率: {final_eval['p0_win_rate']:.2%}")
    print(f"P1 胜率: {final_eval['p1_win_rate']:.2%}")
    print(f"平局率: {final_eval['tie_rate']:.2%}")
    print(f"P0 平均收益: {final_eval['avg_utility_p0']:.2f}")
    
    # 保存最终检查点
    final_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    trainer.save_checkpoint(final_path)
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()
