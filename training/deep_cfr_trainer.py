"""Deep CFR 训练器实现。

Deep CFR（Deep Counterfactual Regret Minimization）是一种将 CFR 算法与
深度神经网络结合的方法，用于求解大规模不完全信息博弈。

主要组件：
- RegretNetwork：学习每个动作的即时遗憾值
- PolicyNetwork：学习长期平均策略
- ReservoirBuffer：使用蓄水池采样的经验回放缓冲区

训练流程：
1. 使用遗憾网络生成策略进行自博弈
2. 精确计算反事实遗憾值
3. 存储样本到缓冲区
4. 批量训练网络
"""

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
from training.reservoir_buffer import ReservoirBuffer
from training.cfr_sampler import CFRSampler, RiverEnumerator


class DeepCFRTrainer:
    """Deep CFR 训练器 - 实现标准 Deep CFR 算法。
    
    训练流程：
    1. 使用遗憾网络生成策略进行自博弈
    2. 精确计算反事实遗憾值
    3. 存储样本到缓冲区
    4. 批量训练网络
    
    Attributes:
        config: 训练配置
        regret_network: 遗憾网络
        policy_network: 策略网络
        regret_buffer: 遗憾值缓冲区
        strategy_buffer: 策略缓冲区
        regret_optimizer: 遗憾网络优化器
        policy_optimizer: 策略网络优化器
        env: 游戏环境
        state_encoder: 状态编码器
        iteration: 当前 CFR 迭代次数
    """
    
    # 行动类型到索引的映射
    # 5 维输出：FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG
    ACTION_TYPE_TO_IDX = {
        ActionType.FOLD: 0,
        ActionType.CHECK: 1,
        ActionType.CALL: 2,
        ActionType.RAISE_SMALL: 3,
        ActionType.RAISE_BIG: 4,
        ActionType.RAISE: 3  # 保留用于向后兼容，映射到 RAISE_SMALL
    }
    
    IDX_TO_ACTION_TYPE = {
        0: ActionType.FOLD,
        1: ActionType.CHECK,
        2: ActionType.CALL,
        3: ActionType.RAISE_SMALL,
        4: ActionType.RAISE_BIG
    }
    
    def __init__(self, config: TrainingConfig):
        """初始化 Deep CFR 训练器。
        
        Args:
            config: 训练配置对象
        """
        self.config = config
        
        # 初始化状态编码器
        self.state_encoder = StateEncoder()
        input_dim = self.state_encoder.encoding_dim  # 370
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
        
        # 初始化经验回放缓冲区
        self.regret_buffer = ReservoirBuffer(config.regret_buffer_size)
        self.strategy_buffer = ReservoirBuffer(config.strategy_buffer_size)
        
        # 初始化游戏环境
        self.env = PokerEnvironment(
            initial_stack=config.initial_stack,
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            max_raises_per_street=config.max_raises_per_street
        )
        
        # 初始化 CFR 采样器
        self.cfr_sampler = CFRSampler(num_flop_buckets=30)
        self.river_enumerator = RiverEnumerator(self.cfr_sampler)
        
        # 迭代计数
        self.iteration = 0
    
    def traverse_game_tree(
        self, 
        state: GameState, 
        player_id: int,
        reach_probs: Tuple[float, float]
    ) -> float:
        """遍历游戏树，计算反事实遗憾值。
        
        使用外部采样 MCCFR：
        - 对当前玩家的所有动作进行遍历
        - 对对手的动作进行采样
        
        Args:
            state: 当前游戏状态
            player_id: 当前遍历的玩家ID（我们要计算其遗憾值的玩家）
            reach_probs: 两个玩家到达当前状态的概率 (p0, p1)
            
        Returns:
            当前状态的期望收益（对于 player_id）
        """
        # 检查是否为终止状态
        if self._is_terminal(state):
            # 返回 player_id 的收益
            return self._get_terminal_utility(state, player_id)
        
        current_player = state.current_player
        
        # 获取合法行动
        legal_actions = self.env.get_legal_actions(state)
        if not legal_actions:
            return 0.0
        
        # 获取当前策略
        strategy = self._get_strategy(state, current_player)
        
        # 过滤策略到合法行动
        legal_strategy = self._filter_strategy_to_legal(strategy, legal_actions)
        
        if current_player == player_id:
            # 当前玩家是我们要计算遗憾值的玩家
            # 遍历所有合法行动
            action_values = {}
            
            for action in legal_actions:
                action_idx = self._action_to_index(action)
                
                # 执行行动，获取下一个状态
                next_state = self._apply_action(state, action)
                
                # 更新到达概率
                new_reach_probs = list(reach_probs)
                new_reach_probs[current_player] *= legal_strategy[action_idx]
                
                # 递归遍历
                action_value = self.traverse_game_tree(
                    next_state, player_id, tuple(new_reach_probs)
                )
                action_values[action_idx] = action_value
            
            # 计算当前策略的期望收益
            expected_value = sum(
                legal_strategy[idx] * value 
                for idx, value in action_values.items()
            )
            
            # 计算即时反事实遗憾值
            instant_regrets = self.compute_counterfactual_regrets(
                state, player_id, action_values, legal_strategy
            )
            
            # 存储遗憾值样本到缓冲区（使用累积遗憾值）
            state_encoding = self.state_encoder.encode(state, player_id)
            # 使用对手的到达概率作为权重
            opponent_reach = reach_probs[1 - player_id]
            if opponent_reach > 0:
                # 获取网络预测的累积遗憾值
                state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
                with torch.no_grad():
                    predicted_regrets = self.regret_network(state_tensor).squeeze(0).numpy()
                
                # 累积遗憾值 = 预测值 + 即时遗憾值
                accumulated_regrets = predicted_regrets + instant_regrets
                
                weighted_regrets = accumulated_regrets * opponent_reach
                self.regret_buffer.add(state_encoding, weighted_regrets, self.iteration)
            
            # 存储策略样本到缓冲区
            # 使用当前玩家的到达概率作为权重
            player_reach = reach_probs[player_id]
            if player_reach > 0:
                self.strategy_buffer.add(state_encoding, legal_strategy, self.iteration)
            
            return expected_value
        else:
            # 当前玩家是对手，使用采样
            # 按策略概率选择一个行动
            action = self._sample_action(legal_actions, legal_strategy)
            action_idx = self._action_to_index(action)
            
            # 执行行动
            next_state = self._apply_action(state, action)
            
            # 更新到达概率
            new_reach_probs = list(reach_probs)
            new_reach_probs[current_player] *= legal_strategy[action_idx]
            
            # 递归遍历
            return self.traverse_game_tree(
                next_state, player_id, tuple(new_reach_probs)
            )
    
    def compute_counterfactual_regrets(
        self, 
        state: GameState,
        player_id: int,
        action_values: Dict[int, float],
        strategy: np.ndarray
    ) -> np.ndarray:
        """计算反事实遗憾值。
        
        反事实遗憾值 = 动作收益 - 当前策略期望收益
        
        Args:
            state: 当前游戏状态
            player_id: 玩家ID
            action_values: 每个动作的收益字典 {action_idx: value}
            strategy: 当前策略概率分布
            
        Returns:
            每个动作的遗憾值数组（长度为 action_dim）
        """
        action_dim = len(strategy)
        
        # 计算当前策略的期望收益
        expected_value = sum(
            strategy[idx] * value 
            for idx, value in action_values.items()
        )
        
        # 计算各动作的遗憾值
        regrets = np.zeros(action_dim, dtype=np.float32)
        for action_idx, action_value in action_values.items():
            # 遗憾值 = 该动作的收益 - 当前策略的期望收益
            regrets[action_idx] = action_value - expected_value
        
        return regrets
    
    def run_cfr_iteration(self, verbose: bool = False) -> Dict[str, float]:
        """执行一次 CFR 迭代。
        
        对两个玩家分别遍历游戏树，收集遗憾值和策略样本。
        
        使用采样策略：
        - Preflop: 均匀采样私牌
        - Flop: 分层采样
        - Turn: 随机采样
        - River: 枚举全部
        
        Args:
            verbose: 是否输出详细采样信息
        
        Returns:
            迭代指标字典
        """
        self.iteration += 1
        
        # 使用采样器采样私牌和公共牌
        player_hands, flop, turn = self.cfr_sampler.sample_game_state()
        
        # 记录采样信息
        sampling_info = self._format_sampling_info(player_hands, flop, turn)
        if verbose:
            print(f"[CFR #{self.iteration}] {sampling_info}")
        
        # 创建初始状态（使用采样的牌）
        initial_state = self._create_initial_state_with_cards(player_hands, flop, turn)
        
        # 为两个玩家分别遍历游戏树
        utility_p0 = self.traverse_game_tree_with_sampling(
            initial_state, player_id=0, reach_probs=(1.0, 1.0)
        )
        
        # 重新创建初始状态（使用相同的牌）
        initial_state = self._create_initial_state_with_cards(player_hands, flop, turn)
        utility_p1 = self.traverse_game_tree_with_sampling(
            initial_state, player_id=1, reach_probs=(1.0, 1.0)
        )
        
        return {
            'iteration': self.iteration,
            'utility_p0': utility_p0,
            'utility_p1': utility_p1,
            'regret_buffer_size': len(self.regret_buffer),
            'strategy_buffer_size': len(self.strategy_buffer),
            'sampling_info': sampling_info
        }
    
    def _format_sampling_info(
        self,
        player_hands: List[List[Card]],
        flop: List[Card],
        turn: Card
    ) -> str:
        """格式化采样信息为可读字符串。
        
        Args:
            player_hands: 两个玩家的私牌
            flop: 翻牌（3张）
            turn: 转牌（1张）
            
        Returns:
            格式化的采样信息字符串
        """
        def card_to_str(card: Card) -> str:
            """将牌转换为字符串表示。"""
            rank_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
            rank_str = rank_map.get(card.rank, str(card.rank))
            suit_map = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}
            suit_str = suit_map.get(card.suit, card.suit)
            return f"{rank_str}{suit_str}"
        
        def hand_to_str(hand: List[Card]) -> str:
            """将手牌转换为字符串。"""
            return ''.join(card_to_str(c) for c in hand)
        
        p0_hand = hand_to_str(player_hands[0])
        p1_hand = hand_to_str(player_hands[1])
        flop_str = ' '.join(card_to_str(c) for c in flop)
        turn_str = card_to_str(turn)
        
        # 获取翻牌的 bucket 信息
        bucket = self.cfr_sampler.flop_classifier.classify(flop)
        
        return f"P0:{p0_hand} P1:{p1_hand} | Flop:[{flop_str}](B{bucket}) Turn:{turn_str}"
    
    def train_networks(self, verbose: bool = True) -> Dict[str, float]:
        """训练遗憾网络和策略网络。
        
        从缓冲区采样数据，使用 MSE 损失训练遗憾网络，
        使用交叉熵损失训练策略网络。
        
        Args:
            verbose: 是否输出训练进度
        
        Returns:
            训练指标字典（损失值等）
        """
        regret_loss_total = 0.0
        policy_loss_total = 0.0
        regret_steps = 0
        policy_steps = 0
        
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
                loss = nn.MSELoss()(predictions, targets_tensor)
                loss.backward()
                self.regret_optimizer.step()
                
                regret_loss_total += loss.item()
                regret_steps += 1
                
                # 每500步输出一次进度
                if verbose and (step + 1) % 500 == 0:
                    print(f" {step + 1}", end="", flush=True)
            
            if verbose:
                print(f" 完成 (损失: {regret_loss_total / max(regret_steps, 1):.6f})")
        else:
            if verbose:
                print(f"跳过遗憾网络训练 (缓冲区大小 {len(self.regret_buffer)} < batch_size {self.config.batch_size})")
        
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
                # 获取策略网络的 logits
                logits = self.policy_network(states_tensor)
                # 使用交叉熵损失（targets 是概率分布）
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = -torch.mean(torch.sum(targets_tensor * log_probs, dim=-1))
                loss.backward()
                self.policy_optimizer.step()
                
                policy_loss_total += loss.item()
                policy_steps += 1
                
                # 每500步输出一次进度
                if verbose and (step + 1) % 500 == 0:
                    print(f" {step + 1}", end="", flush=True)
            
            if verbose:
                print(f" 完成 (损失: {policy_loss_total / max(policy_steps, 1):.6f})")
        else:
            if verbose:
                print(f"跳过策略网络训练 (缓冲区大小 {len(self.strategy_buffer)} < batch_size {self.config.batch_size})")
        
        return {
            'regret_loss': regret_loss_total / max(regret_steps, 1),
            'policy_loss': policy_loss_total / max(policy_steps, 1),
            'regret_train_steps': regret_steps,
            'policy_train_steps': policy_steps
        }
    
    def _get_strategy(self, state: GameState, player_id: int) -> np.ndarray:
        """获取当前状态的策略。
        
        使用遗憾网络的 Regret Matching 输出。
        
        Args:
            state: 游戏状态
            player_id: 玩家ID
            
        Returns:
            策略概率分布
        """
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
        """将策略过滤到合法行动。
        
        Args:
            strategy: 原始策略概率分布
            legal_actions: 合法行动列表
            
        Returns:
            过滤后的策略（只有合法行动有非零概率）
        """
        filtered = np.zeros_like(strategy)
        
        for action in legal_actions:
            idx = self._action_to_index(action)
            filtered[idx] = strategy[idx]
        
        # 归一化
        total = filtered.sum()
        if total > 0:
            filtered = filtered / total
        else:
            # 如果所有合法行动概率为0，使用均匀分布
            for action in legal_actions:
                idx = self._action_to_index(action)
                filtered[idx] = 1.0 / len(legal_actions)
        
        return filtered
    
    def _sample_action(
        self, 
        legal_actions: List[Action], 
        strategy: np.ndarray
    ) -> Action:
        """根据策略采样一个行动。
        
        Args:
            legal_actions: 合法行动列表
            strategy: 策略概率分布
            
        Returns:
            采样的行动
        """
        # 获取合法行动的概率
        probs = []
        for action in legal_actions:
            idx = self._action_to_index(action)
            probs.append(strategy[idx])
        
        probs = np.array(probs)
        total = probs.sum()
        
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(len(legal_actions)) / len(legal_actions)
        
        # 采样
        choice = np.random.choice(len(legal_actions), p=probs)
        return legal_actions[choice]
    
    def _action_to_index(self, action: Action) -> int:
        """将行动转换为索引。"""
        return self.ACTION_TYPE_TO_IDX.get(action.action_type, 0)
    
    def _is_terminal(self, state: GameState) -> bool:
        """检查是否为终止状态。
        
        终止条件：
        1. 有玩家弃牌
        2. 河牌圈结束且下注相等
        """
        # 检查是否有弃牌
        for action in state.action_history:
            if action.action_type == ActionType.FOLD:
                return True
        
        # 检查是否到达摊牌
        if state.stage == GameStage.RIVER:
            # 检查是否所有公共牌都已发出
            if len(state.community_cards) == 5:
                # 检查下注是否相等
                if state.current_bets[0] == state.current_bets[1]:
                    # 检查是否双方都已行动
                    if len(state.action_history) >= 2:
                        last_action = state.action_history[-1]
                        if last_action.action_type in [ActionType.CALL, ActionType.CHECK]:
                            return True
        
        return False
    
    def _get_terminal_utility(self, state: GameState, player_id: int) -> float:
        """获取终止状态的收益。
        
        在终止状态，需要先分配底池，然后计算筹码变化。
        
        Args:
            state: 终止状态
            player_id: 玩家ID
            
        Returns:
            玩家的收益（筹码变化）
        """
        from environment.rule_engine import RuleEngine
        from environment.hand_evaluator import compare_hands
        
        # 确定赢家
        winner = -1  # -1 表示平局
        
        # 检查是否有人弃牌
        # 注意：弃牌后 current_player 会被切换，所以不能用 current_player 判断
        # 需要根据行动历史和行动顺序来确定谁弃牌
        for i, action in enumerate(state.action_history):
            if action.action_type == ActionType.FOLD:
                # 根据行动顺序确定弃牌的玩家
                # 翻牌前：玩家0（小盲位）先行动
                # 翻牌后：玩家1（大盲位）先行动
                # 但这里我们需要考虑整个行动历史
                
                # 简化方法：弃牌后 current_player 被切换到对手
                # 所以弃牌的玩家是 1 - current_player
                folder = 1 - state.current_player
                winner = 1 - folder  # 赢家是没弃牌的玩家
                break
        
        # 如果没有人弃牌，进行摊牌比较
        if winner == -1:
            hand1 = list(state.player_hands[0])
            hand2 = list(state.player_hands[1])
            community = state.community_cards
            
            # compare_hands 返回: 0 表示玩家0赢, 1 表示玩家1赢, -1 表示平局
            winner = compare_hands(hand1, hand2, community)
        
        # 计算收益
        initial_stack = self.config.initial_stack
        
        # 当前筹码（不包括底池）
        current_stack = state.player_stacks[player_id]
        
        # 计算玩家应得的底池份额
        pot = state.pot
        if winner == -1:
            # 平局，平分底池
            pot_share = pot / 2
        elif winner == player_id:
            # 玩家赢，获得全部底池
            pot_share = pot
        else:
            # 玩家输，不获得底池
            pot_share = 0
        
        # 最终筹码 = 当前筹码 + 底池份额
        final_stack = current_stack + pot_share
        
        # 收益 = 最终筹码 - 初始筹码
        return float(final_stack - initial_stack)
    
    def _apply_action(self, state: GameState, action: Action) -> GameState:
        """应用行动到状态，返回新状态。
        
        使用轻量级状态复制，避免深拷贝的性能开销。
        只复制会被修改的字段。
        """
        # 轻量级状态复制
        new_stacks = state.player_stacks.copy()
        new_bets = state.current_bets.copy()
        new_pot = state.pot
        new_stage = state.stage
        current_player = state.current_player
        next_player = 1 - current_player
        
        # 应用行动
        if action.action_type == ActionType.FOLD:
            pass  # 弃牌不改变筹码
        elif action.action_type == ActionType.CHECK:
            pass  # 过牌不改变筹码
        elif action.action_type == ActionType.CALL:
            call_amount = new_bets[1 - current_player] - new_bets[current_player]
            new_stacks[current_player] -= call_amount
            new_bets[current_player] = new_bets[1 - current_player]
            new_pot += call_amount
        elif action.action_type in (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG):
            new_stacks[current_player] -= action.amount
            new_bets[current_player] += action.amount
            new_pot += action.amount
        
        # 检查是否进入下一阶段
        bets_equal = new_bets[0] == new_bets[1]
        if bets_equal and action.action_type in [ActionType.CALL, ActionType.CHECK]:
            # 可能进入下一阶段
            stage_order = [GameStage.PREFLOP, GameStage.FLOP, GameStage.TURN, GameStage.RIVER]
            try:
                idx = stage_order.index(state.stage)
                if idx < len(stage_order) - 1:
                    # 检查是否满足阶段结束条件
                    if action.action_type == ActionType.CALL:
                        new_stage = stage_order[idx + 1]
                        new_bets = [0, 0]
                        next_player = 1  # 翻牌后大盲位先行动
                    elif action.action_type == ActionType.CHECK:
                        # 翻牌前：大盲位 CHECK 结束
                        # 翻牌后：需要双方都 CHECK
                        if state.stage == GameStage.PREFLOP and current_player == 1:
                            new_stage = stage_order[idx + 1]
                            new_bets = [0, 0]
                            next_player = 1
                        elif state.stage != GameStage.PREFLOP:
                            # 检查上一个行动是否也是 CHECK
                            if state.action_history and state.action_history[-1].action_type == ActionType.CHECK:
                                new_stage = stage_order[idx + 1]
                                new_bets = [0, 0]
                                next_player = 1
            except ValueError:
                pass
        
        # 创建新状态
        new_state = GameState(
            player_hands=state.player_hands,  # 共享，不会被修改
            community_cards=state.community_cards.copy() if new_stage != state.stage else state.community_cards,
            pot=new_pot,
            player_stacks=new_stacks,
            current_bets=new_bets,
            button_position=state.button_position,
            stage=new_stage,
            action_history=state.action_history + [action],
            current_player=next_player
        )
        
        # 检查是否需要发公共牌
        if self._should_deal_community_cards(new_state):
            self._deal_community_cards_to_state(new_state)
        
        return new_state
    
    def _should_deal_community_cards(self, state: GameState) -> bool:
        """检查是否需要发公共牌。"""
        stage = state.stage
        num_community = len(state.community_cards)
        
        if stage == GameStage.FLOP and num_community == 0:
            return True
        if stage == GameStage.TURN and num_community == 3:
            return True
        if stage == GameStage.RIVER and num_community == 4:
            return True
        
        return False
    
    def _deal_community_cards_to_state(self, state: GameState) -> None:
        """为状态发公共牌。
        
        使用确定性的方式生成公共牌（基于已知的手牌）。
        在 CFR 中，我们可以使用随机采样或者遍历所有可能的公共牌。
        这里使用随机采样来简化实现。
        """
        # 获取已使用的牌
        used_cards = set()
        for hand in state.player_hands:
            for card in hand:
                used_cards.add((card.rank, card.suit))
        for card in state.community_cards:
            used_cards.add((card.rank, card.suit))
        
        # 创建剩余的牌堆
        remaining_deck = []
        for suit in ['h', 'd', 'c', 's']:
            for rank in range(2, 15):
                if (rank, suit) not in used_cards:
                    remaining_deck.append(Card(rank, suit))
        
        # 随机打乱
        import random
        random.shuffle(remaining_deck)
        
        # 发牌
        stage = state.stage
        num_community = len(state.community_cards)
        
        if stage == GameStage.FLOP and num_community == 0:
            for _ in range(3):
                if remaining_deck:
                    state.community_cards.append(remaining_deck.pop())
        elif stage == GameStage.TURN and num_community == 3:
            if remaining_deck:
                state.community_cards.append(remaining_deck.pop())
        elif stage == GameStage.RIVER and num_community == 4:
            if remaining_deck:
                state.community_cards.append(remaining_deck.pop())
    
    def get_average_strategy(self, state: GameState, player_id: int) -> np.ndarray:
        """获取平均策略（用于最终部署）。
        
        使用策略网络输出的概率分布。
        
        Args:
            state: 游戏状态
            player_id: 玩家ID
            
        Returns:
            平均策略概率分布
        """
        state_encoding = self.state_encoder.encode(state, player_id)
        state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
        
        with torch.no_grad():
            probs = self.policy_network.get_action_probs(state_tensor)
            return probs.squeeze(0).numpy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取当前训练指标。
        
        Returns:
            指标字典
        """
        return {
            'iteration': self.iteration,
            'regret_buffer_size': len(self.regret_buffer),
            'strategy_buffer_size': len(self.strategy_buffer),
            'regret_buffer_total_seen': self.regret_buffer.get_total_seen(),
            'strategy_buffer_total_seen': self.strategy_buffer.get_total_seen()
        }
    
    def reset(self) -> None:
        """重置训练器状态。"""
        self.iteration = 0
        self.regret_buffer.clear()
        self.strategy_buffer.clear()

    def _create_initial_state_with_cards(
        self,
        player_hands: List[List[Card]],
        flop: List[Card],
        turn: Card
    ) -> GameState:
        """使用采样的牌创建初始游戏状态。
        
        Args:
            player_hands: 两个玩家的私牌
            flop: 翻牌（3张）
            turn: 转牌（1张）
            
        Returns:
            游戏状态对象
        """
        # 公共牌包括翻牌和转牌
        community_cards = flop + [turn]
        
        # 将私牌列表转换为元组（GameState 要求）
        hands_as_tuples = [
            (hand[0], hand[1]) for hand in player_hands
        ]
        
        return GameState(
            player_hands=hands_as_tuples,
            community_cards=community_cards,
            current_player=0,
            pot=self.config.small_blind + self.config.big_blind,
            player_stacks=[
                self.config.initial_stack - self.config.small_blind,
                self.config.initial_stack - self.config.big_blind
            ],
            current_bets=[self.config.small_blind, self.config.big_blind],
            button_position=0,
            stage=GameStage.TURN,  # 从转牌阶段开始
            action_history=[]
        )
    
    def traverse_game_tree_with_sampling(
        self, 
        state: GameState, 
        player_id: int,
        reach_probs: Tuple[float, float]
    ) -> float:
        """使用采样策略遍历游戏树。
        
        在河牌阶段使用枚举而非采样。
        
        Args:
            state: 当前游戏状态
            player_id: 当前遍历的玩家ID
            reach_probs: 两个玩家到达当前状态的概率
            
        Returns:
            当前状态的期望收益
        """
        # 检查是否为终止状态
        if self._is_terminal(state):
            return self._get_terminal_utility(state, player_id)
        
        # 检查是否需要进入河牌阶段
        if state.stage == GameStage.TURN and self._should_advance_to_river(state):
            # 河牌阶段：枚举全部河牌
            return self._enumerate_river_and_compute(state, player_id, reach_probs)
        
        current_player = state.current_player
        
        # 获取合法行动
        legal_actions = self.env.get_legal_actions(state)
        if not legal_actions:
            return 0.0
        
        # 获取当前策略
        strategy = self._get_strategy(state, current_player)
        legal_strategy = self._filter_strategy_to_legal(strategy, legal_actions)
        
        if current_player == player_id:
            # 当前玩家是我们要计算遗憾值的玩家
            action_values = {}
            
            for action in legal_actions:
                action_idx = self._action_to_index(action)
                next_state = self._apply_action(state, action)
                
                new_reach_probs = list(reach_probs)
                new_reach_probs[current_player] *= legal_strategy[action_idx]
                
                action_value = self.traverse_game_tree_with_sampling(
                    next_state, player_id, tuple(new_reach_probs)
                )
                action_values[action_idx] = action_value
            
            # 计算期望收益
            expected_value = sum(
                legal_strategy[idx] * value 
                for idx, value in action_values.items()
            )
            
            # 计算即时遗憾值并存储（使用累积遗憾值）
            instant_regrets = self.compute_counterfactual_regrets(
                state, player_id, action_values, legal_strategy
            )
            
            state_encoding = self.state_encoder.encode(state, player_id)
            opponent_reach = reach_probs[1 - player_id]
            if opponent_reach > 0:
                # 获取网络预测的累积遗憾值
                state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
                with torch.no_grad():
                    predicted_regrets = self.regret_network(state_tensor).squeeze(0).numpy()
                
                # 累积遗憾值 = 预测值 + 即时遗憾值
                accumulated_regrets = predicted_regrets + instant_regrets
                
                weighted_regrets = accumulated_regrets * opponent_reach
                self.regret_buffer.add(state_encoding, weighted_regrets, self.iteration)
            
            player_reach = reach_probs[player_id]
            if player_reach > 0:
                self.strategy_buffer.add(state_encoding, legal_strategy, self.iteration)
            
            return expected_value
        else:
            # 对手：采样一个行动
            action = self._sample_action(legal_actions, legal_strategy)
            action_idx = self._action_to_index(action)
            
            next_state = self._apply_action(state, action)
            
            new_reach_probs = list(reach_probs)
            new_reach_probs[current_player] *= legal_strategy[action_idx]
            
            return self.traverse_game_tree_with_sampling(
                next_state, player_id, tuple(new_reach_probs)
            )
    
    def _should_advance_to_river(self, state: GameState) -> bool:
        """检查是否应该进入河牌阶段。
        
        当转牌阶段的下注轮结束时进入河牌。
        """
        if state.stage != GameStage.TURN:
            return False
        
        # 检查下注是否相等且双方都已行动
        if state.current_bets[0] == state.current_bets[1]:
            if len(state.action_history) >= 2:
                last_action = state.action_history[-1]
                if last_action.action_type in [ActionType.CALL, ActionType.CHECK]:
                    return True
        
        return False
    
    def _enumerate_river_and_compute(
        self,
        state: GameState,
        player_id: int,
        reach_probs: Tuple[float, float]
    ) -> float:
        """采样河牌并计算平均收益。
        
        使用采样代替完全枚举，大幅提升性能。
        默认采样 5 张河牌，而不是枚举全部 44-46 张。
        
        Args:
            state: 转牌阶段结束时的状态
            player_id: 玩家ID
            reach_probs: 到达概率
            
        Returns:
            采样河牌的平均收益
        """
        # 收集已使用的牌
        used_cards = set()
        for hand in state.player_hands:
            for card in hand:
                used_cards.add((card.rank, card.suit))
        for card in state.community_cards:
            used_cards.add((card.rank, card.suit))
        
        # 获取所有可能的河牌
        all_river_cards = list(self.cfr_sampler.enumerate_river(used_cards))
        
        if not all_river_cards:
            return 0.0
        
        # 采样河牌数量（默认 5 张，最多不超过可用河牌数）
        num_samples = min(5, len(all_river_cards))
        
        # 随机采样河牌
        import random
        sampled_cards = random.sample(all_river_cards, num_samples)
        
        total_value = 0.0
        
        for river_card in sampled_cards:
            # 创建河牌状态（使用浅拷贝 + 手动复制必要字段，避免深拷贝开销）
            river_state = GameState(
                player_hands=state.player_hands,  # 不变，可以共享
                community_cards=state.community_cards + [river_card],  # 新列表
                pot=state.pot,
                player_stacks=state.player_stacks.copy(),  # 浅拷贝
                current_bets=state.current_bets.copy(),  # 浅拷贝
                button_position=state.button_position,
                stage=GameStage.RIVER,
                action_history=state.action_history,  # 不变，可以共享
                current_player=state.current_player
            )
            
            # 递归计算河牌阶段的收益
            value = self._traverse_river(river_state, player_id, reach_probs)
            total_value += value
        
        return total_value / num_samples
    
    def _traverse_river(
        self,
        state: GameState,
        player_id: int,
        reach_probs: Tuple[float, float]
    ) -> float:
        """遍历河牌阶段的游戏树。
        
        Args:
            state: 河牌阶段的状态
            player_id: 玩家ID
            reach_probs: 到达概率
            
        Returns:
            期望收益
        """
        # 检查是否为终止状态
        if self._is_terminal(state):
            return self._get_terminal_utility(state, player_id)
        
        current_player = state.current_player
        legal_actions = self.env.get_legal_actions(state)
        
        if not legal_actions:
            return 0.0
        
        strategy = self._get_strategy(state, current_player)
        legal_strategy = self._filter_strategy_to_legal(strategy, legal_actions)
        
        if current_player == player_id:
            action_values = {}
            
            for action in legal_actions:
                action_idx = self._action_to_index(action)
                next_state = self._apply_action(state, action)
                
                new_reach_probs = list(reach_probs)
                new_reach_probs[current_player] *= legal_strategy[action_idx]
                
                action_value = self._traverse_river(
                    next_state, player_id, tuple(new_reach_probs)
                )
                action_values[action_idx] = action_value
            
            expected_value = sum(
                legal_strategy[idx] * value 
                for idx, value in action_values.items()
            )
            
            # 存储遗憾值和策略样本（使用累积遗憾值）
            instant_regrets = self.compute_counterfactual_regrets(
                state, player_id, action_values, legal_strategy
            )
            
            state_encoding = self.state_encoder.encode(state, player_id)
            opponent_reach = reach_probs[1 - player_id]
            if opponent_reach > 0:
                # 获取网络预测的累积遗憾值
                state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
                with torch.no_grad():
                    predicted_regrets = self.regret_network(state_tensor).squeeze(0).numpy()
                
                # 累积遗憾值 = 预测值 + 即时遗憾值
                accumulated_regrets = predicted_regrets + instant_regrets
                
                weighted_regrets = accumulated_regrets * opponent_reach
                self.regret_buffer.add(state_encoding, weighted_regrets, self.iteration)
            
            player_reach = reach_probs[player_id]
            if player_reach > 0:
                self.strategy_buffer.add(state_encoding, legal_strategy, self.iteration)
            
            return expected_value
        else:
            action = self._sample_action(legal_actions, legal_strategy)
            action_idx = self._action_to_index(action)
            
            next_state = self._apply_action(state, action)
            
            new_reach_probs = list(reach_probs)
            new_reach_probs[current_player] *= legal_strategy[action_idx]
            
            return self._traverse_river(
                next_state, player_id, tuple(new_reach_probs)
            )
