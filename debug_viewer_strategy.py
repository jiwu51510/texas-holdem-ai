#!/usr/bin/env python3
"""调试viewer策略显示问题的脚本。

检查策略计算和显示的完整流程。
"""

import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from models.core import Card, GameState, GameStage, Action, ActionType
from models.networks import PolicyNetwork
from environment.state_encoder import StateEncoder
from viewer.models import ActionConfig
from viewer.strategy_calculator import StrategyCalculator
from viewer.game_tree import GameTreeNavigator
from analysis.strategy_analyzer import StrategyAnalyzer


def debug_model_output():
    """调试模型原始输出。"""
    print("=" * 60)
    print("1. 检查模型原始输出")
    print("=" * 60)
    
    # 加载检查点
    checkpoint_path = "checkpoints/river_only_fixed/checkpoint_70000.pt"
    if not Path(checkpoint_path).exists():
        # 尝试其他检查点
        checkpoint_dir = Path("checkpoints")
        checkpoints = list(checkpoint_dir.glob("**/*.pt"))
        if checkpoints:
            checkpoint_path = str(checkpoints[0])
            print(f"使用检查点: {checkpoint_path}")
        else:
            print("未找到检查点文件")
            return
    
    # 加载检查点数据
    checkpoint_data = torch.load(checkpoint_path, weights_only=False)
    
    print(f"\n检查点格式: {checkpoint_data.get('checkpoint_format', 'legacy')}")
    print(f"检查点键: {list(checkpoint_data.keys())}")
    
    # 检测动作维度
    if 'policy_network_state_dict' in checkpoint_data:
        state_dict = checkpoint_data['policy_network_state_dict']
        # 找到输出层
        for key in sorted(state_dict.keys()):
            if 'weight' in key:
                print(f"  {key}: {state_dict[key].shape}")
        
        # 获取最后一层的输出维度
        last_weight_key = None
        for key in sorted(state_dict.keys()):
            if 'weight' in key:
                last_weight_key = key
        if last_weight_key:
            action_dim = state_dict[last_weight_key].shape[0]
            print(f"\n检测到的动作维度: {action_dim}")
    
    # 检查action_config
    if 'action_config' in checkpoint_data:
        print(f"\n动作配置: {checkpoint_data['action_config']}")
    else:
        print("\n检查点中没有action_config")
    
    return checkpoint_path


def debug_strategy_analyzer(checkpoint_path: str):
    """调试策略分析器。"""
    print("\n" + "=" * 60)
    print("2. 检查策略分析器输出")
    print("=" * 60)
    
    # 创建策略分析器
    analyzer = StrategyAnalyzer(device="cpu")
    analyzer.load_model(checkpoint_path)
    
    print(f"\n动作配置: {analyzer.action_config}")
    if analyzer.action_config:
        print(f"  动作名称: {analyzer.action_config.action_names}")
        print(f"  动作维度: {analyzer.action_config.action_dim}")
    
    print(f"显示动作: {analyzer.available_actions}")
    
    # 创建测试状态
    hand = (Card(14, 'h'), Card(13, 'h'))  # AhKh
    opponent_hand = (Card(2, 's'), Card(3, 's'))
    board = [Card(10, 'h'), Card(9, 'h'), Card(8, 'h'), Card(2, 'c'), Card(7, 'd')]
    
    state = GameState(
        player_hands=[hand, opponent_hand],
        community_cards=board,
        pot=100,
        player_stacks=[900, 900],
        current_bets=[0, 0],
        button_position=0,
        stage=GameStage.RIVER,
        action_history=[],
        current_player=0
    )
    
    # 获取策略
    probs = analyzer.analyze_state(state, player_id=0)
    print(f"\n测试手牌 AhKh 的策略:")
    for action, prob in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {action}: {prob*100:.2f}%")
    
    # 检查概率总和
    total = sum(probs.values())
    print(f"\n概率总和: {total:.4f}")
    
    return analyzer


def debug_raw_network_output(checkpoint_path: str):
    """调试网络原始输出。"""
    print("\n" + "=" * 60)
    print("3. 检查网络原始输出（未经处理）")
    print("=" * 60)
    
    # 加载检查点
    checkpoint_data = torch.load(checkpoint_path, weights_only=False)
    
    # 检测动作维度
    action_dim = 6  # 默认
    if 'policy_network_state_dict' in checkpoint_data:
        state_dict = checkpoint_data['policy_network_state_dict']
        for key in sorted(state_dict.keys()):
            if 'weight' in key:
                last_weight_key = key
        if last_weight_key:
            action_dim = state_dict[last_weight_key].shape[0]
    
    # 创建网络
    policy_network = PolicyNetwork(
        input_dim=370,
        hidden_dims=[512, 256, 128],
        action_dim=action_dim
    )
    
    # 加载权重
    if 'policy_network_state_dict' in checkpoint_data:
        policy_network.load_state_dict(checkpoint_data['policy_network_state_dict'])
    elif 'model_state_dict' in checkpoint_data:
        policy_network.load_state_dict(checkpoint_data['model_state_dict'])
    
    policy_network.eval()
    
    # 创建测试状态
    encoder = StateEncoder()
    hand = (Card(14, 'h'), Card(13, 'h'))  # AhKh
    opponent_hand = (Card(2, 's'), Card(3, 's'))
    board = [Card(10, 'h'), Card(9, 'h'), Card(8, 'h'), Card(2, 'c'), Card(7, 'd')]
    
    state = GameState(
        player_hands=[hand, opponent_hand],
        community_cards=board,
        pot=100,
        player_stacks=[900, 900],
        current_bets=[0, 0],
        button_position=0,
        stage=GameStage.RIVER,
        action_history=[],
        current_player=0
    )
    
    # 编码状态
    encoding = encoder.encode(state, 0)
    state_tensor = torch.tensor(encoding, dtype=torch.float32)
    
    # 获取原始输出
    with torch.no_grad():
        raw_output = policy_network.network(state_tensor)
        probs = policy_network.get_action_probs(state_tensor)
    
    print(f"\n网络原始输出 (logits): {raw_output.numpy()}")
    print(f"Softmax后的概率: {probs.numpy()}")
    
    # 获取动作配置
    action_config = ActionConfig.from_checkpoint(checkpoint_data)
    print(f"\n动作配置: {action_config.action_names}")
    
    # 映射到动作名称
    print("\n动作概率映射:")
    for i, prob in enumerate(probs.numpy()):
        if i < len(action_config.action_names):
            print(f"  {action_config.action_names[i]}: {prob*100:.2f}%")
        else:
            print(f"  ACTION_{i}: {prob*100:.2f}%")


def debug_strategy_calculator(checkpoint_path: str):
    """调试策略计算器。"""
    print("\n" + "=" * 60)
    print("4. 检查策略计算器输出")
    print("=" * 60)
    
    # 创建策略分析器
    analyzer = StrategyAnalyzer(device="cpu")
    analyzer.load_model(checkpoint_path)
    
    # 获取动作配置
    action_config = analyzer.action_config
    
    # 创建策略计算器
    calculator = StrategyCalculator(
        strategy_analyzer=analyzer,
        action_config=action_config
    )
    
    print(f"\n策略计算器可用动作: {calculator.available_actions}")
    
    # 创建游戏树导航器
    navigator = GameTreeNavigator()
    root = navigator.get_root()
    
    # 设置公共牌
    board = [Card(10, 'h'), Card(9, 'h'), Card(8, 'h'), Card(2, 'c'), Card(7, 'd')]
    
    # 计算策略
    result = calculator.calculate_node_strategy(
        node=root,
        board_cards=board,
        player_id=0
    )
    
    print(f"\n可用动作: {result.available_actions}")
    
    # 检查几个手牌的策略
    test_hands = ['AA', 'AKs', 'AKo', '72o']
    for hand_label in test_hands:
        if hand_label in result.hand_strategies:
            strategy = result.hand_strategies[hand_label]
            print(f"\n{hand_label} 策略:")
            for action, prob in sorted(strategy.action_probabilities.items(), key=lambda x: -x[1]):
                print(f"  {action}: {prob*100:.2f}%")
            
            # 检查概率总和
            total = sum(strategy.action_probabilities.values())
            print(f"  概率总和: {total:.4f}")


def debug_check_call_handling():
    """调试CHECK/CALL处理逻辑。"""
    print("\n" + "=" * 60)
    print("5. 检查CHECK/CALL处理逻辑")
    print("=" * 60)
    
    # 检查ActionConfig的默认配置
    for dim in [4, 5, 6]:
        config = ActionConfig.default_for_dim(dim)
        print(f"\n{dim}维动作配置:")
        print(f"  动作名称: {config.action_names}")
        
        # 检查是否同时有CHECK和CALL
        has_check = 'CHECK' in config.action_names
        has_call = 'CALL' in config.action_names
        print(f"  有CHECK: {has_check}, 有CALL: {has_call}")


def main():
    """主函数。"""
    print("调试Viewer策略显示问题")
    print("=" * 60)
    
    # 1. 检查模型输出
    checkpoint_path = debug_model_output()
    if not checkpoint_path:
        return
    
    # 2. 检查策略分析器
    analyzer = debug_strategy_analyzer(checkpoint_path)
    
    # 3. 检查网络原始输出
    debug_raw_network_output(checkpoint_path)
    
    # 4. 检查策略计算器
    debug_strategy_calculator(checkpoint_path)
    
    # 5. 检查CHECK/CALL处理
    debug_check_call_handling()
    
    print("\n" + "=" * 60)
    print("调试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
