#!/usr/bin/env python3
"""进一步调试viewer策略显示问题。

检查不同状态下的策略输出。
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.core import Card, GameState, GameStage, Action, ActionType
from models.networks import PolicyNetwork
from environment.state_encoder import StateEncoder
from viewer.models import ActionConfig
from analysis.strategy_analyzer import StrategyAnalyzer


def test_different_states():
    """测试不同游戏状态下的策略。"""
    print("=" * 60)
    print("测试不同游戏状态下的策略")
    print("=" * 60)
    
    # 加载检查点
    checkpoint_path = "checkpoints/river_only_fixed/checkpoint_70000.pt"
    if not Path(checkpoint_path).exists():
        checkpoint_dir = Path("checkpoints")
        checkpoints = list(checkpoint_dir.glob("**/*.pt"))
        if checkpoints:
            checkpoint_path = str(checkpoints[0])
        else:
            print("未找到检查点文件")
            return
    
    print(f"使用检查点: {checkpoint_path}")
    
    # 创建策略分析器
    analyzer = StrategyAnalyzer(device="cpu")
    analyzer.load_model(checkpoint_path)
    
    # 测试场景1: 河牌圈，强牌
    print("\n场景1: 河牌圈，强牌 (AhKh，同花顺)")
    hand1 = (Card(14, 'h'), Card(13, 'h'))
    board1 = [Card(10, 'h'), Card(11, 'h'), Card(12, 'h'), Card(2, 'c'), Card(7, 'd')]
    test_state(analyzer, hand1, board1, GameStage.RIVER, [0, 0])
    
    # 测试场景2: 河牌圈，弱牌
    print("\n场景2: 河牌圈，弱牌 (7h2s，无对)")
    hand2 = (Card(7, 'h'), Card(2, 's'))
    board2 = [Card(14, 'c'), Card(13, 'd'), Card(10, 's'), Card(8, 'c'), Card(5, 'd')]
    test_state(analyzer, hand2, board2, GameStage.RIVER, [0, 0])
    
    # 测试场景3: 河牌圈，面对加注
    print("\n场景3: 河牌圈，面对加注 (AhKh)")
    hand3 = (Card(14, 'h'), Card(13, 'h'))
    board3 = [Card(10, 'h'), Card(9, 'h'), Card(8, 'h'), Card(2, 'c'), Card(7, 'd')]
    test_state(analyzer, hand3, board3, GameStage.RIVER, [0, 50])  # 对手下注50
    
    # 测试场景4: 翻牌前
    print("\n场景4: 翻牌前 (AhKh)")
    hand4 = (Card(14, 'h'), Card(13, 'h'))
    board4 = []
    test_state(analyzer, hand4, board4, GameStage.PREFLOP, [5, 10])  # 盲注
    
    # 测试场景5: 翻牌前，弱牌
    print("\n场景5: 翻牌前，弱牌 (7h2s)")
    hand5 = (Card(7, 'h'), Card(2, 's'))
    board5 = []
    test_state(analyzer, hand5, board5, GameStage.PREFLOP, [5, 10])


def test_state(analyzer, hand, board, stage, current_bets):
    """测试特定状态。"""
    opponent_hand = (Card(3, 'c'), Card(4, 'c'))
    
    state = GameState(
        player_hands=[hand, opponent_hand],
        community_cards=board,
        pot=100 + sum(current_bets),
        player_stacks=[1000 - current_bets[0], 1000 - current_bets[1]],
        current_bets=current_bets,
        button_position=0,
        stage=stage,
        action_history=[],
        current_player=0
    )
    
    # 获取策略（不过滤非法动作）
    probs_raw = analyzer.analyze_state(state, player_id=0, filter_illegal=False)
    print(f"  原始策略（不过滤）:")
    for action, prob in sorted(probs_raw.items(), key=lambda x: -x[1]):
        if prob > 0.001:
            print(f"    {action}: {prob*100:.2f}%")
    
    # 获取策略（过滤非法动作）
    probs_filtered = analyzer.analyze_state(state, player_id=0, filter_illegal=True)
    print(f"  过滤后策略:")
    for action, prob in sorted(probs_filtered.items(), key=lambda x: -x[1]):
        if prob > 0.001:
            print(f"    {action}: {prob*100:.2f}%")


def check_state_encoding():
    """检查状态编码。"""
    print("\n" + "=" * 60)
    print("检查状态编码")
    print("=" * 60)
    
    encoder = StateEncoder()
    
    # 创建两个不同的状态
    hand1 = (Card(14, 'h'), Card(13, 'h'))  # AhKh
    hand2 = (Card(7, 'h'), Card(2, 's'))    # 7h2s
    opponent_hand = (Card(3, 'c'), Card(4, 'c'))
    board = [Card(10, 'h'), Card(9, 'h'), Card(8, 'h'), Card(2, 'c'), Card(7, 'd')]
    
    state1 = GameState(
        player_hands=[hand1, opponent_hand],
        community_cards=board,
        pot=100,
        player_stacks=[900, 900],
        current_bets=[0, 0],
        button_position=0,
        stage=GameStage.RIVER,
        action_history=[],
        current_player=0
    )
    
    state2 = GameState(
        player_hands=[hand2, opponent_hand],
        community_cards=board,
        pot=100,
        player_stacks=[900, 900],
        current_bets=[0, 0],
        button_position=0,
        stage=GameStage.RIVER,
        action_history=[],
        current_player=0
    )
    
    enc1 = encoder.encode(state1, 0)
    enc2 = encoder.encode(state2, 0)
    
    print(f"编码维度: {len(enc1)}")
    print(f"AhKh 编码前52个值（手牌）: {enc1[:52]}")
    print(f"7h2s 编码前52个值（手牌）: {enc2[:52]}")
    
    # 检查差异
    import numpy as np
    diff = np.array(enc1) - np.array(enc2)
    diff_indices = np.where(diff != 0)[0]
    print(f"\n编码差异位置: {diff_indices[:20]}...")
    print(f"差异数量: {len(diff_indices)}")


def check_regret_network():
    """检查遗憾网络输出。"""
    print("\n" + "=" * 60)
    print("检查遗憾网络输出")
    print("=" * 60)
    
    checkpoint_path = "checkpoints/river_only_fixed/checkpoint_70000.pt"
    if not Path(checkpoint_path).exists():
        checkpoint_dir = Path("checkpoints")
        checkpoints = list(checkpoint_dir.glob("**/*.pt"))
        if checkpoints:
            checkpoint_path = str(checkpoints[0])
        else:
            print("未找到检查点文件")
            return
    
    checkpoint_data = torch.load(checkpoint_path, weights_only=False)
    
    if 'regret_network_state_dict' not in checkpoint_data:
        print("检查点中没有遗憾网络")
        return
    
    from models.networks import RegretNetwork
    
    regret_network = RegretNetwork(
        input_dim=370,
        hidden_dims=[512, 256, 128],
        action_dim=6
    )
    regret_network.load_state_dict(checkpoint_data['regret_network_state_dict'])
    regret_network.eval()
    
    encoder = StateEncoder()
    
    # 测试不同手牌
    test_hands = [
        ((Card(14, 'h'), Card(13, 'h')), "AhKh"),
        ((Card(7, 'h'), Card(2, 's')), "7h2s"),
    ]
    
    board = [Card(10, 'h'), Card(9, 'h'), Card(8, 'h'), Card(2, 'c'), Card(7, 'd')]
    opponent_hand = (Card(3, 'c'), Card(4, 'c'))
    
    for hand, name in test_hands:
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
        
        encoding = encoder.encode(state, 0)
        state_tensor = torch.tensor(encoding, dtype=torch.float32)
        
        with torch.no_grad():
            regrets = regret_network(state_tensor)
        
        print(f"\n{name} 遗憾值:")
        action_names = ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG', 'ALL_IN']
        for i, (action, regret) in enumerate(zip(action_names, regrets.numpy())):
            print(f"  {action}: {regret:.4f}")


def main():
    """主函数。"""
    test_different_states()
    check_state_encoding()
    check_regret_network()


if __name__ == "__main__":
    main()
