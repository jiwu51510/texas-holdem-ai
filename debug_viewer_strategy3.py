#!/usr/bin/env python3
"""调试策略网络和遗憾网络的关系。"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.core import Card, GameState, GameStage
from models.networks import PolicyNetwork, RegretNetwork
from environment.state_encoder import StateEncoder


def compare_networks():
    """比较策略网络和遗憾网络的输出。"""
    print("=" * 60)
    print("比较策略网络和遗憾网络的输出")
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
    
    print(f"使用检查点: {checkpoint_path}")
    
    # 加载检查点
    checkpoint_data = torch.load(checkpoint_path, weights_only=False)
    
    # 创建网络
    policy_network = PolicyNetwork(input_dim=370, hidden_dims=[512, 256, 128], action_dim=6)
    regret_network = RegretNetwork(input_dim=370, hidden_dims=[512, 256, 128], action_dim=6)
    
    # 加载权重
    policy_network.load_state_dict(checkpoint_data['policy_network_state_dict'])
    regret_network.load_state_dict(checkpoint_data['regret_network_state_dict'])
    
    policy_network.eval()
    regret_network.eval()
    
    # 创建测试状态
    encoder = StateEncoder()
    
    test_cases = [
        ((Card(14, 'h'), Card(13, 'h')), "AhKh (强牌)"),
        ((Card(7, 'h'), Card(2, 's')), "7h2s (弱牌)"),
        ((Card(10, 'h'), Card(10, 's')), "ThTs (中等牌)"),
    ]
    
    board = [Card(10, 'c'), Card(9, 'h'), Card(8, 'h'), Card(2, 'c'), Card(7, 'd')]
    opponent_hand = (Card(3, 'c'), Card(4, 'c'))
    
    action_names = ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG', 'ALL_IN']
    
    for hand, name in test_cases:
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
            # 遗憾网络输出
            regrets = regret_network(state_tensor).numpy()
            regret_strategy = regret_network.get_strategy(state_tensor).numpy()
            
            # 策略网络输出
            policy_logits = policy_network(state_tensor).numpy()
            policy_probs = policy_network.get_action_probs(state_tensor).numpy()
        
        print(f"\n{name}:")
        print(f"  遗憾值: {regrets}")
        print(f"  遗憾网络策略 (Regret Matching): {regret_strategy}")
        print(f"  策略网络 logits: {policy_logits}")
        print(f"  策略网络概率 (Softmax): {policy_probs}")
        
        # 比较
        print(f"\n  动作对比:")
        for i, action in enumerate(action_names):
            print(f"    {action}: 遗憾={regrets[i]:.4f}, RM策略={regret_strategy[i]*100:.2f}%, 策略网络={policy_probs[i]*100:.2f}%")


def check_strategy_buffer_content():
    """检查策略缓冲区应该存储什么内容。"""
    print("\n" + "=" * 60)
    print("分析策略缓冲区内容")
    print("=" * 60)
    
    # 模拟遗憾网络输出
    regrets = np.array([-2.4, 3.0, -7.2, -8.2, -8.5, -1.4])
    
    # Regret Matching
    positive_regrets = np.maximum(regrets, 0)
    regret_sum = positive_regrets.sum()
    
    if regret_sum > 0:
        strategy = positive_regrets / regret_sum
    else:
        strategy = np.ones(6) / 6
    
    print(f"遗憾值: {regrets}")
    print(f"正遗憾值: {positive_regrets}")
    print(f"正遗憾值和: {regret_sum}")
    print(f"Regret Matching 策略: {strategy}")
    
    # 这个策略会被存储到策略缓冲区
    # 然后策略网络会学习这个策略
    
    print("\n策略缓冲区存储的是 Regret Matching 计算出的策略")
    print("策略网络应该学习输出这个策略")
    print("但是策略网络使用 Softmax，而不是 Regret Matching")
    
    # 检查策略网络需要输出什么 logits 才能得到这个策略
    # softmax(logits) = strategy
    # logits = log(strategy) + constant
    
    # 由于 strategy 中有很多 0，log(0) = -inf
    # 这会导致训练问题
    
    print("\n问题分析:")
    print("1. 策略缓冲区存储的策略中，很多动作概率为 0")
    print("2. 策略网络使用交叉熵损失: -sum(target * log(pred))")
    print("3. 当 target=0 时，这一项不贡献损失")
    print("4. 当 target>0 时，网络需要增加对应的 logit")
    print("5. 但是如果只有一个动作有正概率，网络会过度拟合到那个动作")


def analyze_training_issue():
    """分析训练问题。"""
    print("\n" + "=" * 60)
    print("分析训练问题")
    print("=" * 60)
    
    # 假设策略缓冲区中的样本
    # 大多数样本的策略是 [0, 1, 0, 0, 0, 0] (只有 CHECK 有正遗憾)
    
    # 交叉熵损失
    # loss = -sum(target * log(pred))
    # 如果 target = [0, 1, 0, 0, 0, 0]
    # loss = -log(pred[1])
    
    # 为了最小化损失，网络需要最大化 pred[1]
    # 这意味着 logit[1] 需要远大于其他 logits
    
    # 但是如果训练数据中有不同的样本
    # 有些样本 target = [0, 1, 0, 0, 0, 0]
    # 有些样本 target = [0, 0, 0, 0, 1, 0]
    # 网络会尝试平衡这些样本
    
    # 问题可能是：
    # 1. 训练数据不平衡
    # 2. 网络过拟合到某些样本
    # 3. 学习率太高导致不稳定
    
    print("可能的问题:")
    print("1. 策略缓冲区中的样本分布不均衡")
    print("2. 大多数样本可能都是 RAISE_BIG 有高概率")
    print("3. 或者训练过程中出现了问题")
    
    # 检查检查点中的统计信息
    checkpoint_path = "checkpoints/river_only_fixed/checkpoint_70000.pt"
    if Path(checkpoint_path).exists():
        checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        if 'stats' in checkpoint_data:
            print(f"\n检查点统计信息: {checkpoint_data['stats']}")


def test_viewer_should_use_regret_network():
    """测试：viewer 应该使用遗憾网络而不是策略网络。"""
    print("\n" + "=" * 60)
    print("建议：Viewer 应该使用遗憾网络")
    print("=" * 60)
    
    print("""
在 Deep CFR 中：
- 遗憾网络学习每个动作的遗憾值
- 策略通过 Regret Matching 从遗憾值计算
- 策略网络学习长期平均策略（用于最终部署）

但是在训练早期：
- 遗憾网络的输出更准确（直接学习遗憾值）
- 策略网络可能还没有收敛

建议：
1. Viewer 应该使用遗憾网络 + Regret Matching 来显示策略
2. 或者检查策略网络的训练是否正确
""")


def main():
    """主函数。"""
    compare_networks()
    check_strategy_buffer_content()
    analyze_training_issue()
    test_viewer_should_use_regret_network()


if __name__ == "__main__":
    main()
