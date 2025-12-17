"""策略分析示例脚本 - 演示如何分析和可视化训练好的AI策略。

本脚本展示了：
- 加载训练好的模型
- 分析特定状态下的行动概率
- 生成策略热图
- 解释AI决策
- 比较多个模型的策略
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.core import Card, GameState, GameStage
from models.networks import PolicyNetwork
from analysis.strategy_analyzer import StrategyAnalyzer
from utils.checkpoint_manager import CheckpointManager


def demo_analyze_state():
    """演示分析特定游戏状态。"""
    print("=" * 60)
    print("策略分析示例 - 分析游戏状态")
    print("=" * 60)
    
    # 创建策略分析器
    analyzer = StrategyAnalyzer(checkpoint_dir="checkpoints/demo")
    
    # 检查是否有可用的检查点
    checkpoint_dir = "checkpoints/demo"
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if checkpoints:
        # 加载训练好的模型
        latest_checkpoint = checkpoints[-1]
        print(f"\n加载检查点: {latest_checkpoint.path}")
        analyzer.load_model(
            checkpoint_path=latest_checkpoint.path,
            input_dim=370,
            hidden_dims=[256, 128, 64],
            action_dim=6
        )
    else:
        # 使用未训练的模型进行演示
        print("\n没有找到检查点，使用未训练的模型进行演示...")
        
        # 创建临时检查点
        model = PolicyNetwork(
            input_dim=370,
            hidden_dims=[256, 128, 64],
            action_dim=6
        )
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        temp_path = os.path.join(checkpoint_dir, "temp_demo.pt")
        checkpoint_manager.save(
            model=model,
            optimizer=None,
            metadata={'episode_number': 0, 'win_rate': 0.0, 'avg_reward': 0.0}
        )
        
        checkpoints = checkpoint_manager.list_checkpoints()
        if checkpoints:
            analyzer.load_model(
                checkpoint_path=checkpoints[-1].path,
                input_dim=370,
                hidden_dims=[256, 128, 64],
                action_dim=6
            )
    
    # 创建示例游戏状态
    # 场景：翻牌前，玩家持有 A♥ K♥（强起手牌）
    player_hands = [
        (Card(14, 'h'), Card(13, 'h')),  # 玩家0: A♥ K♥
        (Card(7, 'd'), Card(2, 'c'))      # 玩家1: 7♦ 2♣
    ]
    
    state = GameState(
        player_hands=player_hands,
        community_cards=[],
        pot=15,
        player_stacks=[995, 990],
        current_bets=[5, 10],
        button_position=0,
        stage=GameStage.PREFLOP,
        action_history=[],
        current_player=0
    )
    
    print("\n游戏状态:")
    print(f"  阶段: 翻牌前")
    print(f"  手牌: A♥ K♥（同花AK）")
    print(f"  底池: {state.pot}")
    print(f"  位置: 按钮位（小盲）")
    print(f"  当前下注: {state.current_bets}")
    
    # 分析状态
    print("\n分析行动概率...")
    probs = analyzer.analyze_state(state, player_id=0)
    
    print("\n行动概率分布:")
    for action, prob in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {action:15s}: {prob:6.2%} {bar}")
    
    return probs


def demo_explain_decision():
    """演示决策解释功能。"""
    print("\n" + "=" * 60)
    print("策略分析示例 - 决策解释")
    print("=" * 60)
    
    # 创建策略分析器
    analyzer = StrategyAnalyzer(checkpoint_dir="checkpoints/demo")
    
    # 加载模型
    checkpoint_manager = CheckpointManager("checkpoints/demo")
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        print("\n没有找到检查点，请先运行训练示例。")
        return None
    
    analyzer.load_model(
        checkpoint_path=checkpoints[-1].path,
        input_dim=370,
        hidden_dims=[256, 128, 64],
        action_dim=6
    )
    
    # 创建多个场景进行分析
    scenarios = [
        {
            "name": "强起手牌（翻牌前）",
            "hand": (Card(14, 'h'), Card(14, 'd')),  # AA
            "community": [],
            "stage": GameStage.PREFLOP,
            "pot": 15
        },
        {
            "name": "中等牌力（翻牌）",
            "hand": (Card(10, 'h'), Card(10, 'd')),  # TT
            "community": [Card(14, 'c'), Card(7, 's'), Card(3, 'h')],  # A73
            "stage": GameStage.FLOP,
            "pot": 50
        },
        {
            "name": "弱牌（河牌）",
            "hand": (Card(7, 'h'), Card(2, 'd')),  # 72o
            "community": [
                Card(14, 'c'), Card(13, 's'), Card(12, 'h'),
                Card(11, 'd'), Card(10, 'c')
            ],  # AKQJT
            "stage": GameStage.RIVER,
            "pot": 200
        }
    ]
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        print("-" * 40)
        
        # 创建游戏状态
        player_hands = [
            scenario["hand"],
            (Card(8, 'c'), Card(5, 's'))  # 对手虚拟手牌
        ]
        
        state = GameState(
            player_hands=player_hands,
            community_cards=scenario["community"],
            pot=scenario["pot"],
            player_stacks=[1000 - scenario["pot"] // 2] * 2,
            current_bets=[scenario["pot"] // 4, scenario["pot"] // 4],
            button_position=0,
            stage=scenario["stage"],
            action_history=[],
            current_player=0
        )
        
        # 获取决策解释
        explanation = analyzer.explain_decision(state, player_id=0)
        
        print(f"\n{explanation.state_description}")
        print(f"\n推荐行动: {explanation.recommended_action}")
        print(f"期望价值: {explanation.expected_value:.2f}")
        print(f"\n决策理由:\n{explanation.reasoning}")
    
    return explanation


def demo_strategy_heatmap():
    """演示策略热图生成。"""
    print("\n" + "=" * 60)
    print("策略分析示例 - 策略热图")
    print("=" * 60)
    
    # 创建策略分析器
    analyzer = StrategyAnalyzer(checkpoint_dir="checkpoints/demo")
    
    # 加载模型
    checkpoint_manager = CheckpointManager("checkpoints/demo")
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        print("\n没有找到检查点，请先运行训练示例。")
        return None
    
    analyzer.load_model(
        checkpoint_path=checkpoints[-1].path,
        input_dim=370,
        hidden_dims=[256, 128, 64],
        action_dim=6
    )
    
    # 定义要分析的手牌范围
    # 常见的起手牌类型
    hand_range = [
        (Card(14, 'h'), Card(14, 'd')),  # AA
        (Card(13, 'h'), Card(13, 'd')),  # KK
        (Card(12, 'h'), Card(12, 'd')),  # QQ
        (Card(14, 'h'), Card(13, 'h')),  # AKs
        (Card(14, 'h'), Card(13, 'd')),  # AKo
        (Card(14, 'h'), Card(12, 'h')),  # AQs
        (Card(10, 'h'), Card(10, 'd')),  # TT
        (Card(9, 'h'), Card(8, 'h')),    # 98s
        (Card(7, 'h'), Card(6, 'h')),    # 76s
        (Card(2, 'h'), Card(2, 'd')),    # 22
    ]
    
    hand_labels = [
        "AA", "KK", "QQ", "AKs", "AKo",
        "AQs", "TT", "98s", "76s", "22"
    ]
    
    print("\n生成翻牌前策略热图...")
    print(f"分析 {len(hand_range)} 种起手牌")
    
    # 生成热图数据
    heatmap = analyzer.generate_strategy_heatmap(
        hand_range=hand_range,
        community_cards=[],
        stage=GameStage.PREFLOP,
        player_id=0,
        pot=15,
        player_stacks=[995, 990]
    )
    
    # 显示热图数据
    print("\n策略热图数据:")
    print(f"{'手牌':8s} {'FOLD':8s} {'CHECK':8s} {'RAISE_S':8s} {'RAISE_B':8s}")
    print("-" * 40)
    
    action_names = ['FOLD', 'CHECK/CALL', 'RAISE_SMALL', 'RAISE_BIG']
    for i, label in enumerate(hand_labels):
        row = heatmap[i]
        print(f"{label:8s} {row[0]:7.2%} {row[1]:7.2%} {row[2]:7.2%} {row[3]:7.2%}")
    
    # 保存热图（如果matplotlib可用）
    try:
        output_dir = "analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "preflop_heatmap.png")
        
        analyzer.plot_strategy_heatmap(
            heatmap=heatmap,
            hand_labels=hand_labels,
            action_labels=action_names,
            title="翻牌前策略热图",
            save_path=output_path
        )
        print(f"\n热图已保存到: {output_path}")
    except ImportError:
        print("\n注意: 需要安装matplotlib才能生成可视化图表")
    
    return heatmap


def demo_compare_strategies():
    """演示多模型策略比较。"""
    print("\n" + "=" * 60)
    print("策略分析示例 - 策略比较")
    print("=" * 60)
    
    # 检查是否有多个检查点
    checkpoint_manager = CheckpointManager("checkpoints/demo")
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if len(checkpoints) < 2:
        print("\n需要至少2个检查点才能进行比较。")
        print("请多次运行训练示例以生成多个检查点。")
        return None
    
    # 创建策略分析器
    analyzer = StrategyAnalyzer(checkpoint_dir="checkpoints/demo")
    
    # 选择要比较的检查点
    checkpoint_paths = {}
    for i, cp in enumerate(checkpoints[:3]):  # 最多比较3个
        checkpoint_paths[f"模型_{cp.episode_number}回合"] = cp.path
    
    print(f"\n比较 {len(checkpoint_paths)} 个模型:")
    for name, path in checkpoint_paths.items():
        print(f"  - {name}")
    
    # 创建测试状态
    player_hands = [
        (Card(14, 'h'), Card(13, 'h')),  # AKs
        (Card(7, 'd'), Card(2, 'c'))
    ]
    
    state = GameState(
        player_hands=player_hands,
        community_cards=[],
        pot=15,
        player_stacks=[995, 990],
        current_bets=[5, 10],
        button_position=0,
        stage=GameStage.PREFLOP,
        action_history=[],
        current_player=0
    )
    
    # 比较策略
    print("\n比较各模型在相同状态下的策略...")
    comparison = analyzer.compare_strategies(
        checkpoint_paths=checkpoint_paths,
        state=state,
        player_id=0,
        input_dim=370,
        hidden_dims=[256, 128, 64],
        action_dim=6
    )
    
    # 显示比较结果
    print(f"\n状态描述:\n{comparison.state_description}")
    print("\n策略比较:")
    print(f"{'模型':20s} {'FOLD':8s} {'CHECK':8s} {'RAISE_S':8s} {'RAISE_B':8s}")
    print("-" * 56)
    
    for model_name in comparison.models:
        probs = comparison.strategies[model_name]
        print(f"{model_name:20s} "
              f"{probs.get('FOLD', 0):7.2%} "
              f"{probs.get('CHECK/CALL', 0):7.2%} "
              f"{probs.get('RAISE_SMALL', 0):7.2%} "
              f"{probs.get('RAISE_BIG', 0):7.2%}")
    
    # 保存比较图（如果matplotlib可用）
    try:
        output_dir = "analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "strategy_comparison.png")
        
        analyzer.plot_strategy_comparison(
            comparison=comparison,
            save_path=output_path
        )
        print(f"\n比较图已保存到: {output_path}")
    except ImportError:
        print("\n注意: 需要安装matplotlib才能生成可视化图表")
    
    return comparison


def demo_save_analysis():
    """演示保存分析结果。"""
    print("\n" + "=" * 60)
    print("策略分析示例 - 保存分析结果")
    print("=" * 60)
    
    # 创建策略分析器
    analyzer = StrategyAnalyzer(checkpoint_dir="checkpoints/demo")
    
    # 加载模型
    checkpoint_manager = CheckpointManager("checkpoints/demo")
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        print("\n没有找到检查点，请先运行训练示例。")
        return None
    
    analyzer.load_model(
        checkpoint_path=checkpoints[-1].path,
        input_dim=370,
        hidden_dims=[256, 128, 64],
        action_dim=6
    )
    
    # 创建游戏状态
    player_hands = [
        (Card(14, 'h'), Card(14, 'd')),  # AA
        (Card(7, 'd'), Card(2, 'c'))
    ]
    
    state = GameState(
        player_hands=player_hands,
        community_cards=[],
        pot=15,
        player_stacks=[995, 990],
        current_bets=[5, 10],
        button_position=0,
        stage=GameStage.PREFLOP,
        action_history=[],
        current_player=0
    )
    
    # 获取决策解释
    explanation = analyzer.explain_decision(state, player_id=0)
    
    # 保存分析结果
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "decision_analysis.json")
    
    analyzer.save_analysis(explanation, output_path)
    print(f"\n分析结果已保存到: {output_path}")
    
    # 显示保存的内容
    print("\n保存的分析内容:")
    print(f"  推荐行动: {explanation.recommended_action}")
    print(f"  期望价值: {explanation.expected_value:.2f}")
    print(f"  行动概率数量: {len(explanation.action_probabilities)}")
    
    return explanation


if __name__ == "__main__":
    print("德州扑克AI训练系统 - 策略分析示例\n")
    
    # 分析游戏状态
    demo_analyze_state()
    
    # 决策解释
    demo_explain_decision()
    
    # 策略热图
    demo_strategy_heatmap()
    
    # 策略比较
    demo_compare_strategies()
    
    # 保存分析结果
    demo_save_analysis()
    
    print("\n" + "=" * 60)
    print("所有策略分析示例完成!")
    print("=" * 60)
