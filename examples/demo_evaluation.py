"""评估示例脚本 - 演示如何评估训练好的德州扑克AI模型。

本脚本展示了：
- 使用不同对手策略评估模型
- 计算胜率、平均盈利等指标
- 多模型比较
- 保存和加载评估结果
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.networks import PolicyNetwork
from analysis.evaluator import (
    Evaluator,
    RandomStrategy,
    FixedStrategy,
    CallOnlyStrategy,
    AlwaysFoldStrategy
)


def demo_evaluate_against_random():
    """演示对随机策略的评估。"""
    print("=" * 60)
    print("模型评估示例 - 对抗随机策略")
    print("=" * 60)
    
    # 创建评估器
    evaluator = Evaluator(
        initial_stack=1000,
        small_blind=5,
        big_blind=10
    )
    
    # 创建一个未训练的模型（用于演示）
    model = PolicyNetwork(
        input_dim=370,
        hidden_dims=[256, 128, 64],
        action_dim=6
    )
    
    # 创建随机策略对手
    opponent = RandomStrategy(seed=42)
    
    print(f"\n评估配置:")
    print(f"  对手策略: {opponent.name}")
    print(f"  评估对局数: 50")
    print(f"  初始筹码: 1000")
    print(f"  盲注: 5/10")
    
    # 运行评估
    print("\n开始评估...")
    result = evaluator.evaluate(
        model=model,
        opponent=opponent,
        num_games=50,
        model_name="demo_model"
    )
    
    # 显示结果
    print("\n评估结果:")
    print(f"  对局数: {result.num_games}")
    print(f"  胜局: {result.wins}")
    print(f"  负局: {result.losses}")
    print(f"  平局: {result.ties}")
    print(f"  胜率: {result.win_rate:.2%}")
    print(f"  平均盈利: {result.avg_profit:.2f}")
    print(f"  盈利标准差: {result.std_profit:.2f}")
    print(f"  总盈利: {result.total_profit:.2f}")
    
    return result


def demo_evaluate_against_multiple_opponents():
    """演示对多种对手策略的评估。"""
    print("\n" + "=" * 60)
    print("模型评估示例 - 对抗多种策略")
    print("=" * 60)
    
    # 创建评估器
    evaluator = Evaluator(
        initial_stack=1000,
        small_blind=5,
        big_blind=10
    )
    
    # 创建模型
    model = PolicyNetwork(
        input_dim=370,
        hidden_dims=[256, 128, 64],
        action_dim=6
    )
    
    # 定义多种对手策略
    opponents = [
        RandomStrategy(seed=42),
        FixedStrategy(prefer_aggressive=False),
        FixedStrategy(prefer_aggressive=True),
        CallOnlyStrategy(),
        AlwaysFoldStrategy()
    ]
    
    print("\n对各种对手策略进行评估（每种20局）...")
    print("-" * 60)
    
    results = []
    for opponent in opponents:
        result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=20,
            model_name="demo_model"
        )
        results.append(result)
        
        print(f"\n对手: {opponent.name}")
        print(f"  胜率: {result.win_rate:.2%}")
        print(f"  平均盈利: {result.avg_profit:.2f}")
    
    print("-" * 60)
    print("\n评估汇总:")
    for result in results:
        print(f"  {result.opponent_name}: 胜率 {result.win_rate:.2%}, "
              f"平均盈利 {result.avg_profit:.2f}")
    
    return results


def demo_compare_models():
    """演示多模型比较。"""
    print("\n" + "=" * 60)
    print("模型评估示例 - 多模型比较")
    print("=" * 60)
    
    # 创建评估器
    evaluator = Evaluator(
        initial_stack=1000,
        small_blind=5,
        big_blind=10
    )
    
    # 创建多个模型（不同架构）
    models = {
        "small_model": PolicyNetwork(
            input_dim=370,
            hidden_dims=[128, 64],
            action_dim=6
        ),
        "medium_model": PolicyNetwork(
            input_dim=370,
            hidden_dims=[256, 128, 64],
            action_dim=6
        ),
        "large_model": PolicyNetwork(
            input_dim=370,
            hidden_dims=[512, 256, 128],
            action_dim=6
        )
    }
    
    # 创建对手
    opponent = RandomStrategy(seed=42)
    
    print(f"\n比较 {len(models)} 个模型（每个模型评估30局）...")
    print(f"对手策略: {opponent.name}")
    print("-" * 60)
    
    # 运行比较
    comparison = evaluator.compare_models(
        models=models,
        opponent=opponent,
        num_games=30
    )
    
    # 显示结果
    print("\n比较结果:")
    for model_name in comparison.models:
        result = comparison.results[model_name]
        print(f"\n  {model_name}:")
        print(f"    胜率: {result.win_rate:.2%}")
        print(f"    平均盈利: {result.avg_profit:.2f}")
        print(f"    盈利标准差: {result.std_profit:.2f}")
    
    return comparison


def demo_save_and_load_results():
    """演示保存和加载评估结果。"""
    print("\n" + "=" * 60)
    print("模型评估示例 - 结果持久化")
    print("=" * 60)
    
    # 创建结果目录
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建评估器
    evaluator = Evaluator(
        initial_stack=1000,
        small_blind=5,
        big_blind=10
    )
    
    # 创建模型和对手
    model = PolicyNetwork(
        input_dim=370,
        hidden_dims=[256, 128, 64],
        action_dim=6
    )
    opponent = RandomStrategy(seed=42)
    
    # 运行评估
    print("\n运行评估...")
    result = evaluator.evaluate(
        model=model,
        opponent=opponent,
        num_games=20,
        model_name="demo_model"
    )
    
    # 保存结果
    result_path = os.path.join(results_dir, "demo_evaluation.json")
    evaluator.save_results(result, result_path)
    print(f"\n评估结果已保存到: {result_path}")
    
    # 加载结果
    loaded_result = evaluator.load_results(result_path)
    print(f"\n从文件加载的结果:")
    print(f"  模型: {loaded_result.model_name}")
    print(f"  对手: {loaded_result.opponent_name}")
    print(f"  胜率: {loaded_result.win_rate:.2%}")
    print(f"  平均盈利: {loaded_result.avg_profit:.2f}")
    print(f"  评估时间: {loaded_result.timestamp}")
    
    return loaded_result


def demo_evaluate_trained_model():
    """演示评估训练好的模型（如果存在检查点）。"""
    print("\n" + "=" * 60)
    print("模型评估示例 - 评估训练好的模型")
    print("=" * 60)
    
    checkpoint_dir = "checkpoints/demo"
    
    # 检查是否有可用的检查点
    from utils.checkpoint_manager import CheckpointManager
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        print(f"\n没有找到检查点目录: {checkpoint_dir}")
        print("请先运行训练示例 (demo_training.py) 生成检查点。")
        return None
    
    # 使用最新的检查点
    latest_checkpoint = checkpoints[-1]
    print(f"\n加载检查点: {latest_checkpoint.path}")
    print(f"  训练回合数: {latest_checkpoint.episode_number}")
    print(f"  训练胜率: {latest_checkpoint.win_rate:.2%}")
    
    # 创建模型并加载参数
    model = PolicyNetwork(
        input_dim=370,
        hidden_dims=[256, 128, 64],
        action_dim=6
    )
    
    model, _, metadata = checkpoint_manager.load(
        checkpoint_path=latest_checkpoint.path,
        model=model,
        optimizer=None
    )
    
    # 创建评估器
    evaluator = Evaluator(
        initial_stack=1000,
        small_blind=5,
        big_blind=10
    )
    
    # 对多种对手进行评估
    opponents = [
        RandomStrategy(seed=42),
        FixedStrategy(prefer_aggressive=False),
        CallOnlyStrategy()
    ]
    
    print("\n评估训练好的模型...")
    print("-" * 60)
    
    for opponent in opponents:
        result = evaluator.evaluate(
            model=model,
            opponent=opponent,
            num_games=30,
            model_name="trained_model"
        )
        
        print(f"\n对手: {opponent.name}")
        print(f"  胜率: {result.win_rate:.2%}")
        print(f"  平均盈利: {result.avg_profit:.2f}")
    
    return model


if __name__ == "__main__":
    print("德州扑克AI训练系统 - 评估示例\n")
    
    # 对随机策略评估
    demo_evaluate_against_random()
    
    # 对多种策略评估
    demo_evaluate_against_multiple_opponents()
    
    # 多模型比较
    demo_compare_models()
    
    # 保存和加载结果
    demo_save_and_load_results()
    
    # 评估训练好的模型（如果存在）
    demo_evaluate_trained_model()
    
    print("\n" + "=" * 60)
    print("所有评估示例完成!")
    print("=" * 60)
