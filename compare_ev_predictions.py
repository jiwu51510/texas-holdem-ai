#!/usr/bin/env python3
"""
EV预测模型对比脚本

从原始数据中随机选取100组样本，对比：
1. 模型直接预测的EV vs 真实EV
2. 用预测策略加权动作EV计算的EV vs 真实EV
"""

import json
import random
import torch
import numpy as np
from pathlib import Path
from tabulate import tabulate

from training.ev_prediction_network import EVPredictionNetwork
from training.ev_dataset import EVDataset, DEFAULT_ACTIONS


def load_model(checkpoint_path: str):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 从检查点配置中获取模型参数
    config = checkpoint.get('config', {})
    num_actions = config.get('num_actions', 5)
    hidden_dim = config.get('hidden_dim', 256)
    
    print(f"  模型配置: num_actions={num_actions}, hidden_dim={hidden_dim}")
    
    model = EVPredictionNetwork(num_actions=num_actions, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def format_strategy_tensor(strategy: torch.Tensor, action_names: list) -> str:
    """格式化策略张量为简洁字符串"""
    parts = []
    for i, name in enumerate(action_names):
        val = strategy[i].item()
        if val > 0.01:
            short_name = name.split(':')[0][:1] + name.split(':')[1] if ':' in name else name[:3]
            parts.append(f"{short_name}:{val:.0%}")
    return ' '.join(parts) if parts else '-'


class TeeOutput:
    """同时输出到终端和文件"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def main():
    # 配置
    checkpoint_path = "checkpoints/ev_prediction/best_model.pt"
    data_dir = "experiments/validation_data"
    num_samples = 100
    output_file = "experiments/results/ev_prediction_comparison.txt"
    
    # 确保输出目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 同时输出到终端和文件
    import sys
    f = open(output_file, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(sys.stdout, f)
    
    print("=" * 80)
    print("EV预测模型对比分析")
    print("=" * 80)
    
    # 加载模型
    print(f"\n加载模型: {checkpoint_path}")
    model = load_model(checkpoint_path)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加载数据集
    print(f"加载数据: {data_dir}")
    dataset = EVDataset(data_dir, max_files=None)
    print(f"数据集大小: {len(dataset)} 个样本")
    
    # 使用新的随机种子选取不同的样本
    random.seed(12345)  # 新种子，选取不同的100组
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # 收集对比数据
    results = []
    action_names = DEFAULT_ACTIONS
    
    with torch.no_grad():
        for idx in indices:
            features, target_ev, target_action_ev, target_strategy = dataset[idx]
            
            # 模型预测
            features_batch = features.unsqueeze(0)
            pred_ev, pred_action_ev, pred_strategy = model(features_batch)
            
            pred_ev = pred_ev.squeeze()
            pred_action_ev = pred_action_ev.squeeze()
            pred_strategy = pred_strategy.squeeze()
            
            # 计算加权EV：用预测策略加权真实动作EV
            weighted_ev_with_true_action_ev = (pred_strategy * target_action_ev).sum().item()
            
            # 计算加权EV：用预测策略加权预测动作EV
            weighted_ev_with_pred_action_ev = (pred_strategy * pred_action_ev).sum().item()
            
            # 计算加权EV：用真实策略加权真实动作EV（应该等于真实EV）
            weighted_ev_true = (target_strategy * target_action_ev).sum().item()
            
            results.append({
                'idx': idx,
                'hero_eq': features[0].item(),
                'range_eq': features[1].item(),
                'solver_eq': features[2].item(),
                'eqr': features[3].item(),
                'true_ev': target_ev.item(),
                'pred_ev': pred_ev.item(),
                'weighted_ev_true_aev': weighted_ev_with_true_action_ev,  # 预测策略 × 真实动作EV
                'weighted_ev_pred_aev': weighted_ev_with_pred_action_ev,  # 预测策略 × 预测动作EV
                'weighted_ev_true': weighted_ev_true,  # 真实策略 × 真实动作EV
                'true_strategy': target_strategy,
                'pred_strategy': pred_strategy,
                'true_action_ev': target_action_ev,
                'pred_action_ev': pred_action_ev,
            })
    
    # 计算各种误差
    true_evs = np.array([r['true_ev'] for r in results])
    pred_evs = np.array([r['pred_ev'] for r in results])
    weighted_evs_true_aev = np.array([r['weighted_ev_true_aev'] for r in results])
    weighted_evs_pred_aev = np.array([r['weighted_ev_pred_aev'] for r in results])
    
    err_pred_ev = np.abs(pred_evs - true_evs)
    err_weighted_true_aev = np.abs(weighted_evs_true_aev - true_evs)
    err_weighted_pred_aev = np.abs(weighted_evs_pred_aev - true_evs)
    
    # 打印详细对比表格
    print(f"\n新随机选取 {len(results)} 组样本对比:")
    print("-" * 100)
    
    table_data = []
    for r in results[:30]:
        table_data.append([
            r['idx'],
            f"{r['hero_eq']:.3f}",
            f"{r['eqr']:.3f}",
            f"{r['true_ev']:.2f}",
            f"{r['pred_ev']:.2f}",
            f"{r['weighted_ev_true_aev']:.2f}",
            f"{r['weighted_ev_pred_aev']:.2f}",
            f"{abs(r['pred_ev'] - r['true_ev']):.3f}",
            f"{abs(r['weighted_ev_true_aev'] - r['true_ev']):.3f}",
        ])
    
    headers = ['ID', 'Hero_Eq', 'EQR', 'True_EV', 'Pred_EV', 'W_EV(真AEV)', 'W_EV(预AEV)', 'Err1', 'Err2']
    print(tabulate(table_data, headers=headers, tablefmt='simple'))
    print("\n说明:")
    print("  Pred_EV: 网络直接预测的EV")
    print("  W_EV(真AEV): 预测策略 × 真实动作EV")
    print("  W_EV(预AEV): 预测策略 × 预测动作EV")
    print("  Err1: |Pred_EV - True_EV|")
    print("  Err2: |W_EV(真AEV) - True_EV|")
    
    # 打印策略和动作EV对比（前10个）
    print("\n" + "=" * 80)
    print("策略和动作EV详细对比（前10个样本）:")
    print("=" * 80)
    for r in results[:10]:
        print(f"\n样本 {r['idx']}:")
        print(f"  输入: hero_eq={r['hero_eq']:.3f}, eqr={r['eqr']:.3f}")
        print(f"  真实EV: {r['true_ev']:.3f}")
        print(f"  预测EV: {r['pred_ev']:.3f} (误差: {abs(r['pred_ev']-r['true_ev']):.3f})")
        print(f"  加权EV(预测策略×真实动作EV): {r['weighted_ev_true_aev']:.3f} (误差: {abs(r['weighted_ev_true_aev']-r['true_ev']):.3f})")
        print(f"  真实策略: {format_strategy_tensor(r['true_strategy'], action_names)}")
        print(f"  预测策略: {format_strategy_tensor(r['pred_strategy'], action_names)}")
        print(f"  真实动作EV: {[f'{v:.1f}' for v in r['true_action_ev'].tolist()]}")
        print(f"  预测动作EV: {[f'{v:.1f}' for v in r['pred_action_ev'].tolist()]}")
    
    # 打印统计摘要
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)
    
    print(f"\n方法1: 网络直接预测EV")
    print(f"  平均绝对误差 (MAE): {err_pred_ev.mean():.4f}")
    print(f"  标准差: {err_pred_ev.std():.4f}")
    print(f"  中位数: {np.median(err_pred_ev):.4f}")
    print(f"  最大值: {err_pred_ev.max():.4f}")
    
    print(f"\n方法2: 预测策略 × 真实动作EV")
    print(f"  平均绝对误差 (MAE): {err_weighted_true_aev.mean():.4f}")
    print(f"  标准差: {err_weighted_true_aev.std():.4f}")
    print(f"  中位数: {np.median(err_weighted_true_aev):.4f}")
    print(f"  最大值: {err_weighted_true_aev.max():.4f}")
    
    print(f"\n方法3: 预测策略 × 预测动作EV")
    print(f"  平均绝对误差 (MAE): {err_weighted_pred_aev.mean():.4f}")
    print(f"  标准差: {err_weighted_pred_aev.std():.4f}")
    print(f"  中位数: {np.median(err_weighted_pred_aev):.4f}")
    print(f"  最大值: {err_weighted_pred_aev.max():.4f}")
    
    # 误差分布对比
    print(f"\n误差分布对比:")
    print(f"{'阈值':<10} {'直接预测EV':<20} {'预测策略×真实AEV':<20} {'预测策略×预测AEV':<20}")
    print("-" * 70)
    for threshold in [0.5, 1.0, 2.0, 5.0]:
        pct1 = (err_pred_ev < threshold).mean() * 100
        pct2 = (err_weighted_true_aev < threshold).mean() * 100
        pct3 = (err_weighted_pred_aev < threshold).mean() * 100
        print(f"< {threshold:<8} {pct1:>6.1f}%{'':<13} {pct2:>6.1f}%{'':<13} {pct3:>6.1f}%")
    
    # 核心统计：100次总差值
    print("\n" + "=" * 80)
    print("核心统计：预测策略 × 真实动作EV vs 真实EV")
    print("=" * 80)
    
    # 计算差值（带符号）
    diff_weighted = weighted_evs_true_aev - true_evs
    
    print(f"\n100组样本统计:")
    print(f"  真实EV总和: {true_evs.sum():.2f}")
    print(f"  加权EV总和 (预测策略×真实动作EV): {weighted_evs_true_aev.sum():.2f}")
    print(f"  总差值 (加权EV - 真实EV): {diff_weighted.sum():.2f}")
    print(f"  平均每次差值: {diff_weighted.mean():.4f}")
    print(f"  绝对误差总和: {np.abs(diff_weighted).sum():.2f}")
    print(f"  平均绝对误差: {np.abs(diff_weighted).mean():.4f}")
    
    # 正负差值分布
    positive_diff = diff_weighted[diff_weighted > 0]
    negative_diff = diff_weighted[diff_weighted < 0]
    print(f"\n差值分布:")
    print(f"  正差值次数: {len(positive_diff)} (加权EV > 真实EV)")
    print(f"  负差值次数: {len(negative_diff)} (加权EV < 真实EV)")
    print(f"  零差值次数: {(diff_weighted == 0).sum()}")
    if len(positive_diff) > 0:
        print(f"  正差值总和: +{positive_diff.sum():.2f}")
    if len(negative_diff) > 0:
        print(f"  负差值总和: {negative_diff.sum():.2f}")
    
    # 相关性分析
    print(f"\n相关性分析:")
    corr_pred = np.corrcoef(true_evs, pred_evs)[0, 1]
    corr_weighted_true = np.corrcoef(true_evs, weighted_evs_true_aev)[0, 1]
    corr_weighted_pred = np.corrcoef(true_evs, weighted_evs_pred_aev)[0, 1]
    print(f"  真实EV vs 直接预测EV 相关系数: {corr_pred:.4f}")
    print(f"  真实EV vs 加权EV(真实AEV) 相关系数: {corr_weighted_true:.4f}")
    print(f"  真实EV vs 加权EV(预测AEV) 相关系数: {corr_weighted_pred:.4f}")
    
    # 关闭文件
    sys.stdout = original_stdout
    f.close()
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
