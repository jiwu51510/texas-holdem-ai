#!/usr/bin/env python3
"""
EV预测神经网络训练脚本

该脚本用于训练EV预测网络，从验证数据中学习预测：
- 整体期望值（EV）
- 每个动作的期望值（Action EV）
- 动作策略概率分布（Strategy）

使用方法:
    python train_ev_prediction.py --data-dir experiments/validation_data --epochs 100
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split

from training.ev_prediction_network import EVPredictionNetwork
from training.ev_dataset import EVDataset
from training.ev_trainer import EVTrainer


# 配置日志
def setup_logging(log_level: str = "INFO"):
    """配置日志系统"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="训练EV预测神经网络",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据参数
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="experiments/validation_data",
        help="验证数据目录路径"
    )
    parser.add_argument(
        "--extracted-file",
        type=str,
        default=None,
        help="预提取的数据文件路径（如果提供，直接从此文件加载）"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="最大加载文件数（用于调试）"
    )
    
    # 模型参数
    parser.add_argument(
        "--num-actions",
        type=int,
        default=5,
        help="动作数量"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="隐藏层维度"
    )
    
    # 训练参数
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批次大小"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="学习率"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="验证集比例"
    )
    
    # 损失权重
    parser.add_argument(
        "--ev-weight",
        type=float,
        default=1.0,
        help="EV损失权重"
    )
    parser.add_argument(
        "--action-ev-weight",
        type=float,
        default=1.0,
        help="动作EV损失权重"
    )
    parser.add_argument(
        "--strategy-weight",
        type=float,
        default=1.0,
        help="策略损失权重"
    )
    
    # 其他参数
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/ev_prediction",
        help="检查点保存目录"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="日志输出间隔（epoch数）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备（cuda/cpu，默认自动选择）"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("EV预测神经网络训练")
    logger.info("=" * 60)
    
    # 打印配置
    logger.info("配置参数:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # 加载数据
    logger.info("\n加载数据...")
    try:
        dataset = EVDataset(
            args.data_dir, 
            max_files=args.max_files,
            extracted_file=args.extracted_file
        )
    except FileNotFoundError as e:
        logger.error(f"数据目录不存在: {e}")
        sys.exit(1)
    
    if len(dataset) == 0:
        logger.error("数据集为空，请检查数据目录")
        sys.exit(1)
    
    # 打印数据统计
    stats = dataset.get_statistics()
    logger.info(f"数据集大小: {stats['num_samples']} 个样本")
    logger.info(f"加载文件数: {stats['num_files']}")
    logger.info(f"跳过样本数: {stats['skipped_samples']}")
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if args.device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    logger.info("\n创建模型...")
    model = EVPredictionNetwork(
        num_actions=args.num_actions,
        hidden_dim=args.hidden_dim
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}")
    logger.info(f"可训练参数数: {trainable_params:,}")
    
    # 创建训练器
    trainer = EVTrainer(
        model=model,
        learning_rate=args.learning_rate,
        ev_weight=args.ev_weight,
        action_ev_weight=args.action_ev_weight,
        strategy_weight=args.strategy_weight,
        device=args.device
    )
    
    logger.info(f"使用设备: {trainer.device}")
    
    # 恢复训练
    if args.resume:
        logger.info(f"\n从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 创建检查点目录
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    logger.info("\n开始训练...")
    logger.info("-" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_losses = trainer.train_epoch(train_loader)
        
        # 验证
        val_metrics = trainer.evaluate(val_loader)
        val_loss = val_metrics["ev_mse"]["mean"] + val_metrics["action_ev_mse"]["mean"]
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.best_loss = best_val_loss
            best_path = checkpoint_dir / "best_model.pt"
            trainer.save_checkpoint(str(best_path))
        
        # 定期输出日志
        if epoch % args.log_interval == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"Train Loss: {train_losses['total']:.4f} "
                f"(EV: {train_losses['ev']:.4f}, "
                f"ActionEV: {train_losses['action_ev']:.4f}, "
                f"Strategy: {train_losses['strategy']:.4f}) | "
                f"Val EV_MSE: {val_metrics['ev_mse']['mean']:.4f}, "
                f"Val ActionEV_MSE: {val_metrics['action_ev_mse']['mean']:.4f}"
            )
    
    # 保存最终模型
    final_path = checkpoint_dir / "final_model.pt"
    trainer.save_checkpoint(str(final_path))
    
    # 最终评估
    logger.info("\n" + "=" * 60)
    logger.info("训练完成！最终评估结果:")
    logger.info("=" * 60)
    
    final_metrics = trainer.evaluate(val_loader)
    print_evaluation_report(final_metrics)
    
    logger.info(f"\n模型已保存到: {checkpoint_dir}")
    logger.info(f"  - 最佳模型: best_model.pt")
    logger.info(f"  - 最终模型: final_model.pt")


def print_evaluation_report(metrics: dict):
    """打印评估报告"""
    logger.info("\nEV预测指标:")
    logger.info(f"  均方误差 (MSE): {metrics['ev_mse']['mean']:.6f}")
    logger.info(f"  标准差: {metrics['ev_mse']['std']:.6f}")
    logger.info(f"  最小值: {metrics['ev_mse']['min']:.6f}")
    logger.info(f"  最大值: {metrics['ev_mse']['max']:.6f}")
    
    logger.info("\n动作EV预测指标:")
    logger.info(f"  均方误差 (MSE): {metrics['action_ev_mse']['mean']:.6f}")
    logger.info(f"  标准差: {metrics['action_ev_mse']['std']:.6f}")
    logger.info(f"  最小值: {metrics['action_ev_mse']['min']:.6f}")
    logger.info(f"  最大值: {metrics['action_ev_mse']['max']:.6f}")
    
    logger.info("\n策略预测指标:")
    logger.info(f"  KL散度: {metrics['strategy_kl']['mean']:.6f}")
    logger.info(f"  标准差: {metrics['strategy_kl']['std']:.6f}")
    logger.info(f"  最小值: {metrics['strategy_kl']['min']:.6f}")
    logger.info(f"  最大值: {metrics['strategy_kl']['max']:.6f}")
    
    logger.info(f"\n评估样本数: {metrics['num_samples']}")


if __name__ == "__main__":
    main()
