#!/usr/bin/env python3
"""策略查看器启动入口。

本脚本提供命令行启动入口，用于启动策略查看器GUI应用程序。

使用方法:
    # 直接启动（无预加载模型）
    python run_viewer.py
    
    # 启动并加载指定检查点
    python run_viewer.py --checkpoint checkpoints/training/checkpoint_xxx.pt
    
    # 简写形式
    python run_viewer.py -c checkpoints/training/checkpoint_xxx.pt

需求引用:
- 需求 1.1: 加载训练好的模型检查点
"""

import sys
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。
    
    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(
        prog="run_viewer",
        description="策略查看器 - Texas Hold'em AI 策略分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                    # 启动查看器
  %(prog)s -c checkpoint.pt                   # 启动并加载检查点
  %(prog)s --checkpoint checkpoints/model.pt  # 指定检查点路径

快捷键:
  Ctrl+O      加载检查点
  Ctrl+Q      退出
  F5          刷新显示
  Ctrl+E      展开所有节点
  Ctrl+Home   重置到根节点
        """
    )
    
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="检查点文件路径（.pt 或 .pth 文件）"
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    return parser.parse_args()


def check_pyqt6() -> bool:
    """检查PyQt6是否已安装。
    
    Returns:
        PyQt6是否可用
    """
    try:
        from PyQt6.QtWidgets import QApplication
        return True
    except ImportError:
        return False


def main() -> int:
    """主入口函数。
    
    Returns:
        退出码（0表示成功，非0表示失败）
    """
    # 解析命令行参数
    args = parse_args()
    
    # 检查PyQt6依赖
    if not check_pyqt6():
        print("错误: 未安装PyQt6库。", file=sys.stderr)
        print("请运行以下命令安装:", file=sys.stderr)
        print("  pip install PyQt6", file=sys.stderr)
        return 1
    
    # 验证检查点文件（如果指定）
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"错误: 检查点文件不存在: {checkpoint_path}", file=sys.stderr)
            return 1
        if not checkpoint_path.is_file():
            print(f"错误: 指定的路径不是文件: {checkpoint_path}", file=sys.stderr)
            return 1
        checkpoint_path = str(checkpoint_path.resolve())
    
    # 导入并启动应用
    from PyQt6.QtWidgets import QApplication
    from viewer import create_main_window
    
    # 创建应用程序
    app = QApplication(sys.argv)
    app.setApplicationName("策略查看器")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Texas Hold'em AI")
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    # 创建主窗口
    window = create_main_window(checkpoint_path=checkpoint_path)
    window.show()
    
    # 显示启动消息
    if checkpoint_path:
        print(f"策略查看器已启动，正在加载检查点: {checkpoint_path}")
    else:
        print("策略查看器已启动")
    
    # 运行事件循环
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
