# 德州扑克AI训练系统

一个完整的德州扑克单挑（Heads-Up）AI训练平台，使用 Deep CFR 算法训练能够做出最优决策的AI代理。

## 项目简介

本系统采用 **Deep CFR**（Deep Counterfactual Regret Minimization，深度反事实遗憾最小化）算法训练德州扑克AI。Deep CFR 是 Facebook AI Research 在 2019 年提出的算法，成功应用于 Libratus 和 Pluribus 等超人类扑克 AI。

### Deep CFR 架构

系统使用两个神经网络：

- **遗憾网络（Regret Network）**：学习每个动作的即时遗憾值，通过 Regret Matching 生成策略
- **策略网络（Policy Network）**：学习长期平均策略，用于最终部署

训练流程：
1. 使用遗憾网络生成策略进行自博弈
2. 精确计算反事实遗憾值
3. 将样本存储到蓄水池缓冲区
4. 批量训练两个网络

系统提供完整的训练、评估、分析和可视化功能。

### 主要功能

- **模型训练**：自我对弈训练，支持检查点保存和恢复
- **游戏环境**：完整的德州扑克规则实现，包括手牌评估、行动验证
- **实时监控**：训练过程中的指标收集和可视化
- **策略分析**：分析训练好的模型策略，生成决策解释和热图
- **模型评估**：对抗多种基准策略评估模型性能
- **命令行界面**：方便的CLI工具进行训练、评估和分析

## 项目结构

```
.
├── models/          # 核心数据模型和神经网络
│   ├── core.py      # Card, Action, GameState, TrainingConfig等
│   └── networks.py  # RegretNetwork, PolicyNetwork（Deep CFR）
├── environment/     # 游戏环境模块
│   ├── hand_evaluator.py    # 手牌评估器
│   ├── rule_engine.py       # 规则引擎
│   ├── poker_environment.py # 游戏环境
│   └── state_encoder.py     # 状态编码器
├── training/        # 训练模块
│   ├── deep_cfr_trainer.py  # Deep CFR 训练器
│   ├── reservoir_buffer.py  # 蓄水池缓冲区
│   ├── training_engine.py   # 训练引擎
│   ├── cfr_trainer.py       # 基础 CFR 训练器
│   └── parallel_trainer.py  # 并行训练支持
├── monitoring/      # 监控模块
│   ├── metrics_collector.py # 指标收集器
│   └── training_monitor.py  # 训练监控器
├── analysis/        # 分析模块
│   ├── evaluator.py         # 模型评估器
│   └── strategy_analyzer.py # 策略分析器
├── utils/           # 工具模块
│   ├── checkpoint_manager.py # 检查点管理
│   ├── config_manager.py     # 配置管理
│   ├── data_logger.py        # 数据日志
│   └── exceptions.py         # 自定义异常
├── viewer/          # 策略查看器（GUI）
├── examples/        # 示例脚本
├── configs/         # 配置文件示例
├── tests/           # 测试套件
├── cli.py           # 命令行界面
└── requirements.txt # Python依赖
```

## 安装说明

### 环境要求

- Python 3.9+
- PyTorch 2.0+

### 安装步骤

1. 克隆项目：
```bash
git clone <repository-url>
cd texas-holdem-ai
```

2. 创建虚拟环境（推荐）：
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

### 依赖列表

- `torch>=2.0.0` - 深度学习框架
- `numpy>=1.21.0` - 数值计算
- `hypothesis>=6.0.0` - 属性测试
- `pytest>=7.0.0` - 单元测试
- `matplotlib>=3.5.0` - 可视化（可选）

## 快速开始

### 1. 运行示例脚本

```bash
# 训练示例
python3 examples/demo_training.py

# 评估示例
python3 examples/demo_evaluation.py

# 策略分析示例
python3 examples/demo_strategy_analysis.py
```

### 2. 使用命令行界面

```bash
# 查看帮助
python3 cli.py --help

# 启动训练
python3 cli.py train --config configs/default_config.json --episodes 1000

# 评估模型
python3 cli.py evaluate --model checkpoints/checkpoint.pt --games 100

# 分析策略
python3 cli.py analyze --model checkpoints/checkpoint.pt
```

### 3. 在代码中使用

#### 基础训练示例

```python
from models.core import TrainingConfig
from training.training_engine import TrainingEngine

# 创建训练配置
config = TrainingConfig(
    learning_rate=0.001,
    batch_size=2048,
    num_episodes=10000,
    initial_stack=1000,
    small_blind=5,
    big_blind=10
)

# 初始化训练引擎
engine = TrainingEngine(config, checkpoint_dir="checkpoints")

# 开始训练
results = engine.train()

print(f"训练完成！胜率: {results['win_rate']:.2%}")
```

#### Deep CFR 训练示例

```python
from models.core import TrainingConfig
from training.training_engine import TrainingEngine
from training.deep_cfr_trainer import DeepCFRTrainer

# 创建 Deep CFR 配置
config = TrainingConfig(
    learning_rate=0.001,
    batch_size=2048,
    num_episodes=100000,
    initial_stack=1000,
    small_blind=5,
    big_blind=10,
    # Deep CFR 特有参数
    regret_buffer_size=2000000,      # 遗憾缓冲区大小
    strategy_buffer_size=2000000,    # 策略缓冲区大小
    cfr_iterations_per_update=1000,  # 每次网络更新前的 CFR 迭代次数
    network_train_steps=4000         # 每次更新的训练步数
)

# 初始化训练引擎（自动使用 Deep CFR）
engine = TrainingEngine(config, checkpoint_dir="checkpoints")

# 开始训练
results = engine.train()

print(f"训练完成！")
print(f"遗憾网络损失: {results.get('regret_loss', 'N/A')}")
print(f"策略网络损失: {results.get('policy_loss', 'N/A')}")
```

#### 使用遗憾网络生成策略

```python
import torch
from models.networks import RegretNetwork

# 加载遗憾网络
regret_network = RegretNetwork(input_dim=370, action_dim=5)

# 假设 state_encoding 是370维的状态编码
state_encoding = torch.randn(370)

# 获取遗憾值
regrets = regret_network(state_encoding)
print(f"遗憾值: {regrets}")

# 使用 Regret Matching 获取策略
strategy = regret_network.get_strategy(state_encoding)
print(f"策略概率: {strategy}")
# 输出示例: tensor([0.25, 0.35, 0.20, 0.20])
# 分别对应: 弃牌、过牌/跟注、加注、全押
```

## 命令行使用说明

### train - 训练命令

启动AI模型训练。

```bash
python3 cli.py train [选项]
```

选项：
- `--config, -c`: 配置文件路径
- `--episodes, -e`: 训练回合数
- `--checkpoint-dir`: 检查点保存目录
- `--resume`: 从检查点恢复训练
- `--learning-rate`: 学习率
- `--batch-size`: 批次大小

示例：
```bash
# 使用配置文件训练
python3 cli.py train --config configs/default_config.json

# 指定回合数训练
python3 cli.py train --episodes 5000 --checkpoint-dir my_checkpoints

# 从检查点恢复训练
python3 cli.py train --resume checkpoints/checkpoint_1000.pt --episodes 5000
```

### evaluate - 评估命令

评估训练好的模型性能。

```bash
python3 cli.py evaluate [选项]
```

选项：
- `--model, -m`: 模型检查点路径
- `--opponent, -o`: 对手策略（random/fixed/call_only）
- `--games, -g`: 评估对局数
- `--output`: 结果输出文件

示例：
```bash
# 对随机策略评估
python3 cli.py evaluate --model checkpoints/checkpoint.pt --opponent random --games 100

# 保存评估结果
python3 cli.py evaluate --model checkpoints/checkpoint.pt --output results.json
```

### analyze - 分析命令

分析模型策略。

```bash
python3 cli.py analyze [选项]
```

选项：
- `--model, -m`: 模型检查点路径
- `--state`: 游戏状态描述
- `--output`: 输出目录
- `--heatmap`: 生成策略热图

示例：
```bash
# 分析模型策略
python3 cli.py analyze --model checkpoints/checkpoint.pt

# 生成策略热图
python3 cli.py analyze --model checkpoints/checkpoint.pt --heatmap --output analysis/
```

### list - 列出检查点

列出所有可用的检查点。

```bash
python3 cli.py list [选项]
```

选项：
- `--checkpoint-dir`: 检查点目录

示例：
```bash
python3 cli.py list --checkpoint-dir checkpoints
```

## 配置参数说明

训练配置使用JSON格式，包含以下参数：

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `learning_rate` | float | 0.001 | 学习率 |
| `batch_size` | int | 2048 | 批次大小（Deep CFR 通常使用较大批次） |
| `num_episodes` | int | 10000 | 训练回合数 |
| `discount_factor` | float | 1.0 | 折扣因子（CFR 通常不使用折扣） |
| `network_architecture` | list | [512, 256, 128] | 网络隐藏层维度 |
| `checkpoint_interval` | int | 1000 | 检查点保存间隔 |
| `num_parallel_envs` | int | 1 | 并行环境数量 |
| `initial_stack` | int | 1000 | 初始筹码 |
| `small_blind` | int | 5 | 小盲注 |
| `big_blind` | int | 10 | 大盲注 |

### Deep CFR 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `regret_buffer_size` | int | 2000000 | 遗憾缓冲区大小 |
| `strategy_buffer_size` | int | 2000000 | 策略缓冲区大小 |
| `cfr_iterations_per_update` | int | 1000 | 每次网络更新前的 CFR 迭代次数 |
| `network_train_steps` | int | 4000 | 每次更新的训练步数 |

### 配置文件示例

```json
{
  "learning_rate": 0.001,
  "batch_size": 2048,
  "num_episodes": 10000,
  "discount_factor": 1.0,
  "network_architecture": [512, 256, 128],
  "checkpoint_interval": 1000,
  "num_parallel_envs": 1,
  "initial_stack": 1000,
  "small_blind": 5,
  "big_blind": 10,
  "regret_buffer_size": 2000000,
  "strategy_buffer_size": 2000000,
  "cfr_iterations_per_update": 1000,
  "network_train_steps": 4000
}
```

### 预设配置文件

项目提供了几个预设配置文件：

- `configs/default_config.json` - 默认配置，适合一般训练
- `configs/quick_training_config.json` - 快速训练配置，用于测试
- `configs/large_model_config.json` - 大模型配置，用于深度训练
- `configs/high_stakes_config.json` - 高筹码配置，模拟高额游戏

## 核心数据模型

### Card（扑克牌）

```python
from models.core import Card

# 创建一张牌
ace_hearts = Card(rank=14, suit='h')  # 红心A
king_spades = Card(rank=13, suit='s')  # 黑桃K

# rank: 2-14 (2-10, J=11, Q=12, K=13, A=14)
# suit: 'h'(红心), 'd'(方块), 'c'(梅花), 's'(黑桃)
```

### Action（行动）

```python
from models.core import Action, ActionType

fold = Action(ActionType.FOLD)           # 弃牌
check = Action(ActionType.CHECK)         # 过牌
call = Action(ActionType.CALL)           # 跟注
raise_action = Action(ActionType.RAISE, amount=50)  # 加注50
```

### GameState（游戏状态）

```python
from models.core import GameState, GameStage, Card

state = GameState(
    player_hands=[(Card(14, 'h'), Card(13, 'h')), (Card(10, 'd'), Card(9, 'd'))],
    community_cards=[Card(12, 'h'), Card(11, 'h'), Card(2, 'c')],
    pot=150,
    player_stacks=[925, 925],
    current_bets=[75, 75],
    button_position=0,
    stage=GameStage.FLOP,
    current_player=1
)
```

## 运行测试

```bash
# 运行所有测试
python3 -m pytest tests/ -v

# 运行特定测试文件
python3 -m pytest tests/test_hand_evaluator.py -v

# 运行属性测试（更多迭代）
python3 -m pytest tests/ -v --hypothesis-seed=42
```

## 技术栈

- **深度学习框架**：PyTorch
- **数值计算**：NumPy
- **测试框架**：pytest, Hypothesis
- **可视化**：matplotlib
- **算法**：Deep CFR（反事实遗憾最小化）

## 算法说明

### Deep CFR（Deep Counterfactual Regret Minimization）

Deep CFR 是一种将 CFR 算法与深度神经网络结合的方法，用于求解大规模不完全信息博弈。

#### 核心概念

- **反事实遗憾值（Counterfactual Regret）**：假设玩家以概率1到达当前节点时，选择某个动作相对于当前策略期望收益的差值
- **Regret Matching**：根据累积正遗憾值比例选择动作的策略
- **蓄水池采样（Reservoir Sampling）**：保持缓冲区固定大小的采样算法

#### 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                    Deep CFR 训练循环                         │
├─────────────────────────────────────────────────────────────┤
│  1. 使用遗憾网络生成策略（Regret Matching）                   │
│  2. 自博弈打多局，遍历游戏树                                  │
│  3. 精确计算反事实遗憾值                                      │
│  4. 存储样本到遗憾缓冲区和策略缓冲区                          │
│  5. 达到迭代次数后，训练遗憾网络（MSE损失）                   │
│  6. 训练策略网络（交叉熵损失）                                │
│  7. 重复直到收敛                                             │
└─────────────────────────────────────────────────────────────┘
```

### 神经网络架构

#### 遗憾网络（Regret Network）
- **输入**：游戏状态编码（370维）
- **隐藏层**：[512, 256, 128]，ReLU激活
- **输出**：每个动作的遗憾值（4维），无激活函数
- **用途**：学习即时遗憾值，通过 Regret Matching 生成策略

#### 策略网络（Policy Network）
- **输入**：游戏状态编码（370维）
- **隐藏层**：[512, 256, 128]，ReLU激活
- **输出**：动作概率分布（4维），Softmax激活
- **用途**：学习长期平均策略，用于最终部署

### 状态编码

状态编码维度：370维
- 手牌编码：104维（2张牌 × 52维one-hot）
- 公共牌编码：260维（5张牌 × 52维one-hot）
- 筹码信息：4维
- 位置信息：2维

### Regret Matching 算法

```python
def regret_matching(regrets):
    """将遗憾值转换为策略概率。"""
    positive_regrets = max(regrets, 0)
    regret_sum = sum(positive_regrets)
    
    if regret_sum > 0:
        return positive_regrets / regret_sum
    else:
        return uniform_distribution
```

## Potential-Aware 卡牌抽象

### 概述

翻后（Postflop）阶段由于每个不同的公共牌面都代表一个新的游戏状态，策略学习结果非常稀疏。本系统引入 **Potential-Aware 抽象**方法，通过考虑手牌在所有未来轮次的强度分布轨迹（而非仅考虑最终轮的强度分布），将相似的信息集聚类在一起，从而大幅减少需要学习的状态空间，加速训练收敛。

### 与传统抽象的区别

传统的 Distribution-Aware 抽象方法仅考虑手牌在最终轮（河牌）的强度分布。然而，两手牌可能在河牌阶段有相似的强度分布，但在中间轮次实现强度的方式完全不同：

- **手牌A**：翻牌后有顶对，在转牌和河牌阶段强度稳定
- **手牌B**：翻牌后有听牌，在转牌阶段可能变得很强或很弱

Potential-Aware 抽象通过考虑手牌在所有未来轮次的强度分布轨迹，能够区分这两种情况。

### 使用抽象生成命令

```bash
# 生成默认配置的抽象
python3 cli.py generate-abstraction --output abstractions/default

# 自定义桶数量
python3 cli.py generate-abstraction --output abstractions/custom \
    --flop-buckets 1000 \
    --turn-buckets 2000 \
    --river-buckets 3000

# 使用特定随机种子（用于可重复性）
python3 cli.py generate-abstraction --output abstractions/seed42 --seed 42

# 禁用 Potential-Aware，使用传统 Distribution-Aware
python3 cli.py generate-abstraction --output abstractions/dist_aware --no-potential-aware
```

### 查看抽象信息

```bash
# 显示抽象配置和统计信息
python3 cli.py abstraction-info --path abstractions/default

# 显示详细信息
python3 cli.py abstraction-info --path abstractions/default --verbose

# 以 JSON 格式输出
python3 cli.py abstraction-info --path abstractions/default --json
```

### 抽象配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output` | (必需) | 抽象结果输出目录 |
| `--flop-buckets` | 5000 | 翻牌阶段桶数量 |
| `--turn-buckets` | 5000 | 转牌阶段桶数量 |
| `--river-buckets` | 5000 | 河牌阶段桶数量 |
| `--preflop-buckets` | 169 | 翻牌前桶数量（169=无抽象） |
| `--equity-bins` | 50 | Equity 直方图区间数 |
| `--potential-aware` | 启用 | 使用 Potential-Aware 抽象 |
| `--no-potential-aware` | - | 禁用 Potential-Aware，使用 Distribution-Aware |
| `--seed` | 42 | 随机种子（用于可重复性） |
| `--kmeans-restarts` | 25 | k-means 重启次数 |
| `--workers` | 0 | 并行工作进程数（0=自动） |

### 在训练中使用抽象

```python
from models.core import TrainingConfig
from abstraction.data_classes import AbstractionConfig
from training.training_engine import TrainingEngine

# 创建抽象配置
abstraction_config = AbstractionConfig(
    flop_buckets=1000,
    turn_buckets=2000,
    river_buckets=3000,
    use_potential_aware=True
)

# 创建训练配置
config = TrainingConfig(
    learning_rate=0.001,
    batch_size=2048,
    num_episodes=10000,
    use_abstraction=True,
    abstraction_config=abstraction_config,
    abstraction_path="abstractions/default"  # 预计算的抽象路径
)

# 初始化训练引擎
engine = TrainingEngine(config, checkpoint_dir="checkpoints")

# 开始训练
results = engine.train()
```

### 性能优化建议

1. **桶数量选择**：
   - 更多的桶 = 更精确的抽象，但训练更慢
   - 建议从较少的桶开始（如 1000），逐步增加
   - 翻牌阶段通常需要更多的桶

2. **并行计算**：
   - 使用 `--workers` 参数指定并行进程数
   - 默认使用所有 CPU 核心

3. **进度保存**：
   - 抽象生成支持中断后继续
   - 检查点自动保存在输出目录

4. **内存优化**：
   - 大规模抽象可能需要较多内存
   - 可以分阶段生成并保存

## 许可证

MIT License

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m '添加某个功能'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request
