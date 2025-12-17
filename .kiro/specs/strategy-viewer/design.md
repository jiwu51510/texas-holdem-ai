# 设计文档

## 概述

策略查看器是一个基于 Python 的桌面应用程序，使用 PyQt6 构建图形界面。该工具允许用户加载训练好的德州扑克AI模型，通过游戏树导航浏览不同决策节点，并以13x13手牌范围矩阵的形式可视化策略分布。

### 技术选型

- **GUI框架**: PyQt6 - 成熟的跨平台GUI框架，支持丰富的自定义控件
- **数据处理**: NumPy - 高效的数值计算
- **可视化**: Matplotlib（嵌入PyQt） - 用于生成热图和导出图片
- **模型加载**: PyTorch - 与现有训练系统兼容

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Strategy Viewer                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────────────┐  ┌───────────────┐ │
│  │  Game Tree   │  │   Hand Range Matrix  │  │  Info Panel   │ │
│  │   Navigator  │  │      (13x13)         │  │               │ │
│  │              │  │                      │  │ - Board Cards │ │
│  │ Game Begin   │  │  AA AKs AQs ...      │  │ - Player Info │ │
│  │  └─Preflop   │  │  AKo KK  KQs ...     │  │ - Pot/Stack   │ │
│  │    └─P2      │  │  AQo KQo QQ  ...     │  │ - Stage       │ │
│  │      └─Fold  │  │  ...                 │  │               │ │
│  │      └─Call  │  │                      │  │ ┌───────────┐ │ │
│  │      └─Raise │  │                      │  │ │  Detail   │ │ │
│  │              │  │                      │  │ │  Panel    │ │ │
│  └──────────────┘  └──────────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 分层架构

```
┌─────────────────────────────────────────┐
│           Presentation Layer            │
│  (PyQt6 Widgets, Event Handlers)        │
├─────────────────────────────────────────┤
│           Application Layer             │
│  (StrategyViewerController)             │
├─────────────────────────────────────────┤
│            Domain Layer                 │
│  (GameTreeNode, HandRangeCalculator)    │
├─────────────────────────────────────────┤
│         Infrastructure Layer            │
│  (ModelLoader, StrategyAnalyzer)        │
└─────────────────────────────────────────┘
```

## 组件和接口

### 1. GameTreeNode（游戏树节点）

```python
@dataclass
class GameTreeNode:
    """游戏树节点，表示一个决策点。
    
    Attributes:
        node_id: 节点唯一标识符
        stage: 游戏阶段（preflop/flop/turn/river）
        player: 当前行动玩家（0或1）
        action: 到达此节点的行动（可选）
        parent: 父节点（可选）
        children: 子节点列表
        pot: 当前底池大小
        stacks: 玩家筹码列表
        board_cards: 公共牌列表
        action_history: 行动历史
    """
```

### 2. GameTreeNavigator（游戏树导航器）

```python
class GameTreeNavigator:
    """管理游戏树的导航和状态。
    
    Methods:
        get_root() -> GameTreeNode: 获取根节点
        get_children(node: GameTreeNode) -> List[GameTreeNode]: 获取子节点
        navigate_to(node: GameTreeNode) -> None: 导航到指定节点
        get_current_node() -> GameTreeNode: 获取当前节点
        get_path_to_root() -> List[GameTreeNode]: 获取到根节点的路径
        get_available_actions() -> List[str]: 获取当前可用行动
    """
```

### 3. HandRangeCalculator（手牌范围计算器）

```python
class HandRangeCalculator:
    """计算和管理手牌范围的策略。
    
    Methods:
        calculate_range_strategies(node: GameTreeNode) -> Dict[str, Dict[str, float]]:
            计算所有169种手牌组合的策略分布
            返回: {手牌标签: {行动: 概率}}
        
        get_hand_label(card1: Card, card2: Card) -> str:
            获取手牌的标准标签（如"AKs", "AKo", "AA"）
        
        get_all_hand_combinations(hand_label: str) -> List[Tuple[Card, Card]]:
            获取某个手牌标签的所有具体花色组合
        
        filter_by_board(combinations: List, board: List[Card]) -> List:
            过滤与公共牌冲突的手牌组合
    """
```

### 4. StrategyColorMapper（策略颜色映射器）

```python
class StrategyColorMapper:
    """将策略概率映射为颜色。
    
    Methods:
        get_cell_color(strategy: Dict[str, float]) -> QColor:
            根据策略分布返回单元格颜色
            - 纯绿色: 高概率加注/跟注
            - 纯红色: 高概率弃牌
            - 混合色: 混合策略
        
        get_action_color(action: str) -> QColor:
            获取特定行动的颜色
    """
```

### 5. StrategyViewerController（策略查看器控制器）

```python
class StrategyViewerController:
    """协调各组件的主控制器。
    
    Methods:
        load_model(checkpoint_path: str) -> bool: 加载模型
        navigate_to_node(node: GameTreeNode) -> None: 导航到节点
        set_board_cards(cards: List[Card]) -> None: 设置公共牌
        get_hand_strategy(hand_label: str) -> Dict[str, float]: 获取手牌策略
        export_image(path: str) -> None: 导出图片
        export_json(path: str) -> None: 导出JSON
    """
```

### 6. UI组件

#### MainWindow（主窗口）
```python
class MainWindow(QMainWindow):
    """主窗口，包含所有UI组件。"""
```

#### GameTreeWidget（游戏树控件）
```python
class GameTreeWidget(QTreeWidget):
    """显示和交互游戏树的控件。
    
    Signals:
        node_selected(GameTreeNode): 节点被选中时发出
    """
```

#### HandRangeMatrixWidget（手牌矩阵控件）
```python
class HandRangeMatrixWidget(QWidget):
    """显示13x13手牌范围矩阵的控件。
    
    Signals:
        cell_hovered(str): 鼠标悬停在单元格上时发出手牌标签
        cell_clicked(str): 单元格被点击时发出手牌标签
    """
```

#### InfoPanelWidget（信息面板控件）
```python
class InfoPanelWidget(QWidget):
    """显示游戏状态信息的控件。"""
```

#### StrategyDetailWidget（策略详情控件）
```python
class StrategyDetailWidget(QWidget):
    """显示单个手牌详细策略的控件。"""
```

#### BoardCardSelector（公共牌选择器）
```python
class BoardCardSelector(QWidget):
    """选择公共牌的控件。
    
    Signals:
        board_changed(List[Card]): 公共牌改变时发出
    """
```

## 数据模型

### 手牌范围矩阵数据结构

```python
# 13x13矩阵，行和列分别代表两张牌的rank
# 对角线: 对子 (AA, KK, QQ, ...)
# 上三角: 同花 (AKs, AQs, ...)
# 下三角: 非同花 (AKo, AQo, ...)

RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

# 手牌标签矩阵
HAND_LABELS = [
    ['AA',  'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s'],
    ['AKo', 'KK',  'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s'],
    ['AQo', 'KQo', 'QQ',  'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s'],
    # ... 继续到 22
]
```

### 策略数据结构

```python
@dataclass
class HandStrategy:
    """单个手牌的策略信息。
    
    Attributes:
        hand_label: 手牌标签（如"AKs"）
        combinations: 具体花色组合列表
        action_probabilities: 各行动的概率 {action: probability}
        combination_strategies: 每个具体组合的策略（可选）
    """
    hand_label: str
    combinations: List[Tuple[Card, Card]]
    action_probabilities: Dict[str, float]
    combination_strategies: Optional[Dict[str, Dict[str, float]]] = None
```

### 节点状态数据结构

```python
@dataclass  
class NodeState:
    """节点的完整状态信息。
    
    Attributes:
        stage: 游戏阶段
        current_player: 当前玩家
        total_players: 总玩家数
        pot: 底池大小
        stacks: 玩家筹码
        board_cards: 公共牌
        action_history: 行动历史
    """
    stage: GameStage
    current_player: int
    total_players: int
    pot: int
    stacks: List[int]
    board_cards: List[Card]
    action_history: List[Action]
```

## 正确性属性

*属性是系统在所有有效执行中应保持为真的特征或行为——本质上是关于系统应该做什么的形式化陈述。属性作为人类可读规范和机器可验证正确性保证之间的桥梁。*

### 属性 1: 手牌标签位置正确性
*对于任意*两张牌的组合，`get_hand_label`函数应返回正确的标签格式和矩阵位置：
- 对子（两张牌rank相同）返回"XX"格式，位于矩阵对角线
- 同花（两张牌suit相同且rank不同）返回"XYs"格式，位于矩阵上三角
- 非同花（两张牌suit不同且rank不同）返回"XYo"格式，位于矩阵下三角
**验证: 需求 3.3, 3.4, 3.5**

### 属性 2: 策略概率归一化
*对于任意*手牌的策略分布，所有行动概率之和应等于1.0（允许浮点误差±0.001）
**验证: 需求 4.4**

### 属性 3: 手牌组合完整性
*对于任意*手牌标签，`get_all_hand_combinations`返回的组合数量应正确：
- 对子（如"AA"）返回6种组合（C(4,2) = 6）
- 同花（如"AKs"）返回4种组合（4种花色）
- 非同花（如"AKo"）返回12种组合（4×3 = 12）
**验证: 需求 4.3**

### 属性 4: 公共牌过滤正确性
*对于任意*手牌组合列表和公共牌列表，`filter_by_board`应排除所有与公共牌有重复牌的组合，且保留的组合中不存在任何与公共牌相同的牌
**验证: 需求 6.3**

### 属性 5: 游戏树路径一致性
*对于任意*游戏树节点，从该节点到根节点的路径应是唯一的，且路径长度等于该节点的action_history长度加1
**验证: 需求 2.4**

### 属性 6: 颜色映射确定性
*对于任意*相同的策略分布输入，`get_cell_color`应返回完全相同的颜色值（纯函数特性）
**验证: 需求 3.2**

### 属性 7: JSON导出完整性
*对于任意*导出的策略JSON文件，应包含所有169种手牌标签，且每个手牌的策略概率之和为1.0
**验证: 需求 8.2**

### 属性 8: 无效检查点处理
*对于任意*无效或损坏的检查点文件，模型加载应抛出适当的异常而不是崩溃，且系统状态保持不变
**验证: 需求 1.2**

## 错误处理

### 模型加载错误
- 文件不存在：显示"检查点文件不存在"错误
- 格式无效：显示"检查点格式无效"错误
- 版本不兼容：显示"模型版本不兼容"错误

### 导航错误
- 无效节点：忽略操作，保持当前状态
- 空子节点：显示"该节点无可用行动"提示

### 导出错误
- 路径无效：显示"无法写入指定路径"错误
- 权限不足：显示"没有写入权限"错误

## 测试策略

### 单元测试
- 测试`HandRangeCalculator`的手牌标签生成
- 测试`StrategyColorMapper`的颜色映射
- 测试`GameTreeNavigator`的导航逻辑
- 测试数据模型的验证逻辑

### 属性测试
使用 Hypothesis 库进行属性测试：
- 每个属性测试运行至少100次迭代
- 测试标注格式：`**Feature: strategy-viewer, Property {number}: {property_text}**`

### 集成测试
- 测试模型加载和策略计算的完整流程
- 测试UI组件之间的交互
- 测试导出功能的完整性
