# 设计文档

## 概述

本设计实现策略查看器的自适应动作空间功能，使其能够动态获取模型的动作维度和动作名称，而不是使用硬编码的动作列表。同时更新手牌矩阵的显示方式，使用条状组合来可视化策略分布。

### 核心变更

1. **动作配置数据类** - 新增 `ActionConfig` 类封装动作空间信息
2. **模型加载器增强** - 从检查点自动检测动作维度和映射
3. **策略计算器重构** - 使用动态动作列表替代硬编码
4. **颜色映射器更新** - 支持新的颜色方案和任意数量动作
5. **手牌矩阵控件重构** - 实现条状组合显示

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     ActionConfig (新增)                          │
│  - action_names: List[str]                                      │
│  - action_dim: int                                              │
│  - action_colors: Dict[str, Color]                              │
├─────────────────────────────────────────────────────────────────┤
│                     ModelLoader (增强)                           │
│  - detect_action_dim() -> int                                   │
│  - get_action_config() -> ActionConfig                          │
├─────────────────────────────────────────────────────────────────┤
│                  StrategyCalculator (重构)                       │
│  - set_action_config(config: ActionConfig)                      │
│  - get_available_actions() -> List[str]                         │
├─────────────────────────────────────────────────────────────────┤
│                  StrategyColorMapper (更新)                      │
│  - get_action_color(action: str) -> Color                       │
│  - get_bar_segments(strategy: Dict) -> List[BarSegment]         │
└─────────────────────────────────────────────────────────────────┘
```

## 组件和接口


### 1. ActionConfig（动作配置）

```python
@dataclass
class ActionConfig:
    """动作空间配置。
    
    Attributes:
        action_names: 动作名称列表，按索引顺序
        action_dim: 动作空间维度
        display_names: 显示用的动作名称（可选，用于合并CHECK/CALL等）
    """
    action_names: List[str]
    action_dim: int
    display_names: Optional[Dict[str, str]] = None
    
    @classmethod
    def from_checkpoint(cls, checkpoint_data: Dict) -> 'ActionConfig':
        """从检查点数据创建配置。"""
        
    @classmethod
    def default_for_dim(cls, action_dim: int) -> 'ActionConfig':
        """根据维度创建默认配置。"""
```

### 2. 默认动作映射

```python
# 不同维度的默认动作映射
DEFAULT_ACTION_MAPPINGS = {
    4: ['FOLD', 'CHECK', 'CALL', 'RAISE'],
    5: ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG'],
    6: ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG', 'ALL_IN'],
}
```

### 3. 动作颜色方案

```python
# 动作颜色映射
ACTION_COLORS = {
    'FOLD': Color(66, 133, 244),      # 蓝色
    'CHECK': Color(52, 168, 83),      # 绿色
    'CALL': Color(52, 168, 83),       # 绿色
    'RAISE': Color(234, 67, 53),      # 红色
    'RAISE_SMALL': Color(255, 138, 128),  # 浅红色
    'RAISE_BIG': Color(234, 67, 53),      # 深红色
    'ALL_IN': Color(183, 28, 28),         # 最深红色
}
```

### 4. BarSegment（条状段）

```python
@dataclass
class BarSegment:
    """条状图的一个段。
    
    Attributes:
        action: 动作名称
        probability: 概率值
        color: 颜色
        width_ratio: 宽度比例 (0.0 - 1.0)
    """
    action: str
    probability: float
    color: Color
    width_ratio: float
```

### 5. ModelLoader 增强

```python
class ModelLoader:
    def load(self, checkpoint_path, ...) -> ModelMetadata:
        """加载模型，自动检测动作配置。"""
        # 1. 加载检查点数据
        # 2. 检测动作维度
        # 3. 读取或生成动作配置
        # 4. 存储动作配置
    
    @property
    def action_config(self) -> ActionConfig:
        """获取动作配置。"""
    
    def _detect_action_dim(self, checkpoint_data: Dict) -> int:
        """从网络权重检测动作维度。"""
```

### 6. StrategyCalculator 重构

```python
class StrategyCalculator:
    def __init__(self, action_config: Optional[ActionConfig] = None):
        """初始化，可传入动作配置。"""
        self._action_config = action_config or ActionConfig.default_for_dim(6)
    
    def set_action_config(self, config: ActionConfig) -> None:
        """设置动作配置。"""
    
    @property
    def available_actions(self) -> List[str]:
        """获取可用动作列表。"""
        return self._action_config.action_names
```

### 7. StrategyColorMapper 更新

```python
class StrategyColorMapper:
    def get_action_color(self, action: str) -> Color:
        """获取动作对应的颜色。"""
    
    def get_bar_segments(
        self, 
        strategy: Dict[str, float],
        action_order: Optional[List[str]] = None
    ) -> List[BarSegment]:
        """将策略转换为条状段列表。
        
        Args:
            strategy: 动作概率字典
            action_order: 动作显示顺序（可选）
            
        Returns:
            条状段列表，按顺序排列，概率为0的动作不包含
        """
```

## 数据模型

### 检查点元数据扩展

```python
# 检查点中新增的字段
checkpoint_data = {
    # ... 现有字段 ...
    'action_config': {
        'action_names': ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG', 'ALL_IN'],
        'action_dim': 6,
    }
}
```


## 正确性属性

*属性是系统在所有有效执行中应保持为真的特征或行为——本质上是关于系统应该做什么的形式化陈述。属性作为人类可读规范和机器可验证正确性保证之间的桥梁。*

### 属性 1: 动作维度检测正确性
*对于任意*包含策略网络或遗憾网络权重的检查点数据，`detect_action_dim`函数应返回与网络输出层维度一致的动作维度值
**验证: 需求 1.1**

### 属性 2: 动作映射正确性
*对于任意*检查点数据，如果包含`action_config`元数据则使用该配置，否则根据检测到的动作维度返回对应的默认动作映射
**验证: 需求 1.2, 1.3, 3.3**

### 属性 3: 检查点元数据完整性
*对于任意*保存的检查点，检查点数据中应包含`action_config`字段，且该字段包含`action_names`列表和`action_dim`整数值
**验证: 需求 3.1, 3.2**

### 属性 4: 策略动作数量一致性
*对于任意*模型和游戏状态，策略分析器返回的动作概率字典的键数量应等于模型的动作维度
**验证: 需求 2.1, 4.1**

### 属性 5: 动作颜色映射正确性
*对于任意*动作名称：
- FOLD动作返回蓝色系颜色（R < 100, G > 100, B > 200）
- CHECK/CALL动作返回绿色系颜色（R < 100, G > 150, B < 150）
- RAISE类动作返回红色系颜色（R > 180, G < 150, B < 150）
**验证: 需求 6.2, 6.3, 6.4**

### 属性 6: 条状宽度计算正确性
*对于任意*策略概率分布，`get_bar_segments`返回的所有段的`width_ratio`之和应等于1.0（允许浮点误差±0.001），且每个段的`width_ratio`应等于其`probability`值
**验证: 需求 6.5**

### 属性 7: 零概率动作过滤
*对于任意*策略概率分布，`get_bar_segments`返回的段列表中不应包含`probability`为0的动作
**验证: 需求 6.6**

### 属性 8: JSON导出动作完整性
*对于任意*导出的策略JSON，每个手牌的策略应包含与当前模型动作维度相同数量的动作概率
**验证: 需求 5.3**

## 错误处理

### 动作维度检测失败
- 无法从权重检测维度：使用默认维度6
- 维度不在支持范围内：抛出`UnsupportedActionDimError`

### 动作配置不一致
- 检查点中的动作维度与网络权重不匹配：使用网络权重的维度，记录警告

## 测试策略

### 属性测试
使用 Hypothesis 库进行属性测试：
- 每个属性测试运行至少100次迭代
- 测试标注格式：`**Feature: adaptive-action-viewer, Property {number}: {property_text}**`

### 单元测试
- 测试`ActionConfig`的创建和默认值
- 测试`detect_action_dim`对不同网络结构的检测
- 测试`get_bar_segments`的段计算
- 测试颜色映射的正确性

### 集成测试
- 测试加载不同动作维度的检查点
- 测试策略计算和显示的完整流程
