# 设计文档

## 概述

本设计文档描述了修复 regret 网络 action_dim 不一致问题的技术方案。核心问题是训练时使用 5 维动作空间，但模型加载器默认使用 4 维，导致网络权重加载失败或行为异常。

## 架构

修复涉及以下组件：

```
┌─────────────────────────────────────────────────────────────┐
│                     检查点文件 (.pt)                         │
│  - checkpoint_format: 'deep_cfr_v1'                         │
│  - regret_network_state_dict                                │
│  - policy_network_state_dict                                │
│  - action_dim: 5 (新增)                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ModelLoader                               │
│  - load() 方法自动检测 action_dim                            │
│  - 默认 action_dim 改为 5                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 RegretNetwork / PolicyNetwork                │
│  - action_dim = 5                                           │
│  - 输出: [FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG]        │
└─────────────────────────────────────────────────────────────┘
```

## 组件和接口

### 1. ModelLoader 修改

**文件**: `viewer/model_loader.py`

修改 `load()` 方法的默认参数：
- `action_dim: int = 4` → `action_dim: int = 5`

增强自动检测逻辑：
- 优先使用检查点中存储的 `action_dim` 值
- 如果检查点没有 `action_dim`，通过网络权重推断

### 2. DeepCFRTrainer 修改

**文件**: `training/deep_cfr_trainer.py`

在保存检查点时添加 `action_dim` 元数据：
```python
checkpoint_data['action_dim'] = 5
```

### 3. StrategyAnalyzer 修改

**文件**: `analysis/strategy_analyzer.py`

修改 `compare_strategies()` 方法的默认参数：
- `action_dim: int = 4` → `action_dim: int = 5`

### 4. 测试文件修改

**文件**: `tests/test_deep_cfr_trainer.py`

修改常量：
- `ACTION_DIM = 4` → `ACTION_DIM = 5`

## 数据模型

### 检查点数据结构（更新后）

```python
{
    'checkpoint_format': 'deep_cfr_v1',
    'action_dim': 5,  # 新增字段
    'regret_network_state_dict': {...},
    'policy_network_state_dict': {...},
    'episode_number': int,
    'timestamp': str,
    # ... 其他元数据
}
```

## 正确性属性

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. 
Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: 自动检测 action_dim

*For any* Deep CFR 格式的检查点，加载后的 RegretNetwork 和 PolicyNetwork 的 action_dim 应该与检查点中存储的值一致（如果存在），否则应该通过权重推断得到正确的值。

**Validates: Requirements 1.1, 1.3**

### Property 2: 网络架构一致性

*For any* 训练产生的检查点，加载后的网络架构（input_dim, hidden_dims, action_dim）应该与训练时使用的架构完全一致。

**Validates: Requirements 1.2**

### Property 3: 遗憾值维度正确性

*For any* 有效的游戏状态和加载的 Deep CFR 模型，遗憾值计算器返回的遗憾值字典应该包含所有 5 种动作类型的遗憾值。

**Validates: Requirements 3.1, 3.2**

## 错误处理

1. **维度不匹配错误**: 当检查点中的网络权重维度与创建的网络不匹配时，应该抛出 `ModelLoadError` 并提供清晰的错误信息，说明期望的维度和实际的维度。

2. **向后兼容**: 对于旧格式的检查点（action_dim=4），应该继续支持加载，并在日志中记录警告信息。

## 测试策略

### 单元测试

1. 测试 `ModelLoader.load()` 能正确检测 action_dim
2. 测试加载后的网络能正确进行前向传播
3. 测试遗憾值计算器返回正确维度的输出

### 属性测试

使用 Hypothesis 库进行属性测试：

1. **Property 1 测试**: 生成随机的检查点数据（包含不同的 action_dim 值），验证加载器能正确检测
2. **Property 2 测试**: 使用训练器创建检查点，然后加载并比较网络架构
3. **Property 3 测试**: 生成随机的游戏状态，验证遗憾值输出维度

测试框架: pytest + hypothesis

每个属性测试配置运行至少 100 次迭代。
