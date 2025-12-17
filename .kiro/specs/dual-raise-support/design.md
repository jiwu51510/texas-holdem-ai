# 双加注大小支持设计文档

## 架构变更

### 1. 神经网络层 (`models/networks.py`)

**变更：**
- `PolicyNetwork` 和 `RegretNetwork` 的 `action_dim` 从 4 改为 5
- 保持网络结构不变，只修改输出维度

```python
# 新的动作维度
action_dim = 5  # FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG
```

### 2. 训练器 (`training/deep_cfr_trainer.py`)

**变更：**
- 更新 `ACTION_TYPE_TO_IDX` 映射
- 更新 `IDX_TO_ACTION_TYPE` 映射
- 修改 `action_dim` 初始化

```python
ACTION_TYPE_TO_IDX = {
    ActionType.FOLD: 0,
    ActionType.CHECK: 1,
    ActionType.CALL: 2,
    ActionType.RAISE_SMALL: 3,  # 新增
    ActionType.RAISE_BIG: 4     # 新增
}
```

### 3. 核心模型 (`models/core.py`)

**变更：**
- 在 `ActionType` 枚举中添加 `RAISE_SMALL` 和 `RAISE_BIG`
- 保留原有的 `RAISE` 用于向后兼容

```python
class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"        # 保留用于兼容
    RAISE_SMALL = "raise_small"  # 新增
    RAISE_BIG = "raise_big"      # 新增
```

### 4. 游戏环境 (`environment/poker_environment.py`)

**变更：**
- 修改 `get_legal_actions()` 方法
- 生成 RAISE_SMALL 和 RAISE_BIG 两种加注动作
- 实现正确的底池计算逻辑

```python
def get_legal_actions(self, state):
    # ... 现有逻辑 ...
    
    # 计算底池（考虑跟注金额）
    bet_to_call = opponent_bet - current_bet
    effective_pot = state.pot + bet_to_call if bet_to_call > 0 else state.pot
    
    # 半底池加注
    half_pot_raise = bet_to_call + effective_pot // 2
    if half_pot_raise <= player_stack:
        legal_actions.append(Action(ActionType.RAISE_SMALL, half_pot_raise))
    
    # 全底池加注
    full_pot_raise = bet_to_call + effective_pot
    if full_pot_raise <= player_stack:
        legal_actions.append(Action(ActionType.RAISE_BIG, full_pot_raise))
```

### 5. 遗憾值计算器 (`viewer/regret_calculator.py`)

**变更：**
- 更新 `ACTION_NAMES` 为 5 个动作
- 更新 `ACTION_INDEX` 映射

```python
ACTION_NAMES = ['弃牌', '过牌', '跟注', '小加注', '大加注']
ACTION_INDEX = {'弃牌': 0, '过牌': 1, '跟注': 2, '小加注': 3, '大加注': 4}
```

### 6. 策略计算器 (`viewer/strategy_calculator.py`)

**变更：**
- 更新 `DEFAULT_ACTIONS` 列表
- 移除 50/50 分割逻辑

```python
DEFAULT_ACTIONS = ['FOLD', 'CHECK/CALL', 'RAISE_SMALL', 'RAISE_BIG']
```

### 7. 策略分析器 (`analysis/strategy_analyzer.py`)

**变更：**
- 更新 `ACTION_NAMES` 映射
- 修改 `analyze_state()` 方法，直接使用网络输出的 5 维概率

## 向后兼容性

1. 加载旧检查点时检测 `action_dim`
2. 如果是 4 维输出，将 RAISE 概率平均分配给 RAISE_SMALL 和 RAISE_BIG
3. 新训练的模型使用 5 维输出

## 测试计划

1. 单元测试：验证新的动作映射
2. 集成测试：验证训练流程
3. 回归测试：验证旧检查点加载
