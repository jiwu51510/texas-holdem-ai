# 双加注大小支持任务列表

## 任务状态说明
- [ ] 未开始
- [x] 已完成
- [~] 进行中

## 任务列表

### 阶段 1: 核心模型变更

- [ ] **T1.1** 在 `models/core.py` 中添加 `RAISE_SMALL` 和 `RAISE_BIG` 到 `ActionType` 枚举
  - 复杂度: 低
  - 依赖: 无

- [ ] **T1.2** 修改 `models/networks.py`，将默认 `action_dim` 从 4 改为 5
  - 复杂度: 低
  - 依赖: T1.1

### 阶段 2: 游戏环境变更

- [ ] **T2.1** 修改 `environment/poker_environment.py` 的 `get_legal_actions()` 方法
  - 生成 RAISE_SMALL（半底池）和 RAISE_BIG（全底池）
  - 实现正确的底池计算逻辑
  - 复杂度: 中
  - 依赖: T1.1

### 阶段 3: 训练器变更

- [ ] **T3.1** 修改 `training/deep_cfr_trainer.py` 的动作映射
  - 更新 `ACTION_TYPE_TO_IDX` 和 `IDX_TO_ACTION_TYPE`
  - 修改 `action_dim` 初始化为 5
  - 复杂度: 中
  - 依赖: T1.1, T1.2

### 阶段 4: 查看器变更

- [ ] **T4.1** 修改 `viewer/regret_calculator.py`
  - 更新 `ACTION_NAMES` 和 `ACTION_INDEX`
  - 复杂度: 低
  - 依赖: T1.1

- [ ] **T4.2** 修改 `viewer/strategy_calculator.py`
  - 更新 `DEFAULT_ACTIONS`
  - 移除 50/50 分割逻辑
  - 复杂度: 低
  - 依赖: T1.1

- [ ] **T4.3** 修改 `analysis/strategy_analyzer.py`
  - 更新动作映射
  - 修改 `analyze_state()` 方法
  - 复杂度: 中
  - 依赖: T1.1

### 阶段 5: 测试和验证

- [ ] **T5.1** 更新相关测试文件
  - 复杂度: 中
  - 依赖: T1-T4 全部完成

- [ ] **T5.2** 运行完整测试套件
  - 复杂度: 低
  - 依赖: T5.1

## 注意事项

1. 现有检查点将不兼容新的 5 维输出
2. 需要重新训练模型
3. 策略分析器需要支持加载旧格式检查点（4 维）
