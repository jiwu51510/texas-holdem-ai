# 实现计划

- [x] 1. 修复 ModelLoader 的 action_dim 默认值
  - [x] 1.1 修改 `viewer/model_loader.py` 中 `load()` 方法的 `action_dim` 默认值从 4 改为 5
    - 修改函数签名中的默认参数
    - _Requirements: 1.1, 2.1_
  - [x] 1.2 编写属性测试验证 action_dim 自动检测
    - **Property 1: 自动检测 action_dim**
    - **Validates: Requirements 1.1, 1.3**

- [x] 2. 修复 StrategyAnalyzer 的 action_dim 默认值
  - [x] 2.1 修改 `analysis/strategy_analyzer.py` 中 `compare_strategies()` 方法的 `action_dim` 默认值从 4 改为 5
    - 修改函数签名中的默认参数
    - _Requirements: 2.1_

- [x] 3. 修复测试文件的 ACTION_DIM 常量
  - [x] 3.1 修改 `tests/test_deep_cfr_trainer.py` 中的 `ACTION_DIM` 常量从 4 改为 5
    - _Requirements: 2.2_

- [x] 4. 验证修复效果
  - [x] 4.1 运行现有测试确保没有回归
    - 执行 `pytest tests/test_deep_cfr_trainer.py -v`
    - 执行 `pytest tests/test_viewer_properties.py -v`
    - _Requirements: 1.2, 3.1_

- [x] 5. Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题询问用户。
