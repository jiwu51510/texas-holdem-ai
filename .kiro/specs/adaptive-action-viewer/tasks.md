# 实现计划

- [x] 1. 创建动作配置数据类和工具
  - [x] 1.1 在 `viewer/models.py` 中添加 `ActionConfig` 和 `BarSegment` 数据类
    - 实现 `from_checkpoint` 和 `default_for_dim` 类方法
    - 定义默认动作映射 `DEFAULT_ACTION_MAPPINGS`
    - _需求: 1.2, 1.3, 3.3_
  - [x] 1.2 编写属性测试：动作映射正确性
    - **Property 2: 动作映射正确性**
    - **验证: 需求 1.2, 1.3, 3.3**

- [x] 2. 更新模型加载器支持动作配置检测
  - [x] 2.1 在 `viewer/model_loader.py` 中增强 `ModelLoader` 类
    - 添加 `_detect_action_dim` 方法从网络权重检测维度
    - 添加 `action_config` 属性
    - 修改 `load` 方法自动检测和存储动作配置
    - _需求: 1.1, 1.2, 1.3, 1.4_
  - [x] 2.2 编写属性测试：动作维度检测正确性
    - **Property 1: 动作维度检测正确性**
    - **验证: 需求 1.1**

- [x] 3. 更新颜色映射器支持新颜色方案
  - [x] 3.1 在 `viewer/color_mapper.py` 中更新 `StrategyColorMapper` 类
    - 添加 `ACTION_COLORS` 常量定义新颜色方案
    - 更新 `get_action_color` 方法支持所有动作类型
    - 添加 `get_bar_segments` 方法生成条状段列表
    - _需求: 6.2, 6.3, 6.4, 6.5, 6.6_
  - [x] 3.2 编写属性测试：动作颜色映射正确性
    - **Property 5: 动作颜色映射正确性**
    - **验证: 需求 6.2, 6.3, 6.4**
  - [x] 3.3 编写属性测试：条状宽度计算正确性
    - **Property 6: 条状宽度计算正确性**
    - **验证: 需求 6.5**
  - [x] 3.4 编写属性测试：零概率动作过滤
    - **Property 7: 零概率动作过滤**
    - **验证: 需求 6.6**


- [x] 4. 重构策略计算器使用动态动作列表
  - [x] 4.1 在 `viewer/strategy_calculator.py` 中重构 `StrategyCalculator` 类
    - 移除硬编码的 `DEFAULT_ACTIONS` 和 `ACTION_INDEX`
    - 添加 `set_action_config` 方法
    - 添加 `available_actions` 属性
    - 更新策略计算逻辑使用动态动作列表
    - _需求: 2.1, 2.2, 2.3, 2.4_
  - [x] 4.2 编写属性测试：策略动作数量一致性
    - **Property 4: 策略动作数量一致性**
    - **验证: 需求 2.1, 4.1**

- [x] 5. 更新策略分析器支持动态动作
  - [x] 5.1 在 `analysis/strategy_analyzer.py` 中更新 `StrategyAnalyzer` 类
    - 移除硬编码的 `ACTION_NAMES` 和 `DISPLAY_ACTION_NAMES`
    - 添加 `_action_config` 属性
    - 更新 `analyze_state` 方法返回动态动作列表
    - _需求: 4.1, 4.2_

- [x] 6. Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 7. 更新控制器集成动作配置
  - [x] 7.1 在 `viewer/controller.py` 中更新 `StrategyViewerController` 类
    - 在 `load_model` 方法中获取并传递动作配置
    - 添加 `get_action_config` 方法
    - 更新策略计算器和颜色映射器的初始化
    - _需求: 1.4, 2.1_

- [x] 8. 更新手牌矩阵控件实现条状显示
  - [x] 8.1 在 `viewer/widgets/hand_matrix_widget.py` 中更新 `HandRangeMatrixWidget` 类
    - 修改 `paintEvent` 方法实现条状组合绘制
    - 使用 `get_bar_segments` 获取条状段
    - 按概率比例绘制水平条状
    - _需求: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 9. 更新策略详情控件支持动态动作
  - [x] 9.1 在 `viewer/widgets/strategy_detail_widget.py` 中更新控件
    - 动态生成与动作数量匹配的概率条
    - 使用新的颜色方案
    - _需求: 5.1, 5.2_

- [x] 10. 更新导出功能支持动态动作
  - [x] 10.1 在 `viewer/controller.py` 中更新 `export_json` 方法
    - 导出模型实际支持的所有动作
    - 在元数据中包含动作配置信息
    - _需求: 5.3_
  - [x] 10.2 编写属性测试：JSON导出动作完整性
    - **Property 8: JSON导出动作完整性**
    - **验证: 需求 5.3**

- [x] 11. 更新训练系统保存动作配置
  - [x] 11.1 在 `training/deep_cfr_trainer.py` 中更新检查点保存
    - 在保存检查点时包含 `action_config` 字段
    - _需求: 3.1, 3.2_
  - [x] 11.2 编写属性测试：检查点元数据完整性
    - **Property 3: 检查点元数据完整性**
    - **验证: 需求 3.1, 3.2**

- [x] 12. Final Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。
