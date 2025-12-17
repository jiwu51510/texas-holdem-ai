# 实现计划

- [x] 1. 创建核心数据模型和工具类
  - [x] 1.1 创建 `viewer/models.py`，定义 GameTreeNode、HandStrategy、NodeState 数据类
    - 实现节点数据结构和验证逻辑
    - _需求: 2.1, 2.4_
  - [x] 1.2 编写属性测试：手牌标签位置正确性
    - **Property 1: 手牌标签位置正确性**
    - **验证: 需求 3.3, 3.4, 3.5**
  - [x] 1.3 创建 `viewer/hand_range.py`，实现 HandRangeCalculator 类
    - 实现 get_hand_label、get_all_hand_combinations、filter_by_board 方法
    - _需求: 3.3, 3.4, 3.5, 4.3, 6.3_
  - [x] 1.4 编写属性测试：手牌组合完整性
    - **Property 3: 手牌组合完整性**
    - **验证: 需求 4.3**
  - [x] 1.5 编写属性测试：公共牌过滤正确性
    - **Property 4: 公共牌过滤正确性**
    - **验证: 需求 6.3**

- [x] 2. 实现游戏树导航器
  - [x] 2.1 创建 `viewer/game_tree.py`，实现 GameTreeNavigator 类
    - 实现 get_root、get_children、navigate_to、get_path_to_root 方法
    - _需求: 2.1, 2.2, 2.3, 2.4, 2.5_
  - [x] 2.2 编写属性测试：游戏树路径一致性
    - **Property 5: 游戏树路径一致性**
    - **验证: 需求 2.4**

- [x] 3. 实现策略计算和颜色映射
  - [x] 3.1 创建 `viewer/strategy_calculator.py`，实现策略计算逻辑
    - 集成现有的 StrategyAnalyzer，计算节点策略
    - 确保策略概率归一化
    - _需求: 4.1, 4.2, 4.4_
  - [x] 3.2 编写属性测试：策略概率归一化
    - **Property 2: 策略概率归一化**
    - **验证: 需求 4.4**
  - [x] 3.3 创建 `viewer/color_mapper.py`，实现 StrategyColorMapper 类
    - 实现 get_cell_color、get_action_color 方法
    - _需求: 3.2_
  - [x] 3.4 编写属性测试：颜色映射确定性
    - **Property 6: 颜色映射确定性**
    - **验证: 需求 3.2**

- [x] 4. Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 5. 实现模型加载器
  - [x] 5.1 创建 `viewer/model_loader.py`，实现模型加载功能
    - 封装检查点加载逻辑，处理各种错误情况
    - _需求: 1.1, 1.2, 1.3_
  - [x] 5.2 编写属性测试：无效检查点处理
    - **Property 8: 无效检查点处理**
    - **验证: 需求 1.2**

- [x] 6. 实现主控制器
  - [x] 6.1 创建 `viewer/controller.py`，实现 StrategyViewerController 类
    - 协调各组件，提供统一的API接口
    - _需求: 1.1, 2.3, 6.2, 6.4_

- [x] 7. 实现UI组件 - 游戏树控件
  - [x] 7.1 创建 `viewer/widgets/game_tree_widget.py`，实现 GameTreeWidget
    - 显示游戏树结构，支持展开/折叠
    - 发出 node_selected 信号
    - _需求: 2.1, 2.2, 2.3, 2.5_

- [x] 8. 实现UI组件 - 手牌矩阵控件
  - [x] 8.1 创建 `viewer/widgets/hand_matrix_widget.py`，实现 HandRangeMatrixWidget
    - 绘制13x13矩阵，显示颜色编码
    - 支持鼠标悬停和点击事件
    - _需求: 3.1, 3.2, 4.1_

- [x] 9. 实现UI组件 - 信息面板和策略详情
  - [x] 9.1 创建 `viewer/widgets/info_panel_widget.py`，实现 InfoPanelWidget
    - 显示公共牌、玩家信息、底池、筹码、游戏阶段
    - _需求: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  - [x] 9.2 创建 `viewer/widgets/strategy_detail_widget.py`，实现 StrategyDetailWidget
    - 显示单个手牌的详细策略概率表
    - 分别显示每个花色组合的策略
    - _需求: 4.2, 4.3_

- [x] 10. 实现UI组件 - 公共牌选择器
  - [x] 10.1 创建 `viewer/widgets/board_selector_widget.py`，实现 BoardCardSelector
    - 提供公共牌选择界面
    - 发出 board_changed 信号
    - _需求: 6.1, 6.2, 6.3_

- [x] 11. 实现主窗口和布局
  - [x] 11.1 创建 `viewer/main_window.py`，实现 MainWindow
    - 组合所有UI组件
    - 实现菜单栏（文件加载、导出等）
    - _需求: 7.3, 7.4_

- [x] 12. Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 13. 实现导出功能
  - [x] 13.1 在控制器中添加导出方法
    - 实现 export_image 导出手牌矩阵图片
    - 实现 export_json 导出策略数据
    - _需求: 8.1, 8.2, 8.3_
  - [x] 13.2 编写属性测试：JSON导出完整性
    - **Property 7: JSON导出完整性**
    - **验证: 需求 8.2**

- [x] 14. 创建启动入口
  - [x] 14.1 创建 `viewer/__init__.py` 和 `run_viewer.py`
    - 提供命令行启动入口
    - 支持指定检查点文件启动
    - _需求: 1.1_

- [x] 15. Final Checkpoint - 确保所有测试通过
  - [x] 确保所有测试通过，如有问题请询问用户。
  - **状态**: 已完成 - 所有 495 个测试通过

- [x] 16. 修复检查点保存逻辑，支持价值网络
  - [x] 16.1 更新训练引擎的检查点保存方法
    - 修改 `_save_checkpoint` 方法，同时保存价值网络参数
    - 修改 `load_checkpoint` 方法，同时加载价值网络参数
    - _需求: 训练系统需求 5.1, 5.2_
  - [x] 16.2 更新检查点管理器支持多模型保存
    - 修改 `save` 方法支持保存多个模型
    - 修改 `load` 方法支持加载多个模型
    - _需求: 训练系统需求 5.1, 5.2_

- [x] 17. 实现价值网络查看功能
  - [x] 17.1 更新模型加载器支持价值网络
    - 修改 `ModelLoader.load` 方法，检测并加载价值网络参数
    - 添加 `has_value_network` 属性
    - _需求: 9.1, 9.5_
  - [x] 17.2 创建价值计算器
    - 创建 `viewer/value_calculator.py`，实现价值估计计算
    - 实现 `calculate_hand_values` 方法，计算所有手牌的价值估计
    - _需求: 9.2, 9.4_
  - [x] 17.3 创建价值热图控件
    - 创建 `viewer/widgets/value_heatmap_widget.py`
    - 实现13x13价值热图显示
    - 使用颜色编码表示价值高低
    - _需求: 9.2, 9.3_
  - [x] 17.4 更新主窗口集成价值热图
    - 在右侧面板添加"价值估计"标签页
    - 连接信号和更新逻辑
    - _需求: 9.2, 9.4_

- [x] 18. Checkpoint - 验证价值网络功能
  - 确保所有测试通过，如有问题请询问用户。
