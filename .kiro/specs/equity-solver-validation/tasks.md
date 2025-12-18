# 实现计划

- [x] 1. 创建实验框架基础结构
  - [x] 1.1 创建实验目录和数据模型
    - 创建 `experiments/equity_solver_validation/` 目录
    - 实现 SolverConfig、SolverResult、ComparisonResult、ExperimentScenario 数据类
    - 实现 ValidationMetrics 数据类
    - _Requirements: 4.1, 5.1_
  - [x] 1.2 编写属性测试：数据模型序列化往返
    - **Property: 数据模型序列化往返**
    - 验证数据类可以正确序列化和反序列化
    - **Validates: Requirements 5.1**

- [x] 2. 集成开源胜率计算器
  - [x] 2.1 评估并集成 poker-odds-calc
    - 研究 https://github.com/siavashg87/poker-odds-calc 的API
    - 如果满足需求，创建Python封装器
    - 如果不满足需求，使用现有的 EquityCalculator
    - _Requirements: 1.1_
  - [x] 2.2 实现 DeadCardRemover 类
    - 实现 `remove_dead_cards()` 方法
    - 从范围中移除与死牌冲突的手牌组合
    - 保持权重正确
    - _Requirements: 1.2, 2.2_
  - [x] 2.3 编写属性测试：死牌移除正确性
    - **Property 2: 死牌移除正确性**
    - 使用Hypothesis生成随机范围和死牌
    - 验证移除后无冲突且保留所有有效组合
    - **Validates: Requirements 1.2, 2.2**

- [x] 3. 实现范围VS范围胜率计算器
  - [x] 3.1 实现 RangeVsRangeCalculator 类
    - 实现 `calculate_hand_vs_range_equity()` 方法
    - 实现 `calculate_range_vs_range_equity()` 方法
    - 集成死牌移除逻辑和胜率计算器
    - _Requirements: 1.1, 2.1_
  - [x] 3.2 编写属性测试：胜率计算正确性
    - **Property 1: 胜率计算正确性与范围约束**
    - 验证胜率在[0,1]范围内
    - 验证与枚举计算结果一致
    - **Validates: Requirements 1.1, 1.3**
  - [x] 3.3 编写属性测试：范围VS范围完整性
    - **Property 3: 范围VS范围计算完整性**
    - 验证返回向量包含所有有效手牌
    - **Validates: Requirements 2.1, 2.3**

- [x] 4. Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 5. 实现wasm-postflop Solver封装
  - [x] 5.1 研究wasm-postflop API并创建封装器
    - 分析 https://github.com/b-inary/wasm-postflop 的API
    - 实现 WasmPostflopWrapper 类
    - 实现 `solve()` 方法调用Solver
    - _Requirements: 3.1_
  - [x] 5.2 实现策略和EV提取
    - 实现 `get_strategy()` 方法
    - 实现 `get_ev()` 方法
    - 解析Solver输出格式
    - _Requirements: 3.2, 3.3_
  - [x] 5.3 编写属性测试：策略概率归一化
    - **Property 4: 策略概率归一化**
    - 验证提取的策略概率之和为1
    - **Validates: Requirements 3.2, 4.1**

- [x] 6. 实现策略对比器
  - [x] 6.1 实现 StrategyComparator 类
    - 实现 `equity_to_strategy()` 方法（胜率转策略）
    - 实现 `compare_strategies()` 方法
    - 计算总变差距离、KL散度等度量
    - _Requirements: 4.1, 4.2_
  - [x] 6.2 编写属性测试：策略差异度量数学性质
    - **Property 5: 策略差异度量数学性质**
    - 验证总变差距离在[0,1]范围内
    - 验证相同策略距离为0
    - **Validates: Requirements 4.2**

- [x] 7. Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 8. 实现实验运行器
  - [x] 8.1 实现 ExperimentRunner 类
    - 实现场景配置加载
    - 实现批量实验运行逻辑
    - 实现结果记录和保存
    - _Requirements: 5.1, 5.2_
  - [x] 8.2 编写属性测试：批量实验结果完整性
    - **Property 6: 批量实验结果完整性**
    - 验证每个场景都有对应结果
    - **Validates: Requirements 5.2**
  - [x] 8.3 实现结果汇总和统计
    - 计算平均误差、最大误差
    - 计算相关系数
    - 生成汇总报告
    - _Requirements: 5.3_

- [x] 9. 实现可视化模块
  - [x] 9.1 实现 Visualizer 类
    - 实现胜率-策略散点图生成
    - 实现动作分布对比图
    - 支持PNG和SVG导出
    - _Requirements: 6.1, 6.3_

- [x] 10. 创建预定义实验场景
  - [x] 10.1 创建实验场景配置文件
    - 定义干燥牌面场景
    - 定义湿润牌面场景
    - 定义配对牌面场景
    - 定义同花牌面场景
    - _Requirements: 5.1_
  - [x] 10.2 定义标准范围配置
    - 定义宽范围（如BTN open range）
    - 定义窄范围（如3bet range）
    - 定义极化范围和凝聚范围
    - _Requirements: 5.1_

- [x] 11. 创建实验运行脚本
  - [x] 11.1 创建主实验脚本
    - 实现命令行参数解析
    - 支持选择实验场景
    - 支持配置Solver参数
    - _Requirements: 5.1, 5.2_
  - [x] 11.2 实现结果分析和报告生成
    - 生成对比报告
    - 生成可视化图表
    - 输出统计汇总
    - _Requirements: 4.3, 5.3, 6.1_

- [x] 12. Final Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。
