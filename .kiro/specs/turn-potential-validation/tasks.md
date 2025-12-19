# 实现计划

- [x] 1. 创建实验框架基础结构
  - [x] 1.1 创建实验目录和数据模型
    - 创建 `experiments/turn_potential_validation/` 目录
    - 实现 TurnScenario、CorrelationResult、ClusteringComparisonResult 数据类
    - 实现 TurnValidationMetrics 数据类
    - _Requirements: 5.1, 6.1_
  - [x] 1.2 编写属性测试：数据模型序列化往返
    - **Property: 数据模型序列化往返**
    - 验证数据类可以正确序列化和反序列化
    - **Validates: Requirements 6.1**

- [x] 2. 实现河牌枚举器
  - [x] 2.1 实现 RiverCardEnumerator 类
    - 实现 `enumerate_river_cards()` 方法
    - 枚举所有可能的河牌（52-2手牌-4公共牌=46张）
    - 确保不包含与手牌或公共牌重复的牌
    - _Requirements: 1.1_
  - [x] 2.2 编写属性测试：河牌枚举完整性
    - **Property 1: 河牌枚举完整性**
    - 使用Hypothesis生成随机手牌和公共牌
    - 验证枚举数量为46且无重复
    - **Validates: Requirements 1.1**

- [x] 3. 实现Potential直方图计算器
  - [x] 3.1 实现 PotentialHistogramCalculator 类
    - 实现 `calculate_potential_histogram()` 方法
    - 枚举所有河牌，计算每种情况下的Equity
    - 生成归一化的Equity分布直方图
    - _Requirements: 1.1, 1.2, 1.3_
  - [x] 3.2 编写属性测试：Potential直方图归一化
    - **Property 2: Potential直方图归一化**
    - 验证直方图概率和为1
    - 验证所有概率值在[0, 1]范围内
    - **Validates: Requirements 1.3, 8.1**
  - [x] 3.3 实现范围VS范围Potential直方图计算
    - 实现 `calculate_range_potential_histograms()` 方法
    - 对范围内每个手牌计算Potential直方图
    - 应用正确的死牌移除
    - _Requirements: 2.1, 2.2_
  - [x] 3.4 编写属性测试：范围VS范围计算完整性
    - **Property 4: 范围VS范围计算完整性**
    - 验证返回的直方图数量等于有效手牌数
    - 验证每个直方图都是归一化的
    - **Validates: Requirements 2.1, 2.3**

- [x] 4. Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 5. 集成EMD距离计算
  - [x] 5.1 验证现有EMDCalculator的正确性
    - 确认 `abstraction/emd_calculator.py` 满足需求
    - 测试一维EMD计算的正确性
    - _Requirements: 3.1, 3.2_
  - [x] 5.2 编写属性测试：EMD距离非负性和同一性
    - **Property 5: EMD距离非负性和同一性**
    - 验证EMD距离非负
    - 验证相同直方图EMD为0
    - **Validates: Requirements 3.1, 3.3**
  - [x] 5.3 编写属性测试：EMD距离对称性
    - **Property 6: EMD距离对称性**
    - 验证EMD(A, B) = EMD(B, A)
    - **Validates: Requirements 3.1**

- [x] 6. 实现转牌Solver封装
  - [x] 6.1 扩展现有WasmPostflopWrapper支持转牌
    - 修改 `experiments/equity_solver_validation/solver_wrapper.py`
    - 添加转牌阶段的Solver调用支持
    - 处理4张公共牌的情况
    - _Requirements: 4.1_
  - [x] 6.2 实现转牌策略和EV提取
    - 实现转牌节点的策略提取
    - 实现转牌节点的EV提取
    - _Requirements: 4.2, 4.3_
  - [x] 6.3 编写属性测试：策略概率归一化
    - **Property 7: 策略概率归一化**
    - 验证提取的策略概率之和为1
    - **Validates: Requirements 4.2**

- [x] 7. Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 8. 实现Potential分析器
  - [x] 8.1 实现 PotentialAnalyzer 类
    - 实现 `analyze_histogram_strategy_correlation()` 方法
    - 计算Potential直方图特征与策略的相关性
    - _Requirements: 5.1_
  - [x] 8.2 实现基于Potential的聚类
    - 实现 `cluster_by_potential()` 方法
    - 使用EMD距离作为度量进行k-means聚类
    - _Requirements: 5.2_
  - [x] 8.3 实现聚类与策略比较
    - 实现 `compare_clustering_with_strategy()` 方法
    - 计算聚类纯度和归一化互信息
    - _Requirements: 5.2, 5.3_

- [x] 9. 实现实验运行器
  - [x] 9.1 实现 TurnExperimentRunner 类
    - 实现场景配置加载
    - 实现批量实验运行逻辑
    - 实现结果记录和保存
    - _Requirements: 6.1, 6.2_
  - [x] 9.2 编写属性测试：批量实验结果完整性
    - **Property 8: 批量实验结果完整性**
    - 验证每个场景都有对应结果
    - **Validates: Requirements 6.2**
  - [x] 9.3 实现结果汇总和统计
    - 计算平均EMD距离
    - 计算策略相关系数
    - 生成汇总报告
    - _Requirements: 6.3_

- [x] 10. 实现可视化模块
  - [x] 10.1 实现 TurnVisualizer 类
    - 实现Potential直方图热力图生成
    - 实现按动作类型分组显示
    - 支持PNG和SVG导出
    - _Requirements: 7.1, 7.2, 7.3_

- [x] 11. 实现验证功能
  - [x] 11.1 实现手动枚举验证
    - 实现与手动枚举计算结果的对比功能
    - 报告计算误差
    - _Requirements: 8.2, 8.3_
  - [x] 11.2 编写属性测试：Potential直方图计算一致性
    - **Property 9: Potential直方图计算一致性（Round-trip）**
    - 验证计算结果与手动枚举一致
    - **Validates: Requirements 8.2**

- [x] 12. 创建预定义实验场景
  - [x] 12.1 创建转牌实验场景配置文件
    - 定义干燥牌面场景（4张公共牌）
    - 定义湿润牌面场景
    - 定义听牌牌面场景
    - 定义配对牌面场景
    - _Requirements: 6.1_
  - [x] 12.2 定义标准范围配置
    - 复用河牌验证的范围配置
    - 根据转牌特点调整范围
    - _Requirements: 6.1_

- [x] 13. 创建实验运行脚本
  - [x] 13.1 创建主实验脚本
    - 实现命令行参数解析
    - 支持选择实验场景
    - 支持配置Solver参数
    - _Requirements: 6.1, 6.2_
  - [x] 13.2 实现结果分析和报告生成
    - 生成对比报告
    - 生成可视化图表
    - 输出统计汇总
    - _Requirements: 5.3, 6.3, 7.1_

- [x] 14. Final Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

