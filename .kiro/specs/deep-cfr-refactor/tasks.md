# 实现计划

- [x] 1. 实现 ReservoirBuffer（蓄水池缓冲区）
- [x] 1.1 创建 ReservoirBuffer 类
  - 在 training/ 目录下创建 reservoir_buffer.py
  - 实现 __init__ 方法（指定容量）
  - 实现 add 方法（蓄水池采样添加样本）
  - 实现 sample 方法（随机采样指定数量）
  - 实现 __len__ 方法（返回当前大小）
  - 实现 clear 方法（清空缓冲区）
  - _需求：3.1, 3.2, 3.3, 3.4_

- [x] 1.2 编写 ReservoirBuffer 的属性测试
  - **属性 5：缓冲区添加行为（未满）**
  - **属性 6：缓冲区大小上限**
  - **属性 7：缓冲区采样数量正确性**
  - **验证需求：3.2, 3.3, 3.4**

- [x] 2. 实现 RegretNetwork（遗憾网络）
- [x] 2.1 创建 RegretNetwork 类
  - 在 models/networks.py 中添加 RegretNetwork 类
  - 网络架构：输入370维 → 隐藏层[512, 256, 128] → 输出4维
  - 实现 forward 方法（返回遗憾值）
  - 实现 get_strategy 方法（Regret Matching 转换为策略）
  - 无输出激活函数（遗憾值可以是任意实数）
  - _需求：1.1, 1.3_

- [x] 2.2 编写 RegretNetwork 的属性测试
  - **属性 1：遗憾网络输出维度正确性**
  - **属性 8：Regret Matching 输出有效性**
  - **属性 9：Regret Matching 正遗憾值比例**
  - **属性 10：Regret Matching 全非正遗憾值**
  - **验证需求：1.3, 4.3**

- [x] 3. 更新 PolicyNetwork（策略网络）
- [x] 3.1 确保 PolicyNetwork 保持不变
  - 验证现有 PolicyNetwork 满足 Deep CFR 需求
  - 输入370维 → 隐藏层[512, 256, 128] → 输出4维 + Softmax
  - _需求：1.2, 1.4_

- [x] 3.2 编写 PolicyNetwork 的属性测试
  - **属性 2：策略网络输出概率分布有效性**
  - **验证需求：1.4**

- [x] 4. 检查点 - 确保网络和缓冲区测试通过
  - 运行 pytest tests/test_networks.py tests/test_reservoir_buffer.py
  - 确保所有测试通过，如有问题请询问用户

- [x] 5. 更新 TrainingConfig 配置
- [x] 5.1 添加 Deep CFR 配置参数
  - 添加 regret_buffer_size 参数（默认2000000）
  - 添加 strategy_buffer_size 参数（默认2000000）
  - 添加 cfr_iterations_per_update 参数（默认1000）
  - 添加 network_train_steps 参数（默认4000）
  - 移除 cfr_weight 参数
  - 更新配置验证逻辑
  - _需求：6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 5.2 更新配置管理器测试
  - 测试新参数的验证
  - 测试旧配置文件的兼容性加载
  - _需求：6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 6. 实现 DeepCFRTrainer（Deep CFR 训练器）
- [x] 6.1 创建 DeepCFRTrainer 类
  - 在 training/ 目录下创建 deep_cfr_trainer.py
  - 实现 __init__ 方法（初始化网络、缓冲区、优化器）
  - 实现 traverse_game_tree 方法（遍历游戏树）
  - 实现 compute_counterfactual_regrets 方法（计算反事实遗憾值）
  - 实现 run_cfr_iteration 方法（执行一次 CFR 迭代）
  - 实现 train_networks 方法（训练两个网络）
  - _需求：2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3, 4.4_

- [x] 6.2 编写 DeepCFRTrainer 的属性测试
  - **属性 4：遗憾值计算正确性**
  - **验证需求：2.3**

- [x] 6.3 编写 DeepCFRTrainer 的单元测试
  - 测试 CFR 迭代执行
  - 测试样本收集到缓冲区
  - 测试网络训练损失下降
  - _需求：2.1, 2.2, 2.4, 2.5, 4.1, 4.2, 4.4_

- [x] 7. 重构 TrainingEngine
- [x] 7.1 移除 ValueNetwork 相关代码
  - 移除 value_network 初始化
  - 移除 value_optimizer 初始化
  - 移除 update_policy 中的优势函数计算
  - _需求：5.1, 5.2, 5.3_

- [x] 7.2 集成 DeepCFRTrainer
  - 添加 regret_network 初始化
  - 添加 regret_buffer 和 strategy_buffer 初始化
  - 更新 train 方法使用 Deep CFR 流程
  - 更新 self_play_episode 使用遗憾网络生成策略
  - _需求：1.1, 1.2, 2.1_

- [x] 7.3 更新检查点保存和加载
  - 保存 regret_network 和 policy_network 参数
  - 实现旧检查点格式兼容性处理
  - _需求：1.5, 5.4_

- [x] 7.4 编写 TrainingEngine 的属性测试
  - **属性 3：检查点往返一致性**
  - **验证需求：1.5**

- [x] 7.5 编写 TrainingEngine 的单元测试
  - 测试初始化创建正确的网络
  - 测试训练流程执行
  - 测试检查点保存和加载
  - 测试旧检查点兼容性
  - _需求：1.1, 1.2, 1.5, 5.4, 7.1, 7.2, 7.3, 7.4_

- [x] 8. 检查点 - 确保训练引擎测试通过
  - 运行 pytest tests/test_training_engine.py tests/test_deep_cfr_trainer.py
  - 确保所有测试通过，如有问题请询问用户

- [x] 9. 更新策略分析器
- [x] 9.1 更新 StrategyAnalyzer 使用新架构
  - 更新 load_model 方法加载 policy_network
  - 确保 analyze_state 从 policy_network 获取策略
  - 处理旧检查点格式
  - _需求：7.4, 7.5_

- [x] 9.2 更新策略分析器测试
  - 测试加载新格式检查点
  - 测试加载旧格式检查点
  - 测试策略分析功能
  - _需求：7.4, 7.5_

- [x] 10. 更新 CheckpointManager
- [x] 10.1 更新检查点格式
  - 保存 regret_network_state_dict
  - 保存 policy_network_state_dict
  - 保存 regret_optimizer_state_dict
  - 保存 policy_optimizer_state_dict
  - 添加格式版本号
  - _需求：1.5_

- [x] 10.2 实现向后兼容性
  - 检测旧格式检查点
  - 加载旧格式时只加载 policy_network
  - 记录警告信息
  - _需求：5.4_

- [x] 10.3 更新检查点管理器测试
  - 测试新格式保存和加载
  - 测试旧格式兼容性加载
  - _需求：1.5, 5.4_

- [x] 11. 检查点 - 确保所有测试通过
  - 运行完整测试套件 pytest tests/
  - 确保所有测试通过，如有问题请询问用户

- [x] 12. 清理和文档更新
- [x] 12.1 清理废弃代码
  - 移除 ValueNetwork 类（如果不再需要）
  - 移除相关的导入语句
  - 更新 __init__.py 文件

- [x] 12.2 更新文档
  - 更新 README.md 说明 Deep CFR 架构
  - 更新配置参数说明
  - 添加 Deep CFR 训练示例

- [x] 13. 实现 CFR 采样策略
- [x] 13.1 实现 FlopBucketClassifier（翻牌分类器）
  - 在 training/ 目录下创建 flop_bucket_classifier.py
  - 实现 __init__ 方法（指定 bucket 数量，默认 30）
  - 实现 _extract_features 方法（提取翻牌特征）
  - 实现 classify 方法（将翻牌分类到 bucket）
  - 实现 sample_flop_from_bucket 方法（从 bucket 采样翻牌）
  - 实现 get_bucket_for_sampling 方法（随机选择 bucket）
  - _需求：8.5, 8.6, 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 13.2 实现 CFRSampler（CFR 采样器）
  - 在 training/ 目录下创建 cfr_sampler.py
  - 实现 sample_preflop 方法（均匀采样私牌）
  - 实现 sample_flop 方法（分层采样翻牌）
  - 实现 sample_turn 方法（随机采样转牌）
  - 实现 enumerate_river 方法（枚举全部河牌）
  - _需求：8.1, 8.2, 8.3, 8.4_

- [x] 13.3 更新 DeepCFRTrainer 使用采样策略
  - 修改 traverse_game_tree 方法使用 CFRSampler
  - 实现河牌枚举并计算平均收益
  - 更新 run_cfr_iteration 使用新采样策略
  - _需求：8.1, 8.2, 8.3, 8.4_

- [ ]* 13.4 编写 FlopBucketClassifier 的单元测试
  - 测试同花翻牌分类
  - 测试顺子潜力分类
  - 测试牌面高低分类
  - 测试对子翻牌分类
  - 测试从 bucket 采样
  - _需求：9.1, 9.2, 9.3, 9.4, 9.5_

- [ ]* 13.5 编写 CFRSampler 的单元测试
  - 测试私牌采样均匀性
  - 测试翻牌分层采样
  - 测试转牌随机采样
  - 测试河牌枚举完整性
  - _需求：8.1, 8.2, 8.3, 8.4_

- [x] 14. 检查点 - 确保采样策略测试通过
  - 运行 pytest tests/test_flop_bucket_classifier.py tests/test_cfr_sampler.py
  - 确保所有测试通过，如有问题请询问用户

- [x] 15. 最终检查点 - 确保所有测试通过
  - 运行完整测试套件
  - 确保所有属性测试通过（至少100次迭代）
  - 确保所有单元测试通过
  - 修复任何失败的测试
  - 如有问题请询问用户
