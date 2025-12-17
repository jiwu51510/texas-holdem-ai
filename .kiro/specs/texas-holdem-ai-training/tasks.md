# 实现计划

- [x] 1. 搭建项目结构和核心接口
  - 创建目录结构（models、environment、training、monitoring、analysis、utils）
  - 定义核心数据类（Card、Action、GameState、Episode、TrainingConfig）
  - 设置PyTorch和Hypothesis测试框架
  - 创建requirements.txt文件
  - 实现核心数据类的验证逻辑
  - 创建基础单元测试
  - 创建演示示例
  - _需求：所有需求的基础_

- [x] 2. 实现扑克牌和游戏基础组件（环境模块）
- [x] 2.1 实现手牌评估器（HandEvaluator）
  - 在environment/目录下创建hand_evaluator.py
  - 实现evaluate_hand函数（识别所有9种牌型：高牌、对子、两对、三条、顺子、同花、葫芦、四条、同花顺）
  - 实现辅助函数：is_flush、is_straight、find_n_of_a_kind等
  - 处理边界情况（A-2-3-4-5顺子、10-J-Q-K-A顺子）
  - _需求：2.5_

- [x] 2.2 为手牌评估编写边缘情况测试
  - 在tests/目录下创建test_hand_evaluator.py
  - 测试所有9种牌型的识别（每种牌型至少一个具体示例）
  - 测试边界情况（最小顺子A-2-3-4-5、最大顺子10-J-Q-K-A）
  - 测试同花顺vs普通顺子
  - 测试使用7张牌（2张手牌+5张公共牌）找到最佳5张组合
  - _需求：2.5_

- [x] 2.3 实现手牌比较逻辑
  - 在hand_evaluator.py中实现compare_hands函数
  - 比较两手牌的牌型和踢脚牌
  - 返回胜者索引（0、1或-1表示平局）
  - _需求：2.3_

- [x] 2.4 编写手牌比较的属性测试
  - 在test_hand_evaluator.py中添加属性测试
  - **属性7：胜负判定正确性**
  - 使用Hypothesis生成随机手牌和公共牌组合
  - 验证compare_hands的传递性（如果A>B且B>C，则A>C）
  - 验证对称性（如果A>B，则B<A）
  - **验证需求：2.3**

- [-] 3. 实现游戏环境和规则引擎
- [x] 3.1 实现RuleEngine规则引擎
  - 在environment/目录下创建rule_engine.py
  - 实现is_action_legal方法（验证行动合法性）
  - 实现apply_action方法（应用行动并返回新状态）
  - 实现determine_winner方法（判定胜者并返回玩家索引）
  - 实现distribute_pot方法（筹码分配逻辑，处理平局分池）
  - 实现_advance_stage方法（管理游戏阶段转换）
  - _需求：2.2, 2.3, 2.4_

- [x] 3.2 编写规则引擎的单元测试
  - 在tests/目录下创建test_rule_engine.py
  - 测试各种行动的合法性验证（弃牌、跟注、加注、过牌）
  - 测试非法行动被正确拒绝（加注金额不足、跟注金额错误等）
  - 测试游戏阶段转换（翻牌前→翻牌→转牌→河牌）
  - 测试筹码分配（单个胜者、平局分池）
  - _需求：2.2, 2.3, 2.4_

- [x] 3.3 编写规则引擎的属性测试
  - 在test_rule_engine.py中添加属性测试
  - **属性6：行动合法性验证**
  - 生成随机游戏状态和行动，验证is_action_legal的正确性
  - **验证需求：2.2**

- [x] 3.4 编写筹码守恒性的属性测试
  - 在test_rule_engine.py中添加属性测试
  - **属性7：胜负判定正确性**（筹码总和不变）
  - 生成随机游戏状态，执行完整对局，验证筹码总和守恒
  - **验证需求：2.3**

- [x] 3.5 实现PokerEnvironment游戏环境
  - 在environment/目录下创建poker_environment.py
  - 实现reset方法（开始新回合，发牌，设置盲注）
  - 实现step方法（执行行动，返回新状态、奖励、是否结束）
  - 实现get_legal_actions方法（获取当前合法行动列表）
  - 实现_deal_cards方法（发牌逻辑）
  - 集成RuleEngine和HandEvaluator
  - _需求：2.1, 2.2, 2.4_

- [x] 3.6 编写游戏环境的属性测试
  - 在tests/目录下创建test_poker_environment.py
  - **属性5：游戏初始化规则符合性**
  - 验证reset后每个玩家恰好有2张手牌，盲注正确设置
  - **验证需求：2.1**
  - **属性8：游戏阶段转换正确性**
  - 验证翻牌发3张、转牌发1张、河牌发1张，总共5张公共牌
  - **验证需求：2.4**

- [x] 4. 实现状态编码器
- [x] 4.1 实现StateEncoder状态编码器
  - 在environment/目录下创建state_encoder.py
  - 实现encode方法（将GameState编码为固定维度向量）
  - 实现encode_cards方法（牌的one-hot编码：52维/张）
  - 实现encode_betting_history方法（下注历史编码）
  - 编码方案：手牌104维 + 公共牌260维 + 筹码信息4维 + 位置信息2维 = 370维
  - _需求：1.1, 1.2_

- [x] 4.2 编写状态编码器的单元测试
  - 在tests/目录下创建test_state_encoder.py
  - 测试编码输出维度恒为370维
  - 测试one-hot编码正确性（每张牌在52维向量中恰好一个1）
  - 测试不同游戏阶段的编码（翻牌前、翻牌、转牌、河牌）
  - 测试相同状态的编码一致性
  - _需求：1.1_

- [x] 5. 实现神经网络模块
- [x] 5.1 实现PolicyNetwork策略网络
  - 在models/目录下创建networks.py
  - 定义PolicyNetwork类（继承nn.Module）
  - 网络架构：输入370维 → 隐藏层[512, 256, 128] → 输出层（行动空间维度）
  - 实现forward方法（返回行动logits）
  - 实现get_action_probs方法（应用softmax返回概率分布）
  - 使用ReLU激活函数
  - _需求：1.1, 1.2_

- [x] 5.2 实现ValueNetwork价值网络
  - 在networks.py中实现ValueNetwork类
  - 网络架构：输入370维 → 隐藏层[512, 256, 128] → 输出1维
  - 实现forward方法（返回状态价值估计）
  - 使用ReLU激活函数（隐藏层）和Tanh激活函数（输出层）
  - _需求：1.1, 1.2_

- [x] 5.3 编写神经网络的单元测试
  - 在tests/目录下创建test_networks.py
  - 测试PolicyNetwork输入输出维度正确
  - 测试ValueNetwork输入输出维度正确
  - 测试前向传播不抛出异常
  - 测试策略网络输出概率和为1（误差<1e-6）
  - 测试所有概率非负
  - _需求：1.1_

- [x] 5.4 编写策略概率分布的属性测试
  - 在test_networks.py中添加属性测试
  - **属性13：策略概率分布有效性**
  - 生成随机状态编码，验证概率分布有效性（非负、和为1）
  - **验证需求：4.2**

- [x] 6. 实现配置管理
- [x] 6.1 实现ConfigManager配置管理器
  - 在utils/目录下创建config_manager.py
  - 实现load_config方法（从JSON文件加载TrainingConfig）
  - 实现save_config方法（将TrainingConfig保存为JSON）
  - 实现validate_config方法（验证配置有效性，返回错误列表）
  - 处理缺失可选参数（应用默认值）
  - 处理JSON序列化（dataclass → dict → JSON）
  - _需求：7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 6.2 编写配置管理的单元测试
  - 在tests/目录下创建test_config_manager.py
  - 测试保存和加载有效配置
  - 测试无效配置被拒绝（负数学习率、无效盲注等）
  - 测试错误信息清晰性
  - 测试缺失可选参数时应用默认值
  - _需求：7.1, 7.2, 7.5_

- [x] 6.3 编写配置管理的属性测试
  - 在test_config_manager.py中添加属性测试
  - **属性26：配置验证正确性**
  - 生成随机配置，验证validate_config正确识别有效/无效配置
  - **验证需求：7.1**
  - **属性27：配置错误信息清晰性**
  - 验证错误信息包含参数名和原因
  - **验证需求：7.2**
  - **属性28：配置往返一致性**
  - 生成随机配置，保存后加载，验证等价性
  - **验证需求：7.3, 7.4**
  - **属性29：配置默认值应用正确性**
  - 验证缺失可选参数时正确应用默认值
  - **验证需求：7.5**

- [x] 7. 实现检查点管理
- [x] 7.1 实现CheckpointManager检查点管理器
  - 在utils/目录下创建checkpoint_manager.py
  - 实现save方法（保存模型、优化器状态、元数据到.pt文件）
  - 实现load方法（从.pt文件加载检查点）
  - 实现list_checkpoints方法（列出所有检查点及其CheckpointInfo）
  - 实现delete方法（删除指定检查点文件）
  - 使用唯一文件名：checkpoint_{timestamp}_{episode}.pt
  - 元数据包含：episode_number、timestamp、win_rate、avg_reward
  - _需求：5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 7.2 编写检查点管理的单元测试
  - 在tests/目录下创建test_checkpoint_manager.py
  - 测试保存和加载检查点
  - 测试列出检查点返回正确信息
  - 测试删除检查点
  - 测试文件名唯一性（连续保存多个检查点）
  - _需求：5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 7.3 编写检查点管理的属性测试
  - 在test_checkpoint_manager.py中添加属性测试
  - **属性17：检查点往返一致性**
  - 生成随机模型参数，保存后加载，验证参数等价
  - **验证需求：5.1, 5.2**
  - **属性18：检查点列表完整性**
  - 保存N个检查点，验证list_checkpoints返回N个条目
  - **验证需求：5.3**
  - **属性19：检查点删除有效性**
  - 删除检查点后，文件不存在且不在列表中
  - **验证需求：5.4**
  - **属性20：检查点文件名唯一性**
  - 连续保存多个检查点，验证文件名不重复
  - **验证需求：5.5**

- [x] 8. 检查点 - 确保所有测试通过
  - 运行pytest确保所有测试通过
  - 如有问题请询问用户

- [x] 9. 实现CFR训练算法
- [x] 9.1 实现CFRTrainer训练器
  - 在training/目录下创建cfr_trainer.py
  - 实现compute_regrets方法（计算反事实遗憾值）
  - 实现update_strategy方法（基于遗憾值更新策略，使用Regret Matching）
  - 实现get_average_strategy方法（返回累积平均策略）
  - 实现信息集抽象逻辑（将相似状态归为同一信息集）
  - 维护遗憾值表和策略表
  - _需求：1.1, 1.2_

- [x] 9.2 编写CFR算法的单元测试
  - 在tests/目录下创建test_cfr_trainer.py
  - 测试遗憾值计算逻辑
  - 测试策略更新（遗憾值增加时策略概率增加）
  - 测试简单场景下的收敛性（如石头剪刀布游戏）
  - 测试平均策略计算
  - _需求：1.2_

- [x] 10. 实现训练引擎
- [x] 10.1 实现TrainingEngine核心训练循环
  - 在training/目录下创建training_engine.py
  - 实现__init__方法（初始化环境、网络、优化器、CFR训练器等）
  - 实现self_play_episode方法（执行一次自我对弈回合）
  - 实现update_policy方法（使用CFR或神经网络更新策略）
  - 实现train方法（主训练循环，包含进度显示）
  - 集成CheckpointManager（按间隔自动保存）
  - 支持从检查点恢复训练
  - 实现优雅终止（Ctrl+C时保存状态）
  - _需求：1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 10.2 编写训练引擎的单元测试
  - 在tests/目录下创建test_training_engine.py
  - 测试训练引擎初始化
  - 测试自我对弈回合生成Episode
  - 测试训练循环运行指定回合数
  - 测试检查点保存间隔
  - _需求：1.1, 1.3_

- [x] 10.3 编写训练引擎的属性测试
  - 在test_training_engine.py中添加属性测试
  - **属性1：训练初始化完整性**
  - 生成随机配置，验证训练引擎成功初始化且策略网络参数非空
  - **验证需求：1.1**
  - **属性2：训练终止安全性**
  - 运行短期训练，验证终止后模型状态可被加载
  - **验证需求：1.5**
  - **属性3：检查点保存间隔一致性**
  - 运行N个回合，验证至少创建一个检查点
  - **验证需求：1.3**
  - **属性4：配置参数应用正确性**
  - 验证训练器使用的参数与配置一致
  - **验证需求：1.4**

- [x] 11. 实现监控系统
- [x] 11.1 实现MetricsCollector指标收集器
  - 在monitoring/目录下创建metrics_collector.py
  - 实现record_episode方法（记录回合数据：状态、行动、奖励）
  - 实现get_current_metrics方法（返回当前指标字典）
  - 实现get_metric_history方法（返回指定指标的历史数据）
  - 计算指标：胜率、平均奖励、损失值、已完成回合数
  - 维护滑动窗口统计（如最近100回合的胜率）
  - _需求：3.1, 3.2_

- [x] 11.2 实现TrainingMonitor训练监控器
  - 在monitoring/目录下创建training_monitor.py
  - 实现start方法（启动监控线程或进程）
  - 实现update方法（更新显示的指标）
  - 实现plot_metrics方法（使用matplotlib绘制指标曲线）
  - 实现check_anomalies方法（检测损失值发散等异常）
  - 实现日志持久化（写入JSON Lines格式文件）
  - 集成MetricsCollector
  - _需求：3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 11.3 编写监控系统的单元测试
  - 在tests/目录下创建test_monitoring.py
  - 测试MetricsCollector记录和检索指标
  - 测试TrainingMonitor更新和显示
  - 测试日志文件写入和读取
  - 测试异常检测（模拟损失值发散）
  - _需求：3.1, 3.2, 3.3, 3.5_

- [x] 11.4 编写监控系统的属性测试
  - 在test_monitoring.py中添加属性测试
  - **属性9：监控指标更新频率**
  - 验证在指定时间间隔内至少更新一次
  - **验证需求：3.1**
  - **属性10：监控指标完整性**
  - 验证返回的指标包含胜率、平均奖励、损失值、回合数
  - **验证需求：3.2**
  - **属性11：指标持久化一致性**
  - 验证日志文件中的数据与内存中的指标一致
  - **验证需求：3.3**

- [x] 12. 实现数据持久化
- [x] 12.1 实现DataLogger数据日志器
  - 在utils/目录下创建data_logger.py
  - 实现write_episode方法（写入回合数据：状态、行动、奖励）
  - 使用JSON Lines格式（每行一个JSON对象）
  - 每条记录包含：timestamp、episode_number、数据
  - 实现read_episodes方法（读取历史数据）
  - 实现query_episodes方法（按条件过滤数据）
  - _需求：9.1, 9.2, 9.3_

- [x] 12.2 实现数据导出功能
  - 在data_logger.py中实现export_to_csv方法
  - 在data_logger.py中实现export_to_json方法
  - 处理数据格式转换（Episode对象 → CSV/JSON）
  - _需求：9.5_

- [x] 12.3 编写数据持久化的单元测试
  - 在tests/目录下创建test_data_logger.py
  - 测试写入和读取回合数据
  - 测试数据查询和过滤
  - 测试CSV和JSON导出
  - 测试磁盘空间不足错误处理（模拟IO错误）
  - _需求：9.1, 9.2, 9.4, 9.5_

- [x] 12.4 编写数据持久化的属性测试
  - 在test_data_logger.py中添加属性测试
  - **属性32：训练日志记录完整性**
  - 生成随机Episode，写入后验证所有字段都被记录
  - **验证需求：9.1**
  - **属性33：训练数据往返一致性**
  - 写入随机数据，读取后验证数据一致
  - **验证需求：9.2**
  - **属性34：训练指标索引正确性**
  - 验证每条记录包含timestamp和episode_number
  - **验证需求：9.3**
  - **属性35：数据导出格式正确性**
  - 导出数据，验证CSV/JSON格式可被标准解析器解析
  - **验证需求：9.5**

- [x] 13. 检查点 - 确保所有测试通过
  - 运行pytest确保所有测试通过
  - 如有问题请询问用户

- [x] 14. 实现模型评估器
- [x] 14.1 实现Evaluator评估器
  - 在analysis/目录下创建evaluator.py
  - 实现evaluate方法（运行N局评估对局）
  - 实现_play_evaluation_game方法（执行单局评估）
  - 实现指标计算：胜率（胜局数/总局数）、平均盈利、标准差
  - 实现对手策略接口（RandomStrategy、FixedStrategy等）
  - 实现compare_models方法（多模型比较）
  - 实现save_results方法（保存评估结果为JSON）
  - _需求：6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 14.2 编写评估器的单元测试
  - 在tests/目录下创建test_evaluator.py
  - 测试评估运行指定局数
  - 测试指标计算正确性
  - 测试不同对手策略
  - 测试多模型比较
  - 测试结果保存和加载
  - _需求：6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 14.3 编写评估器的属性测试
  - 在test_evaluator.py中添加属性测试
  - **属性21：评估对局数量正确性**
  - 指定N局，验证恰好执行N局
  - **验证需求：6.1**
  - **属性22：评估指标计算正确性**
  - 验证胜率 = 胜局数 / 总局数
  - **验证需求：6.2**
  - **属性23：基准策略应用正确性**
  - 验证对手使用指定策略做决策
  - **验证需求：6.3**
  - **属性24：多模型评估报告完整性**
  - 验证报告包含所有模型的性能数据
  - **验证需求：6.4**
  - **属性25：评估结果持久化正确性**
  - 验证保存的文件包含所有评估指标
  - **验证需求：6.5**

- [x] 15. 实现策略分析器
- [x] 15.1 实现StrategyAnalyzer策略分析器
  - 在analysis/目录下创建strategy_analyzer.py
  - 实现load_model方法（从检查点加载模型）
  - 实现analyze_state方法（分析特定状态，返回行动概率分布）
  - 实现generate_strategy_heatmap方法（生成手牌范围的策略热图）
  - 实现explain_decision方法（解释决策，包含期望价值计算）
  - 实现compare_strategies方法（比较多个模型的策略）
  - 使用matplotlib生成可视化
  - _需求：4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 15.2 编写策略分析器的单元测试
  - 在tests/目录下创建test_strategy_analyzer.py
  - 测试模型加载
  - 测试状态分析返回有效概率分布
  - 测试策略热图生成
  - 测试决策解释包含期望价值
  - 测试多模型比较
  - _需求：4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 15.3 编写策略分析器的属性测试
  - 在test_strategy_analyzer.py中添加属性测试
  - **属性12：模型加载成功性**
  - 验证加载已保存的检查点不抛出异常
  - **验证需求：4.1**
  - **属性13：策略概率分布有效性**（已在网络测试中覆盖）
  - **属性14：策略热图数据完整性**
  - 验证热图覆盖所有请求的手牌组合
  - **验证需求：4.3**
  - **属性15：决策解释完整性**
  - 验证解释包含期望价值
  - **验证需求：4.4**
  - **属性16：多模型比较数据完整性**
  - 验证比较结果包含所有模型的策略信息
  - **验证需求：4.5**

- [x] 16. 实现并行训练支持
- [x] 16.1 实现ParallelTrainer并行训练器
  - 在training/目录下创建parallel_trainer.py
  - 实现create_parallel_envs方法（创建N个并行游戏环境）
  - 实现collect_experiences方法（从所有进程收集经验数据）
  - 实现aggregate_data方法（聚合所有进程的数据）
  - 使用multiprocessing实现进程间通信
  - 实现参数同步机制（主进程更新后广播到工作进程）
  - 实现错误处理（捕获进程异常）
  - 实现cleanup方法（优雅终止所有进程）
  - _需求：8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 16.2 编写并行训练的单元测试
  - 在tests/目录下创建test_parallel_trainer.py
  - 测试创建指定数量的并行环境
  - 测试数据收集和聚合
  - 测试参数同步
  - 测试进程错误处理（模拟进程崩溃）
  - 测试优雅终止
  - _需求：8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 16.3 编写并行训练的属性测试
  - 在test_parallel_trainer.py中添加属性测试
  - **属性30：并行环境创建正确性**
  - 指定N个进程，验证创建恰好N个环境
  - **验证需求：8.1, 8.4**
  - **属性31：并行数据聚合完整性**
  - 验证聚合数据量 = 所有进程数据量之和
  - **验证需求：8.2**

- [x] 17. 实现命令行界面
- [x] 17.1 实现CLI主程序
  - 在项目根目录创建main.py或cli.py
  - 使用argparse创建命令行解析器
  - 实现train子命令（启动训练，参数：--config, --episodes, --checkpoint-dir等）
  - 实现evaluate子命令（运行评估，参数：--model, --opponent, --games等）
  - 实现analyze子命令（策略分析，参数：--model, --state, --output等）
  - 实现list子命令（列出检查点，参数：--checkpoint-dir）
  - 实现help子命令（显示帮助信息）
  - 处理参数验证和错误信息
  - _需求：10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 17.2 编写CLI的单元测试
  - 在tests/目录下创建test_cli.py
  - 测试命令行参数解析
  - 测试各个子命令的调用
  - 测试无效参数的错误处理
  - 测试帮助信息显示
  - _需求：10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 17.3 编写CLI的属性测试
  - 在test_cli.py中添加属性测试
  - **属性36：命令行训练启动正确性**
  - 生成随机有效参数，验证train命令成功启动
  - **验证需求：10.1**
  - **属性37：命令行评估启动正确性**
  - 生成随机有效参数，验证evaluate命令成功启动
  - **验证需求：10.2**
  - **属性38：命令行策略查看启动正确性**
  - 生成随机有效参数，验证analyze命令成功启动
  - **验证需求：10.3**
  - **属性39：命令行错误处理正确性**
  - 生成随机无效参数，验证显示帮助信息而不崩溃
  - **验证需求：10.4**
  - **属性40：命令行帮助信息完整性**
  - 验证帮助信息列出所有命令及说明
  - **验证需求：10.5**

- [x] 18. 实现错误处理和日志系统
- [x] 18.1 实现全局错误处理
  - 实现自定义异常类
  - 实现错误恢复策略
  - 实现日志记录系统（使用Python logging）
  - 配置不同日志级别
  - _需求：所有需求的错误处理_

- [x] 18.2 编写错误处理的单元测试
  - 测试配置错误处理
  - 测试运行时错误处理
  - 测试数据错误处理
  - _需求：7.2, 9.4_

- [x] 19. 创建示例和文档
- [x] 19.1 创建使用示例
  - 创建简单训练示例脚本
  - 创建评估示例脚本
  - 创建策略分析示例脚本
  - 创建配置文件示例

- [x] 19.2 编写README文档
  - 项目介绍
  - 安装说明
  - 快速开始指南
  - 命令行使用说明
  - 配置参数说明

- [x] 20. 最终检查点 - 确保所有测试通过
  - 运行完整测试套件
  - 确保所有属性测试通过（至少100次迭代）
  - 确保所有单元测试通过
  - 修复任何失败的测试
  - 如有问题请询问用户

- [x] 21. 改进CFR集成和添加熵正则化
  - 问题背景：训练300000次后发现所有节点的fold策略概率趋近于0
  - 根本原因：策略梯度只更新被选择的行动，FOLD因为很少被选择而得不到正向更新
  - 解决方案：改进CFR集成 + 添加熵正则化

- [x] 21.1 改进CFR训练器集成
  - 在CFRTrainer中实现get_cfr_guided_target方法
  - 返回基于CFR平均策略的目标分布
  - 确保CFR策略正确计算反事实遗憾值
  - _需求：1.2_

- [x] 21.2 更新TrainingConfig添加新参数
  - 添加entropy_coefficient参数（熵正则化系数，默认0.01）
  - 添加cfr_weight参数（CFR引导权重，默认0.5）
  - 更新配置验证逻辑
  - _需求：7.1_

- [x] 21.3 实现熵正则化损失
  - 在训练引擎中实现compute_entropy_loss函数
  - 熵损失 = -Σ π(a) * log(π(a))
  - 将熵损失添加到总损失中：total_loss = policy_loss - entropy_coefficient * entropy
  - _需求：1.2_

- [x] 21.4 改进update_policy方法
  - 使用CFR平均策略作为训练目标
  - 添加KL散度损失让神经网络逼近CFR策略
  - 集成熵正则化
  - 正确计算反事实遗憾值（不再使用全零遗憾值）
  - _需求：1.2_

- [x] 21.5 编写改进后的单元测试
  - 测试熵正则化损失计算
  - 测试CFR引导目标生成
  - 测试改进后的策略更新
  - 验证FOLD策略不会消失
  - _需求：1.2_

- [x] 22. 检查点 - 验证改进效果
  - 运行短期训练（1000回合）
  - 检查各行动的策略概率分布
  - 确保FOLD策略保持合理的概率
  - 如有问题请询问用户

- [x] 23. 实现Potential-Aware卡牌抽象模块
- [x] 23.1 创建抽象模块目录结构和数据类
  - 在项目根目录创建abstraction/目录
  - 创建abstraction/__init__.py
  - 实现AbstractionConfig数据类（桶数量、区间数、随机种子等配置）
  - 实现EquityHistogram数据类（直方图表示）
  - 实现AbstractionResult数据类（抽象结果）
  - _需求：11.4, 14.5_

- [x] 23.2 实现EquityCalculator（Equity计算器）
  - 在abstraction/目录下创建equity_calculator.py
  - 实现calculate_equity方法（计算手牌对抗随机对手的胜率）
  - 实现calculate_equity_distribution方法（生成Equity分布直方图）
  - 实现calculate_turn_bucket_distribution方法（计算翻牌手牌在转牌桶上的分布）
  - 使用HandEvaluator进行手牌强度比较
  - 优化：使用多进程并行计算
  - _需求：12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 23.3 编写EquityCalculator的属性测试
  - **属性46：Equity分布直方图归一化**
  - 生成随机手牌组合，验证Equity分布直方图概率和为1
  - **验证需求：12.1, 12.2**
  - **属性47：Equity直方图区间覆盖完整性**
  - 验证直方图区间数量正确且覆盖[0,1]范围
  - **验证需求：12.2**
  - **属性48：Potential-Aware特征维度正确性**
  - 验证翻牌手牌的特征向量维度等于转牌桶数量
  - **验证需求：12.3**

- [x] 23.4 实现EMDCalculator（Earth Mover's Distance计算器）
  - 在abstraction/目录下创建emd_calculator.py
  - 实现calculate_emd_1d方法（一维直方图的线性时间EMD计算）
  - 实现calculate_emd_with_ground_distance方法（带地面距离的EMD计算）
  - 实现calculate_emd_fast_approximation方法（快速近似EMD，用于大规模聚类）
  - 利用稀疏性优化计算效率
  - _需求：13.1, 13.2, 13.3, 13.4, 13.5_

- [x] 23.5 编写EMDCalculator的属性测试
  - **属性42：EMD距离度量正确性**
  - 生成随机直方图对，验证EMD满足非负性、对称性、三角不等式
  - **验证需求：11.2**
  - **属性49：一维EMD线性时间计算**
  - 验证一维EMD计算的时间复杂度为O(N)
  - **验证需求：13.1**
  - **属性50：EMD地面距离传递性**
  - 验证使用地面距离的EMD满足三角不等式
  - **验证需求：13.3**

- [x] 23.6 实现PotentialAwareAbstractor（Potential-Aware抽象器）
  - 在abstraction/目录下创建potential_aware_abstractor.py
  - 实现compute_river_abstraction方法（河牌阶段抽象：Equity + k-means）
  - 实现compute_turn_abstraction方法（转牌阶段抽象：河牌桶分布 + EMD + k-means）
  - 实现compute_flop_abstraction方法（翻牌阶段Potential-Aware抽象）
  - 实现_kmeans_with_emd方法（使用EMD作为距离度量的k-means）
  - 支持k-means++初始化和多次重启
  - _需求：11.1, 11.2, 11.3, 11.5_

- [x] 23.7 编写PotentialAwareAbstractor的属性测试
  - **属性41：Potential-Aware抽象考虑未来轮次**
  - 验证翻牌阶段特征向量包含转牌桶分布信息
  - **验证需求：11.1**
  - **属性43：k-means聚类桶数量正确性**
  - 验证k-means生成指定数量的非空桶
  - **验证需求：11.3, 11.4**
  - **属性44：Imperfect Recall抽象允许信息遗忘**
  - 验证翻牌前可区分的手牌在翻牌阶段可能被归入同一桶
  - **验证需求：11.5**

- [x] 24. 实现CardAbstraction管理器和缓存
- [x] 24.1 实现CardAbstraction管理器
  - 在abstraction/目录下创建card_abstraction.py
  - 实现generate_abstraction方法（生成完整的卡牌抽象）
  - 实现get_bucket_id方法（查询手牌组合对应的桶ID）
  - 实现save方法（保存抽象结果到文件）
  - 实现load方法（从文件加载抽象结果）
  - 实现get_abstraction_stats方法（获取抽象统计信息）
  - _需求：11.4, 11.6, 14.1, 14.2, 14.5_

- [x] 24.2 实现AbstractionCache（抽象缓存）
  - 在abstraction/目录下创建abstraction_cache.py
  - 实现O(1)时间复杂度的桶ID查询
  - 实现get_canonical_hand方法（手牌规范化，处理花色同构性）
  - 使用numpy数组作为查找表
  - 支持内存映射（memmap）处理大型映射表
  - _需求：14.3_

- [x] 24.3 编写CardAbstraction和缓存的属性测试
  - **属性45：抽象状态等价性**
  - 验证同一桶内的手牌返回相同的桶ID
  - **验证需求：11.6**
  - **属性52：抽象结果持久化往返一致性**
  - 验证保存后加载的抽象与原始抽象等价
  - **验证需求：14.1, 14.2**
  - **属性53：桶ID查询O(1)时间复杂度**
  - 验证查询时间是常数级别
  - **验证需求：14.3**
  - **属性55：抽象元数据完整性**
  - 验证保存的文件包含完整的配置元数据
  - **验证需求：14.5**

- [x] 25. 实现抽象质量评估
- [x] 25.1 实现AbstractionEvaluator（抽象评估器）
  - 在abstraction/目录下创建abstraction_evaluator.py
  - 实现calculate_wcss方法（计算Within-Cluster Sum of Squares）
  - 实现get_bucket_size_distribution方法（获取桶大小分布统计）
  - 实现compare_abstractions方法（比较不同抽象配置）
  - 实现generate_report方法（生成抽象质量报告）
  - _需求：15.1, 15.2, 15.3, 15.4_

- [x] 25.2 编写抽象评估器的属性测试
  - **属性56：WCSS质量指标计算正确性**
  - 验证WCSS等于所有数据点到聚类中心距离平方和
  - **验证需求：15.1**
  - **属性57：桶大小分布统计完整性**
  - 验证统计报告包含桶数量、平均桶大小、最大桶大小
  - **验证需求：15.2, 15.4**
  - **属性58：抽象可重复性**
  - 验证相同配置（包括随机种子）生成相同结果
  - **验证需求：15.3**

- [x] 26. 检查点 - 确保抽象模块测试通过
  - 运行pytest tests/test_abstraction*.py确保所有测试通过
  - 如有问题请询问用户

- [ ] 27. 集成抽象模块到训练系统
- [x] 27.1 更新TrainingConfig添加抽象配置
  - 添加abstraction_config字段（AbstractionConfig类型）
  - 添加use_abstraction布尔字段（是否启用抽象）
  - 添加abstraction_path字段（预计算抽象文件路径）
  - 更新配置验证逻辑
  - _需求：11.4, 14.4_

- [x] 27.2 更新StateEncoder支持抽象编码
  - 添加encode_with_abstraction方法
  - 将手牌+公共牌编码替换为桶ID的one-hot编码
  - 支持在原始编码和抽象编码之间切换
  - _需求：11.6_

- [x] 27.3 更新CFRTrainer支持抽象信息集
  - 修改信息集计算逻辑，使用桶ID而非具体手牌
  - 同一桶内的手牌共享相同的策略
  - 大幅减少需要维护的信息集数量
  - _需求：11.6_

- [x] 27.4 更新TrainingEngine集成抽象模块
  - 在初始化时加载预计算的抽象
  - 在训练循环中使用抽象后的状态
  - 添加抽象配置变化检测
  - _需求：14.2, 14.4_

- [x] 27.5 编写集成测试
  - 测试使用抽象的训练流程
  - 验证抽象后的训练收敛速度提升
  - 验证抽象配置变化检测功能
  - **属性54：抽象配置变化检测**
  - **验证需求：14.4**

- [x] 28. 实现抽象生成CLI命令
- [x] 28.1 添加generate-abstraction子命令
  - 在cli.py中添加generate-abstraction子命令
  - 参数：--output（输出路径）、--flop-buckets、--turn-buckets、--river-buckets
  - 参数：--potential-aware（是否使用Potential-Aware）、--seed（随机种子）
  - 显示进度条和预计剩余时间
  - _需求：14.1_

- [x] 28.2 添加abstraction-info子命令
  - 显示已生成抽象的配置和统计信息
  - 参数：--path（抽象文件路径）
  - _需求：15.4_

- [x] 29. 检查点 - 确保集成测试通过
  - 运行完整测试套件
  - 验证抽象模块与训练系统正确集成
  - 如有问题请询问用户

- [x] 30. 性能优化和文档更新
- [x] 30.1 优化抽象生成性能
  - 实现多进程并行Equity计算
  - 优化k-means迭代（使用三角不等式剪枝）
  - 添加进度保存和恢复功能（支持中断后继续）
  - _需求：13.2, 13.4_

- [x] 30.2 更新README文档
  - 添加Potential-Aware抽象的说明
  - 添加抽象生成命令的使用示例
  - 添加抽象配置参数说明
  - 添加性能优化建议

- [ ] 31. 最终检查点 - 确保所有测试通过
  - 运行完整测试套件
  - 确保所有属性测试通过（至少100次迭代）
  - 确保所有单元测试通过
  - 修复任何失败的测试
  - 如有问题请询问用户
