# 需求文档

## 简介

本需求文档描述将当前的混合训练架构（CFR + Actor-Critic）重构为纯 Deep CFR 架构。Deep CFR 是一种将 CFR（反事实遗憾最小化）算法与深度神经网络结合的方法，用于求解大规模不完全信息博弈。

当前架构使用策略网络 + 价值网络的 Actor-Critic 混合方式，而标准 Deep CFR 使用：
- **Regret Network（遗憾网络）**：学习每个动作的即时遗憾值
- **Policy Network（策略网络）**：学习长期平均策略

## 术语表

- **Deep CFR**：Deep Counterfactual Regret Minimization，将 CFR 算法与深度神经网络结合的方法
- **遗憾网络（Regret Network）**：预测每个动作遗憾值的神经网络，输出维度等于动作空间大小
- **策略网络（Policy Network）**：学习长期平均策略的神经网络，输出动作概率分布
- **遗憾值（Regret）**：选择某个动作相对于当前策略期望收益的差值
- **反事实遗憾值（Counterfactual Regret）**：假设玩家以概率1到达当前节点时的遗憾值
- **Regret Matching**：根据累积正遗憾值比例选择动作的策略
- **信息集（Information Set）**：玩家在某一时刻所知道的所有信息的集合
- **训练引擎（Training Engine）**：管理训练流程的核心组件
- **遗憾缓冲区（Regret Buffer）**：存储遗憾值训练样本的数据结构
- **策略缓冲区（Strategy Buffer）**：存储策略训练样本的数据结构
- **蓄水池采样（Reservoir Sampling）**：保持缓冲区固定大小的采样算法

## 需求

### 需求 1

**用户故事：** 作为开发者，我希望系统使用遗憾网络和策略网络替代当前的策略网络和价值网络，以符合标准 Deep CFR 实现。

#### 验收标准

1. WHEN 系统初始化训练组件 THEN 训练引擎 SHALL 创建一个遗憾网络（Regret Network）用于学习遗憾值
2. WHEN 系统初始化训练组件 THEN 训练引擎 SHALL 创建一个策略网络（Policy Network）用于学习平均策略
3. WHEN 遗憾网络进行前向传播 THEN 遗憾网络 SHALL 输出每个动作的遗憾值（维度等于动作空间大小）
4. WHEN 策略网络进行前向传播 THEN 策略网络 SHALL 输出动作概率分布（所有概率非负且和为1）
5. WHEN 保存检查点 THEN 检查点管理器 SHALL 保存遗憾网络和策略网络的参数

### 需求 2

**用户故事：** 作为开发者，我希望实现标准的 Deep CFR 训练流程，包括自博弈、遗憾计算和样本收集。

#### 验收标准

1. WHEN 执行训练迭代 THEN 训练引擎 SHALL 使用当前遗憾网络生成策略进行自博弈
2. WHEN 自博弈完成一局 THEN 训练引擎 SHALL 为每个关键决策节点精确计算反事实遗憾值
3. WHEN 计算遗憾值 THEN 训练引擎 SHALL 计算"如果选择其他动作会怎样"的反事实收益差
4. WHEN 收集训练样本 THEN 训练引擎 SHALL 将遗憾值样本存储到遗憾缓冲区
5. WHEN 收集训练样本 THEN 训练引擎 SHALL 将当前策略样本存储到策略缓冲区

### 需求 3

**用户故事：** 作为开发者，我希望实现经验回放缓冲区，使用蓄水池采样保持固定大小。

#### 验收标准

1. WHEN 创建缓冲区 THEN 缓冲区 SHALL 支持指定最大容量
2. WHEN 缓冲区未满时添加样本 THEN 缓冲区 SHALL 直接添加样本
3. WHEN 缓冲区已满时添加样本 THEN 缓冲区 SHALL 使用蓄水池采样决定是否替换旧样本
4. WHEN 从缓冲区采样 THEN 缓冲区 SHALL 返回指定数量的随机样本

### 需求 4

**用户故事：** 作为开发者，我希望遗憾网络和策略网络能够正确学习，以便策略能够收敛到纳什均衡。

#### 验收标准

1. WHEN 训练遗憾网络 THEN 训练引擎 SHALL 使用均方误差损失函数最小化预测遗憾值与目标遗憾值的差异
2. WHEN 训练策略网络 THEN 训练引擎 SHALL 使用交叉熵损失函数学习长期平均策略
3. WHEN 从遗憾网络获取策略 THEN 训练引擎 SHALL 使用 Regret Matching 将遗憾值转换为策略概率
4. WHEN 更新网络参数 THEN 训练引擎 SHALL 在每个 CFR 迭代结束后批量更新两个网络

### 需求 5

**用户故事：** 作为开发者，我希望移除不再需要的价值网络相关代码，以保持代码库整洁。

#### 验收标准

1. WHEN 重构完成 THEN 训练引擎 SHALL 不再包含价值网络（ValueNetwork）的初始化代码
2. WHEN 重构完成 THEN 训练引擎 SHALL 不再包含价值网络的优化器
3. WHEN 重构完成 THEN update_policy 方法 SHALL 不再计算优势函数（advantage）
4. WHEN 加载旧检查点 THEN 检查点管理器 SHALL 能够兼容处理包含价值网络的旧格式检查点

### 需求 6

**用户故事：** 作为开发者，我希望更新配置参数以反映新的 Deep CFR 架构。

#### 验收标准

1. WHEN 配置训练参数 THEN TrainingConfig SHALL 移除 cfr_weight 参数（不再需要混合权重）
2. WHEN 配置训练参数 THEN TrainingConfig SHALL 添加 regret_buffer_size 参数（遗憾缓冲区大小，默认2000000）
3. WHEN 配置训练参数 THEN TrainingConfig SHALL 添加 strategy_buffer_size 参数（策略缓冲区大小，默认2000000）
4. WHEN 配置训练参数 THEN TrainingConfig SHALL 添加 cfr_iterations_per_update 参数（每次网络更新前的 CFR 迭代次数，默认1000）
5. WHEN 配置训练参数 THEN TrainingConfig SHALL 添加 network_train_steps 参数（每次更新的训练步数，默认4000）

### 需求 7

**用户故事：** 作为开发者，我希望现有的测试能够适配新架构，确保重构后系统仍然正确工作。

#### 验收标准

1. WHEN 运行网络测试 THEN 测试 SHALL 验证遗憾网络的输入输出维度正确
2. WHEN 运行网络测试 THEN 测试 SHALL 验证策略网络的输入输出维度正确
3. WHEN 运行训练引擎测试 THEN 测试 SHALL 验证训练流程使用遗憾网络和策略网络
4. WHEN 运行检查点测试 THEN 测试 SHALL 验证检查点包含遗憾网络和策略网络参数
5. WHEN 运行策略分析测试 THEN 测试 SHALL 验证从策略网络正确获取策略概率分布

### 需求 8

**用户故事：** 作为开发者，我希望使用高效的 CFR 采样策略，避免遍历完整游戏树，同时保证训练质量。

#### 验收标准

1. WHEN 执行 CFR 迭代 THEN 训练器 SHALL 使用均匀采样选择私牌组合
2. WHEN 采样翻牌 THEN 训练器 SHALL 使用分层采样（按翻牌纹理分 bucket）选择翻牌
3. WHEN 采样转牌 THEN 训练器 SHALL 从剩余牌堆中均匀随机选择一张转牌
4. WHEN 计算河牌阶段收益 THEN 训练器 SHALL 枚举全部剩余河牌并计算平均收益
5. WHEN 分类翻牌纹理 THEN 翻牌分类器 SHALL 将翻牌分为约 30 个 bucket（范围 20-80）
6. WHEN 分类翻牌纹理 THEN 翻牌分类器 SHALL 考虑同花、顺子潜力、牌面高低、对子、A-high 等特征

### 需求 9

**用户故事：** 作为开发者，我希望翻牌分类器能够正确分类各种翻牌纹理，以支持分层采样。

#### 验收标准

1. WHEN 分类同花翻牌 THEN 翻牌分类器 SHALL 正确识别 2 张或 3 张同花的情况
2. WHEN 分类顺子潜力 THEN 翻牌分类器 SHALL 正确识别已成顺、可成顺、无顺子潜力的情况
3. WHEN 分类牌面高低 THEN 翻牌分类器 SHALL 正确区分高牌（T+）、中牌、低牌翻牌
4. WHEN 分类对子翻牌 THEN 翻牌分类器 SHALL 正确识别有对子的翻牌
5. WHEN 从 bucket 采样 THEN 翻牌分类器 SHALL 能够从指定 bucket 中均匀采样翻牌
