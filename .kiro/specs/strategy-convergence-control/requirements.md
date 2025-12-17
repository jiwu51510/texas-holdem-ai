# 需求文档

## 简介

本文档描述了改进Deep CFR训练中策略网络收敛方向控制的需求。当前河牌阶段训练发现策略网络收敛方向难以控制，表现为策略震荡、遗憾值爆炸、或收敛到非最优策略。本功能旨在通过多种技术手段确保策略网络稳定收敛到近似纳什均衡。

## 术语表

- **Deep CFR**：深度反事实遗憾最小化（Deep Counterfactual Regret Minimization），将CFR算法与深度神经网络结合的方法
- **策略网络（Policy Network）**：学习长期平均策略的神经网络
- **遗憾网络（Regret Network）**：学习每个动作即时遗憾值的神经网络
- **Regret Matching**：根据累积正遗憾值比例分配动作概率的算法
- **CFR+**：CFR的改进版本，使用正遗憾值截断和线性加权
- **LCFR（Linear CFR）**：使用线性加权的CFR变体，后期迭代权重更高
- **可利用度（Exploitability）**：衡量策略与纳什均衡距离的指标
- **蓄水池采样（Reservoir Sampling）**：保持固定大小缓冲区的采样算法

## 需求

### 需求 1

**用户故事：** 作为AI训练工程师，我希望遗憾值累积更加稳定，以便策略网络能够平稳收敛。

#### 验收标准

1. WHEN 计算累积遗憾值 THEN 系统 SHALL 使用CFR+的正遗憾值截断策略，将负遗憾值截断为零
2. WHEN 存储遗憾值样本 THEN 系统 SHALL 支持可配置的遗憾值衰减因子（默认0.99），防止遗憾值无限累积
3. WHEN 遗憾值绝对值超过阈值 THEN 系统 SHALL 对遗憾值进行裁剪，防止数值爆炸
4. WHEN 训练遗憾网络 THEN 系统 SHALL 支持使用Huber损失替代MSE损失，减少异常值影响

### 需求 2

**用户故事：** 作为AI训练工程师，我希望采样方差更低，以便训练信号更加稳定。

#### 验收标准

1. WHEN 执行CFR迭代 THEN 系统 SHALL 支持结果采样（Outcome Sampling）和外部采样（External Sampling）两种模式
2. WHEN 使用外部采样模式 THEN 系统 SHALL 支持配置采样次数，通过多次采样取平均降低方差
3. WHEN 计算动作价值 THEN 系统 SHALL 支持使用基线（Baseline）减少方差
4. WHEN 河牌阶段 THEN 系统 SHALL 继续使用枚举而非采样，保持低方差

### 需求 3

**用户故事：** 作为AI训练工程师，我希望策略更新更加平滑，以便避免策略震荡。

#### 验收标准

1. WHEN 训练策略网络 THEN 系统 SHALL 支持使用指数移动平均（EMA）更新目标网络
2. WHEN 计算策略损失 THEN 系统 SHALL 支持添加KL散度正则化项，限制策略变化幅度
3. WHEN 更新网络参数 THEN 系统 SHALL 支持梯度裁剪，防止梯度爆炸
4. WHEN 训练策略网络 THEN 系统 SHALL 支持使用LCFR线性加权，后期迭代权重更高

### 需求 4

**用户故事：** 作为AI训练工程师，我希望能够监控收敛状态，以便及时发现和诊断问题。

#### 验收标准

1. WHEN 训练过程中 THEN 系统 SHALL 定期计算并记录策略熵值
2. WHEN 训练过程中 THEN 系统 SHALL 定期计算并记录遗憾值的统计信息（均值、方差、最大值）
3. WHEN 训练过程中 THEN 系统 SHALL 定期计算并记录策略变化幅度（与上一检查点的KL散度）
4. WHEN 检测到策略震荡 THEN 系统 SHALL 输出警告信息并建议调整参数

### 需求 5

**用户故事：** 作为AI训练工程师，我希望有更好的缓冲区管理策略，以便训练数据分布更加稳定。

#### 验收标准

1. WHEN 采样训练数据 THEN 系统 SHALL 支持优先采样近期迭代的样本（可配置的时间衰减）
2. WHEN 缓冲区已满 THEN 系统 SHALL 支持基于重要性的样本替换策略
3. WHEN 训练网络 THEN 系统 SHALL 支持分层采样，确保不同游戏阶段的样本均衡
4. WHEN 缓冲区样本过旧 THEN 系统 SHALL 支持定期清理过旧样本

### 需求 6

**用户故事：** 作为AI训练工程师，我希望能够使用更先进的CFR变体，以便加速收敛。

#### 验收标准

1. WHEN 配置训练器 THEN 系统 SHALL 支持选择CFR变体：标准CFR、CFR+、LCFR、DCFR
2. WHEN 使用CFR+ THEN 系统 SHALL 实现正遗憾值截断和即时遗憾值加权
3. WHEN 使用LCFR THEN 系统 SHALL 实现线性迭代加权（权重与迭代次数成正比）
4. WHEN 使用DCFR THEN 系统 SHALL 实现折扣CFR，支持配置折扣因子

