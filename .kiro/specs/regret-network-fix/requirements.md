# 需求文档

## 简介

本文档描述了修复 regret 网络 action_dim 不一致问题的需求。训练时使用 5 维输出（FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG），但模型加载器默认使用 4 维，导致策略查看器无法正确显示训练好的模型策略。

## 术语表

- **RegretNetwork**: 遗憾网络，Deep CFR 算法中用于学习每个动作遗憾值的神经网络
- **action_dim**: 动作空间维度，表示网络输出的动作数量
- **Deep CFR**: 深度反事实遗憾最小化算法，用于训练扑克 AI
- **策略查看器**: 用于可视化和分析训练好的模型策略的工具

## 需求

### 需求 1

**用户故事:** 作为开发者，我希望模型加载器能正确加载 Deep CFR 训练的模型，以便策略查看器能正确显示策略。

#### 验收标准

1. WHEN 加载 Deep CFR 格式的检查点 THEN 模型加载器 SHALL 自动检测并使用正确的 action_dim（5）
2. WHEN 创建 RegretNetwork 实例 THEN 模型加载器 SHALL 使用与训练时相同的网络架构
3. WHEN 检查点包含 action_dim 信息 THEN 模型加载器 SHALL 优先使用检查点中的 action_dim 值

### 需求 2

**用户故事:** 作为开发者，我希望代码库中的 action_dim 默认值保持一致，以避免混淆和错误。

#### 验收标准

1. WHEN 定义 action_dim 默认值 THEN 所有相关模块 SHALL 使用 5 作为默认值
2. WHEN 测试文件定义 ACTION_DIM 常量 THEN 测试文件 SHALL 使用 5 作为值
3. WHEN 文档描述动作空间 THEN 文档 SHALL 明确列出 5 种动作类型

### 需求 3

**用户故事:** 作为用户，我希望策略查看器能正确显示遗憾值热图，以便分析模型的决策过程。

#### 验收标准

1. WHEN 加载有效的 Deep CFR 检查点 THEN 遗憾值计算器 SHALL 返回正确的 5 维遗憾值
2. WHEN 显示遗憾值热图 THEN 查看器 SHALL 正确映射所有 5 种动作的遗憾值
3. WHEN 遗憾网络输出与预期维度不匹配 THEN 系统 SHALL 提供清晰的错误信息
