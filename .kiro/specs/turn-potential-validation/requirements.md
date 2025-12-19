# 需求文档

## 简介

本实验旨在验证转牌阶段的Potential-Aware抽象方法。与河牌阶段使用简单的Equity标量不同，转牌阶段需要使用**Potential直方图**来表示手牌的强度分布。

根据论文"Potential-Aware Imperfect-Recall Abstraction with Earth Mover's Distance in Imperfect-Information Games"（AAAI 2014），Potential直方图捕获了手牌在未来轮次（河牌）的强度分布轨迹，而非仅仅是当前的Equity值。

转牌阶段的验证包括：
1. **手牌VS范围的Potential直方图**：计算单个转牌手牌在所有可能河牌下的Equity分布
2. **范围VS范围的Potential直方图**：计算范围内每个手牌的Potential直方图

由于转牌阶段还有一张河牌未发，不能使用简单的标量Equity，必须使用直方图来表示手牌在河牌阶段的潜在强度分布。

## 术语表

- **Potential直方图（Potential Histogram）**: 手牌在未来轮次（河牌）的Equity分布直方图，捕获手牌的"潜力"
- **EMD（Earth Mover's Distance）**: 用于衡量两个直方图之间距离的度量，也称为Wasserstein距离
- **转牌（Turn）**: 德州扑克中的第四张公共牌
- **河牌（River）**: 德州扑克中的第五张公共牌
- **Equity**: 手牌在特定公共牌面下对抗对手范围的获胜概率
- **Range（范围）**: 玩家可能持有的所有手牌组合及其权重
- **Dead Card Removal（死牌移除）**: 从对手范围中移除与已知牌冲突的手牌组合
- **Solver**: 博弈论求解器，用于计算纳什均衡策略（如wasm-postflop）
- **Distribution-Aware抽象**: 基于当前轮次Equity分布的抽象方法
- **Potential-Aware抽象**: 基于未来轮次Equity分布轨迹的抽象方法

## 需求

### 需求1

**用户故事：** 作为研究者，我想计算转牌阶段单个手牌的Potential直方图，以便了解该手牌在河牌阶段的强度分布。

#### 验收标准

1. WHEN 给定一个具体手牌、4张公共牌（翻牌+转牌）和对手范围 THEN 系统 SHALL 枚举所有可能的河牌并计算每种情况下的Equity
2. WHEN 计算Potential直方图时 THEN 系统 SHALL 将Equity值分配到预定义的区间（如50个区间，每个区间宽度0.02）
3. WHEN Potential直方图计算完成 THEN 系统 SHALL 返回一个归一化的概率分布向量
4. WHEN 计算Equity时 THEN 系统 SHALL 从对手范围中正确移除与手牌和公共牌冲突的组合

### 需求2

**用户故事：** 作为研究者，我想计算转牌阶段"我的范围VS对手范围"的Potential直方图矩阵，以便分析范围对抗范围的情况。

#### 验收标准

1. WHEN 给定我的范围、对手范围和4张公共牌 THEN 系统 SHALL 计算范围内每个手牌组合的Potential直方图
2. WHEN 计算范围VS范围时 THEN 系统 SHALL 对每个手牌组合应用正确的死牌移除
3. WHEN 计算完成 THEN 系统 SHALL 返回一个Potential直方图矩阵，每行对应我的范围中的一个手牌组合

### 需求3

**用户故事：** 作为研究者，我想使用EMD距离比较两个Potential直方图，以便衡量两个手牌的相似程度。

#### 验收标准

1. WHEN 给定两个Potential直方图 THEN 系统 SHALL 使用EMD算法计算两者之间的距离
2. WHEN 计算EMD时 THEN 系统 SHALL 使用一维直方图的线性时间算法
3. WHEN EMD计算完成 THEN 系统 SHALL 返回一个非负浮点数表示距离

### 需求4

**用户故事：** 作为研究者，我想使用wasm-postflop Solver计算转牌阶段的参考策略，以便作为验证的基准。

#### 验收标准

1. WHEN 给定游戏树配置（底池大小、筹码深度、下注选项）和4张公共牌 THEN 系统 SHALL 调用wasm-postflop计算纳什均衡策略
2. WHEN Solver计算完成 THEN 系统 SHALL 提取每个手牌在当前节点的动作概率分布
3. WHEN 提取策略时 THEN 系统 SHALL 同时获取每个动作的EV值

### 需求5

**用户故事：** 作为研究者，我想对比Potential直方图方法和Solver方法的策略差异，以便验证假设的有效性。

#### 验收标准

1. WHEN 给定Potential直方图矩阵和Solver策略 THEN 系统 SHALL 分析两者之间的相关性
2. WHEN 对比策略时 THEN 系统 SHALL 计算基于Potential直方图的手牌聚类与Solver策略的一致性
3. WHEN 分析完成 THEN 系统 SHALL 生成对比报告，包含EMD距离分布、策略相关性等指标

### 需求6

**用户故事：** 作为研究者，我想在多个不同转牌场景下运行验证实验，以便评估方法的通用性。

#### 验收标准

1. WHEN 设计实验时 THEN 系统 SHALL 支持配置不同的公共牌面（4张）、范围和游戏树参数
2. WHEN 运行批量实验时 THEN 系统 SHALL 记录每个场景的详细结果
3. WHEN 实验完成 THEN 系统 SHALL 汇总统计结果，包括平均EMD距离、策略相关系数等

### 需求7

**用户故事：** 作为研究者，我想可视化Potential直方图与Solver策略之间的关系，以便直观理解两者的对应关系。

#### 验收标准

1. WHEN 分析完成 THEN 系统 SHALL 生成Potential直方图的热力图可视化
2. WHEN 可视化时 THEN 系统 SHALL 支持按Solver策略动作类型（fold/call/raise）分组显示
3. WHEN 生成图表时 THEN 系统 SHALL 支持导出为PNG或SVG格式

### 需求8

**用户故事：** 作为研究者，我想验证Potential直方图的计算正确性，以便确保实现与论文描述一致。

#### 验收标准

1. WHEN 计算Potential直方图时 THEN 系统 SHALL 确保直方图是归一化的（概率和为1）
2. WHEN 验证计算时 THEN 系统 SHALL 提供与手动枚举计算结果的对比功能
3. WHEN 验证完成 THEN 系统 SHALL 报告计算误差（如果有）

