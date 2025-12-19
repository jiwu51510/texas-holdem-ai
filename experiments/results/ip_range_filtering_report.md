# IP范围过滤实验报告

## 实验时间
2025-12-19T00:02:19.069Z

## 实验目的
验证当OOP持有特定手牌时，从IP范围中移除与该手牌冲突的组合后，Solver得到的策略是否与使用完整IP范围时相同。

## 实验设置

### 公共牌
`Ks Td 7c 4h 2s`

### OOP范围
`AA,KK,QQ,JJ,TT,99,88,77,66,AKs,AKo,AQs,AQo,KQs,KQo,AJs,KJs,QJs`

**展开为手牌类型：**
- AA
- KK
- QQ
- JJ
- TT
- 99
- 88
- 77
- 66
- AKs
- AKo
- AQs
- AQo
- KQs
- KQo
- AJs
- KJs
- QJs

### IP范围（完整）
`AA,KK,QQ,JJ,TT,99,88,AKs,AKo,AQs,AQo,KQs,KQo,AJs,KJs,QJs,JTs,T9s`

**展开为手牌类型：**
- AA
- KK
- QQ
- JJ
- TT
- 99
- 88
- AKs
- AKo
- AQs
- AQo
- KQs
- KQo
- AJs
- KJs
- QJs
- JTs
- T9s

### Solver参数
| 参数 | 值 |
|------|-----|
| 起始底池 | 100 |
| 有效筹码 | 500 |
| 下注尺寸 | 33%, 50%, 75% |
| 加注尺寸 | 50%, 100% |
| 目标可剥削度 | 0.1% |
| 最大迭代次数 | 1000 |

## 测试结果汇总

| OOP手牌 | 与IP范围冲突 | 被移除组合数 | 策略差异 | 结论 |
|---------|-------------|-------------|---------|------|
| AcAd | 是 | 23 | 99.10% | 策略有显著差异（>5%） |
| QcQd | 是 | 23 | 32.49% | 策略有显著差异（>5%） |
| AhKh | 是 | 23 | 40.55% | 策略有显著差异（>5%） |
| 7d7h | 否 | 0 | 0.00% | 策略基本相同（差异<1%） |
| 6c6d | 否 | 0 | 0.00% | 策略基本相同（差异<1%） |

## 详细测试结果

---

### 测试手牌: AcAd

#### 方案1: 完整IP范围

**IP范围：** `AA,KK,QQ,JJ,TT,99,88,AKs,AKo,AQs,AQo,KQs,KQo,AJs,KJs,QJs,JTs,T9s`

**IP范围展开（18种手牌类型）：**
```
AA, KK, QQ, JJ, TT, 99, 88, AKs, AKo, AQs, AQo, KQs, KQo, AJs, KJs, QJs, JTs, T9s
```

**Solver结果：**
- 策略: `{"Check":0,"Bet":0.98797}`
- 迭代次数: 400
- 可剥削度: 0.0604%

#### 方案2: 过滤后IP范围

**被移除的组合（23个）：**
```
AcAd, AcAh, AcAs, AdAh, AdAs, AcKc, AdKd, AcKd, AcKh, AcKs, AdKc, AdKh, AdKs, AcQc, AdQd, AcQd, AcQh, AcQs, AdQc, AdQh, AdQs, AcJc, AdJd
```

**过滤详情：**
- **AA**: 原6个组合，保留1个，移除: AcAd, AcAh, AcAs, AdAh, AdAs
- **AKs**: 原4个组合，保留2个，移除: AcKc, AdKd
- **AKo**: 原12个组合，保留6个，移除: AcKd, AcKh, AcKs, AdKc, AdKh, AdKs
- **AQs**: 原4个组合，保留2个，移除: AcQc, AdQd
- **AQo**: 原12个组合，保留6个，移除: AcQd, AcQh, AcQs, AdQc, AdQh, AdQs
- **AJs**: 原4个组合，保留2个，移除: AcJc, AdJd

**过滤后IP范围（31种手牌/组合）：**
```
AhAs, KK, QQ, JJ, TT, 99, 88, AhKh, AsKs, AhKc, AhKd, AhKs, AsKc, AsKd, AsKh, AhQh, AsQs, AhQc, AhQd, AhQs, AsQc, AsQd, AsQh, KQs, KQo, AhJh, AsJs, KJs, QJs, JTs, T9s
```

**Solver结果：**
- 策略: `{"Check":0.990974,"Bet":0}`
- 迭代次数: 300
- 可剥削度: 0.0997%

#### 对比结果

| 指标 | 完整范围 | 过滤后范围 | 差异 |
|------|---------|-----------|------|
| Check | 0.00% | 99.10% | 99.10% |
| Bet | 98.80% | 0.00% | 98.80% |

**结论：** 策略有显著差异（>5%）

---

### 测试手牌: QcQd

#### 方案1: 完整IP范围

**IP范围：** `AA,KK,QQ,JJ,TT,99,88,AKs,AKo,AQs,AQo,KQs,KQo,AJs,KJs,QJs,JTs,T9s`

**IP范围展开（18种手牌类型）：**
```
AA, KK, QQ, JJ, TT, 99, 88, AKs, AKo, AQs, AQo, KQs, KQo, AJs, KJs, QJs, JTs, T9s
```

**Solver结果：**
- 策略: `{"Check":0.675139,"Bet":0}`
- 迭代次数: 400
- 可剥削度: 0.0604%

#### 方案2: 过滤后IP范围

**被移除的组合（23个）：**
```
QcQd, QcQh, QcQs, QdQh, QdQs, AcQc, AdQd, AcQd, AdQc, AhQc, AhQd, AsQc, AsQd, KcQc, KdQd, KcQd, KdQc, KhQc, KhQd, KsQc, KsQd, QcJc, QdJd
```

**过滤详情：**
- **QQ**: 原6个组合，保留1个，移除: QcQd, QcQh, QcQs, QdQh, QdQs
- **AQs**: 原4个组合，保留2个，移除: AcQc, AdQd
- **AQo**: 原12个组合，保留6个，移除: AcQd, AdQc, AhQc, AhQd, AsQc, AsQd
- **KQs**: 原4个组合，保留2个，移除: KcQc, KdQd
- **KQo**: 原12个组合，保留6个，移除: KcQd, KdQc, KhQc, KhQd, KsQc, KsQd
- **QJs**: 原4个组合，保留2个，移除: QcJc, QdJd

**过滤后IP范围（31种手牌/组合）：**
```
AA, KK, QhQs, JJ, TT, 99, 88, AKs, AKo, AhQh, AsQs, AcQh, AcQs, AdQh, AdQs, AhQs, AsQh, KhQh, KsQs, KcQh, KcQs, KdQh, KdQs, KhQs, KsQh, AJs, KJs, QhJh, QsJs, JTs, T9s
```

**Solver结果：**
- 策略: `{"Check":1,"Bet":0}`
- 迭代次数: 400
- 可剥削度: 0.0519%

#### 对比结果

| 指标 | 完整范围 | 过滤后范围 | 差异 |
|------|---------|-----------|------|
| Check | 67.51% | 100.00% | 32.49% |
| Bet | 0.00% | 0.00% | 0.00% |

**结论：** 策略有显著差异（>5%）

---

### 测试手牌: AhKh

#### 方案1: 完整IP范围

**IP范围：** `AA,KK,QQ,JJ,TT,99,88,AKs,AKo,AQs,AQo,KQs,KQo,AJs,KJs,QJs,JTs,T9s`

**IP范围展开（18种手牌类型）：**
```
AA, KK, QQ, JJ, TT, 99, 88, AKs, AKo, AQs, AQo, KQs, KQo, AJs, KJs, QJs, JTs, T9s
```

**Solver结果：**
- 策略: `{"Check":0.089725,"Bet":0.124714}`
- 迭代次数: 400
- 可剥削度: 0.0604%

#### 方案2: 过滤后IP范围

**被移除的组合（23个）：**
```
AcAh, AdAh, AhAs, KcKh, KdKh, KhKs, AhKh, AcKh, AdKh, AhKc, AhKd, AhKs, AsKh, AhQh, AhQc, AhQd, AhQs, KhQh, KhQc, KhQd, KhQs, AhJh, KhJh
```

**过滤详情：**
- **AA**: 原6个组合，保留3个，移除: AcAh, AdAh, AhAs
- **KK**: 原6个组合，保留3个，移除: KcKh, KdKh, KhKs
- **AKs**: 原4个组合，保留3个，移除: AhKh
- **AKo**: 原12个组合，保留6个，移除: AcKh, AdKh, AhKc, AhKd, AhKs, AsKh
- **AQs**: 原4个组合，保留3个，移除: AhQh
- **AQo**: 原12个组合，保留9个，移除: AhQc, AhQd, AhQs
- **KQs**: 原4个组合，保留3个，移除: KhQh
- **KQo**: 原12个组合，保留9个，移除: KhQc, KhQd, KhQs
- **AJs**: 原4个组合，保留3个，移除: AhJh
- **KJs**: 原4个组合，保留3个，移除: KhJh

**过滤后IP范围（53种手牌/组合）：**
```
AcAd, AcAs, AdAs, KcKd, KcKs, KdKs, QQ, JJ, TT, 99, 88, AcKc, AdKd, AsKs, AcKd, AcKs, AdKc, AdKs, AsKc, AsKd, AcQc, AdQd, AsQs, AcQd, AcQh, AcQs, AdQc, AdQh, AdQs, AsQc, AsQd, AsQh, KcQc, KdQd, KsQs, KcQd, KcQh, KcQs, KdQc, KdQh, KdQs, KsQc, KsQd, KsQh, AcJc, AdJd, AsJs, KcJc, KdJd, KsJs, QJs, JTs, T9s
```

**Solver结果：**
- 策略: `{"Check":0.495258,"Bet":0.504742}`
- 迭代次数: 300
- 可剥削度: 0.0811%

#### 对比结果

| 指标 | 完整范围 | 过滤后范围 | 差异 |
|------|---------|-----------|------|
| Check | 8.97% | 49.53% | 40.55% |
| Bet | 12.47% | 50.47% | 38.00% |

**结论：** 策略有显著差异（>5%）

---

### 测试手牌: 7d7h

#### 方案1: 完整IP范围

**IP范围：** `AA,KK,QQ,JJ,TT,99,88,AKs,AKo,AQs,AQo,KQs,KQo,AJs,KJs,QJs,JTs,T9s`

**IP范围展开（18种手牌类型）：**
```
AA, KK, QQ, JJ, TT, 99, 88, AKs, AKo, AQs, AQo, KQs, KQo, AJs, KJs, QJs, JTs, T9s
```

**Solver结果：**
- 策略: `{"Check":0.586232,"Bet":0.317604}`
- 迭代次数: 400
- 可剥削度: 0.0604%

#### 方案2: 过滤后IP范围

**被移除的组合（0个）：**
```
无
```

**过滤详情：**
- 无组合被移除

**过滤后IP范围（18种手牌/组合）：**
```
AA, KK, QQ, JJ, TT, 99, 88, AKs, AKo, AQs, AQo, KQs, KQo, AJs, KJs, QJs, JTs, T9s
```

**Solver结果：**
- 策略: `{"Check":0.586232,"Bet":0.317604}`
- 迭代次数: 400
- 可剥削度: 0.0604%

#### 对比结果

| 指标 | 完整范围 | 过滤后范围 | 差异 |
|------|---------|-----------|------|
| Check | 58.62% | 58.62% | 0.00% |
| Bet | 31.76% | 31.76% | 0.00% |

**结论：** 策略基本相同（差异<1%）

---

### 测试手牌: 6c6d

#### 方案1: 完整IP范围

**IP范围：** `AA,KK,QQ,JJ,TT,99,88,AKs,AKo,AQs,AQo,KQs,KQo,AJs,KJs,QJs,JTs,T9s`

**IP范围展开（18种手牌类型）：**
```
AA, KK, QQ, JJ, TT, 99, 88, AKs, AKo, AQs, AQo, KQs, KQo, AJs, KJs, QJs, JTs, T9s
```

**Solver结果：**
- 策略: `{"Check":1,"Bet":0}`
- 迭代次数: 400
- 可剥削度: 0.0604%

#### 方案2: 过滤后IP范围

**被移除的组合（0个）：**
```
无
```

**过滤详情：**
- 无组合被移除

**过滤后IP范围（18种手牌/组合）：**
```
AA, KK, QQ, JJ, TT, 99, 88, AKs, AKo, AQs, AQo, KQs, KQo, AJs, KJs, QJs, JTs, T9s
```

**Solver结果：**
- 策略: `{"Check":1,"Bet":0}`
- 迭代次数: 400
- 可剥削度: 0.0604%

#### 对比结果

| 指标 | 完整范围 | 过滤后范围 | 差异 |
|------|---------|-----------|------|
| Check | 100.00% | 100.00% | 0.00% |
| Bet | 0.00% | 0.00% | 0.00% |

**结论：** 策略基本相同（差异<1%）

## 总结

### ⚠️ 重要发现

当OOP手牌与IP范围存在冲突时（即OOP持有的牌在IP范围中也存在），过滤IP范围后的策略与完整IP范围的策略**显著不同**。

这说明：
1. **Solver不会自动处理card removal**：即使OOP持有AA，Solver仍然会按照IP范围中包含AA来计算
2. **手动过滤IP范围是必要的**：如果要获得正确的策略，需要在调用Solver前手动移除与OOP手牌冲突的IP组合
3. **无冲突时策略相同**：当OOP手牌不在IP范围中时（如77、66），过滤前后策略完全相同

### ✅ 验证结果

当OOP手牌与IP范围无冲突时，过滤IP范围后的策略与完整IP范围的策略**完全相同**，这符合预期。
