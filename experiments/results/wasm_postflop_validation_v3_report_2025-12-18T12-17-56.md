# 跨公共牌四维度胜率-策略验证实验报告 V3

## 实验时间

2025-12-18T12-17-56

## 实验改进

**本版本使用 OMPEval (C++) 计算范围胜率，速度比 poker-odds-calc 快约60倍。**
**新增：每100组实验随机生成一次OOP和IP范围，增加实验多样性。**

## 实验目的

验证：**在不同的（公共牌+固定手牌+随机范围）组合下：**
当以下四个条件同时满足时，策略是否相同？
1. 固定手牌vs对手范围的胜率相近（差异<0.05%）
2. 固定手牌vs对手范围的平局率相近（差异<0.05%）
3. 自己范围vs对手范围的胜率相近（差异<0.05%）
4. 自己范围vs对手范围的平局率相近（差异<0.05%）

## Solver 参数

| 参数 | 值 |
|------|-----|
| 起始底池 | 100 |
| 有效筹码 | 500 |
| 下注尺寸 | 50% pot |
| 加注尺寸 | 100% pot |
| 目标可剥削度 | 0.1% |
| 最大迭代次数 | 1000 |

## 范围策略

- **范围刷新间隔**: 每100组实验随机生成一次
- **OOP范围大小**: 60-120种手牌
- **IP范围大小**: 40-100种手牌
- **使用的范围组数**: 1000

## 实验规模

- 生成场景数: 15898
- 四维度胜率相近的场景对（差异<0.05%）: 14
- 策略差异显著(>15%)的场景对: 4

## 关键发现

### ⚠️ 四维度胜率标量不足以决定最优策略

在 14 对四维度胜率相近的场景中，有 4 对（28.6%）的策略差异显著。

**结论：即使手牌胜率、手牌平局率、范围胜率、范围平局率都精确匹配（差异<0.05%），最优策略仍然可能完全不同。**

### 策略差异显著的反例

---

#### 反例 1

**场景1:**
- 公共牌: `5c Th 7c 2d Qh`
- OOP手牌: `QcQs`
- OOP范围 (66种): `66,A5s,QQ,AA,44,A6s,K6s,33,A2s,Q8s,88,KQs,77,K4s,KTs,JTs,99,Q6s,ATs,QTs,AJs,QJs,K7s,Q5s,K9s,K8s,K2o,J5o,Q4o,T7s,65o,T8s,62o,74o,52s,Q6o,63o,86s,32s,T5o,98o,43o,42o,AQs,K7o,T3o,K3s,98s,T8o,Q4s,J3s,Q9s,54s,QJo,63s,J6o,ATo,T7o,J9o,74s,KK,Q3s,T3s,T5s,76s,J3o`
- IP范围 (74种): `88,KQs,K6s,K9s,J9s,AQs,77,Q9s,Q4s,K2s,33,22,Q5s,ATs,A8s,66,JTs,JJ,AJs,K4s,A6s,TT,A7s,J8s,99,55,QJs,AKs,KK,74s,94o,A8o,44,Q6o,KJo,72o,62s,94s,Q8o,KQo,J3o,T6s,86s,J4o,96s,97s,T4s,J2o,AA,T4o,75s,53s,Q8s,84o,52o,Q3o,92s,T2s,43s,KTs,T9s,J7s,74o,K9o,86o,A9s,K6o,54o,72s,64s,52s,QQ,A9o,T5o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=51.202%, 平局率=0.840%
- EV: 167.22, Solver Equity: 100.00%
- 动作EV: Check:166.84, Bet:167.22
- 加权EV: 167.22
- 策略: `{"Check":0.001004,"Bet":0.998996}`

**场景2:**
- 公共牌: `6d Jd As 8d 9d`
- OOP手牌: `AdJs`
- OOP范围 (104种): `K8s,AJs,QQ,88,Q8s,Q5s,A2s,QTs,99,K2s,JJ,K3s,J8s,A8s,KTs,A5s,KQs,Q2s,TT,J9s,77,Q3s,A6s,KK,Q6s,AKs,44,KJs,55,AQs,J7s,K7s,33,A3s,A7s,ATs,Q7s,A4s,K9s,K4s,22,87s,T2o,ATo,AA,K6o,AJo,93s,T4s,J7o,94s,QJs,98o,Q4o,86s,T3s,54s,QTo,J6o,J8o,Q9s,72o,63o,A2o,A7o,T5o,96o,K8o,Q2o,A8o,84o,64s,K3o,62s,96s,AKo,J2o,75o,J3o,32o,J2s,T9s,Q8o,95o,A5o,62o,T5s,J9o,J4s,A9s,K7o,JTo,K4o,32s,75s,95s,42o,T6s,Q7o,KJo,66,73s,83o,T7o`
- IP范围 (53种): `KTs,J8s,A6s,K3s,AQs,K5s,QQ,KJs,Q9s,KQs,Q4s,JTs,A3s,44,Q5s,J7s,33,QJs,A4s,Q3s,55,A6o,Q2s,66,83s,83o,98s,95s,Q6s,74o,T7o,J3s,82o,KTo,22,K6o,62s,AJs,74s,T8o,KQo,J2o,T5o,KK,J2s,97s,A5s,AQo,A7o,88,52s,K8s,53s`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=51.201%, 平局率=0.846%
- EV: 153.04, Solver Equity: 100.00%
- 动作EV: Check:153.12, Bet:149.67
- 加权EV: 153.04
- 策略: `{"Check":0.975351,"Bet":0.024649}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.000%, 范围平局率差异: 0.006%
- **策略差异: 97.4%**
- **加权EV差异: 14.18**

---

#### 反例 2

**场景1:**
- 公共牌: `8h Qs 3c 6c Ks`
- OOP手牌: `6h6s`
- OOP范围 (82种): `Q6s,K8s,ATs,Q5s,QJs,Q7s,KK,J9s,K6s,99,J7s,A7s,K9s,A6s,TT,33,J8s,K4s,88,A2s,K5s,77,Q2s,A9s,55,Q4s,JTs,QTs,QQ,K2s,A8s,AJs,T3s,54o,J3o,T6o,Q3o,76s,T3o,43s,K3s,AA,T8s,43o,Q8s,32o,Q2o,KQo,A7o,A3s,94o,52s,T4s,Q7o,87s,KQs,J3s,63s,75s,76o,K2o,T4o,Q9s,A9o,K5o,JTo,66,64s,98s,83o,K7s,62o,J4s,95s,73s,K6o,J4o,J5o,98o,65s,86o,QJo`
- IP范围 (48种): `QJs,JTs,KJs,TT,K5s,66,J8s,JJ,44,KK,AKs,KQs,88,K7s,J9s,K8s,K3s,AJs,Q9s,73o,J7s,A5o,43s,86s,KTs,AA,T7s,QQ,J2s,98s,ATs,J3s,A7o,T2s,74s,A9o,K9s,J3o,A4s,T9s,73s,K2s,94s,85s,A2o,J4o,A2s,84s`
- 手牌: 胜率=95.890%, 平局率=0.000%
- 范围: 胜率=53.913%, 平局率=1.602%
- EV: 129.02, Solver Equity: 95.89%
- 动作EV: Check:125.17, Bet:129.02
- 加权EV: 129.02
- 策略: `{"Check":0,"Bet":1}`

**场景2:**
- 公共牌: `4s Qc 4d 3s 2s`
- OOP手牌: `AsKs`
- OOP范围 (86种): `Q8s,JTs,K8s,KTs,AQs,AKs,Q7s,KQs,JJ,Q3s,K4s,K6s,J9s,88,A5s,QJs,Q6s,J7s,QTs,A7s,A4s,AJs,66,QQ,A6s,K9s,99,AA,44,55,Q9s,TT,K7s,A9s,J4o,82o,T9o,QJo,J6o,92s,82s,72s,T8s,83s,AJo,T5o,87o,J2o,J6s,K4o,K5s,ATo,73s,T7o,T2s,77,32s,K5o,62s,J4s,86s,ATs,33,T8o,K3s,75s,KQo,54o,53s,52o,Q3o,63o,52s,A6o,J2s,32o,Q7o,92o,53o,65o,A9o,K7o,73o,T3o,A8s,84s`
- IP范围 (64种): `K8s,A3s,K5s,J8s,Q5s,Q9s,Q2s,Q8s,A4s,KTs,QJs,AKs,33,AQs,A5s,44,A7s,AA,QQ,77,K2s,KJs,Q7s,JJ,66,JTo,55,J6o,75o,93s,94o,54s,T3s,22,J9s,52s,73s,A4o,T6s,T7s,83s,65o,86o,J2s,QTs,A2o,87o,43o,KTo,J4s,KJo,J7s,85o,Q3s,T7o,A3o,J5o,74o,73o,Q6s,K8o,63o,J7o,92o`
- 手牌: 胜率=95.894%, 平局率=0.000%
- 范围: 胜率=53.897%, 平局率=1.556%
- EV: 124.98, Solver Equity: 95.89%
- 动作EV: Check:124.98, Bet:124.12
- 加权EV: 124.98
- 策略: `{"Check":1,"Bet":0}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.004%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.016%, 范围平局率差异: 0.046%
- **策略差异: 100.0%**
- **加权EV差异: 4.04**

---

#### 反例 3

**场景1:**
- 公共牌: `Qh 6s 4s 8s As`
- OOP手牌: `KhKs`
- OOP范围 (107种): `ATs,A3s,Q4s,Q8s,44,AJs,Q9s,K9s,K7s,A4s,KTs,J9s,KQs,K8s,J7s,KK,Q2s,55,Q6s,99,K2s,A9s,AKs,K4s,A7s,A6s,K6s,QJs,K3s,KJs,A2s,TT,J8s,A5s,66,JTs,77,JJ,K5s,33,22,AA,75o,63o,Q7o,94s,82o,94o,87o,J4s,T5o,96o,T8s,Q3o,85s,Q8o,97o,KJo,T3s,86o,43o,KQo,T5s,Q7s,AKo,J3s,K9o,K3o,96s,42o,32o,ATo,A8s,64s,QTs,93o,72o,62o,42s,Q5o,AJo,T2s,98o,Q5s,64o,Q9o,83o,J5s,53s,75s,J7o,92o,Q2o,T7s,A5o,95s,T8o,QQ,82s,54s,T4o,A7o,87s,A3o,AQs,A6o,J4o`
- IP范围 (59种): `K7s,QJs,AKs,JJ,A6s,Q5s,A2s,44,Q3s,AJs,99,J9s,66,88,A8s,J8s,J7s,AQs,KTs,Q6s,Q9s,A5s,K4s,73o,85s,22,A7o,54s,87s,A4s,93o,A4o,T3s,83s,A7s,72o,QTs,Q3o,72s,T9o,55,Q8s,97o,42o,95s,J9o,KJs,84s,AA,73s,Q4o,Q2o,K6o,54o,93s,KK,J3s,98o,J2o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=51.360%, 平局率=0.535%
- EV: 156.89, Solver Equity: 100.00%
- 动作EV: Check:156.89, Bet:153.30
- 加权EV: 156.89
- 策略: `{"Check":1,"Bet":0}`

**场景2:**
- 公共牌: `4s 7h 2h 3s 9h`
- OOP手牌: `AhQh`
- OOP范围 (70种): `K7s,JJ,QQ,22,J9s,AJs,77,AQs,K8s,J8s,ATs,A6s,K3s,33,Q8s,A3s,K6s,Q3s,TT,55,99,KJs,K2s,44,66,K5s,KQs,Q2s,75o,K2o,K9o,J5s,32o,Q4o,A7o,K5o,JTo,AA,85s,92o,98s,42s,T6o,T9s,JTs,62o,43s,Q5s,T2o,74s,T4s,97o,95s,J2o,Q7s,62s,54s,76o,J4o,63o,65o,T5s,Q3o,93o,ATo,A9o,53s,A8s,87o,74o`
- IP范围 (73种): `JTs,Q9s,A9s,KK,KJs,JJ,ATs,K2s,Q2s,77,K8s,Q8s,AKs,A2s,QJs,Q6s,A7s,AA,KTs,33,55,A8s,Q7s,Q4s,Q5s,A6s,KQs,K6s,AQs,94s,96o,72s,93s,96s,T3s,87s,73o,A4s,98s,A4o,42s,75o,T2s,76o,K9s,85s,Q7o,65o,76s,T7s,J4s,J3s,63o,J4o,Q2o,T9s,A8o,52s,97s,K4s,A7o,KTo,82s,82o,J8s,KJo,K3s,J6o,A5o,83o,84o,43o,K6o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=51.381%, 平局率=0.501%
- EV: 158.19, Solver Equity: 100.00%
- 动作EV: Check:158.48, Bet:157.74
- 加权EV: 158.19
- 策略: `{"Check":0.603806,"Bet":0.396194}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.020%, 范围平局率差异: 0.034%
- **策略差异: 39.6%**
- **加权EV差异: 1.30**

---

#### 反例 4

**场景1:**
- 公共牌: `Qc 2c Ac 9c 7s`
- OOP手牌: `KcQh`
- OOP范围 (70种): `22,QTs,55,A3s,KJs,AKs,88,JTs,QQ,99,A5s,A6s,AJs,Q5s,A8s,J8s,A4s,QJs,Q9s,JJ,ATs,K7s,77,KK,AQs,K8s,33,A7s,T5s,K9o,J6o,52o,74o,Q2o,43s,K4o,J2o,86o,QJo,K2s,73o,QTo,84o,Q6s,ATo,K5o,44,63s,62s,42o,KTo,A9s,J2s,Q2s,K4s,92s,Q6o,T8o,J5s,66,Q7o,43o,62o,KQo,82o,65s,T7s,98o,T3s,72o`
- IP范围 (76种): `KTs,TT,A5s,A7s,K8s,AQs,J8s,K5s,J9s,J7s,KJs,K4s,K3s,QJs,88,QQ,55,KK,KQs,22,K9s,Q2s,A8s,JTs,AA,A3s,77,JJ,K6s,AKs,Q3o,K8o,Q9s,A2s,T3s,A4s,74o,98s,93s,Q5s,86s,J3o,94s,JTo,Q7o,84s,J6s,Q4s,87s,T9o,64s,A6s,94o,T2s,K4o,99,95o,K2s,92o,J2s,A5o,72s,J5o,Q2o,T7o,QTs,J9o,T5o,97o,J4o,53o,Q6o,J5s,65o,98o,Q8o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=49.206%, 平局率=0.583%
- EV: 152.56, Solver Equity: 100.00%
- 动作EV: Check:152.69, Bet:152.11
- 加权EV: 152.56
- 策略: `{"Check":0.769471,"Bet":0.230529}`

**场景2:**
- 公共牌: `Ks 7d 4d 5h 2c`
- OOP手牌: `KcKh`
- OOP范围 (113种): `Q9s,99,A9s,K8s,KK,K6s,KQs,K3s,K2s,K4s,KJs,Q6s,KTs,QJs,44,A6s,TT,Q7s,Q8s,JTs,AJs,JJ,Q2s,Q4s,K9s,K7s,Q3s,55,A2s,A5s,K5s,A4s,J7s,A8s,J9s,A3s,QTs,66,AA,A7s,AQs,AKs,77,Q5s,22,32s,A7o,Q8o,95s,T5o,J5o,96o,Q6o,85o,96s,87o,ATs,62s,JTo,87s,J8s,83o,J2o,73o,J6o,86o,95o,QTo,Q3o,K3o,88,74s,J3o,43s,73s,J5s,63s,93o,82s,92o,J4s,63o,42o,62o,K2o,A8o,K4o,52s,76o,84o,72s,52o,85s,53o,ATo,Q2o,A3o,J8o,T6o,98o,86s,T3s,Q4o,97s,QJo,54o,32o,64o,74o,T9o,J7o,75s,QQ`
- IP范围 (56种): `J7s,AJs,KQs,Q8s,A9s,K9s,QJs,ATs,22,K4s,AQs,K7s,55,J9s,K5s,Q2s,33,A7s,K3s,AKs,K2s,TT,Q7o,Q5s,J2s,83o,K8o,T5s,AKo,Q6s,K8s,Q7s,J4s,T8o,K2o,QQ,A9o,J2o,42o,92o,K6s,76s,87o,T3o,76o,97s,KQo,74o,Q3s,J3o,99,96s,84o,A4s,T7s,53o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=49.168%, 平局率=0.607%
- EV: 148.90, Solver Equity: 100.00%
- 动作EV: Check:137.15, Bet:148.90
- 加权EV: 148.90
- 策略: `{"Check":0,"Bet":1}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.038%, 范围平局率差异: 0.024%
- **策略差异: 76.9%**
- **加权EV差异: 3.66**

