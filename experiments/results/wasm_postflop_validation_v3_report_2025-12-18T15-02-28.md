# 跨公共牌四维度胜率-策略验证实验报告 V3

## 实验时间

2025-12-18T15-02-28

## 累计统计

| 指标 | 本次 | 累计 |
|------|------|------|
| 运行次数 | 1 | 4 |
| 场景数 | 100000 | 163564 |
| 胜率相近对数 | 646 | 715 |
| 策略差异显著反例 | 160 | 185 |

## 实验改进

**本版本使用 OMPEval (C++) 计算范围胜率，速度比 poker-odds-calc 快约60倍。**
**新增：每100组实验随机生成一次OOP和IP范围，增加实验多样性。**
**新增：累计数据保存，多次运行结果自动合并。**

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

- 生成场景数: 100000
- 四维度胜率相近的场景对（差异<0.05%）: 646
- 策略差异显著(>15%)的场景对: 160

## 关键发现

### ⚠️ 四维度胜率标量不足以决定最优策略

在 646 对四维度胜率相近的场景中，有 160 对（24.8%）的策略差异显著。

**结论：即使手牌胜率、手牌平局率、范围胜率、范围平局率都精确匹配（差异<0.05%），最优策略仍然可能完全不同。**

### 策略差异显著的反例

---

#### 反例 1

**场景1:**
- 公共牌: `2s 8s 3h Ad Qd`
- OOP手牌: `AcJc`
- OOP范围 (95种): `A3s,K7s,KJs,AJs,A6s,A8s,J9s,K6s,Q8s,77,Q3s,AKs,KTs,A4s,33,K8s,QQ,K5s,TT,K4s,QTs,55,A7s,K9s,88,KQs,99,Q4s,22,Q7s,AQs,JJ,QJs,ATs,66,K2s,A2s,Q2s,JTo,J2s,Q4o,A3o,96s,86o,85s,82s,KJo,43o,A4o,44,A9s,AJo,K3o,A5o,95o,62s,T9s,KK,32o,84o,74o,Q3o,AKo,K8o,92o,42s,QTo,J7s,T4o,98s,T5s,J2o,J4s,95s,64s,76o,T8o,K7o,98o,75o,84s,T4s,J5s,J8s,Q5o,K4o,K9o,82o,54s,63o,54o,87s,QJo,J7o,97s`
- IP范围 (55种): `J7s,Q8s,K8s,K9s,K4s,ATs,K5s,Q7s,A6s,33,TT,JTs,A9s,KJs,JJ,A7s,KQs,66,QJs,A8s,AJs,KK,T6o,43o,K9o,53o,A7o,T3s,Q3s,84s,92o,Q6o,83o,A4o,AA,63s,52s,ATo,T6s,A2s,75s,K7o,83s,85o,J8s,94o,73s,T2s,KTo,J7o,K2o,A5o,QQ,65o,J4o`
- 手牌: 胜率=91.459%, 平局率=0.712%
- 范围: 胜率=52.716%, 平局率=0.738%
- EV: 107.29, Solver Equity: 91.81%
- 动作EV: Check:106.14, Bet:107.29
- 加权EV: 107.29
- 策略: `{"Check":0,"Bet":1}`

**场景2:**
- 公共牌: `8s Ah 9d 4s 2h`
- OOP手牌: `4h2s`
- OOP范围 (84种): `K8s,J7s,AQs,Q5s,Q8s,A7s,K9s,KJs,AA,99,K3s,AKs,Q3s,A9s,22,J9s,77,K2s,JJ,TT,66,KQs,Q7s,QTs,ATs,55,JTs,A4s,88,Q2s,44,K6s,A8s,98o,T6s,95o,74s,75o,94o,T8o,T7o,94s,32s,98s,J8o,86o,83s,Q4o,87o,64s,85o,J4s,Q4s,97s,JTo,65o,Q5o,A3o,33,63s,T5o,A5s,T4o,QJo,53s,QQ,52o,K4s,42o,A7o,AQo,J2o,97o,K9o,76o,J8s,T8s,Q6s,54o,53o,J7o,84o,J6o,32o`
- IP范围 (98种): `AJs,A9s,A2s,A4s,JTs,66,Q2s,A5s,AA,TT,A6s,K5s,A3s,Q6s,JJ,Q3s,K2s,Q9s,55,33,22,J9s,KQs,77,K7s,K4s,A8s,Q5s,KTs,K8s,Q7s,J8s,Q4s,ATs,Q8s,J7s,88,44,KJs,ATo,J3o,43s,AJo,97s,74o,42s,A8o,73o,T8o,85s,K6o,87o,KTo,54o,84s,53s,J5s,QJs,QJo,97o,AQs,JTo,62o,Q5o,K6s,63o,75o,K5o,A5o,92o,K9o,43o,T3s,76o,T8s,76s,A9o,95o,62s,T2o,87s,J6o,KK,42o,K3s,T2s,K3o,65s,94s,Q8o,75s,32s,64o,J6s,A7o,KJo,83s,K7o`
- 手牌: 胜率=91.450%, 平局率=0.743%
- 范围: 胜率=52.726%, 平局率=0.773%
- EV: 107.84, Solver Equity: 91.82%
- 动作EV: Check:108.02, Bet:107.42
- 加权EV: 107.84
- 策略: `{"Check":0.692909,"Bet":0.307091}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.009%, 手牌平局率差异: 0.032%
- 范围胜率差异: 0.010%, 范围平局率差异: 0.035%
- **策略差异: 69.3%**
- **加权EV差异: 0.55**

---

#### 反例 2

**场景1:**
- 公共牌: `Kd 4d 2d 5h 6s`
- OOP手牌: `KhJd`
- OOP范围 (113种): `J7s,Q9s,K9s,AJs,J8s,QJs,K2s,66,KJs,J9s,K8s,A8s,K3s,Q4s,AQs,JTs,A3s,44,Q6s,QTs,K7s,ATs,A5s,A9s,Q3s,88,Q2s,K6s,QQ,JJ,AKs,K5s,22,Q5s,A4s,KTs,TT,Q7s,A6s,Q8s,A2s,K4s,A7s,55,AA,J5s,T8s,T8o,85o,92o,52o,ATo,32s,94s,K5o,99,A4o,Q2o,J2s,73o,AJo,63o,T7o,K4o,62o,82o,J5o,Q7o,Q8o,J4o,Q6o,87o,T9o,J3o,A5o,92s,74o,Q3o,95o,A8o,QTo,62s,T5s,86s,72o,J7o,K3o,72s,K2o,73s,96s,KTo,75o,43s,KK,63s,K9o,76o,97s,J3s,76s,T4s,K7o,T2o,77,J2o,KJo,93s,AKo,32o,KQs,65s,KQo`
- IP范围 (51种): `Q3s,JTs,A7s,J9s,A2s,Q4s,A5s,K9s,A3s,33,88,Q5s,55,K5s,QTs,K4s,Q7s,99,KTs,77,AJs,AA,A8o,Q4o,A6o,98s,95o,J6o,86s,85o,95s,AQo,72o,K9o,T9o,62s,T2s,ATs,Q3o,QQ,JJ,J8s,T2o,J6s,76o,K2s,T7s,K4o,ATo,53s,KTo`
- 手牌: 胜率=77.778%, 平局率=0.000%
- 范围: 胜率=50.846%, 平局率=1.349%
- EV: 79.85, Solver Equity: 77.78%
- 动作EV: Check:78.56, Bet:79.85
- 加权EV: 79.85
- 策略: `{"Check":0,"Bet":1}`

**场景2:**
- 公共牌: `2d 6d 9c Qh Qd`
- OOP手牌: `Ad9s`
- OOP范围 (115种): `55,99,Q6s,Q4s,A2s,Q5s,KQs,A8s,Q2s,K7s,K3s,K9s,QTs,Q7s,K2s,K6s,K8s,JJ,33,A7s,K5s,K4s,A6s,KJs,QJs,A5s,AQs,44,Q8s,A3s,QQ,AA,J9s,Q3s,88,A9s,JTs,AJs,J7s,KTs,77,66,ATs,22,J8s,Q9s,KTo,K8o,J3o,62o,J6o,72s,Q3o,83s,T5s,98o,AJo,T8s,96o,53o,K4o,76o,AKo,32s,KQo,A9o,J6s,A5o,84s,43o,95s,97s,A6o,Q6o,87s,86o,42o,52s,74s,97o,63o,54s,J2s,J5s,85o,95o,J4s,92s,Q5o,J8o,A4s,KJo,76s,Q4o,A8o,T7s,65o,T3s,A3o,65s,87o,A4o,75s,94o,T2o,J5o,53s,T6s,ATo,73s,J9o,Q9o,85s,KK,Q8o`
- IP范围 (90种): `AQs,QQ,KTs,A6s,A4s,J7s,KK,QJs,99,K5s,Q5s,K3s,ATs,66,Q9s,K4s,Q8s,Q4s,A3s,A8s,Q2s,AA,JJ,QTs,J9s,Q7s,K9s,K7s,K8s,K2s,JTs,A2s,TT,J8s,K6s,88,98s,K5o,86o,Q6s,T6s,83o,AKo,J3s,73o,A7o,82o,77,64s,Q2o,T8s,Q4o,52o,A2o,T7s,Q9o,ATo,95s,85o,J7o,54s,K2o,62o,T3o,72o,T4o,A6o,A4o,75o,A7s,T2o,Q8o,55,72s,T6o,Q6o,T2s,KQo,K7o,43s,K9o,85s,T4s,QJo,KJs,64o,87s,87o,KJo,QTo`
- 手牌: 胜率=77.823%, 平局率=0.000%
- 范围: 胜率=50.848%, 平局率=1.330%
- EV: 80.72, Solver Equity: 77.82%
- 动作EV: Check:79.83, Bet:81.03
- 加权EV: 80.72
- 策略: `{"Check":0.256039,"Bet":0.743961}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.046%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.002%, 范围平局率差异: 0.019%
- **策略差异: 25.6%**
- **加权EV差异: 0.87**

---

#### 反例 3

**场景1:**
- 公共牌: `9d Ah Kd 4d Kc`
- OOP手牌: `3h2c`
- OOP范围 (113种): `J7s,Q9s,K9s,AJs,J8s,QJs,K2s,66,KJs,J9s,K8s,A8s,K3s,Q4s,AQs,JTs,A3s,44,Q6s,QTs,K7s,ATs,A5s,A9s,Q3s,88,Q2s,K6s,QQ,JJ,AKs,K5s,22,Q5s,A4s,KTs,TT,Q7s,A6s,Q8s,A2s,K4s,A7s,55,AA,J5s,T8s,T8o,85o,92o,52o,ATo,32s,94s,K5o,99,A4o,Q2o,J2s,73o,AJo,63o,T7o,K4o,62o,82o,J5o,Q7o,Q8o,J4o,Q6o,87o,T9o,J3o,A5o,92s,74o,Q3o,95o,A8o,QTo,62s,T5s,86s,72o,J7o,K3o,72s,K2o,73s,96s,KTo,75o,43s,KK,63s,K9o,76o,97s,J3s,76s,T4s,K7o,T2o,77,J2o,KJo,93s,AKo,32o,KQs,65s,KQo`
- IP范围 (51种): `Q3s,JTs,A7s,J9s,A2s,Q4s,A5s,K9s,A3s,33,88,Q5s,55,K5s,QTs,K4s,Q7s,99,KTs,77,AJs,AA,A8o,Q4o,A6o,98s,95o,J6o,86s,85o,95s,AQo,72o,K9o,T9o,62s,T2s,ATs,Q3o,QQ,JJ,J8s,T2o,J6s,76o,K2s,T7s,K4o,ATo,53s,KTo`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=41.651%, 平局率=3.391%
- EV: 0.00, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.61
- 加权EV: -0.00
- 策略: `{"Check":0.999993,"Bet":0.000007}`

**场景2:**
- 公共牌: `Th 8d 7h Jh 2d`
- OOP手牌: `4c3c`
- OOP范围 (111种): `AKs,JJ,ATs,JTs,A2s,A7s,QQ,33,K8s,QJs,KTs,QTs,K6s,Q8s,J8s,TT,K9s,AJs,J9s,K7s,K5s,66,A3s,77,44,Q7s,KK,Q9s,A6s,Q4s,K2s,A5s,Q5s,A4s,88,AQs,99,K3s,A8s,A9s,Q6s,KJs,55,Q2s,J2s,A4o,T2s,K6o,J4s,84s,62s,53s,93s,64o,A8o,T8o,94o,98o,84o,A7o,K2o,73s,JTo,86s,T3s,ATo,A5o,93o,63s,J8o,A2o,75o,87o,65s,A6o,K5o,22,Q3s,J3o,97s,72s,53o,K7o,92o,42o,QJo,KQo,T5s,T4o,94s,Q7o,K3o,J7o,82s,43s,A3o,62o,T9o,T2o,Q6o,KQs,T5o,98s,54s,T7o,97o,T8s,KTo,AA,AQo,J5o`
- IP范围 (44种): `Q6s,Q7s,Q5s,K8s,77,55,A6s,88,QJs,KTs,A2s,TT,44,J9s,QTs,J8s,33,A5s,Q9o,A8o,76s,A6o,J3o,83s,T4s,96s,32o,K7s,A9o,AKs,82s,92o,J4s,T8s,T3s,A4o,J8o,Q5o,ATs,AJo,98s,84s,95o,A2o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=41.683%, 平局率=3.376%
- EV: 0.02, Solver Equity: 0.00%
- 动作EV: Check:-0.00, Bet:0.09
- 加权EV: 0.02
- 策略: `{"Check":0.767577,"Bet":0.232423}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.032%, 范围平局率差异: 0.015%
- **策略差异: 23.2%**
- **加权EV差异: 0.02**

---

#### 反例 4

**场景1:**
- 公共牌: `Th 5h 9s 9c 3h`
- OOP手牌: `4c2s`
- OOP范围 (109种): `J7s,A8s,KK,K6s,KQs,Q2s,A6s,QJs,A9s,66,QQ,44,K9s,A2s,QTs,JTs,JJ,A3s,J8s,A5s,55,AQs,K3s,Q7s,A7s,Q3s,AA,Q4s,KJs,AKs,K2s,88,K5s,K8s,Q8s,J9s,77,22,TT,Q9s,K7s,K4s,Q6s,85o,98s,Q5s,A4s,42s,A3o,T5o,Q2o,33,T9s,J4s,T6s,K8o,A9o,KJo,84s,QJo,43s,42o,64o,K2o,T2o,AJs,A5o,76s,53s,75s,74s,J9o,75o,T8o,62s,T3o,93o,83o,J3o,K7o,J3s,K5o,ATo,87s,T5s,K4o,94s,65o,53o,J5s,A4o,86s,32o,QTo,Q5o,73s,A6o,KQo,87o,84o,63s,J2s,85s,JTo,Q4o,K6o,T2s,T8s,95o`
- IP范围 (64种): `A6s,66,77,Q5s,A3s,K4s,KQs,K2s,JTs,A8s,KJs,33,A5s,TT,KTs,K7s,88,A4s,J8s,K6s,ATs,99,JJ,AA,QQ,KJo,92s,85o,AKo,63s,A2s,95o,T3o,A3o,Q8s,Q3o,83o,74s,JTo,J2s,T9o,J5s,AJs,87s,44,K9o,Q9o,72o,96s,J4s,T5s,Q7o,ATo,82s,J3s,K9s,43s,A4o,74o,62o,94s,K8o,KTo,K5s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=45.770%, 平局率=0.916%
- EV: 0.00, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-2.60
- 加权EV: 0.00
- 策略: `{"Check":1,"Bet":0}`

**场景2:**
- 公共牌: `4h Ad Qs 2d 9h`
- OOP手牌: `7h6c`
- OOP范围 (79种): `77,TT,KJs,K4s,QTs,A4s,Q5s,A9s,J9s,J8s,A2s,Q8s,AA,K5s,AJs,A5s,QQ,K2s,Q7s,KTs,Q4s,Q2s,ATs,J7s,K9s,22,A3s,Q9s,33,K8s,55,T9o,T9s,74s,KK,JTs,42s,64o,54o,J7o,87s,Q3s,K2o,93o,J3s,J6s,J3o,52o,T7s,J6o,76o,JTo,99,K6o,KQs,JJ,82s,62s,A5o,T7o,Q6s,QTo,95s,88,A3o,T4o,75o,83o,J4o,AKs,32s,K6s,42o,85s,85o,53s,83s,J2o,96s`
- IP范围 (53种): `AQs,K2s,K7s,A8s,J7s,Q2s,K3s,J9s,K5s,66,K6s,Q5s,KTs,QQ,A4s,Q4s,55,44,A3s,TT,K4s,Q2o,93o,K3o,82o,J5o,93s,AJs,96s,74o,94s,T6o,J4s,A3o,AA,A6o,98o,K9o,A2o,T8s,52s,86o,AQo,88,84o,K2o,Q6o,52o,42s,T5o,T7s,J3o,82s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=45.734%, 平局率=0.885%
- EV: -0.01, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.02
- 加权EV: -0.01
- 策略: `{"Check":0.690259,"Bet":0.309741}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.036%, 范围平局率差异: 0.031%
- **策略差异: 31.0%**
- **加权EV差异: 0.01**

---

#### 反例 5

**场景1:**
- 公共牌: `6s 2h Ks 8c Qd`
- OOP手牌: `4h3d`
- OOP范围 (66种): `44,J9s,A4s,Q9s,A8s,A6s,Q7s,Q8s,JJ,ATs,Q2s,Q6s,QTs,K4s,KTs,Q4s,K2s,AQs,K5s,33,K9s,Q5s,66,22,QJs,55,QJo,97o,75o,76s,JTs,54o,A7o,84o,A9o,T9s,63s,Q7o,97s,72o,T4o,A3s,ATo,KJo,K3s,A2s,T5o,Q6o,T8s,65o,99,32s,62s,83o,92o,72s,KQs,K7o,43s,43o,K3o,K5o,K7s,A2o,98o,75s`
- IP范围 (80种): `K4s,K6s,A7s,QQ,JJ,JTs,77,AA,Q5s,A5s,K2s,Q7s,KTs,A8s,ATs,K5s,Q3s,QTs,KJs,AQs,A3s,Q6s,K7s,J9s,A9s,66,A6s,Q4s,44,Q2s,J8s,Q9s,85o,T7o,J8o,T5o,JTo,83s,T3o,J6o,AQo,KTo,94o,94s,AJs,87s,42s,96s,A6o,83o,52o,52s,T9o,J2o,42o,74s,85s,QJs,T4o,Q5o,95s,32s,K4o,T5s,K8o,Q2o,65s,T4s,22,82s,T2s,A3o,75s,62o,Q4o,K9s,65o,Q9o,J6s,53o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=48.033%, 平局率=1.380%
- EV: -0.10, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.29
- 加权EV: -0.10
- 策略: `{"Check":0.667344,"Bet":0.332656}`

**场景2:**
- 公共牌: `Jh Ts 3d 6s Td`
- OOP手牌: `7s4s`
- OOP范围 (95种): `99,Q3s,Q6s,55,AA,Q9s,Q7s,A4s,JTs,QTs,A9s,A8s,AQs,44,KK,AKs,66,Q5s,ATs,A3s,KJs,K2s,22,K9s,77,A2s,K7s,TT,KQs,33,JJ,AJs,J7s,A7s,Q2s,K3s,QQ,K4s,97s,T8o,75s,83s,J7o,74s,96s,Q4o,QJs,T2s,86o,84s,A4o,92s,98o,J9o,Q3o,J4s,84o,T8s,A8o,J4o,T7s,T5o,64s,Q6o,KTs,63o,A7o,J3s,42o,43o,76o,43s,A5s,53s,83o,42s,93o,T3s,AJo,A2o,T7o,52s,T2o,A9o,J3o,AKo,98s,K9o,64o,85o,JTo,96o,K6s,Q5o,KTo`
- IP范围 (70种): `K3s,QQ,66,TT,22,K8s,44,77,99,A4s,J7s,J8s,KTs,K4s,A6s,AJs,Q3s,Q5s,KQs,55,Q2s,ATs,88,KJs,K6s,Q6s,A7s,QJs,Q7o,Q8s,T8s,Q4s,Q8o,65o,KQo,K8o,94o,T9s,AA,JTo,JJ,KJo,96s,J7o,53s,AQs,K2o,Q5o,85o,J6s,T2s,K5s,QJo,Q2o,ATo,Q7s,J4s,K9s,73s,J9o,Q6o,Q3o,KK,A3o,32s,A2s,QTo,AKo,T7o,J9s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=48.043%, 平局率=1.404%
- EV: 0.73, Solver Equity: 0.00%
- 动作EV: Check:-0.00, Bet:0.75
- 加权EV: 0.73
- 策略: `{"Check":0.024747,"Bet":0.975253}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.009%, 范围平局率差异: 0.025%
- **策略差异: 64.3%**
- **加权EV差异: 0.83**

---

#### 反例 6

**场景1:**
- 公共牌: `Js Kh 8d Qd 2c`
- OOP手牌: `4c3h`
- OOP范围 (99种): `KK,88,K6s,AQs,99,Q6s,A5s,K4s,J9s,AKs,Q4s,K7s,QQ,77,J8s,Q2s,Q5s,JJ,Q3s,ATs,Q9s,K3s,TT,A4s,A8s,K8s,44,Q7s,A7s,QJs,66,22,QTs,A6s,AJs,K2s,A2s,KJs,A3s,K9s,92o,52s,84s,93o,76o,J2s,A9s,T3s,KTs,Q8s,K5s,55,43s,J5o,65s,82o,T8o,T4o,Q3o,83s,73o,K5o,73s,95s,A9o,J7s,94s,J3s,T5s,JTo,72s,Q4o,97o,92s,97s,J6s,T9o,AQo,63s,J3o,KQs,J4s,84o,A6o,85o,KTo,75s,43o,Q2o,98o,53o,K2o,A4o,T5o,53s,63o,T9s,65o,T2s`
- IP范围 (63种): `AKs,A7s,J7s,AQs,A4s,KJs,AJs,K8s,J8s,Q6s,J9s,Q9s,JJ,K7s,AA,KK,K2s,QJs,A9s,33,Q5s,TT,KQs,Q4s,22,87o,K3o,J4s,94o,63s,A7o,JTo,A4o,83s,ATs,54s,KJo,T6o,64o,Q3o,95o,97o,T7s,A6o,K6o,ATo,32o,A6s,KQo,J3s,J7o,88,85s,53o,86o,A5s,Q9o,92o,J6o,A5o,Q7s,T8o,J5s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=45.932%, 平局率=3.349%
- EV: 0.13, Solver Equity: 0.00%
- 动作EV: Check:-0.00, Bet:0.38
- 加权EV: 0.13
- 策略: `{"Check":0.64724,"Bet":0.35276}`

**场景2:**
- 公共牌: `2d 9c 9s As 7c`
- OOP手牌: `4s3s`
- OOP范围 (65种): `K4s,KTs,AA,KK,A8s,K7s,J7s,Q8s,K9s,K3s,A2s,AQs,QJs,A4s,77,66,Q4s,A7s,Q2s,A3s,Q5s,QQ,K8s,88,Q6s,A9s,A6o,J4o,33,52s,65o,87s,JTo,92s,ATo,K4o,A4o,Q5o,Q8o,J2o,98s,KQs,T6s,53o,J7o,43s,K7o,TT,32s,J8s,84o,76o,QTs,J3s,Q2o,K5s,53s,KJs,A8o,63s,T9s,J4s,97s,JTs,T2s`
- IP范围 (40种): `A9s,33,Q9s,Q5s,Q7s,K9s,KK,QTs,88,KJs,K4s,A2s,J9s,77,A6s,K2s,32s,J8o,87o,Q6s,A8s,54o,63o,K2o,98s,KQs,Q2s,93s,QJs,87s,64s,KTs,A5o,ATo,A3s,Q3s,T3o,76o,96s,Q8o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=45.914%, 平局率=3.339%
- EV: -0.00, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-1.36
- 加权EV: -0.00
- 策略: `{"Check":0.999999,"Bet":0.000001}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.017%, 范围平局率差异: 0.010%
- **策略差异: 35.3%**
- **加权EV差异: 0.13**

---

#### 反例 7

**场景1:**
- 公共牌: `2h 5h 3d Qc 9c`
- OOP手牌: `7s4h`
- OOP范围 (99种): `AKs,Q3s,33,QQ,99,AQs,JTs,K5s,A6s,Q5s,QTs,K9s,44,Q8s,88,K7s,J8s,Q4s,K6s,J7s,KJs,Q7s,KQs,QJs,Q2s,A7s,AJs,KK,K8s,A2s,A8s,KTs,66,A9s,JJ,A3s,K2s,22,ATs,76o,87o,82s,T6o,74s,98o,J4s,84s,K6o,85o,J6s,65s,T2o,Q4o,JTo,85s,K4o,95s,97o,43o,A7o,J6o,QTo,63o,96o,KTo,75o,K7o,63s,Q6s,K9o,AQo,A5s,KJo,T8s,AKo,T5o,A4o,T2s,T9o,42s,K5o,T8o,65o,T4o,ATo,83o,75s,Q7o,52s,86s,52o,K4s,87s,86o,62s,93o,77,AA,74o`
- IP范围 (41种): `A6s,A5s,Q5s,K7s,QQ,22,44,J9s,A9s,A2s,AJs,Q3s,A3s,J7s,A7s,K4s,32s,Q9s,J3s,KTo,Q8o,Q4o,A9o,AQs,J5s,T9o,J4o,54s,84o,J8s,A4s,J9o,75s,T2o,T6o,K3s,T6s,AKo,63o,82o,52o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=45.758%, 平局率=0.694%
- EV: -0.04, Solver Equity: 0.00%
- 动作EV: Check:-0.00, Bet:-0.08
- 加权EV: -0.04
- 策略: `{"Check":0.539954,"Bet":0.460046}`

**场景2:**
- 公共牌: `6c 8h Kd 4h 3h`
- OOP手牌: `9s2s`
- OOP范围 (64种): `Q5s,QQ,K3s,Q3s,QJs,JTs,KK,K4s,A9s,JJ,AQs,KQs,TT,A7s,88,A3s,A2s,Q9s,AJs,K6s,A6s,99,J9s,77,K9s,T2o,52o,96s,93s,64o,92s,K5o,T5o,82o,87o,QTo,J5s,72o,J6o,A4o,62o,KTo,54o,A8o,KTs,T3s,K6o,Q8s,T8o,Q5o,33,Q8o,97o,AJo,JTo,K3o,K9o,42o,Q6o,T5s,32s,K5s,83s,95s`
- IP范围 (98种): `QJs,KK,K8s,33,77,K2s,A8s,AKs,KTs,A7s,Q7s,22,AQs,Q4s,K5s,A3s,J9s,Q5s,55,A2s,K9s,K7s,KQs,ATs,Q9s,J7s,J8s,TT,Q6s,Q8s,QTs,JJ,K3s,A4s,K6s,44,A6s,KJs,Q3s,93s,A5o,73s,J9o,A9s,AJs,75o,J6o,73o,A8o,Q2s,98o,96o,Q5o,Q7o,J6s,KJo,T5o,JTo,T6s,86o,76s,T6o,93o,84o,A5s,K5o,84s,98s,QJo,T5s,83s,42o,52o,K7o,AKo,82s,K9o,T4o,43s,K3o,J3o,54s,AJo,T2s,99,63o,Q8o,86s,T3s,T9s,74o,T7s,KTo,83o,Q4o,53s,A4o,75s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=45.771%, 平局率=0.689%
- EV: 0.02, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:0.19
- 加权EV: 0.02
- 策略: `{"Check":0.892261,"Bet":0.107739}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.012%, 范围平局率差异: 0.005%
- **策略差异: 35.2%**
- **加权EV差异: 0.06**

---

#### 反例 8

**场景1:**
- 公共牌: `2h 4d Kd Jc 3h`
- OOP手牌: `6c5d`
- OOP范围 (99种): `77,J9s,22,QTs,Q2s,A4s,QJs,KTs,K2s,33,ATs,A2s,K7s,Q8s,QQ,A7s,Q6s,Q3s,KQs,88,A8s,KK,44,JTs,Q9s,K5s,A6s,KJs,AA,Q7s,A3s,AQs,K8s,K4s,JJ,J8s,K3s,AJs,A5s,63o,76s,66,T5o,J5o,KJo,K3o,J6s,Q9o,A2o,96o,42s,95s,Q2o,A4o,55,J4o,93s,T9o,J5s,76o,J7o,J9o,97o,A9o,53s,K5o,T4s,65o,87s,J3o,Q4o,A3o,KTo,AKs,T3s,K7o,J6o,64s,87o,T7o,53o,92s,65s,86s,73s,A8o,T8s,K2o,97s,T9s,T6o,J2s,52o,64o,TT,54s,62s,92o,Q5s`
- IP范围 (69种): `KQs,A4s,Q9s,77,JTs,A8s,AKs,Q2s,AA,33,AQs,Q7s,A2s,KTs,KK,55,A3s,AJs,QTs,QJs,K4s,A7s,Q5s,J9s,J8s,A9s,J7s,T8s,K8s,T9s,96s,K3s,J2o,J4o,96o,KJo,AKo,72o,Q7o,94s,K2s,84o,T7s,JTo,J2s,Q5o,Q2o,43o,62o,J6o,92s,82s,85s,T4o,63s,T3s,A5o,T6s,AQo,85o,J4s,87o,93o,83o,A8o,99,86s,43s,T6o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=50.879%, 平局率=0.628%
- EV: 169.51, Solver Equity: 100.00%
- 动作EV: Check:169.22, Bet:169.75
- 加权EV: 169.51
- 策略: `{"Check":0.447542,"Bet":0.552458}`

**场景2:**
- 公共牌: `8h 4h Qs 2s 5c`
- OOP手牌: `6d3c`
- OOP范围 (77种): `J8s,K6s,Q2s,A9s,A6s,Q9s,K2s,KK,AKs,A8s,Q7s,A5s,AJs,33,J9s,Q4s,66,A3s,Q8s,K9s,K8s,AQs,44,ATs,JTs,Q6s,QTs,22,QQ,AA,63o,94s,97o,K4s,42o,K7s,Q6o,73s,63s,Q8o,J9o,85s,J7s,Q3o,T9s,83s,Q3s,KJs,95o,KJo,K3o,T6s,55,KTs,T6o,J3s,T8o,J4o,Q2o,K7o,J5o,T7s,84o,J4s,62o,54s,53s,A2s,98o,52s,J8o,K9o,KQo,A8o,QJo,94o,82o`
- IP范围 (50种): `22,Q3s,99,A3s,K6s,ATs,QJs,A6s,QQ,K4s,A8s,55,KTs,Q9s,Q8s,Q4s,AKs,Q6s,44,QTs,AKo,T2s,K4o,J3o,QTo,J8o,43o,AJs,T2o,97o,75o,TT,KJs,J2s,T8s,T9o,Q8o,64s,84o,T3o,66,Q6o,Q5o,94o,A2o,75s,Q2o,K9s,A7o,86o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=50.862%, 平局率=0.625%
- EV: 165.15, Solver Equity: 100.00%
- 动作EV: Check:165.14, Bet:165.15
- 加权EV: 165.15
- 策略: `{"Check":0.198999,"Bet":0.801001}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.017%, 范围平局率差异: 0.003%
- **策略差异: 24.9%**
- **加权EV差异: 4.36**

---

#### 反例 9

**场景1:**
- 公共牌: `4c Tc 9c 2c 6h`
- OOP手牌: `Ac7s`
- OOP范围 (65种): `J9s,A8s,A2s,Q5s,AKs,A9s,QTs,A6s,K4s,QJs,KQs,K5s,K7s,A7s,JTs,Q8s,QQ,K9s,Q7s,AQs,KK,K8s,55,J7s,A4s,Q9s,22,J4o,96o,T5s,66,J5o,K4o,A6o,72o,85s,A7o,52s,A5s,T9s,97s,72s,A2o,JJ,KQo,K2o,87s,82s,K6o,Q3s,A5o,94s,74s,99,54s,ATo,95s,76s,J2o,Q2s,Q9o,95o,TT,J8o,75s`
- IP范围 (83种): `K3s,AA,A2s,55,K9s,A3s,K8s,33,TT,K7s,Q8s,Q5s,K6s,K4s,KJs,JTs,K5s,A7s,Q9s,Q4s,Q3s,QTs,KQs,Q7s,J7s,88,KTs,A6s,AKs,Q6s,A9s,AJs,44,ATo,KQo,J6o,32o,72s,J9s,T4o,92o,64o,96o,T2s,QTo,94o,AQo,Q3o,76o,85o,T8o,J4s,T7o,QQ,J4o,74o,T5s,A8s,93o,J3s,99,T4s,86s,Q2s,Q2o,72o,T6s,42s,J9o,T7s,A2o,A8o,86o,K2s,82o,A4o,A7o,62s,63o,Q6o,A3o,T9o,AJo`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=47.471%, 平局率=0.288%
- EV: 145.69, Solver Equity: 100.00%
- 动作EV: Check:145.83, Bet:145.42
- 加权EV: 145.69
- 策略: `{"Check":0.673051,"Bet":0.326949}`

**场景2:**
- 公共牌: `3h 5s Qs 4s 8s`
- OOP手牌: `As7s`
- OOP范围 (77种): `QTs,AKs,QJs,AQs,ATs,QQ,JJ,AA,K8s,AJs,Q7s,88,A6s,99,77,J9s,JTs,K6s,Q6s,J8s,TT,Q3s,KTs,Q5s,A2s,KK,Q9s,66,Q4s,A9s,A4o,J4s,T9s,62o,95o,A7s,T7s,75o,J8o,K9s,76o,K3s,A7o,63o,97o,87o,QJo,84s,52s,A3o,K7s,97s,AJo,KTo,A3s,22,42o,54o,32s,K2o,54s,94o,64o,K2s,86s,T5s,73o,K8o,J5s,94s,Q7o,T7o,74s,T5o,53s,K7o,85s`
- IP范围 (89种): `AKs,QTs,J8s,Q2s,KK,JJ,Q3s,ATs,Q8s,K5s,J7s,77,AJs,KTs,Q9s,Q4s,AQs,33,K9s,44,A6s,A9s,A3s,K2s,J9s,88,KQs,K3s,KJs,55,QJs,K7s,AA,Q6s,A7s,65o,K4o,T6o,97o,K7o,62s,T2o,63s,T9o,43s,Q4o,85s,98o,32o,84o,Q7s,AQo,86o,AJo,A3o,K5o,72o,J5s,J9o,87s,T5s,82s,KQo,93o,T5o,J3s,85o,JTo,Q9o,Q8o,76o,64s,T7s,32s,K6o,K8o,95o,A5o,K9o,84s,86s,K8s,96o,T4s,Q6o,99,J5o,54o,K3o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=47.468%, 平局率=0.296%
- EV: 152.48, Solver Equity: 100.00%
- 动作EV: Check:149.43, Bet:152.48
- 加权EV: 152.48
- 策略: `{"Check":0,"Bet":1}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.004%, 范围平局率差异: 0.008%
- **策略差异: 67.3%**
- **加权EV差异: 6.78**

---

#### 反例 10

**场景1:**
- 公共牌: `4c 2h 7c Ts As`
- OOP手牌: `5c3s`
- OOP范围 (86种): `A4s,AA,22,KK,A3s,AKs,TT,K6s,A8s,K7s,J9s,44,Q7s,88,AJs,JTs,Q9s,99,Q8s,Q3s,J8s,AQs,KQs,A9s,QJs,QQ,66,K9s,KJs,JJ,Q4s,ATs,Q6s,A6s,76s,A7o,J2o,62o,K5o,J6s,Q9o,A9o,74s,K3s,J6o,T7s,83s,A7s,54o,93o,K5s,73o,K3o,T8s,T6s,95o,A4o,QJo,AQo,96s,KTs,82o,54s,T8o,Q8o,98s,76o,86o,A6o,T2o,T6o,K4s,52o,Q7o,82s,32s,53o,Q5s,T2s,87o,A2o,85o,K7o,55,52s,QTs`
- IP范围 (87种): `A2s,K4s,A3s,KQs,J9s,AJs,A7s,66,55,A6s,Q3s,K6s,K5s,J7s,JJ,AA,22,K8s,K3s,J8s,88,JTs,K9s,K7s,KK,KTs,QTs,44,99,QQ,Q9s,Q6s,33,Q4s,98s,J2s,76s,A8s,98o,Q5s,T7o,Q2o,A5s,42s,Q8o,43s,J7o,64s,J3s,96o,K2o,Q9o,AKo,43o,95s,93o,Q6o,A6o,85o,J5o,87s,T9s,85s,K4o,74o,72s,75s,T4s,83s,Q7o,J3o,Q2s,64o,92s,T6o,87o,84o,76o,K9o,T6s,Q4o,A4o,K2s,JTo,J2o,A7o,72o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=51.974%, 平局率=0.670%
- EV: 169.56, Solver Equity: 100.00%
- 动作EV: Check:169.50, Bet:169.59
- 加权EV: 169.56
- 策略: `{"Check":0.363947,"Bet":0.636053}`

**场景2:**
- 公共牌: `5d 3h 6d Th Qh`
- OOP手牌: `Ah6h`
- OOP范围 (92种): `Q6s,A4s,KK,A5s,66,A9s,Q8s,K2s,QTs,Q2s,KTs,KQs,K3s,55,A3s,KJs,TT,AA,A7s,Q3s,ATs,K9s,J9s,77,Q4s,22,K5s,44,A6s,99,QJs,JTs,AJs,JJ,Q5s,J8s,A2s,A8o,A6o,K8s,T7s,Q7o,92o,A7o,86o,97s,52s,A3o,73s,T8o,J6s,T8s,Q6o,K5o,K3o,83o,QJo,33,87s,76o,88,AQo,Q3o,T9s,93o,K8o,K4s,A8s,A9o,63s,KQo,75o,54s,43o,85s,Q7s,83s,74o,Q8o,63o,65o,A5o,T4o,Q5o,T4s,T6o,97o,T6s,96o,T2s,94s,T5o`
- IP范围 (100种): `QQ,AA,K3s,A2s,K9s,A8s,Q6s,AKs,Q5s,A5s,K6s,Q9s,Q2s,Q3s,K7s,K2s,AJs,KTs,99,Q7s,77,55,JJ,JTs,J9s,K4s,33,QJs,J7s,K8s,KQs,K5s,AQs,KJs,22,KK,QTs,66,Q4s,J8s,54o,QJo,JTo,43o,32s,76o,43s,84s,Q8o,ATs,84o,J6o,K5o,T9s,76s,T8o,73s,74s,87o,J2o,87s,K3o,K8o,K6o,Q2o,96o,65o,T3s,J4o,QTo,AJo,94o,64o,K2o,KQo,42s,Q6o,86s,Q9o,63o,J2s,97o,95o,KTo,A8o,65s,42o,T6o,44,75s,62s,J3s,T8s,53s,85s,85o,Q3o,T3o,T5o,T7o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=51.944%, 平局率=0.644%
- EV: 161.30, Solver Equity: 100.00%
- 动作EV: Check:161.54, Bet:161.04
- 加权EV: 161.30
- 策略: `{"Check":0.514766,"Bet":0.485234}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.030%, 范围平局率差异: 0.026%
- **策略差异: 15.1%**
- **加权EV差异: 8.26**

---

#### 反例 11

**场景1:**
- 公共牌: `6h 8d Jd 9h 2h`
- OOP手牌: `Ah7h`
- OOP范围 (86种): `A4s,AA,22,KK,A3s,AKs,TT,K6s,A8s,K7s,J9s,44,Q7s,88,AJs,JTs,Q9s,99,Q8s,Q3s,J8s,AQs,KQs,A9s,QJs,QQ,66,K9s,KJs,JJ,Q4s,ATs,Q6s,A6s,76s,A7o,J2o,62o,K5o,J6s,Q9o,A9o,74s,K3s,J6o,T7s,83s,A7s,54o,93o,K5s,73o,K3o,T8s,T6s,95o,A4o,QJo,AQo,96s,KTs,82o,54s,T8o,Q8o,98s,76o,86o,A6o,T2o,T6o,K4s,52o,Q7o,82s,32s,53o,Q5s,T2s,87o,A2o,85o,K7o,55,52s,QTs`
- IP范围 (87种): `A2s,K4s,A3s,KQs,J9s,AJs,A7s,66,55,A6s,Q3s,K6s,K5s,J7s,JJ,AA,22,K8s,K3s,J8s,88,JTs,K9s,K7s,KK,KTs,QTs,44,99,QQ,Q9s,Q6s,33,Q4s,98s,J2s,76s,A8s,98o,Q5s,T7o,Q2o,A5s,42s,Q8o,43s,J7o,64s,J3s,96o,K2o,Q9o,AKo,43o,95s,93o,Q6o,A6o,85o,J5o,87s,T9s,85s,K4o,74o,72s,75s,T4s,83s,Q7o,J3o,Q2s,64o,92s,T6o,87o,84o,76o,K9o,T6s,Q4o,A4o,K2s,JTo,J2o,A7o,72o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=47.089%, 平局率=1.016%
- EV: 162.99, Solver Equity: 100.00%
- 动作EV: Check:162.87, Bet:163.05
- 加权EV: 162.99
- 策略: `{"Check":0.321279,"Bet":0.678721}`

**场景2:**
- 公共牌: `9d 7d 6h 4c Jc`
- OOP手牌: `5d3d`
- OOP范围 (118种): `AQs,Q8s,A6s,KJs,AJs,KQs,A3s,K4s,99,J7s,A2s,22,KK,77,A8s,JTs,A4s,Q4s,K2s,AKs,Q6s,AA,66,Q3s,K6s,A5s,33,44,K3s,J8s,ATs,Q2s,K9s,55,QQ,K8s,J9s,Q5s,A7s,Q7s,A9s,QTs,QJs,KTs,TT,K7s,88,53o,97o,J3s,74o,T4s,52s,42s,T7s,75s,82o,T8s,43o,92s,73o,76s,QJo,63s,JTo,J6o,T9o,93s,K2o,AKo,T6s,76o,K6o,Q9o,85o,85s,J6s,KJo,K9o,T3o,A8o,75o,83s,Q6o,A9o,64s,92o,K8o,86o,87o,54o,65o,53s,96o,AQo,84s,72s,82s,T2o,42o,J9o,94o,83o,K5o,63o,Q3o,A4o,T4o,T2s,K3o,K4o,A7o,32o,QTo,73s,Q2o,T5s,52o`
- IP范围 (62种): `J8s,K9s,Q3s,K2s,33,A8s,JJ,K3s,KQs,AJs,TT,K6s,K8s,A7s,55,Q2s,KK,JTs,QJs,KJs,Q9s,Q6s,Q4s,QQ,T5s,KQo,Q9o,AQs,96o,87o,T7s,64s,86s,JTo,A3o,63s,J6s,QTs,ATo,ATs,KTo,K7o,J7o,43s,77,88,T6o,87s,Q5o,J4o,54o,72o,93s,A2s,AQo,K4o,Q5s,Q8o,75o,74s,T9o,J5o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=47.061%, 平局率=1.011%
- EV: 163.56, Solver Equity: 100.00%
- 动作EV: Check:163.56, Bet:162.40
- 加权EV: 163.56
- 策略: `{"Check":0.999437,"Bet":0.000563}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.028%, 范围平局率差异: 0.005%
- **策略差异: 67.8%**
- **加权EV差异: 0.56**

---

#### 反例 12

**场景1:**
- 公共牌: `7d Ac 3d 5c Qd`
- OOP手牌: `9c6s`
- OOP范围 (109种): `K2s,77,J8s,A7s,KTs,K7s,Q7s,99,A3s,KJs,44,K8s,Q4s,KQs,A2s,TT,Q9s,KK,K3s,AKs,A5s,Q6s,JTs,Q5s,QJs,66,J7s,55,AA,Q8s,K9s,A4s,QTs,AJs,33,Q3s,A9s,JJ,A8s,ATs,K4s,Q2s,K6s,AQo,K6o,76s,J3s,32s,64s,65o,A5o,A8o,T3o,K8o,T9o,A6s,JTo,KQo,T6o,94o,K4o,87o,22,65s,AKo,82s,T5s,Q5o,72o,98s,75s,T2s,83o,92o,KJo,Q4o,53s,85s,Q8o,J6o,97s,84o,82o,K7o,T6s,32o,63s,43s,42o,74s,95o,88,A2o,Q2o,Q9o,QJo,ATo,73o,K9o,73s,85o,97o,T4s,96s,A9o,52s,J2s,J4o,96o`
- IP范围 (47种): `J9s,K2s,QQ,KK,K9s,A9s,AQs,33,Q2s,66,A2s,A6s,Q4s,Q8s,QJs,JTs,A4s,AJs,J3s,AA,43o,T2o,97o,JJ,72o,54o,A7o,J4s,32o,A7s,98o,T8o,44,Q3s,72s,J8o,QTo,52o,K9o,Q9o,T7o,85s,65s,98s,76o,K8s,T5o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=51.582%, 平局率=0.673%
- EV: -0.01, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.42
- 加权EV: -0.01
- 策略: `{"Check":0.964773,"Bet":0.035227}`

**场景2:**
- 公共牌: `Kd 2h 6c Ac 5d`
- OOP手牌: `7s3s`
- OOP范围 (107种): `K9s,J7s,A2s,KTs,A7s,K7s,ATs,K2s,KK,AA,JJ,77,J8s,A6s,KQs,Q7s,Q8s,A3s,Q4s,Q3s,K4s,A5s,Q9s,QJs,33,88,66,A8s,J9s,22,Q2s,K8s,QQ,44,AKs,A9s,K3s,K6s,55,AJs,99,K5s,T4o,TT,T8s,J7o,A8o,86o,JTo,J5o,K8o,62s,Q9o,Q8o,43o,63s,82s,82o,T6o,65s,AKo,63o,42o,KQo,95s,QJo,Q4o,J3o,Q5s,A4s,T7s,96o,AQo,J6o,A5o,54s,92o,74s,K5o,T8o,Q2o,72o,93o,J5s,KTo,A9o,QTo,97o,T3o,K9o,32o,JTs,53s,K3o,32s,86s,J8o,73s,T9o,A7o,T2s,T5s,85o,73o,Q6o,J2s,A4o`
- IP范围 (41种): `88,JTs,TT,66,KK,J7s,A2s,K2s,K6s,ATs,KJs,Q2s,AKs,Q3s,A3s,Q4s,Q7o,AA,92o,J5o,84s,86s,62s,83o,A8o,64o,K7s,53s,A9o,J6s,JTo,J7o,82o,95o,98o,AQo,T2o,54o,QJs,J9s,T2s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=51.613%, 平局率=0.716%
- EV: 0.22, Solver Equity: 0.00%
- 动作EV: Check:-0.00, Bet:0.31
- 加权EV: 0.22
- 策略: `{"Check":0.274699,"Bet":0.725302}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.032%, 范围平局率差异: 0.043%
- **策略差异: 69.0%**
- **加权EV差异: 0.24**

---

#### 反例 13

**场景1:**
- 公共牌: `6h Ts 4d 3d Qd`
- OOP手牌: `7s2h`
- OOP范围 (64种): `AQs,J9s,KQs,Q7s,QQ,K9s,Q6s,A9s,Q5s,A7s,AA,A2s,K7s,55,K6s,J7s,44,Q4s,A4s,99,AJs,88,KJs,K4s,JTs,J7o,84s,J9o,T7o,T5s,72o,33,J8o,T8s,T7s,J6s,73s,J4s,54s,94s,98o,K5o,75o,A4o,Q5o,K8s,J4o,82o,87s,83o,K5s,A6o,T6o,K8o,76s,A2o,Q9s,J3o,65o,KTo,54o,T2o,K2s,J8s`
- IP范围 (55种): `A4s,K2s,K5s,99,K6s,JJ,QJs,66,Q2s,A9s,22,KTs,A6s,77,Q7s,JTs,A5s,A2s,K4s,Q3s,33,55,93o,J3o,A7s,J2s,KTo,K2o,42o,J4s,85o,T2o,43o,64o,94o,T5o,KQo,82s,94s,63s,88,92o,73s,T2s,53o,98o,K7o,QTo,J9o,AKs,86o,95o,T4o,KJo,A3o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=50.279%, 平局率=0.560%
- EV: -0.07, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.09
- 加权EV: -0.07
- 策略: `{"Check":0.239157,"Bet":0.760843}`

**场景2:**
- 公共牌: `3c Jh 9h 2c 5c`
- OOP手牌: `Th6h`
- OOP范围 (118种): `JJ,QTs,77,Q2s,K5s,K3s,Q5s,J7s,A8s,Q7s,22,AQs,Q3s,A3s,A5s,A7s,JTs,QQ,A2s,99,TT,Q6s,AJs,AA,J9s,66,K9s,J8s,33,A4s,QJs,KJs,Q9s,K4s,55,AKs,KTs,K6s,Q4s,KK,K2s,K8s,88,44,K7s,KQs,Q8s,KQo,54s,K4o,43s,83s,QTo,32s,J8o,97o,42s,98o,JTo,64o,A6o,52s,Q3o,Q9o,T7o,T6o,75s,93s,AJo,K2o,A7o,65s,85o,J6s,T4s,J5s,J6o,97s,74s,73o,J7o,Q4o,J9o,86s,96o,63o,T6s,83o,A3o,A9o,J5o,62s,Q5o,J2o,75o,94o,K5o,82o,A6s,86o,A4o,T3s,A9s,76o,95o,76s,T9o,T2s,96s,T8s,AQo,K7o,ATs,74o,Q6o,53o,T7s,32o`
- IP范围 (53种): `77,QJs,66,Q9s,K2s,KTs,K8s,Q3s,K7s,A2s,33,J8s,99,A4s,Q2s,ATs,JJ,A6s,Q6s,A5s,K3s,KQo,73o,K6o,44,A8s,42o,JTo,A3o,Q7o,Q2o,88,J2s,52o,65s,95s,85o,J6o,82s,Q8s,Q9o,A7o,K5o,J5o,43s,K8o,63s,Q7s,A9o,55,QJo,75s,52s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=50.297%, 平局率=0.557%
- EV: -0.18, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.20
- 加权EV: -0.18
- 策略: `{"Check":0.074729,"Bet":0.925271}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.018%, 范围平局率差异: 0.003%
- **策略差异: 16.4%**
- **加权EV差异: 0.12**

---

#### 反例 14

**场景1:**
- 公共牌: `6h Jc 2s Ks Ac`
- OOP手牌: `5s4h`
- OOP范围 (88种): `66,JJ,Q9s,A5s,Q8s,ATs,K7s,A7s,KJs,K5s,K8s,TT,A3s,KQs,AKs,AQs,88,22,QJs,JTs,K9s,A4s,55,J9s,Q5s,Q7s,44,K6s,K3s,KK,Q4s,99,QQ,K4s,Q6s,97o,74o,T4s,62s,53o,Q7o,J4o,82o,K9o,K2o,JTo,33,KJo,92o,43s,J9o,93s,63o,54o,98s,65o,J3o,76s,J2s,73s,A9s,83o,64o,K7o,T5s,J4s,75o,J8o,84s,42o,J5s,KQo,86o,K5o,A4o,AA,J5o,85s,A6o,97s,Q5o,QTo,72o,T2s,94s,73o,AKo,T6o`
- IP范围 (46种): `QJs,K8s,JTs,A7s,KJs,Q3s,AA,QQ,88,K4s,Q4s,Q7s,77,A2s,A6s,QTs,KTs,Q5s,T3s,QTo,J2o,T2s,87o,A7o,55,98s,44,73s,A8o,A6o,J9o,Q7o,95o,TT,K5s,Q3o,T4o,KQs,62o,A2o,T5s,75s,A9o,J8o,T2o,T3o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=45.507%, 平局率=1.246%
- EV: -0.11, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.53
- 加权EV: -0.11
- 策略: `{"Check":0.794388,"Bet":0.205612}`

**场景2:**
- 公共牌: `7d 9c Kc 3h Tc`
- OOP手牌: `4d2s`
- OOP范围 (60种): `AA,22,K9s,AJs,Q9s,K2s,TT,Q3s,KTs,QTs,JJ,AKs,A7s,Q5s,K6s,Q2s,K7s,QQ,AQs,A8s,Q4s,77,A6s,A3s,J5o,64o,T5o,85o,K8s,AJo,K3o,82s,K7o,AKo,K5s,T9o,T2s,95s,Q7s,74o,QTo,J6s,T3s,A5o,J3o,62s,73o,Q8o,ATs,65o,42o,Q4o,T7s,J8s,74s,K8o,J7s,AQo,85s,54s`
- IP范围 (60种): `88,KQs,Q8s,K8s,JTs,A8s,AKs,Q2s,33,K4s,Q6s,A4s,66,A9s,A5s,K2s,KTs,Q5s,Q7s,A7s,99,AA,ATs,Q3s,85s,J2o,82o,T7s,52o,A3o,63o,QQ,T6o,T3o,K5o,72o,JTo,94s,TT,98o,43s,Q2o,JJ,43o,T5o,73o,T8s,QJs,75o,ATo,AQs,KQo,A2s,94o,J4o,A7o,A2o,K6s,A9o,K8o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=45.465%, 平局率=1.295%
- EV: 0.00, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.60
- 加权EV: 0.00
- 策略: `{"Check":1,"Bet":0}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.043%, 范围平局率差异: 0.049%
- **策略差异: 20.6%**
- **加权EV差异: 0.11**

---

#### 反例 15

**场景1:**
- 公共牌: `3s 5h 6h Td 9d`
- OOP手牌: `8d7c`
- OOP范围 (114种): `66,K9s,J9s,J7s,KK,A5s,A7s,77,K3s,99,Q8s,Q3s,K2s,A6s,A3s,KTs,K7s,A2s,Q4s,A4s,QJs,22,K6s,A9s,AQs,KQs,Q7s,K5s,88,Q9s,A8s,44,AA,TT,J8s,Q6s,55,AJs,K4s,Q2s,ATs,33,Q5s,JJ,AKs,Q5o,A4o,AJo,K2o,KJs,43o,43s,J4o,A3o,53o,76s,K5o,J6s,A8o,Q3o,J3s,87o,83s,52o,ATo,75o,J2s,T9o,J2o,97o,84s,32o,J7o,AKo,98o,K4o,86o,32s,96o,A7o,T7s,T3o,87s,J3o,65s,64s,KJo,K8o,J9o,T7o,J5s,72o,62o,K6o,KTo,53s,T8o,A6o,54s,82s,73o,52s,T4s,QTs,Q7o,T2o,96s,Q6o,T5s,98s,J6o,95s,Q4o,76o`
- IP范围 (47种): `44,K7s,88,AQs,K4s,J8s,99,A3s,A7s,66,KJs,A4s,K9s,J7s,A5s,Q6s,Q3s,AA,AKs,64s,94o,63o,Q5o,95o,53s,J6s,T4s,52o,AKo,Q8s,Q6o,96s,J3s,85s,JTs,Q2s,86s,A6o,55,K5s,T7s,92s,84o,72o,QTs,T6s,A5o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=45.548%, 平局率=0.802%
- EV: 173.22, Solver Equity: 100.00%
- 动作EV: Check:172.57, Bet:173.86
- 加权EV: 173.22
- 策略: `{"Check":0.499329,"Bet":0.500671}`

**场景2:**
- 公共牌: `Tc 9c 4h Qd 2d`
- OOP手牌: `KhJh`
- OOP范围 (107种): `QTs,A6s,K3s,J9s,K2s,KK,44,66,88,JTs,K8s,A8s,Q9s,TT,Q6s,KTs,Q3s,Q5s,55,ATs,Q8s,AJs,A3s,K9s,KQs,QJs,K6s,KJs,J8s,QQ,77,A7s,22,K7s,A5s,JJ,AQs,33,A4s,AKs,K4s,J7s,J4s,K6o,42s,A8o,T4s,T2o,T2s,95s,A9o,76s,52s,K9o,J2s,86s,Q3o,92o,86o,J4o,AJo,K2o,A9s,K7o,T4o,72o,82o,T7o,AA,97s,94o,J3o,A4o,42o,64o,98s,63o,87o,73s,JTo,K8o,74s,T8o,43s,98o,73o,Q2o,KTo,J5o,K4o,82s,J3s,Q8o,T5s,99,83s,K3o,52o,J8o,Q2s,J6o,85o,93o,76o,T8s,J2o,A6o`
- IP范围 (40种): `ATs,A9s,TT,JJ,QQ,AA,K2s,KQs,J7s,AQs,Q2s,QJs,55,33,Q3s,Q6s,Q9o,98s,83o,A7o,J4s,Q3o,K9s,97o,K5s,A3o,J8o,98o,KK,73s,93o,K8o,J6o,75s,52o,Q7o,62o,A7s,44,65s`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=45.548%, 平局率=0.818%
- EV: 172.98, Solver Equity: 100.00%
- 动作EV: Check:169.53, Bet:172.98
- 加权EV: 172.98
- 策略: `{"Check":0.000001,"Bet":0.999999}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.000%, 范围平局率差异: 0.015%
- **策略差异: 49.9%**
- **加权EV差异: 0.23**

---

#### 反例 16

**场景1:**
- 公共牌: `As 2s Kh 6s 3h`
- OOP手牌: `Ks7s`
- OOP范围 (71种): `Q6s,QTs,33,A8s,KQs,J8s,Q2s,JJ,A7s,K7s,88,44,77,A6s,99,QQ,AJs,Q5s,J9s,Q9s,AKs,A5s,JTs,K6s,Q7s,AA,K4s,55,66,K4o,92o,A3s,72o,A9s,J6o,JTo,86o,85s,83s,AQs,64o,84s,52s,76o,J5o,74s,J5s,95o,KJs,87o,Q4s,T4s,92s,64s,K6o,KQo,63s,72s,QJs,K9s,76s,53o,43s,A2s,62o,A2o,J3o,Q7o,Q6o,63o,T4o`
- IP范围 (94种): `88,KTs,A2s,TT,QQ,AA,66,Q4s,Q2s,QJs,33,J7s,A7s,A9s,Q9s,ATs,77,A8s,K4s,A6s,44,KJs,55,K2s,Q5s,K6s,K8s,K9s,K7s,KQs,QTs,Q8s,J8s,AKs,22,K5s,Q7s,K4o,98s,86o,AQo,T3s,A3s,A6o,T5o,73s,T7o,KK,J7o,74o,42s,63s,T6s,42o,K7o,T2s,AKo,97o,Q2o,Q3s,52s,A9o,AJs,K8o,T4o,Q4o,J3o,72o,J9s,92s,75s,J5o,64s,J5s,KJo,A8o,K6o,98o,T8s,53o,52o,63o,A7o,A5s,84o,93o,83s,65s,62o,32o,32s,95o,T3o,43o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=51.773%, 平局率=0.639%
- EV: 160.35, Solver Equity: 100.00%
- 动作EV: Check:160.23, Bet:160.79
- 加权EV: 160.35
- 策略: `{"Check":0.779778,"Bet":0.220222}`

**场景2:**
- 公共牌: `4s 3d 7h 3h 2d`
- OOP手牌: `7d3c`
- OOP范围 (65种): `KTs,K8s,QQ,A4s,KK,A7s,TT,K9s,Q8s,AKs,33,A9s,K5s,Q5s,AQs,J7s,66,AA,A6s,QJs,A5s,KJs,AJs,JJ,K3s,A3s,J7o,97o,J8o,54s,73o,63o,A7o,65s,44,ATs,T2s,83s,62o,Q4o,75s,43s,J6s,T4o,55,Q9s,T3o,74s,85o,76o,64s,52s,72o,KQs,T7s,QTo,82o,53s,98o,Q6s,J6o,K2s,K4o,87o,AJo`
- IP范围 (48种): `K9s,KTs,K7s,J7s,Q5s,KQs,A7s,66,22,AQs,55,JTs,AJs,A9s,A4s,AKs,Q9s,99,Q7s,K4o,T8s,75s,K8s,95o,A4o,A5o,87o,K5o,A2s,T2s,AKo,88,J5o,Q2o,94s,52s,A6o,K6o,A3o,A5s,J6s,K6s,62s,QQ,T2o,T6s,65s,T5s`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=51.740%, 平局率=0.620%
- EV: 169.74, Solver Equity: 100.00%
- 动作EV: Check:167.03, Bet:169.76
- 加权EV: 169.74
- 策略: `{"Check":0.006169,"Bet":0.993831}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.033%, 范围平局率差异: 0.019%
- **策略差异: 77.4%**
- **加权EV差异: 9.39**

---

#### 反例 17

**场景1:**
- 公共牌: `Kh 5d 3c Qc 2c`
- OOP手牌: `8d4h`
- OOP范围 (73种): `AKs,88,AQs,Q3s,44,55,A6s,A5s,K2s,Q9s,KJs,K3s,22,JTs,K8s,QQ,ATs,A8s,AJs,JJ,QJs,J9s,J8s,A4s,KQs,A2s,Q6s,KTs,KK,J4o,86s,K6s,83o,82o,33,77,76o,A7s,QTs,75s,53s,T6o,75o,J6s,65o,97o,43s,J7s,A9s,K9o,T7o,95o,93s,Q8o,KTo,K6o,72s,J6o,Q6o,T8s,95s,T8o,32s,T9o,K7s,T5o,96o,84o,J3s,64o,97s,74s,43o`
- IP范围 (49种): `66,AQs,A9s,Q5s,KTs,Q8s,Q3s,K6s,A4s,K8s,K4s,J8s,K9s,A6s,J7s,Q9s,Q4s,K3s,99,82o,K5o,K4o,AKo,Q6s,JTs,87s,AJs,A6o,K9o,A4o,83s,92o,Q2s,ATs,75s,A2s,22,T9o,64o,75o,KQo,33,85o,KJo,92s,T2o,96o,J3o,JJ`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=39.006%, 平局率=0.596%
- EV: -0.05, Solver Equity: 0.00%
- 动作EV: Check:-0.00, Bet:-0.11
- 加权EV: -0.05
- 策略: `{"Check":0.569652,"Bet":0.430348}`

**场景2:**
- 公共牌: `Jc Th 5c 2c Ah`
- OOP手牌: `6c3d`
- OOP范围 (70种): `K6s,77,J7s,KJs,A4s,AKs,QJs,44,AJs,K9s,QTs,33,Q9s,J8s,99,JJ,K5s,Q3s,K8s,J9s,66,A6s,KQs,KK,Q5s,KTs,TT,Q2s,74o,K5o,62o,K4o,J5s,62s,85s,J4s,22,Q8o,54o,32o,A3s,KTo,92s,T8s,T4o,T6o,AQs,AA,AQo,63o,K2o,A7s,82o,JTo,96s,K7s,97o,64o,93o,74s,73o,52s,J6s,63s,Q5o,AJo,QQ,J7o,42o,72o`
- IP范围 (71种): `K5s,KJs,AKs,J7s,77,22,K8s,Q9s,AQs,J9s,A5s,J8s,A7s,99,A2s,88,A9s,K6s,A6s,Q7s,Q8s,Q2s,K9s,A4s,Q6s,44,QTs,Q3s,T8o,JJ,66,AQo,J4s,85o,J2o,86s,QQ,52s,84o,93s,98s,43o,J5o,J3s,83o,65s,T7s,J5s,T7o,32s,87o,T8s,T5o,95s,T3o,KQo,96o,T9s,J6s,72s,T2o,ATo,KK,T5s,K7o,AA,T9o,J8o,95o,KQs,QTo`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=39.004%, 平局率=0.585%
- EV: -0.05, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.36
- 加权EV: -0.05
- 策略: `{"Check":0.847028,"Bet":0.152972}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.002%, 范围平局率差异: 0.011%
- **策略差异: 27.7%**
- **加权EV差异: 0.01**

---

#### 反例 18

**场景1:**
- 公共牌: `Jh 5d Jd 9d 7d`
- OOP手牌: `3h2h`
- OOP范围 (73种): `AKs,88,AQs,Q3s,44,55,A6s,A5s,K2s,Q9s,KJs,K3s,22,JTs,K8s,QQ,ATs,A8s,AJs,JJ,QJs,J9s,J8s,A4s,KQs,A2s,Q6s,KTs,KK,J4o,86s,K6s,83o,82o,33,77,76o,A7s,QTs,75s,53s,T6o,75o,J6s,65o,97o,43s,J7s,A9s,K9o,T7o,95o,93s,Q8o,KTo,K6o,72s,J6o,Q6o,T8s,95s,T8o,32s,T9o,K7s,T5o,96o,84o,J3s,64o,97s,74s,43o`
- IP范围 (49种): `66,AQs,A9s,Q5s,KTs,Q8s,Q3s,K6s,A4s,K8s,K4s,J8s,K9s,A6s,J7s,Q9s,Q4s,K3s,99,82o,K5o,K4o,AKo,Q6s,JTs,87s,AJs,A6o,K9o,A4o,83s,92o,Q2s,ATs,75s,A2s,22,T9o,64o,75o,KQo,33,85o,KJo,92s,T2o,96o,J3o,JJ`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=50.087%, 平局率=0.949%
- EV: 0.06, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:0.10
- 加权EV: 0.06
- 策略: `{"Check":0.413195,"Bet":0.586805}`

**场景2:**
- 公共牌: `6s 2h 7s 4d 4c`
- OOP手牌: `8h3d`
- OOP范围 (92种): `J8s,J9s,33,KJs,Q3s,AKs,A8s,55,A4s,A5s,KQs,77,JJ,QJs,Q7s,Q2s,AA,66,K2s,88,ATs,A3s,JTs,AJs,KK,K7s,QTs,K3s,K6s,Q8s,KTs,A9s,AQs,99,Q9s,K4s,53o,K9s,82o,A3o,T4s,J4o,J6o,83o,Q6s,A7o,J2s,Q8o,86s,KQo,TT,T5o,K8o,93s,T3o,Q7o,K7o,K6o,97s,J3s,K5o,74s,T8o,83s,65o,J8o,KTo,Q3o,32o,KJo,J6s,73o,T7o,87s,A2s,T4o,JTo,92o,94o,22,J7o,93o,32s,AJo,J3o,Q6o,A6s,Q5s,A9o,Q4s,AQo,75o`
- IP范围 (68种): `ATs,KJs,A5s,KK,A8s,QQ,QJs,88,22,K4s,A2s,KQs,K8s,K5s,66,J9s,A9s,K6s,Q2s,Q5s,JTs,J8s,J7s,AQs,99,A4s,55,44,Q3o,J2s,T7s,65o,J9o,A2o,J4s,92s,87o,42o,63s,A8o,A7o,T9s,QJo,A3o,85s,82s,Q4s,J3s,K9s,AKo,AJo,93o,77,62s,86s,ATo,J5o,85o,T5o,A9o,Q2o,KTs,Q6o,AA,Q5o,K5o,43o,86o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=50.129%, 平局率=0.974%
- EV: 0.04, Solver Equity: 0.00%
- 动作EV: Check:-0.00, Bet:0.04
- 加权EV: 0.04
- 策略: `{"Check":0.027489,"Bet":0.972511}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.042%, 范围平局率差异: 0.025%
- **策略差异: 38.6%**
- **加权EV差异: 0.02**

---

#### 反例 19

**场景1:**
- 公共牌: `Jh 5d Jd 9d 7d`
- OOP手牌: `3h2h`
- OOP范围 (73种): `AKs,88,AQs,Q3s,44,55,A6s,A5s,K2s,Q9s,KJs,K3s,22,JTs,K8s,QQ,ATs,A8s,AJs,JJ,QJs,J9s,J8s,A4s,KQs,A2s,Q6s,KTs,KK,J4o,86s,K6s,83o,82o,33,77,76o,A7s,QTs,75s,53s,T6o,75o,J6s,65o,97o,43s,J7s,A9s,K9o,T7o,95o,93s,Q8o,KTo,K6o,72s,J6o,Q6o,T8s,95s,T8o,32s,T9o,K7s,T5o,96o,84o,J3s,64o,97s,74s,43o`
- IP范围 (49种): `66,AQs,A9s,Q5s,KTs,Q8s,Q3s,K6s,A4s,K8s,K4s,J8s,K9s,A6s,J7s,Q9s,Q4s,K3s,99,82o,K5o,K4o,AKo,Q6s,JTs,87s,AJs,A6o,K9o,A4o,83s,92o,Q2s,ATs,75s,A2s,22,T9o,64o,75o,KQo,33,85o,KJo,92s,T2o,96o,J3o,JJ`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=50.087%, 平局率=0.949%
- EV: 0.06, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:0.10
- 加权EV: 0.06
- 策略: `{"Check":0.413195,"Bet":0.586805}`

**场景2:**
- 公共牌: `9h 2d Td Js 4d`
- OOP手牌: `6h5c`
- OOP范围 (107种): `A7s,K5s,A3s,KQs,QTs,55,K7s,K2s,ATs,K4s,TT,A9s,QJs,Q7s,K8s,Q6s,KJs,A8s,QQ,Q9s,Q5s,33,A4s,Q3s,JTs,J7s,K9s,44,AJs,KTs,99,Q4s,A5s,A6s,AQs,Q2s,KK,88,J9s,A2s,AA,66,T6s,K7o,52s,J9o,AQo,97s,95o,T8s,52o,Q8s,A3o,83s,A7o,QTo,97o,98o,84o,JJ,62s,T9o,42s,KQo,Q2o,65o,T4s,A9o,T5s,T2s,AKo,83o,65s,K2o,J5s,T3s,42o,87s,32o,T7s,43o,A5o,J3o,K8o,Q3o,87o,53o,Q6o,J4s,J7o,T8o,A4o,T7o,J6s,Q4o,75s,92s,J6o,KJo,J2s,T6o,73s,75o,85s,53s,K6s,84s`
- IP范围 (53种): `Q7s,K6s,A2s,A8s,K9s,A4s,A6s,KK,QQ,JJ,66,KTs,88,77,Q3s,TT,22,A7s,J9s,A3s,QTs,74o,55,T7s,86s,J2s,43o,AQo,52s,Q3o,K7o,AKo,Q2s,AA,K6o,93s,QJo,K2s,K5o,KJo,42s,J6s,J3o,54o,A5o,KTo,64o,T4o,54s,Q6o,43s,J6o,KQo`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=50.048%, 平局率=0.976%
- EV: -0.00, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.12
- 加权EV: -0.00
- 策略: `{"Check":0.997951,"Bet":0.002049}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.038%, 范围平局率差异: 0.027%
- **策略差异: 58.5%**
- **加权EV差异: 0.06**

---

#### 反例 20

**场景1:**
- 公共牌: `Jh 5d Jd 9d 7d`
- OOP手牌: `3h2h`
- OOP范围 (73种): `AKs,88,AQs,Q3s,44,55,A6s,A5s,K2s,Q9s,KJs,K3s,22,JTs,K8s,QQ,ATs,A8s,AJs,JJ,QJs,J9s,J8s,A4s,KQs,A2s,Q6s,KTs,KK,J4o,86s,K6s,83o,82o,33,77,76o,A7s,QTs,75s,53s,T6o,75o,J6s,65o,97o,43s,J7s,A9s,K9o,T7o,95o,93s,Q8o,KTo,K6o,72s,J6o,Q6o,T8s,95s,T8o,32s,T9o,K7s,T5o,96o,84o,J3s,64o,97s,74s,43o`
- IP范围 (49种): `66,AQs,A9s,Q5s,KTs,Q8s,Q3s,K6s,A4s,K8s,K4s,J8s,K9s,A6s,J7s,Q9s,Q4s,K3s,99,82o,K5o,K4o,AKo,Q6s,JTs,87s,AJs,A6o,K9o,A4o,83s,92o,Q2s,ATs,75s,A2s,22,T9o,64o,75o,KQo,33,85o,KJo,92s,T2o,96o,J3o,JJ`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=50.087%, 平局率=0.949%
- EV: 0.06, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:0.10
- 加权EV: 0.06
- 策略: `{"Check":0.413195,"Bet":0.586805}`

**场景2:**
- 公共牌: `Tc Ks 3d Qc 5s`
- OOP手牌: `4d2d`
- OOP范围 (81种): `K9s,AQs,KQs,99,JJ,A2s,K6s,33,AJs,AKs,A4s,TT,J9s,K4s,Q2s,Q6s,88,55,JTs,Q4s,ATs,J7s,K3s,A3s,KJs,A9s,A8s,A7s,K2s,Q3s,66,A5s,95o,73s,T3s,92s,K7s,K8s,32o,53s,Q7s,64s,J5s,AKo,AJo,J2o,Q8s,T8o,32s,75s,65o,QJo,A3o,73o,95s,42s,T7o,A8o,76s,K7o,T8s,Q7o,T6s,96s,98s,72o,KK,72s,53o,43s,97o,T5s,J3o,AA,J3s,Q8o,Q6o,QTs,96o,J8o,K3o`
- IP范围 (62种): `AKs,66,K8s,JTs,QJs,K6s,AA,A3s,77,KTs,J8s,99,JJ,Q2s,Q3s,J9s,K3s,A4s,K7s,QTs,A6s,A2s,K4s,K2s,K5o,A5o,44,95o,Q4o,97o,KK,KQs,QTo,Q8o,J2o,Q7o,K8o,62o,T9o,82o,93o,AQs,T7s,T4o,T6o,87s,52s,88,K6o,Q9s,86s,A6o,75s,QJo,KTo,J3o,A8s,64o,K7o,J4o,T7o,J3s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=50.132%, 平局率=0.955%
- EV: 1.31, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:1.31
- 加权EV: 1.31
- 策略: `{"Check":0,"Bet":1}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.045%, 范围平局率差异: 0.006%
- **策略差异: 41.3%**
- **加权EV差异: 1.25**

