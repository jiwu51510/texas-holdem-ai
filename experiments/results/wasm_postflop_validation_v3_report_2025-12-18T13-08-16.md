# 跨公共牌四维度胜率-策略验证实验报告 V3

## 实验时间

2025-12-18T13-08-16

## 累计统计

| 指标 | 本次 | 累计 |
|------|------|------|
| 运行次数 | 1 | 1 |
| 场景数 | 23733 | 23733 |
| 胜率相近对数 | 20 | 20 |
| 策略差异显著反例 | 5 | 5 |

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

- 生成场景数: 23733
- 四维度胜率相近的场景对（差异<0.05%）: 20
- 策略差异显著(>15%)的场景对: 5

## 关键发现

### ⚠️ 四维度胜率标量不足以决定最优策略

在 20 对四维度胜率相近的场景中，有 5 对（25.0%）的策略差异显著。

**结论：即使手牌胜率、手牌平局率、范围胜率、范围平局率都精确匹配（差异<0.05%），最优策略仍然可能完全不同。**

### 策略差异显著的反例

---

#### 反例 1

**场景1:**
- 公共牌: `6d Qd Kd 3c Ac`
- OOP手牌: `5d2s`
- OOP范围 (107种): `77,J8s,A5s,Q6s,A2s,66,TT,K2s,A3s,KJs,K7s,AQs,Q8s,K4s,KQs,AKs,J9s,KK,Q2s,Q4s,QQ,ATs,22,55,A6s,Q3s,44,88,KTs,K5s,AJs,Q5s,JJ,K3s,K9s,Q7s,33,A7s,AA,JTs,A8s,K6s,J7s,T5s,A7o,64s,96o,Q3o,KJo,QJs,97o,J2s,94o,QTo,K2o,75s,92o,98o,53s,T9s,AKo,T7o,A9o,52o,J8o,65s,T7s,93o,93s,J3s,54s,T4s,A3o,98s,92s,J6s,86o,62o,Q8o,73o,72s,A9s,76s,AQo,74o,84o,A5o,QJo,T3s,62s,K5o,J2o,J5o,K6o,T2o,A8o,A4s,95o,52s,T4o,87o,65o,97s,87s,Q6o,74s,T5o`
- IP范围 (62种): `A2s,Q4s,TT,AKs,AA,33,J8s,K7s,22,A7s,A4s,J9s,88,J7s,A9s,55,QQ,JJ,A8s,A5s,Q8s,QTs,JTs,Q2s,AQo,A4o,32s,83s,K3o,43o,KK,62s,62o,QJs,94o,85s,95o,J2o,76o,K8s,K2o,A7o,K6o,87s,T9s,K8o,KTo,82s,93o,73s,Q8o,97o,63s,65s,72s,T7s,K4o,AJs,A3s,64o,KTs,KJo`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=40.960%, 平局率=1.741%
- EV: 0.21, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:0.64
- 加权EV: 0.21
- 策略: `{"Check":0.667299,"Bet":0.332701}`

**场景2:**
- 公共牌: `Kh 7d Qc 8h 3s`
- OOP手牌: `4s2h`
- OOP范围 (110种): `QQ,Q4s,K7s,JJ,TT,K3s,66,QJs,99,33,J7s,KQs,22,44,KJs,AJs,A3s,K5s,JTs,QTs,Q7s,Q9s,Q3s,55,A6s,AKs,J9s,ATs,K2s,A4s,AA,J8s,K8s,88,A8s,Q6s,A7s,K6s,Q2s,77,A2s,A9s,KK,A5s,K4s,A6o,QJo,82s,T8o,KTo,95s,98o,J6o,94s,T8s,94o,53o,64o,QTo,32o,63s,T5o,82o,J5s,T3s,A7o,42o,92s,AKo,AJo,JTo,65s,J5o,J6s,A9o,86o,T3o,J4o,54s,T7o,63o,97o,AQo,52o,96o,J2s,43o,95o,AQs,T4o,92o,87o,42s,76s,72o,83s,53s,86s,J7o,J3s,A8o,T6s,Q5o,ATo,K2o,87s,Q8s,84o,73s,J2o`
- IP范围 (88种): `AJs,A7s,99,33,Q2s,QQ,A4s,K4s,Q3s,JTs,66,K7s,KTs,AA,K3s,Q7s,Q8s,K8s,A3s,K9s,55,KK,Q9s,K5s,KQs,A9s,QJs,A8s,ATs,J7s,Q6s,AQs,JJ,K6s,44,84o,J8s,T3s,Q7o,54s,AJo,73o,65o,83s,J6o,T2o,88,J9s,J6s,86o,83o,A6o,62o,T5s,T8o,K9o,52s,Q2o,82s,J3s,63s,43o,74o,Q5s,87o,85o,64s,T6o,A4o,TT,J7o,AQo,J8o,Q6o,95s,KQo,QTo,J2o,22,Q3o,QJo,72s,76s,84s,62s,54o,Q8o,53s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=40.938%, 平局率=1.696%
- EV: 0.00, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.86
- 加权EV: 0.00
- 策略: `{"Check":1,"Bet":0}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.022%, 范围平局率差异: 0.045%
- **策略差异: 33.3%**
- **加权EV差异: 0.21**

---

#### 反例 2

**场景1:**
- 公共牌: `Jh Kc 5s As 2d`
- OOP手牌: `Kd5d`
- OOP范围 (72种): `J7s,J8s,A7s,K9s,AJs,Q5s,KK,AQs,44,A2s,33,KQs,A9s,AKs,A5s,AA,55,A3s,Q2s,QQ,Q3s,K6s,Q6s,KTs,K3s,88,77,K5s,66,92o,76s,32o,T9s,T4s,32s,K9o,J3o,T2s,43s,J4o,J3s,75o,Q8o,94s,96s,84o,75s,K2s,42s,J5o,K2o,QTo,J9o,72o,A3o,Q7o,KQo,83s,ATo,A7o,Q6o,95o,T4o,86o,65s,Q4s,T6o,J4s,63s,J7o,53s,JJ`
- IP范围 (64种): `K4s,QQ,A2s,K8s,KK,44,J7s,A3s,K2s,AA,AQs,J9s,55,A8s,A5s,Q7s,AKs,77,KQs,Q8s,A7s,Q9s,QJs,A9s,KTs,Q8o,J6o,64o,75o,76s,ATs,T5s,K7o,J4s,66,76o,97s,K6o,T7s,Q3s,Q9o,K3s,J7o,AJo,KJs,T3o,J5o,T6s,J6s,T2o,86s,75s,K8o,63s,JTo,J8o,T8o,K2o,ATo,J2o,84o,J4o,A6o,T6o`
- 手牌: 胜率=94.690%, 平局率=0.000%
- 范围: 胜率=50.319%, 平局率=1.192%
- EV: 120.54, Solver Equity: 94.69%
- 动作EV: Check:120.47, Bet:120.67
- 加权EV: 120.54
- 策略: `{"Check":0.677367,"Bet":0.322633}`

**场景2:**
- 公共牌: `6h 8s 3s 3d 7s`
- OOP手牌: `Js6s`
- OOP范围 (92种): `33,A7s,KJs,J8s,KTs,88,QJs,ATs,K9s,K8s,K3s,QQ,JJ,J7s,KK,Q6s,K6s,77,AJs,K4s,Q7s,A9s,JTs,99,TT,J9s,KQs,Q2s,A6s,44,A8s,Q9s,55,A4s,A5s,QTs,75o,85o,AKs,K7o,QJo,Q5o,J2o,76o,32s,K9o,64s,J7o,42s,A2s,43o,74o,98s,Q6o,T2s,96o,KJo,A3s,KTo,Q9o,86s,95o,98o,J4o,54o,T3s,Q5s,92o,83s,66,A3o,T5s,75s,J6s,AKo,K5s,T2o,T3o,T8s,72o,32o,ATo,82s,63s,K4o,73s,AA,K6o,85s,A5o,52s,83o`
- IP范围 (97种): `Q4s,JTs,Q5s,K3s,Q7s,Q9s,AKs,Q2s,QJs,K8s,A4s,66,77,A9s,K5s,AA,TT,QQ,J8s,A8s,JJ,99,KJs,A5s,Q8s,55,KK,K6s,Q3s,33,K2s,44,J9s,ATs,A7s,A3s,K9s,A2s,K8o,72o,42o,64o,Q6o,22,Q2o,93o,AQo,72s,A6o,KTo,AJs,32s,98o,T8s,98s,A6s,87s,T5o,AQs,A2o,T3o,63s,43s,T6o,T9s,Q3o,97s,74s,J4s,75s,T9o,T8o,84o,96s,A9o,K5o,73o,85o,52s,K4s,76o,AJo,A7o,K7o,J5o,T2o,T4o,J6o,Q9o,A3o,73s,KQs,86s,64s,92o,J3s,ATo`
- 手牌: 胜率=94.697%, 平局率=0.000%
- 范围: 胜率=50.324%, 平局率=1.220%
- EV: 122.11, Solver Equity: 94.70%
- 动作EV: Check:122.11, Bet:119.74
- 加权EV: 122.11
- 策略: `{"Check":0.999996,"Bet":0.000004}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.007%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.005%, 范围平局率差异: 0.028%
- **策略差异: 32.3%**
- **加权EV差异: 1.58**

---

#### 反例 3

**场景1:**
- 公共牌: `3c 4h 8h 3s 2d`
- OOP手牌: `3d3h`
- OOP范围 (83种): `JJ,KK,K7s,A4s,J7s,K5s,55,Q3s,K2s,KQs,KJs,K4s,J8s,AQs,A8s,22,88,33,A3s,99,A6s,TT,AJs,JTs,AKs,KTs,QTs,QQ,A9s,Q6s,Q7s,Q4s,44,93s,A6o,J2s,43s,72s,32s,Q9s,T7s,82s,K5o,AA,T5s,QJs,64s,K9s,93o,J9o,86s,82o,87o,Q3o,85o,77,83o,Q5s,A9o,K4o,65s,A8o,73s,63o,83s,T5o,T7o,T3o,QJo,K2o,54s,Q7o,T6s,87s,A4o,92o,84o,T9o,J2o,A2o,JTo,J9s,T9s`
- IP范围 (69种): `Q7s,55,88,33,AQs,K3s,JTs,A8s,K8s,QJs,K2s,A3s,TT,J9s,44,Q6s,K5s,AA,QTs,A6s,A4s,ATs,JJ,Q8s,J8s,K9s,KK,93s,J9o,K9o,A2o,94s,A7s,64s,52o,72s,T7o,KJo,Q4o,83s,T2o,K5o,Q5s,96s,J5s,J5o,K6o,AJo,KQo,84s,94o,KTs,97o,T6o,T4o,83o,75o,43o,T6s,A3o,A2s,65s,77,AKs,63o,ATo,42o,Q2s,42s`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=54.856%, 平局率=0.726%
- EV: 149.72, Solver Equity: 100.00%
- 动作EV: Check:149.72, Bet:147.22
- 加权EV: 149.72
- 策略: `{"Check":1,"Bet":0}`

**场景2:**
- 公共牌: `2s 5c 4s 7s 9c`
- OOP手牌: `AsKs`
- OOP范围 (99种): `A6s,KQs,QTs,K3s,QJs,TT,K4s,A5s,AA,AJs,33,QQ,A4s,Q4s,K6s,55,Q7s,K9s,J8s,AKs,KK,KJs,Q6s,A8s,A3s,ATs,K2s,Q2s,K5s,A7s,K7s,22,Q3s,AQs,Q9s,66,Q5s,99,J9s,T6o,93o,K3o,K8s,J3o,43o,KTs,93s,A9s,Q4o,T2s,43s,32s,62s,T8s,32o,94o,53s,Q3o,85o,AKo,Q2o,T6s,65o,98s,76o,T3s,A2o,T7o,K8o,J8o,95s,J9o,52s,A9o,K2o,76s,44,JJ,85s,QTo,62o,53o,A3o,84o,95o,72s,T4s,64o,AJo,83o,JTs,AQo,96s,63o,88,98o,Q9o,ATo,86s`
- IP范围 (55种): `44,A7s,J7s,QTs,Q9s,QJs,K5s,99,KTs,AA,AKs,A9s,JTs,AQs,A5s,K3s,ATs,Q5s,Q3s,A8s,K8s,A2s,T7o,T3s,Q3o,K3o,98o,KJo,73s,J4s,75s,63s,T2s,T2o,KTo,64o,54o,94s,Q5o,J2s,K2s,97s,96s,Q6o,97o,AJs,76o,86s,32o,Q8o,QTo,87s,K7s,54s,T6o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=54.849%, 平局率=0.690%
- EV: 157.77, Solver Equity: 100.00%
- 动作EV: Check:156.53, Bet:157.77
- 加权EV: 157.77
- 策略: `{"Check":0.000007,"Bet":0.999993}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.006%, 范围平局率差异: 0.036%
- **策略差异: 100.0%**
- **加权EV差异: 8.05**

---

#### 反例 4

**场景1:**
- 公共牌: `3d 4h 5c 6h Qh`
- OOP手牌: `Tc8d`
- OOP范围 (107种): `A2s,88,Q2s,66,J8s,Q7s,Q9s,Q6s,K5s,QTs,55,AA,A9s,K4s,K7s,J9s,99,KK,KQs,AKs,K3s,ATs,KJs,K8s,A7s,A8s,JJ,Q8s,Q5s,Q3s,AJs,A5s,AQs,J7s,K6s,33,QJs,QQ,Q4s,A4s,77,JTs,K7o,J8o,52s,63o,K9o,85s,95s,74o,64s,22,76o,AJo,95o,Q9o,T8o,J6o,75o,76s,43s,J6s,J3s,82s,J4o,96s,T8s,53s,T6o,T7o,73o,92o,84o,72o,94o,98o,Q5o,K4o,86s,53o,KTs,T3o,Q7o,QTo,44,83s,63s,K8o,87o,86o,KTo,K3o,A7o,A4o,62s,65o,82o,97o,65s,A6s,T4o,AKo,A6o,J2s,AQo,KJo,A3s`
- IP范围 (75种): `KK,66,K3s,88,A5s,KQs,A3s,22,QTs,K8s,Q6s,K7s,QQ,TT,Q2s,A8s,Q8s,K2s,K6s,J8s,A6s,JTs,33,44,AJs,AA,A2s,A4s,99,Q7s,T4o,K2o,84s,ATo,87s,AQo,65s,J5s,K6o,73s,QTo,T7s,96s,A5o,43o,J8o,75s,76s,Q3s,T4s,72o,J3o,J4o,Q9s,Q7o,A9s,JJ,T5o,QJs,52s,KTs,55,ATs,94s,AKo,74o,83s,A7s,T2s,Q9o,J6s,52o,J7o,76o,63o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=44.386%, 平局率=3.977%
- EV: -0.00, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.89
- 加权EV: -0.00
- 策略: `{"Check":0.999967,"Bet":0.000033}`

**场景2:**
- 公共牌: `4h 4c 6d 7d 5c`
- OOP手牌: `Ts2h`
- OOP范围 (98种): `99,AQs,77,AA,A5s,J9s,K7s,K8s,AKs,33,J8s,K2s,AJs,A8s,Q5s,66,Q7s,A6s,Q9s,KQs,22,QQ,Q8s,A2s,A9s,KJs,88,Q4s,A7s,JTs,Q6s,Q3s,KK,K3s,KTs,JJ,K4s,QTs,QJs,76s,74s,Q6o,A2o,74o,94o,K8o,97s,A8o,96s,92s,K5s,32s,T7s,ATs,K3o,T2s,53o,54s,72s,83o,65o,Q5o,72o,63s,93s,AQo,44,T2o,A3s,T8s,T9o,95s,J6o,82s,J2s,54o,J6s,85s,A6o,Q9o,96o,K9o,85o,T6o,42o,K9s,J7s,J7o,43s,52s,94s,JTo,J5o,T5s,75o,J4o,A9o,82o`
- IP范围 (86种): `KTs,A6s,99,JTs,QQ,A4s,K8s,Q8s,TT,A9s,A3s,A7s,A2s,AJs,K9s,J8s,Q5s,K6s,J7s,77,33,KQs,44,K4s,Q6s,QTs,KK,J9s,22,ATs,Q4s,88,KJs,AQs,T8o,J8o,Q3s,KJo,53o,AA,93s,K2o,A4o,74s,75o,K7s,T3o,72s,84o,43s,98s,AKs,42s,85o,K8o,J9o,32s,QJo,J3o,73o,Q4o,A5s,94o,86s,T4s,JJ,A8s,J6o,QJs,K4o,Q3o,K2s,72o,T3s,K7o,52o,74o,86o,97o,ATo,J6s,T9o,KQo,83s,AKo,62o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=44.365%, 平局率=3.993%
- EV: -0.00, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.00
- 加权EV: -0.00
- 策略: `{"Check":0.481537,"Bet":0.518463}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.021%, 范围平局率差异: 0.016%
- **策略差异: 51.8%**
- **加权EV差异: 0.00**

---

#### 反例 5

**场景1:**
- 公共牌: `4c 7s Kd Ad 8c`
- OOP手牌: `AhAs`
- OOP范围 (83种): `99,K3s,AQs,K2s,55,66,K5s,A6s,A3s,A8s,QJs,AJs,A4s,QTs,J7s,J8s,K8s,AA,ATs,KTs,Q4s,AKs,K7s,Q5s,44,TT,A2s,77,K4s,33,KQs,KJs,JJ,A3o,A6o,72s,T9o,KK,82o,A7s,63o,83o,AJo,Q3o,J8o,T6o,T8o,T2o,K8o,98o,T5o,T2s,52s,J5o,ATo,92s,K6o,22,J6o,J3s,53o,94s,42o,32s,AQo,T4o,KQo,63s,53s,74o,93o,T7o,62o,54s,Q2s,A2o,75o,76o,K3o,65o,K9s,96o,97s`
- IP范围 (71种): `KTs,ATs,A8s,A4s,QQ,AA,88,Q6s,A7s,99,44,K2s,22,AKs,JJ,AQs,QTs,J7s,TT,JTs,Q8s,K9s,Q4s,K6s,55,K4s,Q7s,A3s,76s,KTo,A6s,Q9o,Q3o,K8o,92s,83o,54s,T4o,Q3s,73o,93s,K7o,K3s,AQo,J9o,A5s,QJo,64s,82s,Q2s,Q6o,A7o,J9s,32s,Q4o,86s,J6s,Q9s,AKo,K4o,AJo,A9o,KK,T3o,85s,ATo,J5o,A2o,T9s,T5s,52s`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=44.865%, 平局率=1.686%
- EV: 158.96, Solver Equity: 100.00%
- 动作EV: Check:150.95, Bet:158.96
- 加权EV: 158.96
- 策略: `{"Check":0,"Bet":1}`

**场景2:**
- 公共牌: `6s Jc Th Ah 2d`
- OOP手牌: `TcTd`
- OOP范围 (65种): `A6s,TT,QQ,QTs,A4s,22,Q3s,Q4s,77,JJ,66,Q5s,Q6s,K5s,J7s,J8s,K7s,K3s,AJs,K9s,Q9s,55,Q2s,KTs,KQs,K6s,52s,43s,83o,Q2o,J2s,76o,Q4o,33,K2o,K7o,98s,53s,Q6o,ATo,A5s,KJs,AKo,T7o,K2s,74s,T9o,J5s,JTs,AQo,K5o,J9s,K9o,76s,97s,T5o,Q9o,KJo,72s,75o,T7s,75s,A4o,A8o,KTo`
- IP范围 (45种): `AKs,KJs,A6s,44,AQs,K7s,Q8s,K2s,22,A3s,ATs,A5s,K3s,66,Q4s,Q5s,TT,K5s,T4s,53o,85o,AQo,AKo,A5o,A6o,J8o,QJo,93s,T9o,82o,ATo,98o,J6o,K4o,76o,72o,63s,A9s,42s,Q9o,J6s,J8s,T5s,J7o,86s`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=44.834%, 平局率=1.729%
- EV: 162.58, Solver Equity: 100.00%
- 动作EV: Check:162.58, Bet:156.14
- 加权EV: 162.58
- 策略: `{"Check":1,"Bet":0}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.031%, 范围平局率差异: 0.043%
- **策略差异: 100.0%**
- **加权EV差异: 3.61**

