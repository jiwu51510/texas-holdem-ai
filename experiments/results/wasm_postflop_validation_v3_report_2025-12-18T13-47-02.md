# 跨公共牌四维度胜率-策略验证实验报告 V3

## 实验时间

2025-12-18T13-47-02

## 累计统计

| 指标 | 本次 | 累计 |
|------|------|------|
| 运行次数 | 1 | 3 |
| 场景数 | 23933 | 63564 |
| 胜率相近对数 | 35 | 69 |
| 策略差异显著反例 | 16 | 25 |

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

- 生成场景数: 23933
- 四维度胜率相近的场景对（差异<0.05%）: 35
- 策略差异显著(>15%)的场景对: 16

## 关键发现

### ⚠️ 四维度胜率标量不足以决定最优策略

在 35 对四维度胜率相近的场景中，有 16 对（45.7%）的策略差异显著。

**结论：即使手牌胜率、手牌平局率、范围胜率、范围平局率都精确匹配（差异<0.05%），最优策略仍然可能完全不同。**

### 策略差异显著的反例

---

#### 反例 1

**场景1:**
- 公共牌: `9s 7h 3d 2d Ad`
- OOP手牌: `6h5s`
- OOP范围 (111种): `QTs,Q7s,AQs,Q9s,KK,K7s,TT,J9s,QJs,A8s,AJs,55,88,A3s,Q8s,Q6s,22,K8s,K3s,A5s,Q5s,K5s,A9s,QQ,K6s,Q2s,33,KJs,AKs,99,J8s,A6s,Q4s,44,Q3s,KTs,66,JTs,77,J7s,K9s,K2s,K4s,JJ,96s,97o,42o,Q4o,94s,76s,KTo,T6o,AJo,94o,97s,QJo,T9s,T3o,K5o,63s,K7o,92s,KQo,AA,A8o,T5o,Q6o,63o,T9o,72s,74s,J6s,Q9o,86s,ATo,J9o,A6o,62o,73s,A7o,65o,72o,A4o,76o,K2o,Q5o,A3o,Q3o,A9o,64s,T2s,A2o,K6o,52o,A2s,95s,32s,A5o,K3o,J5o,T3s,65s,87o,J5s,43o,85o,J3s,J2s,43s,64o,ATs`
- IP范围 (75种): `QTs,A4s,K9s,Q5s,88,ATs,44,K5s,KK,KTs,J9s,TT,QJs,JJ,Q8s,Q3s,K2s,K7s,JTs,K3s,33,AJs,KJs,66,77,A5s,K4s,A6s,22,J7s,84o,95s,K6s,52s,A3s,AKo,AQs,J8s,82o,K3o,K9o,QQ,A2s,73o,J2s,J4o,99,A5o,K6o,96s,T3s,K4o,K5o,72s,T2s,97s,T8s,82s,Q2s,A2o,T7o,85o,T9o,J5s,54o,73s,76s,Q7o,95o,62s,74s,86s,87s,42o,53o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=48.236%, 平局率=0.535%
- EV: -0.00, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-1.95
- 加权EV: -0.00
- 策略: `{"Check":0.999998,"Bet":0.000002}`

**场景2:**
- 公共牌: `2d Js 7s Ah 3s`
- OOP手牌: `6d5c`
- OOP范围 (118种): `A4s,ATs,QTs,KK,K4s,A9s,A2s,K2s,K9s,J9s,K6s,TT,Q3s,AQs,22,QQ,Q9s,AJs,JTs,KQs,J8s,99,A3s,J7s,55,77,Q4s,A7s,Q6s,Q2s,KTs,K3s,A6s,AA,Q8s,44,Q5s,KJs,33,A5s,AKs,JJ,K8s,Q7s,66,K5s,QJs,Q6o,J4o,87s,52o,T7s,K9o,72o,K7o,98s,T9s,T3s,84o,J6s,42s,AQo,KTo,84s,72s,T6o,Q5o,K8o,74s,92o,Q7o,KJo,86o,A8s,A9o,76o,QJo,43s,J8o,Q9o,A7o,T8o,54s,J3o,AJo,Q3o,92s,65o,53s,53o,T5o,ATo,T9o,64s,A8o,75s,85o,J2o,83o,T3o,J7o,A6o,74o,97o,96o,82s,T6s,96s,86s,63s,K3o,K5o,QTo,73s,88,73o,T2s,J3s`
- IP范围 (77种): `AKs,TT,99,A7s,K8s,22,A4s,AJs,Q5s,KTs,88,K3s,K2s,K7s,JTs,KJs,ATs,Q9s,Q8s,J7s,J8s,33,QJs,77,A5s,A2s,K5s,66,K6s,Q6s,83s,52o,95s,QTs,A9o,JTo,T6s,J4s,85o,K4s,92s,T6o,87o,95o,J7o,42o,T8o,72o,Q3o,74s,74o,97s,K9s,J4o,T8s,QJo,93o,87s,AQo,76s,AKo,Q8o,K2o,42s,Q3s,32s,A2o,J2o,J9o,98s,Q4s,86o,J5s,T4o,J3s,43o,85s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=48.279%, 平局率=0.568%
- EV: -0.29, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.93
- 加权EV: -0.29
- 策略: `{"Check":0.692275,"Bet":0.307725}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.044%, 范围平局率差异: 0.033%
- **策略差异: 30.8%**
- **加权EV差异: 0.29**

---

#### 反例 2

**场景1:**
- 公共牌: `Jc Tc 3h 5c 6h`
- OOP手牌: `Ac6c`
- OOP范围 (70种): `KK,66,KJs,J8s,Q8s,33,A6s,AQs,JJ,Q7s,KTs,K4s,AJs,99,K9s,Q6s,JTs,K2s,ATs,A4s,55,AKs,K8s,K3s,TT,88,K6s,A2s,J6s,J7o,72s,KQs,J8o,A4o,64s,A9o,96s,K5o,Q9s,Q4o,T3s,T5s,K3o,K5s,Q6o,86s,J7s,62s,76o,93o,KTo,73o,85o,A9s,A7s,52o,32o,77,AQo,43s,32s,75s,52s,T6o,A8s,98s,QTs,53s,K4o,J9s`
- IP范围 (97种): `Q5s,AQs,TT,Q2s,KTs,K2s,K4s,A3s,K8s,Q7s,KQs,99,Q9s,J9s,QTs,KK,J7s,66,AJs,Q3s,Q6s,88,QQ,A9s,ATs,K5s,QJs,A2s,Q8s,AKs,K7s,A8s,A4s,44,55,Q4s,K3s,AA,77,KJo,J5s,A5s,98s,64s,72s,K6s,K7o,98o,33,52s,T5o,J2o,A6s,K2o,Q4o,JTo,75o,73s,Q9o,87o,T8s,AQo,54o,43s,86o,K5o,T8o,73o,94o,53o,AJo,A8o,92s,J4o,KTo,T5s,65s,J9o,53s,K9s,52o,KJs,T9s,J5o,T2s,QTo,63s,K3o,T3s,97s,96s,74s,22,T6o,AKo,Q6o,75s`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=49.945%, 平局率=0.842%
- EV: 157.67, Solver Equity: 100.00%
- 动作EV: Check:157.67, Bet:156.48
- 加权EV: 157.67
- 策略: `{"Check":1,"Bet":0}`

**场景2:**
- 公共牌: `7s 2d 3h Ac 9c`
- OOP手牌: `9d9h`
- OOP范围 (70种): `Q8s,44,Q7s,K4s,KQs,A8s,Q4s,K6s,A2s,A9s,QQ,K2s,AA,Q3s,KTs,A7s,33,55,A6s,JTs,K8s,22,Q9s,KK,QTs,ATs,J9s,A3s,62s,72s,87o,T5o,T6s,83o,Q6o,84s,32s,75s,98s,96s,54s,65s,QJo,95o,94o,84o,Q5s,T2o,K3o,T4o,86o,J8o,97s,A8o,93s,99,AKo,96o,87s,95s,J3o,43o,42s,A4o,86s,74s,63o,52o,88,82o`
- IP范围 (84种): `KK,KJs,J8s,Q6s,A3s,A6s,Q4s,K9s,A8s,A9s,A7s,KQs,AQs,A5s,AKs,33,K2s,99,Q8s,JJ,66,Q9s,77,44,22,JTs,A2s,K4s,K3s,K8s,J9s,Q5s,K5s,QTo,AKo,KQo,53o,J2s,T7o,84s,75s,J6o,ATs,A9o,Q4o,K4o,K7o,JTo,KTs,87o,K6s,Q3s,84o,K6o,T6s,94s,J3s,62s,J6s,Q3o,K7s,72o,43o,A3o,TT,85s,J4o,96o,86o,76s,J5o,A4s,83s,73s,A2o,64o,32s,42s,T9o,87s,K9o,ATo,T5o,A8o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=49.933%, 平局率=0.874%
- EV: 164.93, Solver Equity: 100.00%
- 动作EV: Check:164.70, Bet:164.93
- 加权EV: 164.93
- 策略: `{"Check":0.006361,"Bet":0.993639}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.012%, 范围平局率差异: 0.032%
- **策略差异: 99.4%**
- **加权EV差异: 7.26**

---

#### 反例 3

**场景1:**
- 公共牌: `5c 7h Ah Qc 2h`
- OOP手牌: `Ts7d`
- OOP范围 (112种): `J7s,JTs,KJs,33,Q8s,AA,KTs,K9s,ATs,77,Q2s,K3s,JJ,66,22,J8s,QQ,A4s,K6s,A8s,A5s,55,99,A6s,Q9s,K2s,Q3s,QTs,K4s,A2s,KK,Q4s,Q5s,TT,AKs,A7s,Q7s,K7s,A3s,AQs,J9s,44,88,A9s,72s,97s,J9o,J5o,J4o,Q9o,T5o,AQo,J6s,Q5o,T8o,T8s,54o,T7o,A8o,T5s,Q3o,43s,KQo,98o,72o,T9o,73s,J5s,32o,96s,J8o,85s,K8o,T7s,94s,Q6o,K7o,J6o,75o,65s,J3s,A4o,A9o,65o,92s,K3o,86o,J4s,76s,97o,52o,QJs,AJo,87o,85o,84s,73o,Q4o,JTo,64s,83s,AJs,42s,K2o,T9s,95s,84o,AKo,83o,92o,82o,K5o`
- IP范围 (89种): `Q2s,AQs,A8s,AA,AKs,Q8s,K7s,K6s,KQs,J8s,K2s,A2s,QTs,66,KJs,A4s,Q6s,J9s,K8s,J7s,55,Q3s,K4s,33,A7s,ATs,99,A3s,K9s,QQ,44,JJ,QJs,KTs,Q7s,A6o,95o,73o,62o,63o,54s,53o,Q5s,64s,K8o,Q9s,76o,K7o,82o,T4s,84s,77,T4o,J4o,T3s,K3o,54o,93o,94s,86s,Q4s,K2o,J3o,65s,T6s,ATo,73s,Q3o,52s,AJs,63s,85s,A7o,52o,92s,43s,K3s,87o,92o,KK,KJo,82s,32s,JTs,85o,KTo,TT,64o,J5o`
- 手牌: 胜率=65.646%, 平局率=0.000%
- 范围: 胜率=52.682%, 平局率=0.794%
- EV: 54.62, Solver Equity: 65.65%
- 动作EV: Check:54.67, Bet:54.32
- 加权EV: 54.62
- 策略: `{"Check":0.860922,"Bet":0.139078}`

**场景2:**
- 公共牌: `3c Qd 5s Ac 8d`
- OOP手牌: `9c9d`
- OOP范围 (73种): `Q7s,K6s,A7s,K2s,66,QTs,KTs,J7s,JJ,AQs,Q4s,55,K8s,ATs,77,A2s,88,44,KQs,K3s,K5s,Q3s,J9s,99,AA,TT,KJs,22,A8s,J6s,73s,97o,AJo,A6o,83o,A2o,96s,T4o,AKo,62o,76o,J4s,J4o,97s,AQo,33,Q9s,A8o,42o,84o,T6o,92o,63s,AJs,Q2s,96o,64s,94s,T8o,K6o,J5o,JTs,T2s,75s,QQ,53o,Q6s,QJo,A3s,Q8o,87s,A5o,87o`
- IP范围 (55种): `QJs,Q7s,44,Q2s,K6s,A8s,ATs,KTs,33,AQs,TT,KQs,AKs,Q3s,A9s,22,K3s,Q8s,JJ,K8s,88,J9s,QQ,43o,94s,KK,43s,K5s,52o,96o,K7s,J2o,98s,QTs,76s,72o,K3o,AJs,ATo,86o,A3s,T2o,93s,T7o,KJo,J9o,83o,54s,JTs,J2s,J8o,A3o,53o,K8o,Q4s`
- 手牌: 胜率=65.603%, 平局率=0.000%
- 范围: 胜率=52.731%, 平局率=0.788%
- EV: 58.96, Solver Equity: 65.60%
- 动作EV: Check:56.50, Bet:58.96
- 加权EV: 58.96
- 策略: `{"Check":0,"Bet":1}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.043%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.049%, 范围平局率差异: 0.006%
- **策略差异: 86.1%**
- **加权EV差异: 4.34**

---

#### 反例 4

**场景1:**
- 公共牌: `Td 6c Kc 5h Js`
- OOP手牌: `AhQh`
- OOP范围 (61种): `Q2s,99,JJ,AQs,K7s,KK,TT,J7s,A3s,88,Q8s,ATs,AA,Q7s,KTs,A6s,K8s,K4s,Q9s,55,QTs,33,66,J9s,JTs,75s,Q8o,96o,62o,AJo,92s,A9s,65s,54s,T2o,K3s,64o,K4o,93s,95s,53o,92o,42o,73s,22,J2o,K7o,72s,86o,Q5s,A3o,82o,Q9o,54o,74s,A5s,Q3s,T4s,T7o,44,J7o`
- IP范围 (76种): `Q5s,JTs,Q7s,JJ,J9s,J7s,77,44,QJs,99,KQs,QQ,QTs,J8s,K6s,A9s,K5s,A8s,K8s,AA,KJs,K4s,A2s,KTs,A7s,K9s,22,55,AJs,66,93o,K4o,J7o,J2s,63o,T9s,K8o,T9o,Q3s,Q8o,88,42s,A8o,Q6s,92o,82o,T4o,Q2o,94s,T7o,33,83o,65o,53s,A9o,QJo,53o,K3s,J6o,ATs,T6o,QTo,73o,92s,74s,97s,65s,J5s,96o,Q5o,J6s,87o,52s,62s,43s,A5s`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=51.673%, 平局率=1.839%
- EV: 161.77, Solver Equity: 100.00%
- 动作EV: Check:162.09, Bet:161.41
- 加权EV: 161.77
- 策略: `{"Check":0.536329,"Bet":0.463671}`

**场景2:**
- 公共牌: `Qc Jd 3s 6d 8c`
- OOP手牌: `QhQs`
- OOP范围 (87种): `JTs,AJs,Q2s,66,A7s,99,A3s,AKs,Q4s,33,KK,A5s,ATs,Q3s,TT,55,QQ,77,KQs,J8s,Q7s,Q8s,A4s,A8s,K7s,A6s,K5s,AQs,JJ,K3s,Q9s,A9s,22,K2s,88,A2o,95o,52s,64s,83o,K2o,53o,32o,T7s,J4s,J8o,82o,95s,T7o,J2o,K9o,J6s,32s,76o,QTs,73o,QTo,T9s,94o,64o,84o,96o,82s,J2s,98o,K4s,65o,KQo,A3o,87o,K8s,94s,93o,85s,A4o,43s,ATo,KTo,62s,53s,93s,JTo,75s,76s,42o,44,63o`
- IP范围 (55种): `A2s,A8s,K8s,QTs,66,K7s,K9s,22,K3s,44,Q7s,Q4s,AKs,ATs,K2s,QJs,A4s,AA,Q2s,JTs,KJs,Q9s,32o,84o,K6s,T8s,TT,Q5o,QQ,Q3o,T4s,T8o,A7s,K5o,54o,32s,K3o,62o,75o,85o,72s,Q9o,ATo,KJo,72o,Q5s,62s,Q7o,K9o,95s,64s,43o,82s,88,63o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=51.651%, 平局率=1.819%
- EV: 159.36, Solver Equity: 100.00%
- 动作EV: Check:158.01, Bet:159.36
- 加权EV: 159.36
- 策略: `{"Check":0,"Bet":1}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.022%, 范围平局率差异: 0.020%
- **策略差异: 53.6%**
- **加权EV差异: 2.42**

---

#### 反例 5

**场景1:**
- 公共牌: `Qs Kd 2d 3d Tc`
- OOP手牌: `Ad8d`
- OOP范围 (74种): `K8s,Q3s,22,Q9s,77,K2s,Q4s,TT,JTs,QTs,K6s,K5s,66,A5s,K4s,A6s,A4s,K9s,AQs,A9s,K3s,QQ,33,J9s,Q6s,Q7s,A8s,AKs,Q8s,QTo,K5o,J7s,T9o,KTs,84o,T2s,T6s,82s,Q7o,92s,T7s,A7o,82o,Q9o,K7s,98s,32o,Q3o,96s,98o,64o,99,A9o,43o,43s,K4o,55,T7o,85o,85s,KK,Q8o,T9s,Q4o,T3s,83o,62o,T6o,86o,J5o,K6o,42o,95s,AJo`
- IP范围 (70种): `A2s,K8s,QJs,A5s,KQs,KJs,A9s,22,K9s,QTs,K5s,66,77,K2s,ATs,TT,55,99,88,K7s,Q9s,JTs,K4s,KK,K3s,Q2s,QQ,A3s,Q5s,A9o,95s,AJo,J2o,86o,K6o,Q8o,K4o,AQs,65o,KTs,T4o,43s,75s,42o,98o,T5s,44,T3o,Q6s,A4o,AQo,Q6o,Q7s,73s,J4s,T9o,54o,A6o,A5o,87o,J6s,93s,94o,K5o,T6s,T7s,98s,J8s,QTo,97s`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=52.820%, 平局率=0.768%
- EV: 163.92, Solver Equity: 100.00%
- 动作EV: Check:163.39, Bet:164.23
- 加权EV: 163.92
- 策略: `{"Check":0.371662,"Bet":0.628339}`

**场景2:**
- 公共牌: `Ah 4d 7d Kh 5c`
- OOP手牌: `3d2h`
- OOP范围 (93种): `Q4s,Q7s,KQs,K4s,A8s,K5s,K6s,AA,KTs,J7s,A7s,Q2s,QTs,Q5s,A6s,K2s,88,A9s,A4s,J9s,QJs,66,ATs,J8s,33,22,44,AKs,Q6s,TT,QQ,K7s,A5s,K9s,K8s,Q9s,JTs,73s,73o,72s,J5s,97o,42s,K6o,84o,KTo,A7o,94s,65o,Q8s,52o,KK,J4o,92o,43o,A2s,AJs,QTo,63o,53o,55,T5s,T9o,T7o,32o,K2o,86o,96o,93o,77,82o,Q3s,82s,98o,72o,J5o,A3o,Q4o,97s,A4o,T6s,J6s,T9s,ATo,J2s,T2s,54s,A6o,94o,75s,75o,95o,T7s`
- IP范围 (50种): `Q3s,A3s,A2s,AA,A4s,55,KJs,J9s,33,AQs,AKs,A5s,K6s,99,Q8s,88,Q9s,K4s,JJ,TT,95o,Q9o,J4o,K2o,66,J5o,A2o,KTs,Q7o,84o,92s,87s,K5s,96o,A9o,J2o,K7s,Q5s,62s,72o,T9s,Q8o,QQ,98o,K8s,74o,K5o,54s,J3o,73s`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=52.791%, 平局率=0.737%
- EV: 149.71, Solver Equity: 100.00%
- 动作EV: Check:149.12, Bet:149.71
- 加权EV: 149.71
- 策略: `{"Check":0.003211,"Bet":0.996789}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.029%, 范围平局率差异: 0.031%
- **策略差异: 36.8%**
- **加权EV差异: 14.22**

---

#### 反例 6

**场景1:**
- 公共牌: `8s Kc 7d Kd 4d`
- OOP手牌: `6c2d`
- OOP范围 (74种): `K8s,Q3s,22,Q9s,77,K2s,Q4s,TT,JTs,QTs,K6s,K5s,66,A5s,K4s,A6s,A4s,K9s,AQs,A9s,K3s,QQ,33,J9s,Q6s,Q7s,A8s,AKs,Q8s,QTo,K5o,J7s,T9o,KTs,84o,T2s,T6s,82s,Q7o,92s,T7s,A7o,82o,Q9o,K7s,98s,32o,Q3o,96s,98o,64o,99,A9o,43o,43s,K4o,55,T7o,85o,85s,KK,Q8o,T9s,Q4o,T3s,83o,62o,T6o,86o,J5o,K6o,42o,95s,AJo`
- IP范围 (70种): `A2s,K8s,QJs,A5s,KQs,KJs,A9s,22,K9s,QTs,K5s,66,77,K2s,ATs,TT,55,99,88,K7s,Q9s,JTs,K4s,KK,K3s,Q2s,QQ,A3s,Q5s,A9o,95s,AJo,J2o,86o,K6o,Q8o,K4o,AQs,65o,KTs,T4o,43s,75s,42o,98o,T5s,44,T3o,Q6s,A4o,AQo,Q6o,Q7s,73s,J4s,T9o,54o,A6o,A5o,87o,J6s,93s,94o,K5o,T6s,T7s,98s,J8s,QTo,97s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=47.810%, 平局率=1.663%
- EV: -0.01, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.02
- 加权EV: -0.01
- 策略: `{"Check":0.514829,"Bet":0.485171}`

**场景2:**
- 公共牌: `3s Qs 8s 5c 3c`
- OOP手牌: `6c4h`
- OOP范围 (91种): `JTs,33,Q4s,QTs,K4s,22,QQ,77,KK,K5s,J9s,AQs,A6s,ATs,A8s,A5s,A4s,AKs,Q6s,KJs,J8s,K9s,K8s,TT,55,A3s,K7s,QJs,KTs,J7s,K6s,A2s,Q5s,Q9s,A9s,AJs,KJo,K7o,64s,T3o,74o,A7s,95o,82o,88,T4s,93s,J2s,Q6o,T7o,64o,J2o,J6o,K4o,AA,A6o,J8o,65o,63o,KQo,32o,Q5o,JTo,Q3o,J6s,K3o,43s,T2o,96o,86o,75o,Q7s,93o,AQo,T7s,T8o,T5o,T8s,66,T6s,42o,T9s,Q3s,T9o,84s,75s,85s,A9o,T4o,J7o,52o`
- IP范围 (54种): `AKs,99,KJs,J7s,Q7s,QJs,QTs,AJs,88,K2s,K7s,33,A8s,Q9s,QQ,A9s,JJ,Q6s,66,A2s,TT,QTo,J4o,92s,Q3s,98s,J3s,K6s,96s,A8o,53o,74s,A9o,T2s,Q8s,K4s,83s,72s,K4o,72o,43o,JTo,77,T9o,94o,K8o,82s,KQo,ATo,Q7o,Q5s,T6o,K8s,J2s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=47.785%, 平局率=1.712%
- EV: 0.00, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.80
- 加权EV: 0.00
- 策略: `{"Check":1,"Bet":0}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.025%, 范围平局率差异: 0.049%
- **策略差异: 48.5%**
- **加权EV差异: 0.01**

---

#### 反例 7

**场景1:**
- 公共牌: `Ks Qd 5c 7h Js`
- OOP手牌: `Jh8d`
- OOP范围 (99种): `Q8s,J9s,A7s,K2s,55,A5s,66,Q3s,AKs,A4s,K5s,K4s,A6s,K7s,K3s,AJs,88,Q7s,Q5s,77,QJs,KK,Q4s,JTs,ATs,J8s,Q2s,A9s,TT,A2s,K6s,KQs,AA,A3s,22,99,Q9s,J7s,AQs,J8o,QQ,75s,K4o,A6o,Q7o,82o,83s,87s,73s,KJo,97s,J3s,97o,63o,Q5o,T7s,A9o,A3o,93s,76s,Q2o,95o,54o,Q4o,73o,J6s,92o,T3s,JTo,75o,J4s,82s,T8o,86o,96o,KQo,33,J4o,72s,95s,KJs,T4o,98o,53s,T3o,T6s,44,A8s,T6o,QTs,63s,74o,KTo,T9s,32o,52s,52o,T2s,T8s`
- IP范围 (45种): `Q6s,Q8s,K3s,QTs,JTs,44,A5s,55,99,AA,88,A8s,Q4s,KJs,K5s,K2s,J7s,KK,KTo,52s,54o,43o,A6o,33,A8o,73o,85s,K3o,J3o,TT,JTo,32o,ATo,77,J9s,J2o,J4s,72o,65o,T6s,Q7o,63o,98o,Q4o,A4s`
- 手牌: 胜率=63.035%, 平局率=0.000%
- 范围: 胜率=44.367%, 平局率=2.241%
- EV: 52.40, Solver Equity: 63.03%
- 动作EV: Check:52.40, Bet:51.90
- 加权EV: 52.40
- 策略: `{"Check":1,"Bet":0}`

**场景2:**
- 公共牌: `6h 4c 2h 3s 6d`
- OOP手牌: `8d8h`
- OOP范围 (111种): `K2s,J7s,AA,Q9s,Q3s,KK,K8s,J9s,A9s,A8s,Q5s,K3s,22,K5s,88,TT,A6s,AKs,Q2s,KTs,Q8s,Q6s,44,77,33,KJs,AQs,A4s,A7s,K7s,A5s,K4s,QTs,99,ATs,A3s,Q7s,J8s,AJs,JJ,55,QJs,JTs,KQs,T9o,T3s,74s,K9s,Q4s,AKo,A4o,75o,95s,85o,64s,86o,A3o,Q3o,ATo,76o,84s,65s,T5s,96s,T8s,JTo,J4s,95o,J3s,T7o,Q7o,T2o,94s,K6s,43s,QQ,K4o,AQo,54s,QJo,43o,KTo,Q2o,53s,A9o,QTo,K9o,KQo,J6o,86s,52o,J7o,T4o,82s,J5o,96o,K7o,83s,93o,Q9o,K6o,Q8o,Q6o,T2s,94o,92s,T7s,62o,A6o,84o,64o`
- IP范围 (42种): `A6s,AKs,ATs,A8s,K3s,33,AA,Q7s,J9s,K4s,JJ,Q9s,QTs,66,K5s,K7s,T6o,ATo,96o,J4o,73o,T8s,KQs,J7s,J6o,Q6s,A9o,K5o,97o,T4s,55,94s,A5o,K6s,T2s,A5s,82s,Q4o,54s,93o,T8o,J3s`
- 手牌: 胜率=63.014%, 平局率=0.000%
- 范围: 胜率=44.416%, 平局率=2.237%
- EV: 57.01, Solver Equity: 63.01%
- 动作EV: Check:55.67, Bet:57.01
- 加权EV: 57.01
- 策略: `{"Check":0,"Bet":1}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.021%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.049%, 范围平局率差异: 0.004%
- **策略差异: 100.0%**
- **加权EV差异: 4.62**

---

#### 反例 8

**场景1:**
- 公共牌: `4s Jd 7h Th 3c`
- OOP手牌: `5c2s`
- OOP范围 (91种): `JTs,33,Q4s,QTs,K4s,22,QQ,77,KK,K5s,J9s,AQs,A6s,ATs,A8s,A5s,A4s,AKs,Q6s,KJs,J8s,K9s,K8s,TT,55,A3s,K7s,QJs,KTs,J7s,K6s,A2s,Q5s,Q9s,A9s,AJs,KJo,K7o,64s,T3o,74o,A7s,95o,82o,88,T4s,93s,J2s,Q6o,T7o,64o,J2o,J6o,K4o,AA,A6o,J8o,65o,63o,KQo,32o,Q5o,JTo,Q3o,J6s,K3o,43s,T2o,96o,86o,75o,Q7s,93o,AQo,T7s,T8o,T5o,T8s,66,T6s,42o,T9s,Q3s,T9o,84s,75s,85s,A9o,T4o,J7o,52o`
- IP范围 (54种): `AKs,99,KJs,J7s,Q7s,QJs,QTs,AJs,88,K2s,K7s,33,A8s,Q9s,QQ,A9s,JJ,Q6s,66,A2s,TT,QTo,J4o,92s,Q3s,98s,J3s,K6s,96s,A8o,53o,74s,A9o,T2s,Q8s,K4s,83s,72s,K4o,72o,43o,JTo,77,T9o,94o,K8o,82s,KQo,ATo,Q7o,Q5s,T6o,K8s,J2s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=48.089%, 平局率=0.659%
- EV: -0.35, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.46
- 加权EV: -0.35
- 策略: `{"Check":0.251456,"Bet":0.748544}`

**场景2:**
- 公共牌: `2s 5h 7d 9d Kh`
- OOP手牌: `6d3d`
- OOP范围 (92种): `Q3s,JJ,22,A7s,Q2s,A6s,AKs,K6s,KK,Q6s,KQs,77,QTs,Q4s,K4s,J7s,AJs,TT,88,KJs,55,JTs,44,K5s,K8s,A5s,A4s,Q8s,A9s,AA,K9s,Q7s,99,Q5s,KTs,A3s,96o,53s,J8o,97s,87o,J5o,K7s,KJo,K2s,64s,73s,93o,65s,T9s,94o,QJs,32o,T4s,33,J6o,AQo,94s,AKo,J2o,AQs,63s,43s,QJo,J6s,83s,75s,A8s,86s,K4o,76o,95o,84o,ATo,66,T7o,32s,T4o,J9s,74o,T2s,52s,T8s,A6o,54s,T6s,52o,A3o,Q2o,76s,86o,92s`
- IP范围 (54种): `JJ,Q5s,33,QQ,QTs,K6s,AQs,KQs,A3s,QJs,44,K7s,AKs,A5s,AJs,A6s,Q9s,TT,77,JTs,KTs,J5s,73o,75s,AJo,Q7s,96o,85o,K6o,A7s,76o,88,T6s,KTo,K8s,A2o,A8o,JTo,AQo,T5o,82o,K8o,74s,93s,85s,Q3o,86o,99,J2o,Q4o,K5s,62s,55,62o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=48.107%, 平局率=0.703%
- EV: 0.91, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:0.91
- 加权EV: 0.91
- 策略: `{"Check":0,"Bet":1}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.018%, 范围平局率差异: 0.044%
- **策略差异: 25.1%**
- **加权EV差异: 1.26**

---

#### 反例 9

**场景1:**
- 公共牌: `3d 7h Ks 5h 7s`
- OOP手牌: `Kh9s`
- OOP范围 (100种): `KTs,Q9s,KK,A2s,K7s,A9s,AJs,K5s,J9s,AQs,K4s,22,88,99,A3s,JTs,77,K8s,K6s,A8s,A5s,TT,Q6s,KJs,Q3s,Q5s,A6s,AKs,KQs,33,ATs,J7s,K2s,JJ,Q8s,66,Q4s,AA,QTs,Q2s,T5o,K7o,82o,75o,K9o,T2o,J6s,Q4o,62s,64o,93o,J8s,86o,K3s,52s,92o,T4o,Q8o,AQo,72s,ATo,Q7o,73s,53o,J3o,53s,J8o,63s,96s,93s,32s,T3s,62o,87s,84s,95s,K2o,K5o,T6o,T9o,84o,74o,85o,KJo,K6o,K3o,J2s,A3o,J4s,T4s,KTo,55,K4o,43s,T6s,43o,T8s,T2s,J5o,Q6o`
- IP范围 (45种): `QQ,AKs,QJs,JTs,K3s,A6s,A3s,Q6s,A5s,KQs,A4s,K2s,J7s,K5s,Q5s,QTs,KTs,Q3s,ATs,J7o,J5o,93o,53s,K6s,62o,KJs,J9s,Q4o,T8o,43o,66,53o,J2o,KJo,83o,K7s,AQo,Q9o,22,83s,AA,Q2o,A2o,42s,J4o`
- 手牌: 胜率=88.142%, 平局率=0.000%
- 范围: 胜率=49.347%, 平局率=1.996%
- EV: 101.02, Solver Equity: 88.14%
- 动作EV: Check:98.94, Bet:101.02
- 加权EV: 101.02
- 策略: `{"Check":0,"Bet":1}`

**场景2:**
- 公共牌: `6d 8h Js 4c 6s`
- OOP手牌: `AcAh`
- OOP范围 (113种): `99,K6s,K8s,AA,A3s,A2s,Q6s,QTs,K5s,33,KJs,Q9s,ATs,22,55,K4s,J7s,K2s,JJ,AJs,AQs,KTs,66,J9s,K9s,A9s,A7s,Q7s,JTs,K7s,77,K3s,J8s,A8s,44,TT,Q5s,Q4s,A5s,KK,Q3s,AKs,KQs,QQ,Q2s,Q9o,T4s,A6s,86s,A3o,43s,52s,QJo,A7o,K9o,T7s,J2o,T2o,J5s,A2o,T2s,A5o,T9s,J2s,85o,32o,64o,85s,AKo,88,65s,63o,J6s,J3o,T5s,J6o,54s,62s,95s,72s,K4o,K7o,Q8o,A4o,K8o,Q6o,74o,84s,K3o,72o,93o,87s,42s,94s,T6s,Q3o,75s,T8s,A9o,A6o,Q7o,42o,K6o,A8o,65o,53s,96o,KQo,74s,T9o,Q8s,98o,T3s`
- IP范围 (67种): `A8s,JJ,K3s,A7s,QTs,KK,AQs,TT,Q8s,A4s,ATs,AKs,Q4s,JTs,J9s,22,J8s,K4s,Q6s,AJs,33,44,K6s,Q3s,QQ,A6s,J5s,94s,AQo,Q5o,85o,88,96o,63o,T9s,KJs,A9s,K9o,T2s,K8o,73o,A7o,QTo,92o,A4o,Q5s,K7s,Q7o,54s,T4o,98s,77,84s,97s,Q7s,Q9s,95s,93s,94o,Q4o,T8s,86s,32s,Q6o,T6o,K9s,AJo`
- 手牌: 胜率=88.125%, 平局率=0.000%
- 范围: 胜率=49.331%, 平局率=2.021%
- EV: 101.74, Solver Equity: 88.13%
- 动作EV: Check:101.74, Bet:101.58
- 加权EV: 101.74
- 策略: `{"Check":0.999835,"Bet":0.000165}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.017%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.015%, 范围平局率差异: 0.025%
- **策略差异: 100.0%**
- **加权EV差异: 0.72**

---

#### 反例 10

**场景1:**
- 公共牌: `Jh 3h 9h 6h Ks`
- OOP手牌: `AhJc`
- OOP范围 (73种): `A5s,K6s,Q2s,KJs,33,AA,K5s,Q8s,KQs,A7s,K7s,J9s,ATs,Q3s,J7s,22,99,KTs,Q6s,K4s,Q9s,66,Q4s,TT,JJ,A8s,A3s,K2s,77,83o,K9o,98o,87o,94o,88,76o,T5s,J2s,QJs,72s,73s,95o,52s,93o,J4s,T6s,A4s,T6o,Q6o,JTs,82s,53o,A9o,T3o,T2o,A2o,54s,A8o,44,AJo,AKo,A6s,75s,J3o,KTo,A7o,T4s,Q7s,55,J4o,64o,62s,84s`
- IP范围 (82种): `55,JTs,K8s,J8s,A2s,K5s,ATs,K7s,QJs,Q4s,Q8s,A3s,KTs,AKs,J7s,Q9s,KJs,QQ,A5s,KK,JJ,AJs,A8s,QTs,K3s,A7s,AQs,KQs,99,Q3s,Q7s,44,A4s,K6s,84o,J9o,JTo,A5o,J4s,83s,T7o,64s,J6s,92o,97s,95s,K7o,K4s,A7o,A4o,T9s,KQo,A3o,A6s,J5o,54o,Q6o,K9s,75o,T5s,T2s,ATo,T8o,85s,J3s,KTo,J2s,87s,83o,62o,86s,J3o,Q3o,QTo,93o,J6o,64o,T6o,A6o,T9o,43s,J7o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=47.886%, 平局率=0.656%
- EV: 148.83, Solver Equity: 100.00%
- 动作EV: Check:148.83, Bet:146.37
- 加权EV: 148.83
- 策略: `{"Check":1,"Bet":0}`

**场景2:**
- 公共牌: `4d Ad 7s Ks 3s`
- OOP手牌: `As5s`
- OOP范围 (77种): `AA,A4s,QJs,QTs,K5s,A8s,Q2s,Q3s,K6s,A7s,44,66,KJs,77,A3s,J7s,KQs,KTs,A2s,K7s,55,ATs,J8s,Q6s,KK,Q5s,JTs,A5s,QQ,AQs,95s,Q8o,Q9s,Q7o,52o,T5o,Q3o,97o,Q2o,82o,KQo,72s,42o,T6o,J8o,A6o,A5o,76s,96s,43s,T4s,A9o,42s,94o,K8o,T5s,64o,QJo,J5s,J6s,64s,KTo,K5o,T6s,22,ATo,J3s,J3o,92s,Q7s,TT,94s,J2s,T9o,74s,99,J2o`
- IP范围 (98种): `A9s,TT,A4s,ATs,K8s,K5s,88,22,A6s,Q3s,A5s,Q9s,44,K6s,JJ,77,K4s,Q2s,55,QQ,KK,K3s,99,A7s,JTs,Q7s,K9s,K7s,QJs,Q6s,A3s,AQs,J8s,Q4s,AA,K2s,AKs,KQs,KTs,A3o,64o,T7o,K7o,K9o,92o,J8o,42s,98s,K5o,A9o,86s,J3s,53o,Q5s,83o,ATo,J2s,86o,54o,76s,72o,QJo,65o,QTs,KJs,73o,J7o,Q9o,52o,97s,62s,J6o,T9s,54s,Q8o,74s,76o,J5o,A2s,T8s,T4s,K2o,62o,94s,85s,K4o,33,T7s,A6o,64s,66,J5s,96o,T2o,97o,82s,AQo,T3o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=47.882%, 平局率=0.666%
- EV: 161.70, Solver Equity: 100.00%
- 动作EV: Check:161.63, Bet:161.70
- 加权EV: 161.70
- 策略: `{"Check":0.016327,"Bet":0.983673}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.004%, 范围平局率差异: 0.010%
- **策略差异: 98.4%**
- **加权EV差异: 12.86**

---

#### 反例 11

**场景1:**
- 公共牌: `5c 2c 3s Ts 9h`
- OOP手牌: `7c4c`
- OOP范围 (113种): `99,K6s,K8s,AA,A3s,A2s,Q6s,QTs,K5s,33,KJs,Q9s,ATs,22,55,K4s,J7s,K2s,JJ,AJs,AQs,KTs,66,J9s,K9s,A9s,A7s,Q7s,JTs,K7s,77,K3s,J8s,A8s,44,TT,Q5s,Q4s,A5s,KK,Q3s,AKs,KQs,QQ,Q2s,Q9o,T4s,A6s,86s,A3o,43s,52s,QJo,A7o,K9o,T7s,J2o,T2o,J5s,A2o,T2s,A5o,T9s,J2s,85o,32o,64o,85s,AKo,88,65s,63o,J6s,J3o,T5s,J6o,54s,62s,95s,72s,K4o,K7o,Q8o,A4o,K8o,Q6o,74o,84s,K3o,72o,93o,87s,42s,94s,T6s,Q3o,75s,T8s,A9o,A6o,Q7o,42o,K6o,A8o,65o,53s,96o,KQo,74s,T9o,Q8s,98o,T3s`
- IP范围 (67种): `A8s,JJ,K3s,A7s,QTs,KK,AQs,TT,Q8s,A4s,ATs,AKs,Q4s,JTs,J9s,22,J8s,K4s,Q6s,AJs,33,44,K6s,Q3s,QQ,A6s,J5s,94s,AQo,Q5o,85o,88,96o,63o,T9s,KJs,A9s,K9o,T2s,K8o,73o,A7o,QTo,92o,A4o,Q5s,K7s,Q7o,54s,T4o,98s,77,84s,97s,Q7s,Q9s,95s,93s,94o,Q4o,T8s,86s,32s,Q6o,T6o,K9s,AJo`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=45.505%, 平局率=0.677%
- EV: 0.05, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:0.27
- 加权EV: 0.05
- 策略: `{"Check":0.814511,"Bet":0.185489}`

**场景2:**
- 公共牌: `4d 3c 3h 5h 9d`
- OOP手牌: `Th6c`
- OOP范围 (63种): `KJs,44,A4s,66,Q2s,33,KTs,QTs,JJ,AA,QJs,AKs,ATs,Q7s,Q9s,J8s,Q3s,K7s,A9s,99,55,JTs,AJs,K4s,J9s,98o,85s,KTo,A6s,83s,QQ,K6s,QTo,T8o,97o,93s,Q7o,A3s,KK,Q8o,64s,62o,J3s,A2s,A3o,T6o,52s,72s,K4o,76s,AQs,Q5o,K2o,J4s,J2o,86s,77,J7o,83o,76o,K8o,65s,A8s`
- IP范围 (71种): `Q6s,K4s,KTs,22,A3s,Q4s,A8s,ATs,QQ,44,Q2s,J7s,QTs,JJ,A4s,66,J9s,A6s,AQs,Q5s,KJs,K5s,K8s,K6s,Q8s,K3s,AKs,AJs,A7o,J2s,JTo,97o,J6o,98s,96o,K2s,55,99,A7s,K9s,53o,AA,84o,94s,A4o,T5o,KJo,97s,K5o,QJs,T7s,K9o,85s,J6s,93o,95s,A3o,QJo,AQo,K3o,T8o,32o,KQs,K2o,A2s,62s,T3o,J5s,42o,88,92o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=45.458%, 平局率=0.645%
- EV: 0.12, Solver Equity: 0.00%
- 动作EV: Check:-0.00, Bet:0.21
- 加权EV: 0.12
- 策略: `{"Check":0.447669,"Bet":0.552331}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.046%, 范围平局率差异: 0.031%
- **策略差异: 36.7%**
- **加权EV差异: 0.07**

---

#### 反例 12

**场景1:**
- 公共牌: `Js 8s 3s 5h Th`
- OOP手牌: `7h2s`
- OOP范围 (113种): `99,K6s,K8s,AA,A3s,A2s,Q6s,QTs,K5s,33,KJs,Q9s,ATs,22,55,K4s,J7s,K2s,JJ,AJs,AQs,KTs,66,J9s,K9s,A9s,A7s,Q7s,JTs,K7s,77,K3s,J8s,A8s,44,TT,Q5s,Q4s,A5s,KK,Q3s,AKs,KQs,QQ,Q2s,Q9o,T4s,A6s,86s,A3o,43s,52s,QJo,A7o,K9o,T7s,J2o,T2o,J5s,A2o,T2s,A5o,T9s,J2s,85o,32o,64o,85s,AKo,88,65s,63o,J6s,J3o,T5s,J6o,54s,62s,95s,72s,K4o,K7o,Q8o,A4o,K8o,Q6o,74o,84s,K3o,72o,93o,87s,42s,94s,T6s,Q3o,75s,T8s,A9o,A6o,Q7o,42o,K6o,A8o,65o,53s,96o,KQo,74s,T9o,Q8s,98o,T3s`
- IP范围 (67种): `A8s,JJ,K3s,A7s,QTs,KK,AQs,TT,Q8s,A4s,ATs,AKs,Q4s,JTs,J9s,22,J8s,K4s,Q6s,AJs,33,44,K6s,Q3s,QQ,A6s,J5s,94s,AQo,Q5o,85o,88,96o,63o,T9s,KJs,A9s,K9o,T2s,K8o,73o,A7o,QTo,92o,A4o,Q5s,K7s,Q7o,54s,T4o,98s,77,84s,97s,Q7s,Q9s,95s,93s,94o,Q4o,T8s,86s,32s,Q6o,T6o,K9s,AJo`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=49.595%, 平局率=0.816%
- EV: 0.10, Solver Equity: 0.00%
- 动作EV: Check:-0.00, Bet:0.10
- 加权EV: 0.10
- 策略: `{"Check":0.000103,"Bet":0.999897}`

**场景2:**
- 公共牌: `7c 3h 5s Js Ks`
- OOP手牌: `9s8c`
- OOP范围 (112种): `ATs,K2s,Q8s,Q4s,K8s,A3s,AKs,Q7s,K9s,K5s,Q2s,QQ,33,44,99,QJs,A2s,JTs,J8s,Q3s,77,22,K7s,A7s,Q5s,A4s,QTs,AA,66,Q6s,J9s,88,KJs,KQs,K3s,Q9s,55,AJs,TT,K4s,K6s,KK,JJ,J7s,K9o,T6o,T8s,96o,42o,T9o,82o,T2o,83o,65o,42s,54o,AQs,K6o,63s,96s,53o,T3s,52o,K2o,92o,87s,KTo,43s,84o,63o,72o,AKo,J6o,Q3o,K7o,73s,52s,64o,JTo,A6s,J2s,J3s,A9o,75o,A9s,74o,43o,T5s,J3o,95o,J4o,T7o,92s,97s,J8o,Q8o,Q6o,K4o,97o,T5o,A7o,85s,Q4o,J2o,Q5o,98o,QJo,94s,82s,T3o,A8s,J6s`
- IP范围 (44种): `AJs,K9s,A7s,J8s,KJs,A8s,Q6s,Q7s,ATs,KTs,K6s,88,QTs,TT,Q9s,A4s,KK,T6s,43o,53o,95o,83s,T7o,77,64s,J7s,85s,J2o,A8o,T7s,KTo,K8o,Q9o,T5o,T8s,T6o,AKs,74o,93s,97o,T2o,54o,A6o,A3o`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=49.634%, 平局率=0.822%
- EV: -0.12, Solver Equity: 0.00%
- 动作EV: Check:-0.00, Bet:-0.19
- 加权EV: -0.12
- 策略: `{"Check":0.364683,"Bet":0.635317}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.039%, 范围平局率差异: 0.006%
- **策略差异: 36.5%**
- **加权EV差异: 0.22**

---

#### 反例 13

**场景1:**
- 公共牌: `4d 9d Qd 5d 3s`
- OOP手牌: `KdJd`
- OOP范围 (90种): `A8s,99,33,A5s,J8s,Q2s,K9s,A2s,Q6s,Q7s,K6s,55,A3s,Q5s,AKs,Q4s,K5s,77,A9s,K4s,KJs,KQs,K7s,A4s,88,KK,K2s,Q9s,66,J7s,QQ,Q8s,TT,44,KTs,A7s,T3o,K9o,95s,62s,T6o,K2o,T2s,73s,A7o,84o,J4o,Q6o,K3o,AJs,K4o,KTo,T4o,85o,Q8o,T5s,KJo,63s,AQo,T7o,T9s,62o,ATo,72o,A6o,J9o,97o,76s,A8o,96o,82o,AJo,92s,JJ,J6s,J6o,64s,K5o,96s,K3s,86s,A2o,87o,QTo,32o,A6s,A9o,J4s,98s,T2o`
- IP范围 (48种): `Q5s,88,KTs,K3s,Q9s,Q6s,A4s,J9s,A8s,A9s,K7s,K2s,66,KJs,44,ATs,K6s,K8s,AQs,62s,93o,94o,QTo,62o,52o,KK,J3o,J4o,JJ,53o,AKo,64o,K3o,T8o,K7o,JTs,86o,Q2s,T3o,K5s,J6s,T9o,98s,JTo,J6o,A3s,96s,33`
- 手牌: 胜率=97.647%, 平局率=0.000%
- 范围: 胜率=51.065%, 平局率=0.268%
- EV: 128.22, Solver Equity: 97.65%
- 动作EV: Check:125.09, Bet:128.22
- 加权EV: 128.22
- 策略: `{"Check":0,"Bet":1}`

**场景2:**
- 公共牌: `4c 3s 9s Ks 2s`
- OOP手牌: `QhQs`
- OOP范围 (91种): `KTs,AA,K7s,J8s,K9s,K4s,Q7s,44,K8s,QJs,Q9s,22,QQ,ATs,J7s,JJ,A2s,KJs,A5s,AQs,JTs,A7s,Q3s,Q2s,88,QTs,KK,K2s,77,Q6s,K5s,TT,A8s,A4s,A6s,99,72s,92s,J6o,32o,J9s,53s,T9s,J8o,63o,Q8s,K5o,T4o,42o,Q4o,J7o,JTo,QJo,85s,AQo,J3o,52o,64s,42s,54s,Q6o,93s,A4o,84o,Q4s,K2o,83s,94s,65s,A3o,AJo,98o,Q9o,J3s,K3o,K3s,96o,63s,J9o,AJs,J2s,43s,ATo,82o,A2o,AKs,T6o,62s,K8o,J4o,KQs`
- IP范围 (81种): `KK,ATs,77,33,44,A6s,J8s,K2s,KJs,A2s,Q7s,A3s,Q6s,Q2s,QTs,K9s,KQs,A4s,55,K3s,A5s,QQ,K6s,AA,Q3s,66,Q4s,KTs,Q5s,K8s,A7s,TT,85o,83o,92o,T2s,J4s,Q9s,T3s,94o,KJo,T8s,T4s,J9s,K8o,83s,74o,T3o,99,J2s,K6o,88,Q8s,K3o,Q9o,53o,97o,J3s,86s,T2o,J7s,T9s,K2o,T7s,K9o,K5s,KQo,94s,Q7o,A8s,73s,K5o,75o,95s,Q2o,87o,K4o,AQs,54o,54s,QTo`
- 手牌: 胜率=97.619%, 平局率=0.000%
- 范围: 胜率=51.051%, 平局率=0.237%
- EV: 131.56, Solver Equity: 97.62%
- 动作EV: Check:131.56, Bet:130.14
- 加权EV: 131.56
- 策略: `{"Check":1,"Bet":0}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.028%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.014%, 范围平局率差异: 0.030%
- **策略差异: 100.0%**
- **加权EV差异: 3.34**

---

#### 反例 14

**场景1:**
- 公共牌: `7d 2c Kd Ad 3d`
- OOP手牌: `QdTs`
- OOP范围 (81种): `J7s,88,77,Q9s,AKs,Q8s,44,K6s,22,K8s,AA,QJs,K3s,A2s,K2s,Q7s,A9s,AQs,A7s,Q2s,K4s,KQs,JJ,A8s,AJs,K7s,A6s,99,K5s,KK,A4s,J8s,T2o,K7o,T2s,A5s,T8o,T5s,97o,97s,T9o,T5o,ATo,TT,Q4o,85s,33,94s,76o,55,75s,52o,95s,T7o,AQo,83o,62o,43s,J8o,K4o,A2o,T6o,63s,J6s,T7s,87s,64s,KJo,93o,Q3s,T4o,QTo,K3o,93s,72o,84o,64o,32s,63o,T3s,Q5o`
- IP范围 (48种): `KQs,Q4s,Q2s,44,22,Q9s,TT,Q7s,88,A2s,J8s,Q5s,K5s,JJ,J7s,K9s,Q6s,AKs,Q3s,J5s,74s,A6s,KK,A5s,T4o,85s,J4o,97s,J2s,85o,63s,43s,K8o,92o,J3o,73s,87o,AA,QTs,AJs,T7s,A9o,86s,J8o,73o,66,A4o,T7o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=50.661%, 平局率=0.256%
- EV: 159.97, Solver Equity: 100.00%
- 动作EV: Check:158.85, Bet:159.97
- 加权EV: 159.97
- 策略: `{"Check":0.000146,"Bet":0.999854}`

**场景2:**
- 公共牌: `4s 6h Ah Kh 3h`
- OOP手牌: `Qh6s`
- OOP范围 (100种): `Q6s,AKs,99,66,55,QJs,Q4s,22,K2s,77,A3s,AQs,Q7s,KTs,Q2s,J9s,A7s,JTs,ATs,J7s,QQ,A8s,Q3s,KQs,KK,K5s,A9s,K3s,Q8s,A5s,A6s,88,33,KJs,K7s,K9s,K4s,QTs,JJ,A4s,73s,T3s,T6o,44,Q2o,T5s,K6o,K2o,A6o,QJo,J3s,43s,Q6o,AA,83o,A8o,94s,A3o,84o,82s,J5o,AJo,K8o,ATo,T8s,T4o,93s,T7o,73o,K5o,74o,J8o,J4s,65s,KQo,A7o,Q5s,87o,Q4o,43o,T4s,Q5o,A9o,T3o,63s,42o,84s,75o,85s,AQo,T7s,T2s,A5o,J2o,54s,JTo,92o,83s,75s,J7o`
- IP范围 (90种): `J7s,66,K6s,AJs,Q4s,33,K3s,K8s,A7s,44,K7s,A5s,KTs,Q6s,Q5s,55,A4s,ATs,AA,Q2s,77,Q9s,QJs,A3s,K4s,JJ,99,A6s,88,KK,A2s,22,K9s,J9s,QQ,Q8s,76s,K6o,J7o,A2o,AQo,K8o,ATo,A8o,A5o,97s,83s,92s,85s,J6s,85o,65s,KQo,53o,74s,Q6o,K2o,JTo,K2s,A9o,86o,97o,72s,62o,K7o,Q4o,73o,32o,A9s,98o,72o,AKs,J8o,43s,42s,Q7o,94o,T8o,32s,84s,54o,J4o,96o,83o,J4s,82s,T4o,AKo,Q3s,86s`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=50.709%, 平局率=0.265%
- EV: 152.22, Solver Equity: 100.00%
- 动作EV: Check:152.22, Bet:150.50
- 加权EV: 152.22
- 策略: `{"Check":1,"Bet":0}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.048%, 范围平局率差异: 0.008%
- **策略差异: 100.0%**
- **加权EV差异: 7.74**

---

#### 反例 15

**场景1:**
- 公共牌: `3h 4s Qs 2c 8s`
- OOP手牌: `AsJs`
- OOP范围 (117种): `J9s,AQs,A5s,Q3s,QTs,KQs,QQ,55,K7s,KK,TT,99,AJs,K6s,A9s,A4s,Q6s,A7s,Q9s,JJ,K9s,Q4s,Q2s,KTs,K2s,KJs,A8s,A2s,Q5s,QJs,AKs,ATs,K3s,A6s,88,Q7s,44,K4s,22,AA,K8s,66,77,K5s,Q8s,33,54s,T3o,53s,T6o,A2o,A4o,86o,T2s,Q7o,86s,74s,J8o,43s,87s,94s,62s,73s,84o,Q9o,63o,75o,K6o,J7s,75s,A9o,A6o,82s,KQo,32s,84s,32o,42o,65s,T4s,92o,82o,53o,J6s,95o,Q6o,43o,52o,83s,T5s,52s,T3s,J7o,Q2o,A3o,63s,JTs,J5o,J6o,74o,KTo,85o,83o,T2o,J2o,87o,Q3o,96o,T5o,85s,QTo,T4o,J3o,T7o,64s,K9o,72s`
- IP范围 (45种): `A6s,K3s,Q5s,22,KJs,K7s,Q8s,K2s,K9s,A9s,KQs,88,QQ,J7s,TT,A8s,99,K8s,KQo,JTs,Q2s,AKs,52o,KTo,A2o,A4s,96s,52s,T5o,62o,86o,T7o,86s,98s,74s,94o,Q3s,T3s,54s,J9o,K6s,J3s,42s,T2s,72o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=53.693%, 平局率=0.673%
- EV: 162.43, Solver Equity: 100.00%
- 动作EV: Check:162.43, Bet:161.65
- 加权EV: 162.43
- 策略: `{"Check":0.996948,"Bet":0.003052}`

**场景2:**
- 公共牌: `4c 8h 7s 2s 9h`
- OOP手牌: `JdTd`
- OOP范围 (65种): `K2s,K9s,K7s,TT,55,Q4s,A7s,JTs,Q3s,K8s,A8s,QTs,QJs,A6s,Q8s,K3s,AA,KQs,KK,AQs,QQ,K4s,44,K6s,Q5s,J7s,ATo,93s,54s,75s,K3o,87o,96o,Q7o,98s,KJs,Q6o,93o,T4s,Q3o,54o,T4o,92o,K5o,74s,97o,AQo,74o,A5o,ATs,72o,77,A5s,KTs,95s,J4s,K4o,J2o,65o,A4o,84s,Q6s,J6s,A8o,A3s`
- IP范围 (94种): `88,A8s,A9s,55,A7s,KQs,K6s,K3s,Q8s,Q7s,K4s,Q4s,66,Q6s,22,A4s,Q9s,44,KTs,A2s,J9s,Q3s,QQ,A5s,ATs,K9s,K2s,J8s,QJs,QTs,33,KK,Q5s,K7s,A3s,JJ,K5s,82o,K6o,74o,K3o,83o,76s,72o,KQo,T4s,AKs,53s,J3s,AJs,96s,84o,J5s,86o,Q2s,T4o,TT,54s,T8s,KJs,Q2o,A5o,62o,K9o,43s,77,A6o,T9s,T7s,92s,A4o,95o,32s,K5o,K2o,Q7o,63o,J6o,T5o,42s,82s,94o,QTo,62s,73o,96o,J2s,J4o,J8o,T5s,A2o,85o,52o,Q8o`
- 手牌: 胜率=100.000%, 平局率=0.000%
- 范围: 胜率=53.743%, 平局率=0.691%
- EV: 165.94, Solver Equity: 100.00%
- 动作EV: Check:166.17, Bet:165.66
- 加权EV: 165.94
- 策略: `{"Check":0.544669,"Bet":0.455331}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.050%, 范围平局率差异: 0.017%
- **策略差异: 45.2%**
- **加权EV差异: 3.51**

---

#### 反例 16

**场景1:**
- 公共牌: `Kd Jc 5s 6c 2h`
- OOP手牌: `8h4d`
- OOP范围 (79种): `AJs,AQs,AKs,JJ,QJs,55,Q9s,A7s,Q5s,QQ,A3s,Q6s,A2s,22,K2s,K8s,Q2s,K6s,33,77,Q3s,J8s,99,A9s,A8s,KJs,K4s,A5s,K3s,QTs,J9s,T3o,J7o,82o,KQo,43s,97o,J4s,97s,54o,A9o,54s,86o,Q5o,75s,J3o,84o,K5o,AKo,J4o,K7o,J2o,72s,K2o,T2o,92o,KTs,76o,52o,A4s,A8o,T9o,62o,85o,K6o,A6o,32s,T8o,T4s,63s,TT,A7o,T7o,T2s,T7s,53o,ATo,73o,A2o`
- IP范围 (63种): `Q7s,A8s,Q5s,A2s,K9s,A7s,A9s,J8s,K3s,AJs,JJ,66,AKs,Q4s,Q3s,A4s,K5s,KJs,J7s,K7s,Q8s,KTs,Q2s,K2s,77,T3o,96o,A8o,42o,97o,A6o,98o,T9o,ATo,75o,76o,T7o,T6s,KQs,63s,K6o,T9s,J6s,JTo,54o,J9o,K6s,42s,52s,J7o,K7o,86o,32s,75s,44,J5s,53s,97s,J4s,99,43o,85s,64s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=48.613%, 平局率=0.999%
- EV: -0.38, Solver Equity: 0.00%
- 动作EV: Check:-0.00, Bet:-0.53
- 加权EV: -0.38
- 策略: `{"Check":0.280367,"Bet":0.719633}`

**场景2:**
- 公共牌: `3d 8s Js Qs As`
- OOP手牌: `5d2h`
- OOP范围 (90种): `Q3s,J8s,A6s,QJs,Q8s,K6s,K3s,KK,Q7s,55,AQs,K8s,J9s,K5s,K4s,Q2s,K7s,A7s,Q9s,A4s,Q5s,QQ,A8s,Q4s,AA,JTs,QTs,J7s,A9s,A5s,22,K9s,99,KTs,K2s,A3s,72o,64o,63s,73o,J4s,53s,95s,63o,43s,QJo,76s,A9o,J8o,T5o,T5s,T3s,Q5o,52o,T7s,T6s,KJo,A3o,98o,98s,T2o,87s,Q9o,96o,J9o,86s,TT,84s,94s,A2o,88,97s,ATo,85o,92s,Q8o,K8o,T3o,J2o,32s,66,KQs,AJs,AQo,54o,82s,A8o,Q3o,62o,75o`
- IP范围 (44种): `ATs,A5s,QJs,JJ,33,Q9s,K6s,A4s,K8s,88,AKs,K4s,99,AJs,J8s,A6s,K9s,AQo,95o,JTo,93o,22,43s,QTo,Q7o,AQs,T6s,J6o,JTs,Q7s,K5o,Q2o,43o,Q3s,75s,J6s,87o,Q5s,T5s,64s,K7o,J9o,AJo,65s`
- 手牌: 胜率=0.000%, 平局率=0.000%
- 范围: 胜率=48.633%, 平局率=1.000%
- EV: -0.00, Solver Equity: 0.00%
- 动作EV: Check:0.00, Bet:-0.01
- 加权EV: -0.00
- 策略: `{"Check":0.678808,"Bet":0.321192}`

**对比:**
- 范围相同: 否
- 手牌胜率差异: 0.000%, 手牌平局率差异: 0.000%
- 范围胜率差异: 0.019%, 范围平局率差异: 0.000%
- **策略差异: 39.8%**
- **加权EV差异: 0.38**

