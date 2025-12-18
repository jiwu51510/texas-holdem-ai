# 跨公共牌双维度胜率-策略验证实验报告 V2

## 实验改进

**本版本使用 poker-odds-calc 库独立计算胜率，而不是依赖 solver 的结果。**

这确保了胜率计算的独立性和准确性。

## 实验目的

验证：**在不同的（公共牌+固定手牌）组合下：**
当以下两个条件同时满足时，策略是否相同？
1. 固定手牌vs对手范围的胜率相近（差异<1%）
2. 自己范围vs对手范围的胜率相近（差异<1%）

## 实验方法

1. 随机生成公共牌场景
2. 为每个公共牌场景随机选择一个固定手牌
3. **使用 poker-odds-calc 完整枚举计算胜率**：
   - 手牌胜率：遍历对手范围内所有有效组合（排除与固定手牌冲突的组合），计算胜率
   - 范围胜率：完整枚举OOP范围和IP范围的所有组合对，计算平均胜率（排除与固定OOP手牌冲突的组合）
4. 使用 wasm-postflop solver 获取最优策略
5. 比较双维度胜率相近的场景对的策略差异

## 范围定义

- **OOP范围**: AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o
- **IP范围**: AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo

## 实验规模

- 生成场景数: 100
- 双维度胜率相近的场景对（差异<1%）: 17
- 策略差异显著(>15%)的场景对: 7

## 关键发现

### 发现双维度胜率相近但策略不同的反例

以下是策略差异显著的反例详情（包含IP去掉死牌后的真实范围）：

---

#### 反例 1

**场景1:**
- 公共牌: `9c Js 7h Qh 4d`
- OOP手牌: `AhQd`
- 手牌胜率: 78.409%
- 范围胜率: 胜率: 45.095%, 平局率: 1.815% (综合: 46.003%)
- 策略: `{"Check":0,"Bet":0.243213,"Allin":0.756787}`
- IP有效范围 (176个组合):
  `AcAd, AcAs, AdAs, AcKc, AcKd, AcKh, AcKs, AdKc, AdKd, AdKh, AdKs, AsKc, AsKd, AsKh, AsKs, AcQc, AcQs, AdQc, AdQs, AsQc, AsQs, AcJc, AcJd, AcJh, AdJc, AdJd, AdJh, AsJc, AsJd, AsJh, AcTc, AcTd, AcTh, AcTs, AdTc, AdTd, AdTh, AdTs, AsTc, AsTd, AsTh, AsTs, Ad9d, As9s, Ac8c, Ad8d, As8s, Ac7c, Ad7d, As7s, Ac6c, Ad6d, As6s, Ac5c, Ad5d, As5s, Ac4c, As4s, Ac3c, Ad3d, As3s, Ac2c, Ad2d, As2s, KcKd, KcKh, KcKs, KdKh, KdKs, KhKs, KcQc, KcQs, KdQc, KdQs, KhQc, KhQs, KsQc, KsQs, KcJc, KcJd, KcJh, KdJc, KdJd, KdJh, KhJc, KhJd, KhJh, KsJc, KsJd, KsJh, KcTc, KdTd, KhTh, KsTs, Kd9d, Kh9h, Ks9s, QcQs, QcJc, QcJd, QcJh, QsJc, QsJd, QsJh, QcTc, QsTs, Qs9s, JcJd, JcJh, JdJh, JcTc, JdTd, JhTh, Jd9d, Jh9h, TcTd, TcTh, TcTs, TdTh, TdTs, ThTs, Td9d, Th9h, Ts9s, 9d9h, 9d9s, 9h9s, 9d8d, 9h8h, 9s8s, 8c8d, 8c8h, 8c8s, 8d8h, 8d8s, 8h8s, 8c7c, 8d7d, 8s7s, 7c7d, 7c7s, 7d7s, 7c6c, 7d6d, 7s6s, 6c6d, 6c6h, 6c6s, 6d6h, 6d6s, 6h6s, 6c5c, 6d5d, 6h5h, 6s5s, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4h, 4c4s, 4h4s, 3c3d, 3c3h, 3c3s, 3d3h, 3d3s, 3h3s, 2c2d, 2c2h, 2c2s, 2d2h, 2d2s, 2h2s`

**场景2:**
- 公共牌: `Js 2s 3s 4s Jd`
- OOP手牌: `QhQs`
- 手牌胜率: 77.604%
- 范围胜率: 胜率: 46.032%, 平局率: 0.413% (综合: 46.238%)
- 策略: `{"Check":0.454773,"Bet":0.214969,"Allin":0.330258}`
- IP有效范围 (192个组合):
  `AcAd, AcAh, AcAs, AdAh, AdAs, AhAs, AcKc, AcKd, AcKh, AcKs, AdKc, AdKd, AdKh, AdKs, AhKc, AhKd, AhKh, AhKs, AsKc, AsKd, AsKh, AsKs, AcQc, AcQd, AdQc, AdQd, AhQc, AhQd, AsQc, AsQd, AcJc, AcJh, AdJc, AdJh, AhJc, AhJh, AsJc, AsJh, AcTc, AcTd, AcTh, AcTs, AdTc, AdTd, AdTh, AdTs, AhTc, AhTd, AhTh, AhTs, AsTc, AsTd, AsTh, AsTs, Ac9c, Ad9d, Ah9h, As9s, Ac8c, Ad8d, Ah8h, As8s, Ac7c, Ad7d, Ah7h, As7s, Ac6c, Ad6d, Ah6h, As6s, Ac5c, Ad5d, Ah5h, As5s, Ac4c, Ad4d, Ah4h, Ac3c, Ad3d, Ah3h, Ac2c, Ad2d, Ah2h, KcKd, KcKh, KcKs, KdKh, KdKs, KhKs, KcQc, KcQd, KdQc, KdQd, KhQc, KhQd, KsQc, KsQd, KcJc, KcJh, KdJc, KdJh, KhJc, KhJh, KsJc, KsJh, KcTc, KdTd, KhTh, KsTs, Kc9c, Kd9d, Kh9h, Ks9s, QcQd, QcJc, QcJh, QdJc, QdJh, QcTc, QdTd, Qc9c, Qd9d, JcJh, JcTc, JhTh, Jc9c, Jh9h, TcTd, TcTh, TcTs, TdTh, TdTs, ThTs, Tc9c, Td9d, Th9h, Ts9s, 9c9d, 9c9h, 9c9s, 9d9h, 9d9s, 9h9s, 9c8c, 9d8d, 9h8h, 9s8s, 8c8d, 8c8h, 8c8s, 8d8h, 8d8s, 8h8s, 8c7c, 8d7d, 8h7h, 8s7s, 7c7d, 7c7h, 7c7s, 7d7h, 7d7s, 7h7s, 7c6c, 7d6d, 7h6h, 7s6s, 6c6d, 6c6h, 6c6s, 6d6h, 6d6s, 6h6s, 6c5c, 6d5d, 6h5h, 6s5s, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4d, 4c4h, 4d4h, 3c3d, 3c3h, 3d3h, 2c2d, 2c2h, 2d2h`

**对比:**
- 手牌胜率差异: 0.805%
- 范围胜率差异: 0.236%
- **策略差异: 30.3%**

---

#### 反例 2

**场景1:**
- 公共牌: `Ks 8d Ac 4d 2c`
- OOP手牌: `AsJc`
- 手牌胜率: 80.117%
- 范围胜率: 胜率: 45.456%, 平局率: 1.110% (综合: 46.011%)
- 策略: `{"Check":0.600069,"Bet":0.173622,"Allin":0.226309}`
- IP有效范围 (171个组合):
  `AdAh, AdKc, AdKd, AdKh, AhKc, AhKd, AhKh, AdQc, AdQd, AdQh, AdQs, AhQc, AhQd, AhQh, AhQs, AdJd, AdJh, AdJs, AhJd, AhJh, AhJs, AdTc, AdTd, AdTh, AdTs, AhTc, AhTd, AhTh, AhTs, Ad9d, Ah9h, Ah8h, Ad7d, Ah7h, Ad6d, Ah6h, Ad5d, Ah5h, Ah4h, Ad3d, Ah3h, Ad2d, Ah2h, KcKd, KcKh, KdKh, KcQc, KcQd, KcQh, KcQs, KdQc, KdQd, KdQh, KdQs, KhQc, KhQd, KhQh, KhQs, KcJd, KcJh, KcJs, KdJd, KdJh, KdJs, KhJd, KhJh, KhJs, KcTc, KdTd, KhTh, Kc9c, Kd9d, Kh9h, QcQd, QcQh, QcQs, QdQh, QdQs, QhQs, QcJd, QcJh, QcJs, QdJd, QdJh, QdJs, QhJd, QhJh, QhJs, QsJd, QsJh, QsJs, QcTc, QdTd, QhTh, QsTs, Qc9c, Qd9d, Qh9h, Qs9s, JdJh, JdJs, JhJs, JdTd, JhTh, JsTs, Jd9d, Jh9h, Js9s, TcTd, TcTh, TcTs, TdTh, TdTs, ThTs, Tc9c, Td9d, Th9h, Ts9s, 9c9d, 9c9h, 9c9s, 9d9h, 9d9s, 9h9s, 9c8c, 9h8h, 9s8s, 8c8h, 8c8s, 8h8s, 8c7c, 8h7h, 8s7s, 7c7d, 7c7h, 7c7s, 7d7h, 7d7s, 7h7s, 7c6c, 7d6d, 7h6h, 7s6s, 6c6d, 6c6h, 6c6s, 6d6h, 6d6s, 6h6s, 6c5c, 6d5d, 6h5h, 6s5s, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4h, 4c4s, 4h4s, 3c3d, 3c3h, 3c3s, 3d3h, 3d3s, 3h3s, 2d2h, 2d2s, 2h2s`

**场景2:**
- 公共牌: `Qd Js 8s 2c Ts`
- OOP手牌: `Tc9c`
- 手牌胜率: 81.026%
- 范围胜率: 胜率: 45.619%, 平局率: 2.422% (综合: 46.829%)
- 策略: `{"Check":0,"Bet":0.142312,"Allin":0.857688}`
- IP有效范围 (195个组合):
  `AcAd, AcAh, AcAs, AdAh, AdAs, AhAs, AcKc, AcKd, AcKh, AcKs, AdKc, AdKd, AdKh, AdKs, AhKc, AhKd, AhKh, AhKs, AsKc, AsKd, AsKh, AsKs, AcQc, AcQh, AcQs, AdQc, AdQh, AdQs, AhQc, AhQh, AhQs, AsQc, AsQh, AsQs, AcJc, AcJd, AcJh, AdJc, AdJd, AdJh, AhJc, AhJd, AhJh, AsJc, AsJd, AsJh, AcTd, AcTh, AdTd, AdTh, AhTd, AhTh, AsTd, AsTh, Ad9d, Ah9h, As9s, Ac8c, Ad8d, Ah8h, Ac7c, Ad7d, Ah7h, As7s, Ac6c, Ad6d, Ah6h, As6s, Ac5c, Ad5d, Ah5h, As5s, Ac4c, Ad4d, Ah4h, As4s, Ac3c, Ad3d, Ah3h, As3s, Ad2d, Ah2h, As2s, KcKd, KcKh, KcKs, KdKh, KdKs, KhKs, KcQc, KcQh, KcQs, KdQc, KdQh, KdQs, KhQc, KhQh, KhQs, KsQc, KsQh, KsQs, KcJc, KcJd, KcJh, KdJc, KdJd, KdJh, KhJc, KhJd, KhJh, KsJc, KsJd, KsJh, KdTd, KhTh, Kd9d, Kh9h, Ks9s, QcQh, QcQs, QhQs, QcJc, QcJd, QcJh, QhJc, QhJd, QhJh, QsJc, QsJd, QsJh, QhTh, Qh9h, Qs9s, JcJd, JcJh, JdJh, JdTd, JhTh, Jd9d, Jh9h, TdTh, Td9d, Th9h, 9d9h, 9d9s, 9h9s, 9d8d, 9h8h, 8c8d, 8c8h, 8d8h, 8c7c, 8d7d, 8h7h, 7c7d, 7c7h, 7c7s, 7d7h, 7d7s, 7h7s, 7c6c, 7d6d, 7h6h, 7s6s, 6c6d, 6c6h, 6c6s, 6d6h, 6d6s, 6h6s, 6c5c, 6d5d, 6h5h, 6s5s, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4d, 4c4h, 4c4s, 4d4h, 4d4s, 4h4s, 3c3d, 3c3h, 3c3s, 3d3h, 3d3s, 3h3s, 2d2h, 2d2s, 2h2s`

**对比:**
- 手牌胜率差异: 0.909%
- 范围胜率差异: 0.819%
- **策略差异: 42.1%**

---

#### 反例 3

**场景1:**
- 公共牌: `8h 3h Qh Th 2s`
- OOP手牌: `AsKs`
- 手牌胜率: 17.204%
- 范围胜率: 胜率: 45.107%, 平局率: 0.429% (综合: 45.321%)
- 策略: `{"Check":0.000001,"Bet":0.282858,"Allin":0.717141}`
- IP有效范围 (186个组合):
  `AcAd, AcAh, AdAh, AcKc, AcKd, AcKh, AdKc, AdKd, AdKh, AhKc, AhKd, AhKh, AcQc, AcQd, AcQs, AdQc, AdQd, AdQs, AhQc, AhQd, AhQs, AcJc, AcJd, AcJh, AcJs, AdJc, AdJd, AdJh, AdJs, AhJc, AhJd, AhJh, AhJs, AcTc, AcTd, AcTs, AdTc, AdTd, AdTs, AhTc, AhTd, AhTs, Ac9c, Ad9d, Ah9h, Ac8c, Ad8d, Ac7c, Ad7d, Ah7h, Ac6c, Ad6d, Ah6h, Ac5c, Ad5d, Ah5h, Ac4c, Ad4d, Ah4h, Ac3c, Ad3d, Ac2c, Ad2d, Ah2h, KcKd, KcKh, KdKh, KcQc, KcQd, KcQs, KdQc, KdQd, KdQs, KhQc, KhQd, KhQs, KcJc, KcJd, KcJh, KcJs, KdJc, KdJd, KdJh, KdJs, KhJc, KhJd, KhJh, KhJs, KcTc, KdTd, Kc9c, Kd9d, Kh9h, QcQd, QcQs, QdQs, QcJc, QcJd, QcJh, QcJs, QdJc, QdJd, QdJh, QdJs, QsJc, QsJd, QsJh, QsJs, QcTc, QdTd, QsTs, Qc9c, Qd9d, Qs9s, JcJd, JcJh, JcJs, JdJh, JdJs, JhJs, JcTc, JdTd, JsTs, Jc9c, Jd9d, Jh9h, Js9s, TcTd, TcTs, TdTs, Tc9c, Td9d, Ts9s, 9c9d, 9c9h, 9c9s, 9d9h, 9d9s, 9h9s, 9c8c, 9d8d, 9s8s, 8c8d, 8c8s, 8d8s, 8c7c, 8d7d, 8s7s, 7c7d, 7c7h, 7c7s, 7d7h, 7d7s, 7h7s, 7c6c, 7d6d, 7h6h, 7s6s, 6c6d, 6c6h, 6c6s, 6d6h, 6d6s, 6h6s, 6c5c, 6d5d, 6h5h, 6s5s, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4d, 4c4h, 4c4s, 4d4h, 4d4s, 4h4s, 3c3d, 3c3s, 3d3s, 2c2d, 2c2h, 2d2h`

**场景2:**
- 公共牌: `4d As 2d 3s Ts`
- OOP手牌: `KhJh`
- 手牌胜率: 16.578%
- 范围胜率: 胜率: 45.022%, 平局率: 1.166% (综合: 45.606%)
- 策略: `{"Check":0,"Bet":0.003774,"Allin":0.996226}`
- IP有效范围 (187个组合):
  `AcAd, AcAh, AdAh, AcKc, AcKd, AcKs, AdKc, AdKd, AdKs, AhKc, AhKd, AhKs, AcQc, AcQd, AcQh, AcQs, AdQc, AdQd, AdQh, AdQs, AhQc, AhQd, AhQh, AhQs, AcJc, AcJd, AcJs, AdJc, AdJd, AdJs, AhJc, AhJd, AhJs, AcTc, AcTd, AcTh, AdTc, AdTd, AdTh, AhTc, AhTd, AhTh, Ac9c, Ad9d, Ah9h, Ac8c, Ad8d, Ah8h, Ac7c, Ad7d, Ah7h, Ac6c, Ad6d, Ah6h, Ac5c, Ad5d, Ah5h, Ac4c, Ah4h, Ac3c, Ad3d, Ah3h, Ac2c, Ah2h, KcKd, KcKs, KdKs, KcQc, KcQd, KcQh, KcQs, KdQc, KdQd, KdQh, KdQs, KsQc, KsQd, KsQh, KsQs, KcJc, KcJd, KcJs, KdJc, KdJd, KdJs, KsJc, KsJd, KsJs, KcTc, KdTd, Kc9c, Kd9d, Ks9s, QcQd, QcQh, QcQs, QdQh, QdQs, QhQs, QcJc, QcJd, QcJs, QdJc, QdJd, QdJs, QhJc, QhJd, QhJs, QsJc, QsJd, QsJs, QcTc, QdTd, QhTh, Qc9c, Qd9d, Qh9h, Qs9s, JcJd, JcJs, JdJs, JcTc, JdTd, Jc9c, Jd9d, Js9s, TcTd, TcTh, TdTh, Tc9c, Td9d, Th9h, 9c9d, 9c9h, 9c9s, 9d9h, 9d9s, 9h9s, 9c8c, 9d8d, 9h8h, 9s8s, 8c8d, 8c8h, 8c8s, 8d8h, 8d8s, 8h8s, 8c7c, 8d7d, 8h7h, 8s7s, 7c7d, 7c7h, 7c7s, 7d7h, 7d7s, 7h7s, 7c6c, 7d6d, 7h6h, 7s6s, 6c6d, 6c6h, 6c6s, 6d6h, 6d6s, 6h6s, 6c5c, 6d5d, 6h5h, 6s5s, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4h, 4c4s, 4h4s, 3c3d, 3c3h, 3d3h, 2c2h, 2c2s, 2h2s`

**对比:**
- 手牌胜率差异: 0.627%
- 范围胜率差异: 0.285%
- **策略差异: 18.6%**

---

#### 反例 4

**场景1:**
- 公共牌: `8h 3h Qh Th 2s`
- OOP手牌: `AsKs`
- 手牌胜率: 17.204%
- 范围胜率: 胜率: 45.107%, 平局率: 0.429% (综合: 45.321%)
- 策略: `{"Check":0.000001,"Bet":0.282858,"Allin":0.717141}`
- IP有效范围 (186个组合):
  `AcAd, AcAh, AdAh, AcKc, AcKd, AcKh, AdKc, AdKd, AdKh, AhKc, AhKd, AhKh, AcQc, AcQd, AcQs, AdQc, AdQd, AdQs, AhQc, AhQd, AhQs, AcJc, AcJd, AcJh, AcJs, AdJc, AdJd, AdJh, AdJs, AhJc, AhJd, AhJh, AhJs, AcTc, AcTd, AcTs, AdTc, AdTd, AdTs, AhTc, AhTd, AhTs, Ac9c, Ad9d, Ah9h, Ac8c, Ad8d, Ac7c, Ad7d, Ah7h, Ac6c, Ad6d, Ah6h, Ac5c, Ad5d, Ah5h, Ac4c, Ad4d, Ah4h, Ac3c, Ad3d, Ac2c, Ad2d, Ah2h, KcKd, KcKh, KdKh, KcQc, KcQd, KcQs, KdQc, KdQd, KdQs, KhQc, KhQd, KhQs, KcJc, KcJd, KcJh, KcJs, KdJc, KdJd, KdJh, KdJs, KhJc, KhJd, KhJh, KhJs, KcTc, KdTd, Kc9c, Kd9d, Kh9h, QcQd, QcQs, QdQs, QcJc, QcJd, QcJh, QcJs, QdJc, QdJd, QdJh, QdJs, QsJc, QsJd, QsJh, QsJs, QcTc, QdTd, QsTs, Qc9c, Qd9d, Qs9s, JcJd, JcJh, JcJs, JdJh, JdJs, JhJs, JcTc, JdTd, JsTs, Jc9c, Jd9d, Jh9h, Js9s, TcTd, TcTs, TdTs, Tc9c, Td9d, Ts9s, 9c9d, 9c9h, 9c9s, 9d9h, 9d9s, 9h9s, 9c8c, 9d8d, 9s8s, 8c8d, 8c8s, 8d8s, 8c7c, 8d7d, 8s7s, 7c7d, 7c7h, 7c7s, 7d7h, 7d7s, 7h7s, 7c6c, 7d6d, 7h6h, 7s6s, 6c6d, 6c6h, 6c6s, 6d6h, 6d6s, 6h6s, 6c5c, 6d5d, 6h5h, 6s5s, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4d, 4c4h, 4c4s, 4d4h, 4d4s, 4h4s, 3c3d, 3c3s, 3d3s, 2c2d, 2c2h, 2d2h`

**场景2:**
- 公共牌: `Kc Jc Ts 7c 9h`
- OOP手牌: `5h5s`
- 手牌胜率: 16.414%
- 范围胜率: 胜率: 42.412%, 平局率: 3.998% (综合: 44.411%)
- 策略: `{"Check":0,"Bet":0.998138,"Allin":0.001862}`
- IP有效范围 (198个组合):
  `AcAd, AcAh, AcAs, AdAh, AdAs, AhAs, AcKd, AcKh, AcKs, AdKd, AdKh, AdKs, AhKd, AhKh, AhKs, AsKd, AsKh, AsKs, AcQc, AcQd, AcQh, AcQs, AdQc, AdQd, AdQh, AdQs, AhQc, AhQd, AhQh, AhQs, AsQc, AsQd, AsQh, AsQs, AcJd, AcJh, AcJs, AdJd, AdJh, AdJs, AhJd, AhJh, AhJs, AsJd, AsJh, AsJs, AcTc, AcTd, AcTh, AdTc, AdTd, AdTh, AhTc, AhTd, AhTh, AsTc, AsTd, AsTh, Ac9c, Ad9d, As9s, Ac8c, Ad8d, Ah8h, As8s, Ad7d, Ah7h, As7s, Ac6c, Ad6d, Ah6h, As6s, Ac5c, Ad5d, Ac4c, Ad4d, Ah4h, As4s, Ac3c, Ad3d, Ah3h, As3s, Ac2c, Ad2d, Ah2h, As2s, KdKh, KdKs, KhKs, KdQc, KdQd, KdQh, KdQs, KhQc, KhQd, KhQh, KhQs, KsQc, KsQd, KsQh, KsQs, KdJd, KdJh, KdJs, KhJd, KhJh, KhJs, KsJd, KsJh, KsJs, KdTd, KhTh, Kd9d, Ks9s, QcQd, QcQh, QcQs, QdQh, QdQs, QhQs, QcJd, QcJh, QcJs, QdJd, QdJh, QdJs, QhJd, QhJh, QhJs, QsJd, QsJh, QsJs, QcTc, QdTd, QhTh, Qc9c, Qd9d, Qs9s, JdJh, JdJs, JhJs, JdTd, JhTh, Jd9d, Js9s, TcTd, TcTh, TdTh, Tc9c, Td9d, 9c9d, 9c9s, 9d9s, 9c8c, 9d8d, 9s8s, 8c8d, 8c8h, 8c8s, 8d8h, 8d8s, 8h8s, 8d7d, 8h7h, 8s7s, 7d7h, 7d7s, 7h7s, 7d6d, 7h6h, 7s6s, 6c6d, 6c6h, 6c6s, 6d6h, 6d6s, 6h6s, 6c5c, 6d5d, 5c5d, 4c4d, 4c4h, 4c4s, 4d4h, 4d4s, 4h4s, 3c3d, 3c3h, 3c3s, 3d3h, 3d3s, 3h3s, 2c2d, 2c2h, 2c2s, 2d2h, 2d2s, 2h2s`

**对比:**
- 手牌胜率差异: 0.790%
- 范围胜率差异: 0.910%
- **策略差异: 47.7%**

---

#### 反例 5

**场景1:**
- 公共牌: `Qc 7d Jh 8s 9h`
- OOP手牌: `AhQd`
- 手牌胜率: 58.286%
- 范围胜率: 胜率: 43.294%, 平局率: 3.621% (综合: 45.105%)
- 策略: `{"Check":0.25733,"Bet":0.74267,"Allin":0}`
- IP有效范围 (175个组合):
  `AcAd, AcAs, AdAs, AcKc, AcKd, AcKh, AcKs, AdKc, AdKd, AdKh, AdKs, AsKc, AsKd, AsKh, AsKs, AcQh, AcQs, AdQh, AdQs, AsQh, AsQs, AcJc, AcJd, AcJs, AdJc, AdJd, AdJs, AsJc, AsJd, AsJs, AcTc, AcTd, AcTh, AcTs, AdTc, AdTd, AdTh, AdTs, AsTc, AsTd, AsTh, AsTs, Ac9c, Ad9d, As9s, Ac8c, Ad8d, Ac7c, As7s, Ac6c, Ad6d, As6s, Ac5c, Ad5d, As5s, Ac4c, Ad4d, As4s, Ac3c, Ad3d, As3s, Ac2c, Ad2d, As2s, KcKd, KcKh, KcKs, KdKh, KdKs, KhKs, KcQh, KcQs, KdQh, KdQs, KhQh, KhQs, KsQh, KsQs, KcJc, KcJd, KcJs, KdJc, KdJd, KdJs, KhJc, KhJd, KhJs, KsJc, KsJd, KsJs, KcTc, KdTd, KhTh, KsTs, Kc9c, Kd9d, Ks9s, QhQs, QhJc, QhJd, QhJs, QsJc, QsJd, QsJs, QhTh, QsTs, Qs9s, JcJd, JcJs, JdJs, JcTc, JdTd, JsTs, Jc9c, Jd9d, Js9s, TcTd, TcTh, TcTs, TdTh, TdTs, ThTs, Tc9c, Td9d, Ts9s, 9c9d, 9c9s, 9d9s, 9c8c, 9d8d, 8c8d, 8c8h, 8d8h, 8c7c, 8h7h, 7c7h, 7c7s, 7h7s, 7c6c, 7h6h, 7s6s, 6c6d, 6c6h, 6c6s, 6d6h, 6d6s, 6h6s, 6c5c, 6d5d, 6h5h, 6s5s, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4d, 4c4h, 4c4s, 4d4h, 4d4s, 4h4s, 3c3d, 3c3h, 3c3s, 3d3h, 3d3s, 3h3s, 2c2d, 2c2h, 2c2s, 2d2h, 2d2s, 2h2s`

**场景2:**
- 公共牌: `6d 3c Ks 2h Ac`
- OOP手牌: `KdQs`
- 手牌胜率: 57.459%
- 范围胜率: 胜率: 43.615%, 平局率: 1.130% (综合: 44.180%)
- 策略: `{"Check":1,"Bet":0,"Allin":0}`
- IP有效范围 (181个组合):
  `AdAh, AdAs, AhAs, AdKc, AdKh, AhKc, AhKh, AsKc, AsKh, AdQc, AdQd, AdQh, AhQc, AhQd, AhQh, AsQc, AsQd, AsQh, AdJc, AdJd, AdJh, AdJs, AhJc, AhJd, AhJh, AhJs, AsJc, AsJd, AsJh, AsJs, AdTc, AdTd, AdTh, AdTs, AhTc, AhTd, AhTh, AhTs, AsTc, AsTd, AsTh, AsTs, Ad9d, Ah9h, As9s, Ad8d, Ah8h, As8s, Ad7d, Ah7h, As7s, Ah6h, As6s, Ad5d, Ah5h, As5s, Ad4d, Ah4h, As4s, Ad3d, Ah3h, As3s, Ad2d, As2s, KcKh, KcQc, KcQd, KcQh, KhQc, KhQd, KhQh, KcJc, KcJd, KcJh, KcJs, KhJc, KhJd, KhJh, KhJs, KcTc, KhTh, Kc9c, Kh9h, QcQd, QcQh, QdQh, QcJc, QcJd, QcJh, QcJs, QdJc, QdJd, QdJh, QdJs, QhJc, QhJd, QhJh, QhJs, QcTc, QdTd, QhTh, Qc9c, Qd9d, Qh9h, JcJd, JcJh, JcJs, JdJh, JdJs, JhJs, JcTc, JdTd, JhTh, JsTs, Jc9c, Jd9d, Jh9h, Js9s, TcTd, TcTh, TcTs, TdTh, TdTs, ThTs, Tc9c, Td9d, Th9h, Ts9s, 9c9d, 9c9h, 9c9s, 9d9h, 9d9s, 9h9s, 9c8c, 9d8d, 9h8h, 9s8s, 8c8d, 8c8h, 8c8s, 8d8h, 8d8s, 8h8s, 8c7c, 8d7d, 8h7h, 8s7s, 7c7d, 7c7h, 7c7s, 7d7h, 7d7s, 7h7s, 7c6c, 7h6h, 7s6s, 6c6h, 6c6s, 6h6s, 6c5c, 6h5h, 6s5s, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4d, 4c4h, 4c4s, 4d4h, 4d4s, 4h4s, 3d3h, 3d3s, 3h3s, 2c2d, 2c2s, 2d2s`

**对比:**
- 手牌胜率差异: 0.827%
- 范围胜率差异: 0.925%
- **策略差异: 49.5%**

---

#### 反例 6

**场景1:**
- 公共牌: `6c 3d 8h Td Qs`
- OOP手牌: `KcQh`
- 手牌胜率: 80.105%
- 范围胜率: 胜率: 46.223%, 平局率: 1.687% (综合: 47.067%)
- 策略: `{"Check":0.188519,"Bet":0.378592,"Allin":0.432889}`
- IP有效范围 (191个组合):
  `AcAd, AcAh, AcAs, AdAh, AdAs, AhAs, AcKd, AcKh, AcKs, AdKd, AdKh, AdKs, AhKd, AhKh, AhKs, AsKd, AsKh, AsKs, AcQc, AcQd, AdQc, AdQd, AhQc, AhQd, AsQc, AsQd, AcJc, AcJd, AcJh, AcJs, AdJc, AdJd, AdJh, AdJs, AhJc, AhJd, AhJh, AhJs, AsJc, AsJd, AsJh, AsJs, AcTc, AcTh, AcTs, AdTc, AdTh, AdTs, AhTc, AhTh, AhTs, AsTc, AsTh, AsTs, Ac9c, Ad9d, Ah9h, As9s, Ac8c, Ad8d, As8s, Ac7c, Ad7d, Ah7h, As7s, Ad6d, Ah6h, As6s, Ac5c, Ad5d, Ah5h, As5s, Ac4c, Ad4d, Ah4h, As4s, Ac3c, Ah3h, As3s, Ac2c, Ad2d, Ah2h, As2s, KdKh, KdKs, KhKs, KdQc, KdQd, KhQc, KhQd, KsQc, KsQd, KdJc, KdJd, KdJh, KdJs, KhJc, KhJd, KhJh, KhJs, KsJc, KsJd, KsJh, KsJs, KhTh, KsTs, Kd9d, Kh9h, Ks9s, QcQd, QcJc, QcJd, QcJh, QcJs, QdJc, QdJd, QdJh, QdJs, QcTc, Qc9c, Qd9d, JcJd, JcJh, JcJs, JdJh, JdJs, JhJs, JcTc, JhTh, JsTs, Jc9c, Jd9d, Jh9h, Js9s, TcTh, TcTs, ThTs, Tc9c, Th9h, Ts9s, 9c9d, 9c9h, 9c9s, 9d9h, 9d9s, 9h9s, 9c8c, 9d8d, 9s8s, 8c8d, 8c8s, 8d8s, 8c7c, 8d7d, 8s7s, 7c7d, 7c7h, 7c7s, 7d7h, 7d7s, 7h7s, 7d6d, 7h6h, 7s6s, 6d6h, 6d6s, 6h6s, 6d5d, 6h5h, 6s5s, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4d, 4c4h, 4c4s, 4d4h, 4d4s, 4h4s, 3c3h, 3c3s, 3h3s, 2c2d, 2c2h, 2c2s, 2d2h, 2d2s, 2h2s`

**场景2:**
- 公共牌: `Qd Js 8s 2c Ts`
- OOP手牌: `Tc9c`
- 手牌胜率: 81.026%
- 范围胜率: 胜率: 45.619%, 平局率: 2.422% (综合: 46.829%)
- 策略: `{"Check":0,"Bet":0.142312,"Allin":0.857688}`
- IP有效范围 (195个组合):
  `AcAd, AcAh, AcAs, AdAh, AdAs, AhAs, AcKc, AcKd, AcKh, AcKs, AdKc, AdKd, AdKh, AdKs, AhKc, AhKd, AhKh, AhKs, AsKc, AsKd, AsKh, AsKs, AcQc, AcQh, AcQs, AdQc, AdQh, AdQs, AhQc, AhQh, AhQs, AsQc, AsQh, AsQs, AcJc, AcJd, AcJh, AdJc, AdJd, AdJh, AhJc, AhJd, AhJh, AsJc, AsJd, AsJh, AcTd, AcTh, AdTd, AdTh, AhTd, AhTh, AsTd, AsTh, Ad9d, Ah9h, As9s, Ac8c, Ad8d, Ah8h, Ac7c, Ad7d, Ah7h, As7s, Ac6c, Ad6d, Ah6h, As6s, Ac5c, Ad5d, Ah5h, As5s, Ac4c, Ad4d, Ah4h, As4s, Ac3c, Ad3d, Ah3h, As3s, Ad2d, Ah2h, As2s, KcKd, KcKh, KcKs, KdKh, KdKs, KhKs, KcQc, KcQh, KcQs, KdQc, KdQh, KdQs, KhQc, KhQh, KhQs, KsQc, KsQh, KsQs, KcJc, KcJd, KcJh, KdJc, KdJd, KdJh, KhJc, KhJd, KhJh, KsJc, KsJd, KsJh, KdTd, KhTh, Kd9d, Kh9h, Ks9s, QcQh, QcQs, QhQs, QcJc, QcJd, QcJh, QhJc, QhJd, QhJh, QsJc, QsJd, QsJh, QhTh, Qh9h, Qs9s, JcJd, JcJh, JdJh, JdTd, JhTh, Jd9d, Jh9h, TdTh, Td9d, Th9h, 9d9h, 9d9s, 9h9s, 9d8d, 9h8h, 8c8d, 8c8h, 8d8h, 8c7c, 8d7d, 8h7h, 7c7d, 7c7h, 7c7s, 7d7h, 7d7s, 7h7s, 7c6c, 7d6d, 7h6h, 7s6s, 6c6d, 6c6h, 6c6s, 6d6h, 6d6s, 6h6s, 6c5c, 6d5d, 6h5h, 6s5s, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4d, 4c4h, 4c4s, 4d4h, 4d4s, 4h4s, 3c3d, 3c3h, 3c3s, 3d3h, 3d3s, 3h3s, 2d2h, 2d2s, 2h2s`

**对比:**
- 手牌胜率差异: 0.921%
- 范围胜率差异: 0.238%
- **策略差异: 28.3%**

---

#### 反例 7

**场景1:**
- 公共牌: `6c 7c 9c 8h Ad`
- OOP手牌: `KsJs`
- 手牌胜率: 8.242%
- 范围胜率: 胜率: 49.616%, 平局率: 2.661% (综合: 50.946%)
- 策略: `{"Check":0,"Bet":0,"Allin":1}`
- IP有效范围 (182个组合):
  `AcAh, AcAs, AhAs, AcKc, AcKd, AcKh, AhKc, AhKd, AhKh, AsKc, AsKd, AsKh, AcQc, AcQd, AcQh, AcQs, AhQc, AhQd, AhQh, AhQs, AsQc, AsQd, AsQh, AsQs, AcJc, AcJd, AcJh, AhJc, AhJd, AhJh, AsJc, AsJd, AsJh, AcTc, AcTd, AcTh, AcTs, AhTc, AhTd, AhTh, AhTs, AsTc, AsTd, AsTh, AsTs, Ah9h, As9s, Ac8c, As8s, Ah7h, As7s, Ah6h, As6s, Ac5c, Ah5h, As5s, Ac4c, Ah4h, As4s, Ac3c, Ah3h, As3s, Ac2c, Ah2h, As2s, KcKd, KcKh, KdKh, KcQc, KcQd, KcQh, KcQs, KdQc, KdQd, KdQh, KdQs, KhQc, KhQd, KhQh, KhQs, KcJc, KcJd, KcJh, KdJc, KdJd, KdJh, KhJc, KhJd, KhJh, KcTc, KdTd, KhTh, Kd9d, Kh9h, QcQd, QcQh, QcQs, QdQh, QdQs, QhQs, QcJc, QcJd, QcJh, QdJc, QdJd, QdJh, QhJc, QhJd, QhJh, QsJc, QsJd, QsJh, QcTc, QdTd, QhTh, QsTs, Qd9d, Qh9h, Qs9s, JcJd, JcJh, JdJh, JcTc, JdTd, JhTh, Jd9d, Jh9h, TcTd, TcTh, TcTs, TdTh, TdTs, ThTs, Td9d, Th9h, Ts9s, 9d9h, 9d9s, 9h9s, 9d8d, 9s8s, 8c8d, 8c8s, 8d8s, 8d7d, 8s7s, 7d7h, 7d7s, 7h7s, 7d6d, 7h6h, 7s6s, 6d6h, 6d6s, 6h6s, 6d5d, 6h5h, 6s5s, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4d, 4c4h, 4c4s, 4d4h, 4d4s, 4h4s, 3c3d, 3c3h, 3c3s, 3d3h, 3d3s, 3h3s, 2c2d, 2c2h, 2c2s, 2d2h, 2d2s, 2h2s`

**场景2:**
- 公共牌: `6s Th 2d 7h 9d`
- OOP手牌: `KcJd`
- 手牌胜率: 8.333%
- 范围胜率: 胜率: 49.095%, 平局率: 2.160% (综合: 50.174%)
- 策略: `{"Check":0,"Bet":0.440838,"Allin":0.559162}`
- IP有效范围 (198个组合):
  `AcAd, AcAh, AcAs, AdAh, AdAs, AhAs, AcKd, AcKh, AcKs, AdKd, AdKh, AdKs, AhKd, AhKh, AhKs, AsKd, AsKh, AsKs, AcQc, AcQd, AcQh, AcQs, AdQc, AdQd, AdQh, AdQs, AhQc, AhQd, AhQh, AhQs, AsQc, AsQd, AsQh, AsQs, AcJc, AcJh, AcJs, AdJc, AdJh, AdJs, AhJc, AhJh, AhJs, AsJc, AsJh, AsJs, AcTc, AcTd, AcTs, AdTc, AdTd, AdTs, AhTc, AhTd, AhTs, AsTc, AsTd, AsTs, Ac9c, Ah9h, As9s, Ac8c, Ad8d, Ah8h, As8s, Ac7c, Ad7d, As7s, Ac6c, Ad6d, Ah6h, Ac5c, Ad5d, Ah5h, As5s, Ac4c, Ad4d, Ah4h, As4s, Ac3c, Ad3d, Ah3h, As3s, Ac2c, Ah2h, As2s, KdKh, KdKs, KhKs, KdQc, KdQd, KdQh, KdQs, KhQc, KhQd, KhQh, KhQs, KsQc, KsQd, KsQh, KsQs, KdJc, KdJh, KdJs, KhJc, KhJh, KhJs, KsJc, KsJh, KsJs, KdTd, KsTs, Kh9h, Ks9s, QcQd, QcQh, QcQs, QdQh, QdQs, QhQs, QcJc, QcJh, QcJs, QdJc, QdJh, QdJs, QhJc, QhJh, QhJs, QsJc, QsJh, QsJs, QcTc, QdTd, QsTs, Qc9c, Qh9h, Qs9s, JcJh, JcJs, JhJs, JcTc, JsTs, Jc9c, Jh9h, Js9s, TcTd, TcTs, TdTs, Tc9c, Ts9s, 9c9h, 9c9s, 9h9s, 9c8c, 9h8h, 9s8s, 8c8d, 8c8h, 8c8s, 8d8h, 8d8s, 8h8s, 8c7c, 8d7d, 8s7s, 7c7d, 7c7s, 7d7s, 7c6c, 7d6d, 6c6d, 6c6h, 6d6h, 6c5c, 6d5d, 6h5h, 5c5d, 5c5h, 5c5s, 5d5h, 5d5s, 5h5s, 4c4d, 4c4h, 4c4s, 4d4h, 4d4s, 4h4s, 3c3d, 3c3h, 3c3s, 3d3h, 3d3s, 3h3s, 2c2h, 2c2s, 2h2s`

**对比:**
- 手牌胜率差异: 0.092%
- 范围胜率差异: 0.772%
- **策略差异: 29.4%**


## 结论

### ⚠️ 双维度胜率标量不足以决定最优策略

使用独立的胜率计算方法验证后，实验发现：在 17 对双维度胜率相近的场景中，有 7 对（41.2%）的策略差异显著。

**结论：即使手牌胜率和范围胜率都精确匹配，最优策略仍然可能完全不同。**

### 分析

从反例中可以看出，即使两个场景的手牌胜率和范围胜率都非常接近，但由于：
1. IP的有效范围组合不同（死牌不同导致）
2. 公共牌结构不同（顺子/同花可能性不同）
3. 手牌与公共牌的互动不同

这些因素导致最优策略可能完全不同。这证明了**仅靠两个胜率标量无法替代完整的博弈论求解**。
