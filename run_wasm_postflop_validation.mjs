#!/usr/bin/env node
/**
 * 使用wasm-postflop进行跨公共牌双维度胜率-策略验证实验
 * 
 * 核心问题：在不同的（公共牌+固定手牌）组合下：
 * 当以下两个条件同时满足时，策略是否相同？
 * 1. 固定手牌vs对手范围的胜率相近（差异<5%）
 * 2. 自己范围vs对手范围的胜率相近（差异<5%）
 * 
 * 关键约束：每个公共牌场景只能选择一个固定手牌，因为一旦固定了手牌，
 * 就不能再从自己范围内采样其他手牌了。
 */

import { solveRiver, parseRange, parseBoard, cardIndexToString } from './solver_tools/postflop_solver.mjs';
import { writeFileSync } from 'fs';

// 范围定义
const OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o';
const IP_RANGE = 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo';

/**
 * 将牌索引转换为字符串
 */
function cardToString(cardIdx) {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    const rank = Math.floor(cardIdx / 4);
    const suit = cardIdx % 4;
    return ranks[rank] + suits[suit];
}

/**
 * 从范围权重数组中排除指定的死牌
 * @param {Float32Array} weights - 1326长度的范围权重数组
 * @param {number[]} deadCards - 死牌索引数组
 * @returns {Float32Array} - 排除死牌后的权重数组
 */
function removeDeadCardsFromWeights(weights, deadCards) {
    const deadSet = new Set(deadCards);
    const newWeights = new Float32Array(weights.length);
    
    let idx = 0;
    for (let c1 = 0; c1 < 52; c1++) {
        for (let c2 = c1 + 1; c2 < 52; c2++) {
            // 如果任一牌是死牌，权重设为0
            if (deadSet.has(c1) || deadSet.has(c2)) {
                newWeights[idx] = 0;
            } else {
                newWeights[idx] = weights[idx];
            }
            idx++;
        }
    }
    
    return newWeights;
}

// 生成随机公共牌
function generateRandomBoard() {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    const cards = [];
    const usedCards = new Set();
    
    while (cards.length < 5) {
        const rank = Math.floor(Math.random() * 13);
        const suit = Math.floor(Math.random() * 4);
        const cardIdx = rank * 4 + suit;
        
        if (!usedCards.has(cardIdx)) {
            usedCards.add(cardIdx);
            cards.push(ranks[rank] + suits[suit]);
        }
    }
    
    return cards.join(' ');
}

// 为给定公共牌随机选择一个不冲突的手牌
function selectRandomHeroHand(board) {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    const boardCards = parseBoard(board);
    const boardSet = new Set(Array.from(boardCards));
    
    // 生成所有可能的手牌组合（在OOP范围内且不与公共牌冲突）
    const validHands = [];
    
    // 口袋对
    for (let r = 0; r < 13; r++) {
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = s1 + 1; s2 < 4; s2++) {
                const c1 = r * 4 + s1;
                const c2 = r * 4 + s2;
                if (!boardSet.has(c1) && !boardSet.has(c2)) {
                    validHands.push({
                        hand: ranks[r] + suits[s1] + ranks[r] + suits[s2],
                        cards: [c1, c2]
                    });
                }
            }
        }
    }
    
    // 同花连接和高牌
    const suitedCombos = [
        [12, 11], [11, 10], [10, 9], [9, 8], [8, 7], [7, 6], // AKs, KQs, QJs, JTs, T9s, 98s
        [12, 10], [12, 9], [11, 9], // AQs, AJs, KJs
    ];
    
    for (const [r1, r2] of suitedCombos) {
        for (let s = 0; s < 4; s++) {
            const c1 = r1 * 4 + s;
            const c2 = r2 * 4 + s;
            if (!boardSet.has(c1) && !boardSet.has(c2)) {
                validHands.push({
                    hand: ranks[r1] + suits[s] + ranks[r2] + suits[s],
                    cards: [c1, c2]
                });
            }
        }
    }
    
    // 非同花高牌
    const offsuitCombos = [
        [12, 11], [12, 10], [12, 9], [11, 10], [11, 9], // AKo, AQo, AJo, KQo, KJo
    ];
    
    for (const [r1, r2] of offsuitCombos) {
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = 0; s2 < 4; s2++) {
                if (s1 !== s2) {
                    const c1 = r1 * 4 + s1;
                    const c2 = r2 * 4 + s2;
                    if (!boardSet.has(c1) && !boardSet.has(c2)) {
                        validHands.push({
                            hand: ranks[r1] + suits[s1] + ranks[r2] + suits[s2],
                            cards: [c1, c2]
                        });
                    }
                }
            }
        }
    }
    
    if (validHands.length === 0) return null;
    
    // 随机选择一个
    const idx = Math.floor(Math.random() * validHands.length);
    return validHands[idx];
}

/**
 * 解析手牌字符串为两张牌的索引
 */
function parseHeroHand(handStr) {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    
    const card1 = handStr.substring(0, 2);
    const card2 = handStr.substring(2, 4);
    
    const r1 = ranks.indexOf(card1[0].toUpperCase());
    const s1 = suits.indexOf(card1[1].toLowerCase());
    const r2 = ranks.indexOf(card2[0].toUpperCase());
    const s2 = suits.indexOf(card2[1].toLowerCase());
    
    return [r1 * 4 + s1, r2 * 4 + s2];
}

/**
 * 在oopCards中找到手牌索引
 */
function findHandIndex(oopCards, heroCards) {
    const hero = heroCards.slice().sort((a, b) => a - b);
    
    for (let i = 0; i < oopCards.length; i++) {
        const handIdx = oopCards[i];
        const c1 = handIdx & 0xFF;
        const c2 = (handIdx >> 8) & 0xFF;
        const cards = [c1, c2].sort((a, b) => a - b);
        
        if (cards[0] === hero[0] && cards[1] === hero[1]) {
            return i;
        }
    }
    
    return -1;
}

/**
 * 从solver结果中提取特定手牌的策略和胜率
 */
function extractHandData(solverResult, heroCards) {
    const { oopCards, ipCards, results, numActions, actions } = solverResult;
    
    const heroIndex = findHandIndex(oopCards, heroCards);
    if (heroIndex === -1) {
        return null;
    }
    
    const actionList = actions.split('/').map(a => {
        const match = a.match(/(\w+):(\d+)/);
        if (match) {
            return { name: match[1], amount: parseInt(match[2]) };
        }
        return { name: a, amount: 0 };
    });
    
    const oopLen = oopCards.length;
    const ipLen = ipCards.length;
    
    // 解析results数组
    let offset = 3; // header
    offset += oopLen + ipLen; // weights
    offset += oopLen + ipLen; // normalizer
    
    // equity (胜率 + 平分率/2)
    const equityOop = results.slice(offset, offset + oopLen);
    offset += oopLen;
    offset += ipLen;
    
    // ev
    const evOop = results.slice(offset, offset + oopLen);
    offset += oopLen;
    offset += ipLen;
    
    // eqr
    offset += oopLen + ipLen;
    
    // strategy
    const strategyData = results.slice(offset, offset + numActions * oopLen);
    
    const heroEquity = equityOop[heroIndex];
    const heroEv = evOop[heroIndex];
    
    // 计算范围平均胜率
    let totalEquity = 0;
    let count = 0;
    for (let i = 0; i < oopLen; i++) {
        const eq = equityOop[i];
        if (!isNaN(eq) && eq >= 0 && eq <= 1) {
            totalEquity += eq;
            count++;
        }
    }
    const rangeEquity = count > 0 ? totalEquity / count : 0.5;
    
    // 提取策略
    const strategy = {};
    for (let i = 0; i < numActions; i++) {
        strategy[actionList[i].name] = strategyData[i * oopLen + heroIndex];
    }
    
    return {
        heroEquity,
        rangeEquity,
        heroEv,
        strategy,
        actions: actionList,
    };
}

/**
 * 运行单个场景的求解
 * 
 * 关于死牌处理的说明：
 * - wasm-postflop solver在求解时会自动排除公共牌冲突
 * - solver返回的equity数组中，每个OOP手牌的胜率是基于IP的有效范围计算的
 *   （即排除了与该OOP手牌冲突的IP组合）
 * - 这是因为solver在计算每个OOP手牌的胜率时，会考虑到该手牌的存在使得
 *   某些IP组合不可能出现
 */
async function solveScenario(board, oopRange, ipRange, heroCards) {
    try {
        const result = await solveRiver({
            oopRange,
            ipRange,
            board,
            startingPot: 100,
            effectiveStack: 100,
            oopBetSizes: '50',
            ipBetSizes: '50',
            targetExploitability: 0.3,
            maxIterations: 500,
        });
        
        result.heroCards = heroCards;
        
        return result;
    } catch (e) {
        console.error(`Error solving ${board}:`, e.message);
        return null;
    }
}

/**
 * 计算策略差异
 */
function calculateStrategyDiff(strategy1, strategy2) {
    const keys = new Set([...Object.keys(strategy1), ...Object.keys(strategy2)]);
    let diff = 0;
    for (const key of keys) {
        diff += Math.abs((strategy1[key] || 0) - (strategy2[key] || 0));
    }
    return diff / keys.size;
}

/**
 * 主实验函数
 */
async function runExperiment() {
    console.log('='.repeat(80));
    console.log('使用wasm-postflop进行跨公共牌双维度胜率-策略验证实验');
    console.log('='.repeat(80));
    console.log('\n核心问题: 在不同的（公共牌+固定手牌）组合下：');
    console.log('  当手牌胜率和范围胜率都相近时，策略是否相同？');
    console.log('\n关键约束: 每个公共牌场景只选择一个固定手牌');
    console.log('\n范围定义:');
    console.log(`  OOP: ${OOP_RANGE}`);
    console.log(`  IP: ${IP_RANGE}`);
    
    // 生成大量场景
    const NUM_SCENARIOS = 10000; // 生成10000个场景
    const scenarios = [];
    
    console.log(`\n生成 ${NUM_SCENARIOS} 个（公共牌+固定手牌）场景...`);
    
    const startTime = Date.now();
    let lastProgressTime = startTime;
    
    for (let i = 0; i < NUM_SCENARIOS; i++) {
        const board = generateRandomBoard();
        const heroHandInfo = selectRandomHeroHand(board);
        
        if (!heroHandInfo) {
            continue;
        }
        
        // 每100个场景输出一次进度
        if ((i + 1) % 100 === 0) {
            const elapsed = (Date.now() - startTime) / 1000;
            const rate = (i + 1) / elapsed;
            const remaining = (NUM_SCENARIOS - i - 1) / rate;
            console.log(`进度: ${i+1}/${NUM_SCENARIOS} (${((i+1)/NUM_SCENARIOS*100).toFixed(1)}%) - 已用时: ${elapsed.toFixed(0)}s - 预计剩余: ${remaining.toFixed(0)}s`);
        }
        
        const result = await solveScenario(board, OOP_RANGE, IP_RANGE, heroHandInfo.cards);
        if (!result) continue;
        
        const handData = extractHandData(result, heroHandInfo.cards);
        if (!handData) {
            continue;
        }
        
        scenarios.push({
            board,
            heroHand: heroHandInfo.hand,
            heroCards: heroHandInfo.cards,
            heroEquity: handData.heroEquity,
            rangeEquity: handData.rangeEquity,
            strategy: handData.strategy,
            actions: handData.actions,
        });
    }
    
    const totalTime = (Date.now() - startTime) / 1000;
    console.log(`\n\n成功生成 ${scenarios.length} 个场景，总用时: ${totalTime.toFixed(1)}s`);
    
    // 分析结果：找出双维度胜率相近的场景对
    console.log('\n' + '='.repeat(80));
    console.log('分析结果：寻找双维度胜率相近的场景对（阈值: 0.1%）');
    console.log('='.repeat(80));
    
    const threshold = 0.001; // 0.1%差异阈值（千分之一）
    const pairs = [];
    
    for (let i = 0; i < scenarios.length; i++) {
        for (let j = i + 1; j < scenarios.length; j++) {
            const s1 = scenarios[i];
            const s2 = scenarios[j];
            
            const heroEqDiff = Math.abs(s1.heroEquity - s2.heroEquity);
            const rangeEqDiff = Math.abs(s1.rangeEquity - s2.rangeEquity);
            
            if (heroEqDiff < threshold && rangeEqDiff < threshold) {
                const strategyDiff = calculateStrategyDiff(s1.strategy, s2.strategy);
                
                pairs.push({
                    scenario1: s1,
                    scenario2: s2,
                    heroEqDiff,
                    rangeEqDiff,
                    strategyDiff,
                });
            }
        }
    }
    
    console.log(`\n找到 ${pairs.length} 对双维度胜率相近的场景`);
    
    // 统计策略差异
    const significantPairs = pairs.filter(p => p.strategyDiff > 0.15);
    console.log(`其中策略差异显著(>15%)的: ${significantPairs.length} 对`);
    
    // 输出一些反例
    if (significantPairs.length > 0) {
        console.log('\n【策略差异显著的反例】');
        for (const p of significantPairs.slice(0, 10)) {
            console.log(`\n  场景1: ${p.scenario1.board} + ${p.scenario1.heroHand}`);
            console.log(`  场景2: ${p.scenario2.board} + ${p.scenario2.heroHand}`);
            console.log(`  手牌胜率: ${(p.scenario1.heroEquity*100).toFixed(3)}% vs ${(p.scenario2.heroEquity*100).toFixed(3)}%`);
            console.log(`  范围胜率: ${(p.scenario1.rangeEquity*100).toFixed(3)}% vs ${(p.scenario2.rangeEquity*100).toFixed(3)}%`);
            console.log(`  策略1: ${JSON.stringify(p.scenario1.strategy)}`);
            console.log(`  策略2: ${JSON.stringify(p.scenario2.strategy)}`);
            console.log(`  策略差异: ${(p.strategyDiff*100).toFixed(1)}%`);
        }
    }
    
    // 保存结果
    const outputData = {
        oopRange: OOP_RANGE,
        ipRange: IP_RANGE,
        numScenarios: scenarios.length,
        scenarios,
        pairs: pairs.map(p => ({
            board1: p.scenario1.board,
            heroHand1: p.scenario1.heroHand,
            board2: p.scenario2.board,
            heroHand2: p.scenario2.heroHand,
            heroEq1: p.scenario1.heroEquity,
            heroEq2: p.scenario2.heroEquity,
            rangeEq1: p.scenario1.rangeEquity,
            rangeEq2: p.scenario2.rangeEquity,
            strategy1: p.scenario1.strategy,
            strategy2: p.scenario2.strategy,
            strategyDiff: p.strategyDiff,
        })),
    };
    
    const outputPath = 'experiments/results/wasm_postflop_validation.json';
    writeFileSync(outputPath, JSON.stringify(outputData, null, 2));
    console.log(`\n结果已保存到: ${outputPath}`);
    
    // 生成报告
    generateReport(scenarios, pairs, significantPairs);
}

/**
 * 生成实验报告
 */
function generateReport(scenarios, pairs, significantPairs) {
    let report = `# 跨公共牌双维度胜率-策略验证实验报告

## 实验目的

验证：**在不同的（公共牌+固定手牌）组合下：**
当以下两个条件同时满足时，策略是否相同？
1. 固定手牌vs对手范围的胜率相近（差异<0.1%）
2. 自己范围vs对手范围的胜率相近（差异<0.1%）

## 实验方法

1. 随机生成公共牌场景
2. 为每个公共牌场景随机选择一个固定手牌（不与公共牌冲突）
3. 对每个（公共牌+固定手牌）组合：
   - 定义双方范围（IP范围和OOP范围）
   - 使用范围vs范围进行CFR求解
   - **Solver自动排除死牌**：计算固定手牌胜率时，自动排除IP范围中与固定手牌冲突的组合
   - 只提取固定手牌的策略（其他手牌策略不参与比较）
   - 计算固定手牌vs对手范围的胜率（胜率 + 平分率/2，已排除冲突组合）
   - 计算范围vs范围的胜率（胜率 + 平分率/2）
4. 在所有场景中，找出双维度胜率都相近的场景对（阈值: 0.1%）
5. 比较这些场景对中固定手牌的策略是否相同

**关键约束：每个公共牌场景只选择一个固定手牌，因为一旦固定了手牌，就不能再从自己范围内采样其他手牌了。**

**死牌处理：wasm-postflop solver在计算每个OOP手牌的胜率时，会自动排除IP范围中与该手牌冲突的组合。**

**胜率比对阈值：0.1%（千分之一）**

## 范围定义

- **OOP范围（位置劣势方）**: ${OOP_RANGE}
- **IP范围（位置优势方）**: ${IP_RANGE}

## 实验规模

- 生成场景数: ${scenarios.length}
- 双维度胜率相近的场景对（差异<0.1%）: ${pairs.length}
- 策略差异显著(>15%)的场景对: ${significantPairs.length}

## 关键发现

`;

    if (significantPairs.length > 0) {
        report += `### 发现双维度胜率相近但策略不同的反例

| 场景1 (公共牌+手牌) | 场景2 (公共牌+手牌) | 手牌胜率 | 范围胜率 | 策略差异 |
|---------------------|---------------------|----------|----------|----------|
`;
        
        for (const p of significantPairs.slice(0, 20)) {
            report += `| ${p.scenario1.board} + ${p.scenario1.heroHand} | ${p.scenario2.board} + ${p.scenario2.heroHand} | `;
            report += `${(p.scenario1.heroEquity*100).toFixed(3)}% vs ${(p.scenario2.heroEquity*100).toFixed(3)}% | `;
            report += `${(p.scenario1.rangeEquity*100).toFixed(3)}% vs ${(p.scenario2.rangeEquity*100).toFixed(3)}% | `;
            report += `${(p.strategyDiff*100).toFixed(1)}% |\n`;
        }
        
        report += `\n### 详细反例分析\n\n`;
        
        for (const p of significantPairs.slice(0, 5)) {
            report += `#### 反例: ${p.scenario1.heroHand} vs ${p.scenario2.heroHand}\n\n`;
            report += `**场景1:**\n`;
            report += `- 公共牌: ${p.scenario1.board}\n`;
            report += `- 固定手牌: ${p.scenario1.heroHand}\n`;
            report += `- 手牌胜率: ${(p.scenario1.heroEquity*100).toFixed(3)}%\n`;
            report += `- 范围胜率: ${(p.scenario1.rangeEquity*100).toFixed(3)}%\n`;
            report += `- 策略: ${JSON.stringify(p.scenario1.strategy)}\n\n`;
            
            report += `**场景2:**\n`;
            report += `- 公共牌: ${p.scenario2.board}\n`;
            report += `- 固定手牌: ${p.scenario2.heroHand}\n`;
            report += `- 手牌胜率: ${(p.scenario2.heroEquity*100).toFixed(3)}%\n`;
            report += `- 范围胜率: ${(p.scenario2.rangeEquity*100).toFixed(3)}%\n`;
            report += `- 策略: ${JSON.stringify(p.scenario2.strategy)}\n\n`;
            
            report += `**策略差异: ${(p.strategyDiff*100).toFixed(1)}%**\n\n`;
            report += `---\n\n`;
        }
    } else {
        report += `在测试的 ${scenarios.length} 个场景中，未发现双维度胜率相近（差异<0.1%）但策略显著不同的反例。\n\n`;
        report += `这表明：**当手牌胜率和范围胜率都精确匹配时（差异<0.1%），策略也趋于一致。**\n\n`;
    }
    
    report += `## 结论

`;
    
    if (significantPairs.length > 0) {
        const ratio = (significantPairs.length / pairs.length * 100).toFixed(1);
        report += `### ⚠️ 双维度胜率标量不足以决定最优策略

实验发现：在 ${pairs.length} 对双维度胜率相近（差异<0.1%）的场景中，有 ${significantPairs.length} 对（${ratio}%）的策略差异显著。

这证明了：**即使固定手牌胜率和范围胜率都精确匹配（差异<0.1%），最优策略仍然可能完全不同。仅靠两个胜率标量无法替代solver。**
`;
    } else {
        report += `### ✓ 在测试的场景中，双维度胜率精确匹配时策略也相近

在 ${scenarios.length} 个场景中，当双维度胜率差异小于0.1%时，未发现策略显著不同的反例。

这表明：**当手牌胜率和范围胜率都精确匹配时，策略可能趋于一致。但需要注意，在实际应用中很难找到胜率如此精确匹配的场景。**
`;
    }
    
    const reportPath = 'experiments/results/wasm_postflop_validation_report.md';
    writeFileSync(reportPath, report);
    console.log(`\n报告已保存到: ${reportPath}`);
}

// 运行实验
runExperiment().catch(console.error);
