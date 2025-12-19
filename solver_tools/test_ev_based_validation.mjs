#!/usr/bin/env node
/**
 * 基于EV的四维度胜率验证实验
 * 
 * 核心思路：
 * 当动作EV非常接近时，策略的"抖动"是正常的（纳什均衡的性质）。
 * 真正需要验证的是：当四维度胜率相同时，各动作的EV是否相同？
 * 
 * 如果EV相同，说明四维度胜率足以决定"最优EV"，只是具体策略可以有多种选择。
 * 如果EV不同，说明四维度胜率确实不足以决定最优策略。
 */

import { solveRiver } from './postflop_solver.mjs';
import { parseBoardString } from './equity_calculator.mjs';
import { writeFileSync, readFileSync, existsSync } from 'fs';
import { execSync } from 'child_process';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const OMPEVAL_PATH = join(__dirname, 'OMPEval', 'equity_calc');

// 所有可能的手牌类型
const ALL_HANDS = [];
const RANKS = '23456789TJQKA';

function initAllHands() {
    for (let r = 12; r >= 0; r--) {
        ALL_HANDS.push(RANKS[r] + RANKS[r]);
    }
    for (let r1 = 12; r1 >= 1; r1--) {
        for (let r2 = r1 - 1; r2 >= 0; r2--) {
            ALL_HANDS.push(RANKS[r1] + RANKS[r2] + 's');
        }
    }
    for (let r1 = 12; r1 >= 1; r1--) {
        for (let r2 = r1 - 1; r2 >= 0; r2--) {
            ALL_HANDS.push(RANKS[r1] + RANKS[r2] + 'o');
        }
    }
}
initAllHands();

function generateRandomRange(minHands = 50, maxHands = 150) {
    const numHands = minHands + Math.floor(Math.random() * (maxHands - minHands + 1));
    const selectedHands = new Set();
    const strongHandsCount = Math.floor(numHands * 0.4);
    const strongHandsPool = ALL_HANDS.slice(0, Math.floor(ALL_HANDS.length * 0.3));
    
    while (selectedHands.size < strongHandsCount && strongHandsPool.length > 0) {
        const idx = Math.floor(Math.random() * strongHandsPool.length);
        selectedHands.add(strongHandsPool[idx]);
    }
    while (selectedHands.size < numHands) {
        const idx = Math.floor(Math.random() * ALL_HANDS.length);
        selectedHands.add(ALL_HANDS[idx]);
    }
    return Array.from(selectedHands).join(',');
}

function expandRangeToOmpeval(rangeStr) {
    const ranks = '23456789TJQKA';
    const result = [];
    const parts = rangeStr.split(',');
    for (const part of parts) {
        const p = part.trim();
        if (!p) continue;
        if (p.includes('-')) {
            const [start, end] = p.split('-');
            if (start.length === 2 && start[0] === start[1]) {
                const startRank = ranks.indexOf(start[0]);
                const endRank = ranks.indexOf(end[0]);
                const minRank = Math.min(startRank, endRank);
                const maxRank = Math.max(startRank, endRank);
                for (let r = minRank; r <= maxRank; r++) {
                    result.push(ranks[r] + ranks[r]);
                }
            }
        } else {
            result.push(p);
        }
    }
    return result.join(',');
}

function calculateEquityOmpeval(range1, range2, board, dead = '') {
    const range1Expanded = expandRangeToOmpeval(range1);
    const range2Expanded = expandRangeToOmpeval(range2);
    const boardCompact = board.replace(/\s+/g, '');
    
    let cmd = `"${OMPEVAL_PATH}" "${range1Expanded}" "${range2Expanded}" "${boardCompact}"`;
    if (dead) cmd += ` "${dead}"`;
    
    try {
        const output = execSync(cmd, { encoding: 'utf8' });
        const cleanOutput = output.replace(/nan/g, 'null');
        const data = JSON.parse(cleanOutput);
        
        const wins = data.wins[0];
        const losses = data.wins[1];
        const ties = data.tieCount;
        const total = wins + losses + ties;
        
        if (total === 0) return { equity: 0.5, winRate: 0.5, tieRate: 0 };
        
        return {
            equity: (wins + ties * 0.5) / total,
            winRate: wins / total,
            tieRate: ties / total,
        };
    } catch (e) {
        throw new Error(`OMPEval error: ${e.message}`);
    }
}

function generateRandomBoard() {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    const cards = [];
    const usedCards = new Set();
    
    while (cards.length < 5) {
        const rank = Math.floor(Math.random() * 13);
        const suit = Math.floor(Math.random() * 4);
        const cardStr = ranks[rank] + suits[suit];
        if (!usedCards.has(cardStr)) {
            usedCards.add(cardStr);
            cards.push(cardStr);
        }
    }
    return cards.join(' ');
}

function expandHandType(handType) {
    const suits = 'cdhs';
    const result = [];
    
    if (handType.length === 2 && handType[0] === handType[1]) {
        const r = handType[0];
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = s1 + 1; s2 < 4; s2++) {
                result.push({ hand: r + suits[s1] + r + suits[s2], cards: [r + suits[s1], r + suits[s2]] });
            }
        }
    } else if (handType.endsWith('s')) {
        const r1 = handType[0], r2 = handType[1];
        for (let s = 0; s < 4; s++) {
            result.push({ hand: r1 + suits[s] + r2 + suits[s], cards: [r1 + suits[s], r2 + suits[s]] });
        }
    } else if (handType.endsWith('o')) {
        const r1 = handType[0], r2 = handType[1];
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = 0; s2 < 4; s2++) {
                if (s1 !== s2) {
                    result.push({ hand: r1 + suits[s1] + r2 + suits[s2], cards: [r1 + suits[s1], r2 + suits[s2]] });
                }
            }
        }
    }
    return result;
}

function selectRandomHeroHand(board, oopRange) {
    const boardCards = new Set(parseBoardString(board));
    const handTypes = oopRange.split(',').map(h => h.trim());
    const allHands = [];
    
    for (const handType of handTypes) {
        const expanded = expandHandType(handType);
        for (const hand of expanded) {
            if (!boardCards.has(hand.cards[0]) && !boardCards.has(hand.cards[1])) {
                allHands.push(hand);
            }
        }
    }
    
    if (allHands.length === 0) return null;
    return allHands[Math.floor(Math.random() * allHands.length)];
}

function findHandIndex(oopCards, heroCards) {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    
    const heroIndices = heroCards.map(card => {
        const rank = ranks.indexOf(card[0]);
        const suit = suits.indexOf(card[1]);
        return rank * 4 + suit;
    }).sort((a, b) => a - b);
    
    for (let i = 0; i < oopCards.length; i++) {
        const handIdx = oopCards[i];
        const c1 = handIdx & 0xFF;
        const c2 = (handIdx >> 8) & 0xFF;
        const cards = [c1, c2].sort((a, b) => a - b);
        if (cards[0] === heroIndices[0] && cards[1] === heroIndices[1]) return i;
    }
    return -1;
}

function extractStrategyAndEV(solverResult, heroCards) {
    const { oopCards, ipCards, results, numActions, actions } = solverResult;
    
    const heroIndex = findHandIndex(oopCards, heroCards);
    if (heroIndex === -1) return null;
    
    const actionList = actions.split('/').map(a => {
        const match = a.match(/(\w+):(\d+)/);
        return match ? { name: match[1], amount: parseInt(match[2]) } : { name: a, amount: 0 };
    });
    
    const oopLen = oopCards.length;
    const ipLen = ipCards.length;
    
    let offset = 3;
    offset += oopLen + ipLen; // weights
    offset += oopLen + ipLen; // normalizer
    
    const heroEquity = results[offset + heroIndex];
    offset += oopLen + ipLen; // equity
    
    const heroEV = results[offset + heroIndex];
    offset += oopLen + ipLen; // ev
    
    offset += oopLen + ipLen; // eqr
    
    // 策略
    const strategy = {};
    for (let i = 0; i < numActions; i++) {
        strategy[actionList[i].name] = results[offset + i * oopLen + heroIndex];
    }
    offset += numActions * oopLen;
    
    // 动作EV
    const actionEV = {};
    for (let i = 0; i < numActions; i++) {
        actionEV[actionList[i].name] = results[offset + i * oopLen + heroIndex];
    }
    
    return { strategy, actionEV, ev: heroEV, equity: heroEquity };
}

async function solveScenario(board, oopRange, ipRange) {
    try {
        return await solveRiver({
            oopRange, ipRange, board,
            startingPot: 100, effectiveStack: 500,
            oopBetSizes: '33,50,75,100', ipBetSizes: '33,50,75,100',
            oopRaiseSizes: '50,100', ipRaiseSizes: '50,100',
            targetExploitability: 0.05,  // 更严格的收敛
            maxIterations: 2000,  // 更多迭代
        });
    } catch (e) {
        return null;
    }
}

async function runExperiment() {
    console.log('='.repeat(80));
    console.log('基于EV的四维度胜率验证实验');
    console.log('='.repeat(80));
    console.log('\n核心问题：当四维度胜率相同时，各动作的EV是否相同？');
    console.log('如果EV相同，策略抖动是正常的（纳什均衡性质）');
    console.log('如果EV不同，说明四维度胜率不足以决定最优策略\n');
    
    const NUM_SCENARIOS = 5000;
    const RANGE_REFRESH_INTERVAL = 100;
    const scenarios = [];
    
    console.log(`生成 ${NUM_SCENARIOS} 个场景...\n`);
    
    const startTime = Date.now();
    let currentOopRange = generateRandomRange(60, 120);
    let currentIpRange = generateRandomRange(40, 100);
    
    for (let i = 0; i < NUM_SCENARIOS; i++) {
        if (i > 0 && i % RANGE_REFRESH_INTERVAL === 0) {
            currentOopRange = generateRandomRange(60, 120);
            currentIpRange = generateRandomRange(40, 100);
        }
        
        const board = generateRandomBoard();
        const heroHandInfo = selectRandomHeroHand(board, currentOopRange);
        if (!heroHandInfo) continue;
        
        if ((i + 1) % 100 === 0) {
            const elapsed = (Date.now() - startTime) / 1000;
            console.log(`进度: ${i+1}/${NUM_SCENARIOS} - 成功: ${scenarios.length} - 已用时: ${elapsed.toFixed(0)}s`);
        }
        
        const deadCards = heroHandInfo.cards.join('');
        
        try {
            const heroEquity = calculateEquityOmpeval(heroHandInfo.hand, currentIpRange, board, '');
            const rangeEquity = calculateEquityOmpeval(currentOopRange, currentIpRange, board, deadCards);
            
            const solverResult = await solveScenario(board, currentOopRange, currentIpRange);
            if (!solverResult) continue;
            
            const data = extractStrategyAndEV(solverResult, heroHandInfo.cards);
            if (!data) continue;
            
            scenarios.push({
                board, heroHand: heroHandInfo.hand,
                heroEquity, rangeEquity,
                strategy: data.strategy, actionEV: data.actionEV,
                ev: data.ev, exploitability: solverResult.exploitability,
            });
        } catch (e) {
            // 忽略错误
        }
    }
    
    console.log(`\n成功生成 ${scenarios.length} 个场景`);
    
    analyzeResults(scenarios);
}

function analyzeResults(scenarios) {
    console.log('\n' + '='.repeat(80));
    console.log('分析结果：基于EV的验证');
    console.log('='.repeat(80));
    
    const threshold = 0.0005; // 0.05% 胜率阈值
    const evThreshold = 0.5;  // EV差异阈值（0.5 chips）
    
    const pairs = [];
    
    for (let i = 0; i < scenarios.length; i++) {
        for (let j = i + 1; j < scenarios.length; j++) {
            const s1 = scenarios[i];
            const s2 = scenarios[j];
            
            const heroWinDiff = Math.abs(s1.heroEquity.winRate - s2.heroEquity.winRate);
            const heroTieDiff = Math.abs(s1.heroEquity.tieRate - s2.heroEquity.tieRate);
            const rangeWinDiff = Math.abs(s1.rangeEquity.winRate - s2.rangeEquity.winRate);
            const rangeTieDiff = Math.abs(s1.rangeEquity.tieRate - s2.rangeEquity.tieRate);
            
            if (heroWinDiff < threshold && heroTieDiff < threshold && 
                rangeWinDiff < threshold && rangeTieDiff < threshold) {
                
                // 计算EV差异
                const evDiff = Math.abs(s1.ev - s2.ev);
                
                // 计算各动作EV的最大差异
                const actions = new Set([...Object.keys(s1.actionEV), ...Object.keys(s2.actionEV)]);
                let maxActionEvDiff = 0;
                const actionEvDiffs = {};
                
                for (const action of actions) {
                    const ev1 = s1.actionEV[action] || 0;
                    const ev2 = s2.actionEV[action] || 0;
                    const diff = Math.abs(ev1 - ev2);
                    actionEvDiffs[action] = { ev1, ev2, diff };
                    maxActionEvDiff = Math.max(maxActionEvDiff, diff);
                }
                
                pairs.push({
                    s1, s2,
                    heroWinDiff, heroTieDiff, rangeWinDiff, rangeTieDiff,
                    evDiff, maxActionEvDiff, actionEvDiffs,
                });
            }
        }
    }
    
    console.log(`\n找到 ${pairs.length} 对四维度胜率相近的场景`);
    
    // 分析EV差异
    const significantEvDiff = pairs.filter(p => p.evDiff > evThreshold);
    const significantActionEvDiff = pairs.filter(p => p.maxActionEvDiff > evThreshold);
    
    console.log(`\n【EV分析】`);
    console.log(`  总体EV差异显著(>${evThreshold}): ${significantEvDiff.length} 对`);
    console.log(`  动作EV差异显著(>${evThreshold}): ${significantActionEvDiff.length} 对`);
    
    if (significantActionEvDiff.length > 0) {
        console.log('\n【动作EV差异显著的案例】');
        for (const p of significantActionEvDiff.slice(0, 5)) {
            console.log(`\n${'─'.repeat(70)}`);
            console.log(`场景1: ${p.s1.board} + ${p.s1.heroHand}`);
            console.log(`  手牌胜率: ${(p.s1.heroEquity.winRate*100).toFixed(3)}%`);
            console.log(`  范围胜率: ${(p.s1.rangeEquity.winRate*100).toFixed(3)}%`);
            console.log(`  总体EV: ${p.s1.ev.toFixed(2)}`);
            console.log(`  动作EV: ${JSON.stringify(Object.fromEntries(Object.entries(p.s1.actionEV).map(([k,v]) => [k, v.toFixed(2)])))}`);
            
            console.log(`场景2: ${p.s2.board} + ${p.s2.heroHand}`);
            console.log(`  手牌胜率: ${(p.s2.heroEquity.winRate*100).toFixed(3)}%`);
            console.log(`  范围胜率: ${(p.s2.rangeEquity.winRate*100).toFixed(3)}%`);
            console.log(`  总体EV: ${p.s2.ev.toFixed(2)}`);
            console.log(`  动作EV: ${JSON.stringify(Object.fromEntries(Object.entries(p.s2.actionEV).map(([k,v]) => [k, v.toFixed(2)])))}`);
            
            console.log(`【差异】 总体EV差: ${p.evDiff.toFixed(2)}, 最大动作EV差: ${p.maxActionEvDiff.toFixed(2)}`);
        }
    }
    
    // 生成报告
    generateReport(scenarios, pairs, evThreshold);
}

function generateReport(scenarios, pairs, evThreshold) {
    const significantEvDiff = pairs.filter(p => p.evDiff > evThreshold);
    const significantActionEvDiff = pairs.filter(p => p.maxActionEvDiff > evThreshold);
    
    let report = `# 基于EV的四维度胜率验证实验报告

## 实验时间
${new Date().toISOString()}

## 实验目的

验证：当四维度胜率（手牌胜率、手牌平局率、范围胜率、范围平局率）相同时，各动作的EV是否相同？

**核心逻辑：**
- 如果EV相同，策略的"抖动"是正常的（纳什均衡中，当多个动作EV相等时，任何混合策略都是最优的）
- 如果EV不同，说明四维度胜率确实不足以决定最优策略

## 实验参数

| 参数 | 值 |
|------|-----|
| 场景数 | ${scenarios.length} |
| 胜率匹配阈值 | 0.05% |
| EV差异阈值 | ${evThreshold} chips |
| Solver目标可剥削度 | 0.05% |
| Solver最大迭代 | 2000 |

## 结果统计

| 指标 | 数量 |
|------|------|
| 四维度胜率相近的场景对 | ${pairs.length} |
| 总体EV差异显著(>${evThreshold}) | ${significantEvDiff.length} |
| 动作EV差异显著(>${evThreshold}) | ${significantActionEvDiff.length} |

## 结论

`;

    if (significantActionEvDiff.length === 0 && pairs.length > 0) {
        report += `### ✅ 四维度胜率可以决定最优EV

在 ${pairs.length} 对四维度胜率相近的场景中，**没有发现**动作EV差异显著的案例。

这说明：
1. 当四维度胜率相同时，各动作的EV也基本相同
2. 策略的"抖动"是正常的纳什均衡性质（当多个动作EV相等时，任何混合策略都是最优的）
3. **四维度胜率足以决定最优EV**，只是具体的混合策略可以有多种选择

**实际应用意义：**
- 可以使用四维度胜率作为状态抽象的依据
- 不同的策略选择不会影响最终的期望收益
`;
    } else if (significantActionEvDiff.length > 0) {
        const ratio = (significantActionEvDiff.length / pairs.length * 100).toFixed(1);
        report += `### ⚠️ 四维度胜率不足以决定最优EV

在 ${pairs.length} 对四维度胜率相近的场景中，有 ${significantActionEvDiff.length} 对（${ratio}%）的动作EV差异显著。

这说明：
1. 即使四维度胜率完全相同，不同场景的最优动作EV可能不同
2. 四维度胜率不足以完全描述博弈状态
3. 需要更多的特征来准确预测最优策略

### 动作EV差异显著的案例

`;
        for (const p of significantActionEvDiff.slice(0, 10)) {
            report += `---

**场景1:** \`${p.s1.board}\` + \`${p.s1.heroHand}\`
- 手牌胜率: ${(p.s1.heroEquity.winRate*100).toFixed(3)}%, 平局率: ${(p.s1.heroEquity.tieRate*100).toFixed(3)}%
- 范围胜率: ${(p.s1.rangeEquity.winRate*100).toFixed(3)}%, 平局率: ${(p.s1.rangeEquity.tieRate*100).toFixed(3)}%
- 总体EV: ${p.s1.ev.toFixed(2)}
- 动作EV: ${JSON.stringify(Object.fromEntries(Object.entries(p.s1.actionEV).map(([k,v]) => [k, v.toFixed(2)])))}
- 策略: ${JSON.stringify(Object.fromEntries(Object.entries(p.s1.strategy).map(([k,v]) => [k, (v*100).toFixed(1)+'%'])))}

**场景2:** \`${p.s2.board}\` + \`${p.s2.heroHand}\`
- 手牌胜率: ${(p.s2.heroEquity.winRate*100).toFixed(3)}%, 平局率: ${(p.s2.heroEquity.tieRate*100).toFixed(3)}%
- 范围胜率: ${(p.s2.rangeEquity.winRate*100).toFixed(3)}%, 平局率: ${(p.s2.rangeEquity.tieRate*100).toFixed(3)}%
- 总体EV: ${p.s2.ev.toFixed(2)}
- 动作EV: ${JSON.stringify(Object.fromEntries(Object.entries(p.s2.actionEV).map(([k,v]) => [k, v.toFixed(2)])))}
- 策略: ${JSON.stringify(Object.fromEntries(Object.entries(p.s2.strategy).map(([k,v]) => [k, (v*100).toFixed(1)+'%'])))}

**差异:**
- 总体EV差: ${p.evDiff.toFixed(2)} chips
- 最大动作EV差: ${p.maxActionEvDiff.toFixed(2)} chips

`;
        }
    } else {
        report += `### 需要更多数据

未找到足够的四维度胜率相近的场景对进行分析。
`;
    }
    
    const reportPath = join(__dirname, '..', 'experiments', 'results', 'ev_based_validation_report.md');
    writeFileSync(reportPath, report);
    console.log(`\n报告已保存到: ${reportPath}`);
}

runExperiment().catch(console.error);
