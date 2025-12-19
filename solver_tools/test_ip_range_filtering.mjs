#!/usr/bin/env node
/**
 * 测试：固定OOP手牌时，过滤IP范围 vs 完整IP范围 的策略对比
 * 
 * 实验目的：验证当OOP持有特定手牌时，从IP范围中移除与该手牌冲突的组合后，
 * Solver得到的策略是否与使用完整IP范围时相同。
 * 
 * 理论上应该相同，因为Solver内部会自动处理死牌（card removal）。
 */

import { solveRiver } from './postflop_solver.mjs';
import { writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// 将手牌类型展开为具体组合
function expandHandType(handType) {
    const suits = 'cdhs';
    const result = [];
    
    if (handType.length === 2 && handType[0] === handType[1]) {
        // 对子: AA -> AcAd, AcAh, AcAs, AdAh, AdAs, AhAs
        const r = handType[0];
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = s1 + 1; s2 < 4; s2++) {
                result.push(r + suits[s1] + r + suits[s2]);
            }
        }
    } else if (handType.endsWith('s')) {
        // 同花: AKs -> AcKc, AdKd, AhKh, AsKs
        const r1 = handType[0];
        const r2 = handType[1];
        for (let s = 0; s < 4; s++) {
            result.push(r1 + suits[s] + r2 + suits[s]);
        }
    } else if (handType.endsWith('o')) {
        // 非同花: AKo -> AcKd, AcKh, AcKs, AdKc, ...
        const r1 = handType[0];
        const r2 = handType[1];
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = 0; s2 < 4; s2++) {
                if (s1 !== s2) {
                    result.push(r1 + suits[s1] + r2 + suits[s2]);
                }
            }
        }
    } else if (handType.length === 2) {
        // 无后缀的非对子: AK -> 包含同花和非同花
        const r1 = handType[0];
        const r2 = handType[1];
        for (let s = 0; s < 4; s++) {
            result.push(r1 + suits[s] + r2 + suits[s]);
        }
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = 0; s2 < 4; s2++) {
                if (s1 !== s2) {
                    result.push(r1 + suits[s1] + r2 + suits[s2]);
                }
            }
        }
    }
    return result;
}

// 从范围中移除与指定牌冲突的组合，返回详细信息
function filterRangeByDeadCards(rangeStr, deadCards) {
    const deadSet = new Set(deadCards);
    const handTypes = rangeStr.split(',').map(h => h.trim());
    const filteredHands = [];
    const removedCombos = [];  // 记录被移除的组合
    const details = [];  // 详细信息
    
    for (const handType of handTypes) {
        const combos = expandHandType(handType);
        const validCombos = [];
        const invalidCombos = [];
        
        for (const combo of combos) {
            const card1 = combo.slice(0, 2);
            const card2 = combo.slice(2, 4);
            if (!deadSet.has(card1) && !deadSet.has(card2)) {
                validCombos.push(combo);
            } else {
                invalidCombos.push(combo);
            }
        }
        
        if (invalidCombos.length > 0) {
            removedCombos.push(...invalidCombos);
            details.push({
                handType,
                totalCombos: combos.length,
                validCombos: validCombos.length,
                removedCombos: invalidCombos,
            });
        }
        
        if (validCombos.length > 0) {
            // 如果所有组合都有效，保留原始手牌类型
            if (validCombos.length === combos.length) {
                filteredHands.push(handType);
            } else {
                // 否则添加具体的有效组合
                filteredHands.push(...validCombos);
            }
        }
    }
    
    return {
        filteredRange: filteredHands.join(','),
        filteredHands,
        removedCombos,
        details,
    };
}

// 在oopCards中找到手牌索引
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
        
        if (cards[0] === heroIndices[0] && cards[1] === heroIndices[1]) {
            return i;
        }
    }
    return -1;
}

// 从solver结果中提取特定手牌的策略
function extractStrategy(solverResult, heroCards) {
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
    offset += oopLen + ipLen; // equity
    offset += oopLen + ipLen; // ev
    offset += oopLen + ipLen; // eqr
    
    // 提取策略
    const strategy = {};
    for (let i = 0; i < numActions; i++) {
        strategy[actionList[i].name] = results[offset + i * oopLen + heroIndex];
    }
    
    return { strategy, actions: actionList };
}

// 计算策略差异
function calculateStrategyDiff(strategy1, strategy2) {
    const keys = new Set([...Object.keys(strategy1), ...Object.keys(strategy2)]);
    let maxDiff = 0;
    let totalDiff = 0;
    
    for (const key of keys) {
        const diff = Math.abs((strategy1[key] || 0) - (strategy2[key] || 0));
        maxDiff = Math.max(maxDiff, diff);
        totalDiff += diff;
    }
    
    return { maxDiff, avgDiff: totalDiff / keys.size };
}

async function runTest() {
    console.log('='.repeat(80));
    console.log('测试：固定OOP手牌时，过滤IP范围 vs 完整IP范围 的策略对比');
    console.log('='.repeat(80));
    
    // 测试场景
    const board = 'Ks Td 7c 4h 2s';
    const oopRange = 'AA,KK,QQ,JJ,TT,99,88,77,66,AKs,AKo,AQs,AQo,KQs,KQo,AJs,KJs,QJs';
    const ipRange = 'AA,KK,QQ,JJ,TT,99,88,AKs,AKo,AQs,AQo,KQs,KQo,AJs,KJs,QJs,JTs,T9s';
    
    // 测试多个OOP手牌
    const testHands = [
        ['Ac', 'Ad'],  // AA - 与IP范围的AA有冲突
        ['Qc', 'Qd'],  // QQ - 与IP范围的QQ有冲突
        ['Ah', 'Kh'],  // AKs - 与IP范围的AKs有冲突
        ['7d', '7h'],  // 77 - 不在IP范围中，无冲突
        ['6c', '6d'],  // 66 - 不在IP范围中，无冲突
    ];
    
    console.log(`\n公共牌: ${board}`);
    console.log(`OOP范围: ${oopRange}`);
    console.log(`IP范围: ${ipRange}`);
    console.log('\nSolver参数: pot=100, stack=500, bet=33%,50%,75%, raise=50%,100%');
    
    const solverConfig = {
        board,
        startingPot: 100,
        effectiveStack: 500,
        oopBetSizes: '33,50,75',
        ipBetSizes: '33,50,75',
        oopRaiseSizes: '50,100',
        ipRaiseSizes: '50,100',
        targetExploitability: 0.1,
        maxIterations: 1000,
    };
    
    // 报告数据
    const reportData = {
        timestamp: new Date().toISOString(),
        board,
        oopRange,
        ipRange,
        solverConfig: {
            startingPot: 100,
            effectiveStack: 500,
            betSizes: '33%, 50%, 75%',
            raiseSizes: '50%, 100%',
            targetExploitability: '0.1%',
            maxIterations: 1000,
        },
        tests: [],
    };
    
    console.log('\n' + '─'.repeat(80));
    
    for (const heroCards of testHands) {
        const heroHand = heroCards.join('');
        console.log(`\n【测试手牌: ${heroHand}】`);
        
        const testResult = {
            heroHand,
            heroCards,
        };
        
        // 死牌 = 公共牌 + OOP手牌
        const boardCards = board.split(' ');
        const deadCards = [...boardCards, ...heroCards];
        
        // 方案1: 使用完整IP范围
        console.log('\n方案1: 完整IP范围');
        const result1 = await solveRiver({
            ...solverConfig,
            oopRange,
            ipRange,
        });
        
        const strategy1 = extractStrategy(result1, heroCards);
        if (!strategy1) {
            console.log('  错误: 无法提取策略');
            continue;
        }
        console.log(`  策略: ${JSON.stringify(strategy1.strategy)}`);
        console.log(`  迭代次数: ${result1.iterations}, 可剥削度: ${result1.exploitability.toFixed(4)}%`);
        
        testResult.fullRange = {
            ipRange,
            ipRangeHandTypes: ipRange.split(','),
            strategy: strategy1.strategy,
            iterations: result1.iterations,
            exploitability: result1.exploitability,
        };
        
        // 方案2: 过滤后的IP范围（移除与OOP手牌冲突的组合）
        const filterResult = filterRangeByDeadCards(ipRange, heroCards);
        console.log(`\n方案2: 过滤后IP范围`);
        console.log(`  过滤前: ${ipRange.split(',').length} 种手牌类型`);
        console.log(`  过滤后: ${filterResult.filteredHands.length} 种手牌/组合`);
        console.log(`  被移除的组合: ${filterResult.removedCombos.join(', ') || '无'}`);
        
        const result2 = await solveRiver({
            ...solverConfig,
            oopRange,
            ipRange: filterResult.filteredRange,
        });
        
        const strategy2 = extractStrategy(result2, heroCards);
        if (!strategy2) {
            console.log('  错误: 无法提取策略');
            continue;
        }
        console.log(`  策略: ${JSON.stringify(strategy2.strategy)}`);
        console.log(`  迭代次数: ${result2.iterations}, 可剥削度: ${result2.exploitability.toFixed(4)}%`);
        
        testResult.filteredRange = {
            ipRange: filterResult.filteredRange,
            ipRangeHandTypes: filterResult.filteredHands,
            removedCombos: filterResult.removedCombos,
            filterDetails: filterResult.details,
            strategy: strategy2.strategy,
            iterations: result2.iterations,
            exploitability: result2.exploitability,
        };
        
        // 对比
        const diff = calculateStrategyDiff(strategy1.strategy, strategy2.strategy);
        console.log(`\n【对比结果】`);
        console.log(`  最大差异: ${(diff.maxDiff * 100).toFixed(4)}%`);
        console.log(`  平均差异: ${(diff.avgDiff * 100).toFixed(4)}%`);
        
        let conclusion = '';
        if (diff.maxDiff < 0.01) {
            conclusion = '策略基本相同（差异<1%）';
            console.log(`  ✅ ${conclusion}`);
        } else if (diff.maxDiff < 0.05) {
            conclusion = '策略有轻微差异（1%-5%）';
            console.log(`  ⚠️ ${conclusion}`);
        } else {
            conclusion = '策略有显著差异（>5%）';
            console.log(`  ❌ ${conclusion}`);
        }
        
        testResult.comparison = {
            maxDiff: diff.maxDiff,
            avgDiff: diff.avgDiff,
            conclusion,
            hasConflict: filterResult.removedCombos.length > 0,
        };
        
        reportData.tests.push(testResult);
        
        console.log('\n' + '─'.repeat(80));
    }
    
    // 生成报告
    generateReport(reportData);
    
    console.log('\n实验完成！');
}

function generateReport(data) {
    let report = `# IP范围过滤实验报告

## 实验时间
${data.timestamp}

## 实验目的
验证当OOP持有特定手牌时，从IP范围中移除与该手牌冲突的组合后，Solver得到的策略是否与使用完整IP范围时相同。

## 实验设置

### 公共牌
\`${data.board}\`

### OOP范围
\`${data.oopRange}\`

**展开为手牌类型：**
${data.oopRange.split(',').map(h => `- ${h}`).join('\n')}

### IP范围（完整）
\`${data.ipRange}\`

**展开为手牌类型：**
${data.ipRange.split(',').map(h => `- ${h}`).join('\n')}

### Solver参数
| 参数 | 值 |
|------|-----|
| 起始底池 | ${data.solverConfig.startingPot} |
| 有效筹码 | ${data.solverConfig.effectiveStack} |
| 下注尺寸 | ${data.solverConfig.betSizes} |
| 加注尺寸 | ${data.solverConfig.raiseSizes} |
| 目标可剥削度 | ${data.solverConfig.targetExploitability} |
| 最大迭代次数 | ${data.solverConfig.maxIterations} |

## 测试结果汇总

| OOP手牌 | 与IP范围冲突 | 被移除组合数 | 策略差异 | 结论 |
|---------|-------------|-------------|---------|------|
`;

    for (const test of data.tests) {
        const hasConflict = test.comparison.hasConflict ? '是' : '否';
        const removedCount = test.filteredRange.removedCombos.length;
        const diffPercent = (test.comparison.maxDiff * 100).toFixed(2) + '%';
        report += `| ${test.heroHand} | ${hasConflict} | ${removedCount} | ${diffPercent} | ${test.comparison.conclusion} |\n`;
    }

    report += `\n## 详细测试结果\n`;

    for (const test of data.tests) {
        report += `
---

### 测试手牌: ${test.heroHand}

#### 方案1: 完整IP范围

**IP范围：** \`${test.fullRange.ipRange}\`

**IP范围展开（${test.fullRange.ipRangeHandTypes.length}种手牌类型）：**
\`\`\`
${test.fullRange.ipRangeHandTypes.join(', ')}
\`\`\`

**Solver结果：**
- 策略: \`${JSON.stringify(test.fullRange.strategy)}\`
- 迭代次数: ${test.fullRange.iterations}
- 可剥削度: ${test.fullRange.exploitability.toFixed(4)}%

#### 方案2: 过滤后IP范围

**被移除的组合（${test.filteredRange.removedCombos.length}个）：**
\`\`\`
${test.filteredRange.removedCombos.length > 0 ? test.filteredRange.removedCombos.join(', ') : '无'}
\`\`\`

**过滤详情：**
`;
        if (test.filteredRange.filterDetails.length > 0) {
            for (const detail of test.filteredRange.filterDetails) {
                report += `- **${detail.handType}**: 原${detail.totalCombos}个组合，保留${detail.validCombos}个，移除: ${detail.removedCombos.join(', ')}\n`;
            }
        } else {
            report += `- 无组合被移除\n`;
        }

        report += `
**过滤后IP范围（${test.filteredRange.ipRangeHandTypes.length}种手牌/组合）：**
\`\`\`
${test.filteredRange.ipRangeHandTypes.join(', ')}
\`\`\`

**Solver结果：**
- 策略: \`${JSON.stringify(test.filteredRange.strategy)}\`
- 迭代次数: ${test.filteredRange.iterations}
- 可剥削度: ${test.filteredRange.exploitability.toFixed(4)}%

#### 对比结果

| 指标 | 完整范围 | 过滤后范围 | 差异 |
|------|---------|-----------|------|
`;
        const actions = Object.keys(test.fullRange.strategy);
        for (const action of actions) {
            const v1 = test.fullRange.strategy[action] || 0;
            const v2 = test.filteredRange.strategy[action] || 0;
            const diff = Math.abs(v1 - v2);
            report += `| ${action} | ${(v1 * 100).toFixed(2)}% | ${(v2 * 100).toFixed(2)}% | ${(diff * 100).toFixed(2)}% |\n`;
        }

        report += `
**结论：** ${test.comparison.conclusion}
`;
    }

    report += `
## 总结

`;
    const conflictTests = data.tests.filter(t => t.comparison.hasConflict);
    const noConflictTests = data.tests.filter(t => !t.comparison.hasConflict);
    
    if (conflictTests.length > 0) {
        const allConflictHaveDiff = conflictTests.every(t => t.comparison.maxDiff > 0.01);
        if (allConflictHaveDiff) {
            report += `### ⚠️ 重要发现

当OOP手牌与IP范围存在冲突时（即OOP持有的牌在IP范围中也存在），过滤IP范围后的策略与完整IP范围的策略**显著不同**。

这说明：
1. **Solver不会自动处理card removal**：即使OOP持有AA，Solver仍然会按照IP范围中包含AA来计算
2. **手动过滤IP范围是必要的**：如果要获得正确的策略，需要在调用Solver前手动移除与OOP手牌冲突的IP组合
3. **无冲突时策略相同**：当OOP手牌不在IP范围中时（如77、66），过滤前后策略完全相同
`;
        }
    }

    if (noConflictTests.length > 0) {
        const allNoConflictSame = noConflictTests.every(t => t.comparison.maxDiff < 0.01);
        if (allNoConflictSame) {
            report += `
### ✅ 验证结果

当OOP手牌与IP范围无冲突时，过滤IP范围后的策略与完整IP范围的策略**完全相同**，这符合预期。
`;
        }
    }

    // 保存报告
    const reportPath = join(__dirname, '..', 'experiments', 'results', 'ip_range_filtering_report.md');
    writeFileSync(reportPath, report);
    console.log(`\n报告已保存到: ${reportPath}`);
    
    // 保存JSON数据
    const jsonPath = join(__dirname, '..', 'experiments', 'results', 'ip_range_filtering_data.json');
    writeFileSync(jsonPath, JSON.stringify(data, null, 2));
    console.log(`数据已保存到: ${jsonPath}`);
}

runTest().catch(console.error);
