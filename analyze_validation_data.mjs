#!/usr/bin/env node
/**
 * 数据分析脚本 - 读取验证数据并分析找反例
 * 
 * 功能：
 * 1. 从批次文件中提取比对所需的关键信息（四维度胜率 + 策略）
 * 2. 分析四维度胜率相近的场景对，找出策略差异显著的反例
 * 3. 保存索引信息方便回查原始数据
 * 
 * 用法：
 *   提取数据：node analyze_validation_data.mjs extract
 *   分析数据：node analyze_validation_data.mjs analyze [胜率阈值%] [策略差异阈值%]
 */

import { readFileSync, writeFileSync, existsSync, readdirSync, mkdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DATA_DIR = join(__dirname, 'experiments', 'validation_data');
const RESULTS_DIR = join(__dirname, 'experiments', 'results');
const EXTRACTED_DIR = join(__dirname, 'experiments', 'extracted');
const EXTRACTED_INDEX_FILE = join(EXTRACTED_DIR, 'extracted_index.json');

/**
 * 从场景中提取比对所需的关键信息
 * @returns {Object} 精简的场景数据，包含索引信息
 */
function extractKeyInfo(scenario, sourceFile) {
    return {
        // 索引信息（用于回查原始数据）
        id: scenario.id,
        src: sourceFile,
        
        // 场景标识
        board: scenario.board,
        hand: scenario.heroHand,
        
        // 四维度胜率（比对核心数据）
        hWin: scenario.heroEquity?.winRate ?? 0,
        hTie: scenario.heroEquity?.tieRate ?? 0,
        rWin: scenario.rangeEquity?.winRate ?? 0,
        rTie: scenario.rangeEquity?.tieRate ?? 0,
        
        // 策略（比对核心数据）
        strat: scenario.strategy,
        
        // 各动作的EV（用于判断策略差异是否有意义）
        actEV: scenario.actionEV,
        
        // 辅助信息
        ev: scenario.ev,
        oopSize: scenario.oopRangeStats?.validCombos ?? 0,
        ipSize: scenario.ipRangeStats?.validCombos ?? 0,
    };
}

/**
 * 提取批次文件中的关键信息，跳过已提取的文件
 */
function extractBatchFiles() {
    if (!existsSync(EXTRACTED_DIR)) mkdirSync(EXTRACTED_DIR, { recursive: true });
    
    // 加载已提取文件索引
    let extractedIndex = { files: [], totalScenarios: 0 };
    if (existsSync(EXTRACTED_INDEX_FILE)) {
        try {
            extractedIndex = JSON.parse(readFileSync(EXTRACTED_INDEX_FILE, 'utf8'));
        } catch (e) {}
    }
    const extractedFiles = new Set(extractedIndex.files.map(f => f.source));
    
    // 查找所有批次文件
    const batchFiles = readdirSync(DATA_DIR).filter(f => f.endsWith('.json') && f.includes('batch'));
    console.log(`找到 ${batchFiles.length} 个批次文件`);
    
    let newExtracted = 0;
    let skipped = 0;
    
    for (const filename of batchFiles) {
        if (extractedFiles.has(filename)) {
            skipped++;
            continue;
        }
        
        const filePath = join(DATA_DIR, filename);
        try {
            const data = JSON.parse(readFileSync(filePath, 'utf8'));
            const scenarios = data.scenarios || [];
            
            // 提取关键信息
            const extracted = scenarios.map(s => extractKeyInfo(s, filename));
            
            // 保存提取后的文件
            const extractedFilename = filename.replace('.json', '_extracted.json');
            writeFileSync(join(EXTRACTED_DIR, extractedFilename), JSON.stringify(extracted, null, 2));
            
            // 更新索引
            extractedIndex.files.push({
                source: filename,
                extracted: extractedFilename,
                count: extracted.length,
            });
            extractedIndex.totalScenarios += extracted.length;
            
            console.log(`  提取: ${filename} -> ${extractedFilename} (${extracted.length} 场景)`);
            newExtracted++;
        } catch (e) {
            console.log(`  失败: ${filename} - ${e.message}`);
        }
    }
    
    // 保存索引
    extractedIndex.lastUpdated = new Date().toISOString();
    writeFileSync(EXTRACTED_INDEX_FILE, JSON.stringify(extractedIndex, null, 2));
    
    console.log(`\n提取完成: 新提取 ${newExtracted} 个文件，跳过 ${skipped} 个已提取文件`);
    console.log(`总计: ${extractedIndex.files.length} 个文件，${extractedIndex.totalScenarios} 个场景`);
    
    return extractedIndex;
}

/**
 * 从已提取的文件加载关键信息
 */
function loadExtractedData() {
    if (!existsSync(EXTRACTED_INDEX_FILE)) {
        console.log('未找到提取索引，请先运行: node analyze_validation_data.mjs extract');
        return [];
    }
    
    const extractedIndex = JSON.parse(readFileSync(EXTRACTED_INDEX_FILE, 'utf8'));
    console.log(`找到 ${extractedIndex.files.length} 个已提取文件，共 ${extractedIndex.totalScenarios} 个场景`);
    
    const scenarios = [];
    for (const fileInfo of extractedIndex.files) {
        const filePath = join(EXTRACTED_DIR, fileInfo.extracted);
        if (existsSync(filePath)) {
            try {
                const data = JSON.parse(readFileSync(filePath, 'utf8'));
                scenarios.push(...data);
                console.log(`  已加载: ${fileInfo.extracted} (${data.length} 场景)`);
            } catch (e) {
                console.log(`  加载失败: ${fileInfo.extracted} - ${e.message}`);
            }
        }
    }
    
    console.log(`\n总计加载 ${scenarios.length} 个场景`);
    return scenarios;
}

/**
 * 计算策略差异（基于每个动作的概率差异）
 * 策略格式: { "Check:0": 0.5, "Bet:33": 0.2, "Bet:50": 0.3, ... }
 * 返回值范围: 0-1，表示策略差异程度（所有动作概率差异的最大值）
 */
function calculateStrategyDiff(strategy1, strategy2) {
    if (!strategy1 || !strategy2) return 1;
    
    // 获取所有动作的并集
    const allActions = new Set([...Object.keys(strategy1), ...Object.keys(strategy2)]);
    
    // 计算每个动作的概率差异，取最大值
    let maxDiff = 0;
    for (const action of allActions) {
        const prob1 = strategy1[action] || 0;
        const prob2 = strategy2[action] || 0;
        const diff = Math.abs(prob1 - prob2);
        maxDiff = Math.max(maxDiff, diff);
    }
    
    return maxDiff;
}

/**
 * 计算场景的 EV 信息
 * 返回: { bestAction, bestEV, allEV }
 */
function calculateEVInfo(actionEV) {
    if (!actionEV) return { bestAction: 'unknown', bestEV: 0, allEV: {} };
    
    // 按 EV 排序所有动作
    const sortedActions = Object.entries(actionEV)
        .map(([action, ev]) => ({ action, ev: ev || 0 }))
        .sort((a, b) => b.ev - a.ev);
    
    if (sortedActions.length === 0) {
        return { bestAction: 'unknown', bestEV: 0, allEV: {} };
    }
    
    return {
        bestAction: sortedActions[0].action,
        bestEV: sortedActions[0].ev,
        allEV: actionEV,
    };
}

/**
 * 计算两个场景在策略差异最大的动作上的 EV 差距
 * 
 * 逻辑：
 * 1. 找出两个场景策略差异最大的动作（如 Check vs Bet:75）
 * 2. 对于每个场景，计算该动作与最优动作的 EV 差距
 * 3. 返回两个场景中较小的 EV 差距（因为只有两个场景都有显著 EV 差距才算有效反例）
 * 
 * @returns {{ action1, action2, s1_evGap, s2_evGap, minEvGapPct, explanation }}
 */
function calculateCrossActionEVGap(s1_strat, s2_strat, s1_actEV, s2_actEV) {
    if (!s1_strat || !s2_strat || !s1_actEV || !s2_actEV) {
        return { action1: 'unknown', action2: 'unknown', s1_evGap: 0, s2_evGap: 0, minEvGapPct: 0, explanation: '数据缺失' };
    }
    
    // 找出策略差异最大的两个动作
    const allActions = new Set([...Object.keys(s1_strat), ...Object.keys(s2_strat)]);
    let maxDiffAction1 = null;  // 场景1 概率高的动作
    let maxDiffAction2 = null;  // 场景2 概率高的动作
    let maxDiff = 0;
    
    for (const action of allActions) {
        const prob1 = s1_strat[action] || 0;
        const prob2 = s2_strat[action] || 0;
        const diff = prob1 - prob2;  // 正值表示场景1更倾向这个动作
        
        if (Math.abs(diff) > maxDiff) {
            maxDiff = Math.abs(diff);
            if (diff > 0) {
                maxDiffAction1 = action;  // 场景1 更倾向
            } else {
                maxDiffAction2 = action;  // 场景2 更倾向
            }
        }
    }
    
    // 找出两个场景各自倾向的动作
    // 场景1 倾向 action1，场景2 倾向 action2
    let action1 = maxDiffAction1;
    let action2 = maxDiffAction2;
    
    // 如果只找到一个方向的差异，找另一个方向
    if (!action1) {
        // 场景1 没有明显倾向的动作，找场景1概率最高的动作
        action1 = Object.entries(s1_strat).sort((a, b) => b[1] - a[1])[0]?.[0];
    }
    if (!action2) {
        // 场景2 没有明显倾向的动作，找场景2概率最高的动作
        action2 = Object.entries(s2_strat).sort((a, b) => b[1] - a[1])[0]?.[0];
    }
    
    if (!action1 || !action2) {
        return { action1: 'unknown', action2: 'unknown', s1_evGap: 0, s2_evGap: 0, minEvGapPct: 0, explanation: '无法确定差异动作' };
    }
    
    // 计算 EV 差距
    // 场景1: 选择 action1，我们看 action1 和 action2 的 EV 差距
    // 场景2: 选择 action2，我们看 action2 和 action1 的 EV 差距
    const s1_ev_action1 = s1_actEV[action1] || 0;
    const s1_ev_action2 = s1_actEV[action2] || 0;
    const s2_ev_action1 = s2_actEV[action1] || 0;
    const s2_ev_action2 = s2_actEV[action2] || 0;
    
    // 场景1 选择 action1 而不是 action2 的 EV 优势
    const s1_evGap = s1_ev_action1 - s1_ev_action2;
    // 场景2 选择 action2 而不是 action1 的 EV 优势
    const s2_evGap = s2_ev_action2 - s2_ev_action1;
    
    // 转换为底池百分比
    const s1_evGapPct = s1_evGap / 100;
    const s2_evGapPct = s2_evGap / 100;
    
    // 取两者中较小的（因为两个场景都需要有显著 EV 差距才算有效反例）
    const minEvGapPct = Math.min(s1_evGapPct, s2_evGapPct);
    
    const explanation = `场景1倾向${action1}(EV差距${(s1_evGapPct*100).toFixed(1)}%), 场景2倾向${action2}(EV差距${(s2_evGapPct*100).toFixed(1)}%)`;
    
    return {
        action1,
        action2,
        s1_evGap,
        s2_evGap,
        s1_evGapPct,
        s2_evGapPct,
        minEvGapPct,
        explanation,
    };
}

/**
 * 分析场景对，找出四维度胜率相近但策略差异显著的反例
 * @param {number} evGapThresholdPct - EV差距阈值（百分比，相对于底池），只有当两个场景的EV差距都大于此值时才算有效反例
 */
function analyzeScenarios(scenarios, equityThreshold, strategyThreshold, evGapThresholdPct = 0.01) {
    console.log(`\n分析 ${scenarios.length} 个场景...`);
    console.log(`四维度胜率阈值: ${(equityThreshold * 100).toFixed(3)}%`);
    console.log(`策略差异阈值: ${(strategyThreshold * 100).toFixed(1)}%`);
    console.log(`EV差距阈值: ${(evGapThresholdPct * 100).toFixed(1)}% 底池（Check vs Bet的EV差距需大于此值才算有效）`);
    
    const pairs = [];
    const startTime = Date.now();
    const totalPairs = scenarios.length * (scenarios.length - 1) / 2;
    let checkedPairs = 0;
    let lastProgress = 0;
    
    for (let i = 0; i < scenarios.length; i++) {
        for (let j = i + 1; j < scenarios.length; j++) {
            checkedPairs++;
            
            const progress = Math.floor(checkedPairs / totalPairs * 100);
            if (progress > lastProgress && progress % 10 === 0) {
                console.log(`  分析进度: ${progress}%`);
                lastProgress = progress;
            }
            
            const s1 = scenarios[i];
            const s2 = scenarios[j];
            
            // 计算四维度差异
            const heroWinDiff = Math.abs(s1.hWin - s2.hWin);
            const heroTieDiff = Math.abs(s1.hTie - s2.hTie);
            const rangeWinDiff = Math.abs(s1.rWin - s2.rWin);
            const rangeTieDiff = Math.abs(s1.rTie - s2.rTie);
            
            // 检查是否满足四维度阈值
            if (heroWinDiff < equityThreshold && heroTieDiff < equityThreshold && 
                rangeWinDiff < equityThreshold && rangeTieDiff < equityThreshold) {
                
                const strategyDiff = calculateStrategyDiff(s1.strat, s2.strat);
                
                // 计算两个场景的基本 EV 信息
                const ev1 = calculateEVInfo(s1.actEV);
                const ev2 = calculateEVInfo(s2.actEV);
                
                // 计算跨动作 EV 差距（核心改进：比较策略差异最大的动作的 EV）
                const crossEV = calculateCrossActionEVGap(s1.strat, s2.strat, s1.actEV, s2.actEV);
                
                pairs.push({
                    // 索引信息
                    s1_id: s1.id,
                    s1_src: s1.src,
                    s2_id: s2.id,
                    s2_src: s2.src,
                    
                    // 场景标识
                    s1_board: s1.board,
                    s1_hand: s1.hand,
                    s2_board: s2.board,
                    s2_hand: s2.hand,
                    
                    // 四维度数据
                    s1_hWin: s1.hWin, s1_hTie: s1.hTie, s1_rWin: s1.rWin, s1_rTie: s1.rTie,
                    s2_hWin: s2.hWin, s2_hTie: s2.hTie, s2_rWin: s2.rWin, s2_rTie: s2.rTie,
                    
                    // 差异
                    hWinDiff: heroWinDiff, hTieDiff: heroTieDiff, 
                    rWinDiff: rangeWinDiff, rTieDiff: rangeTieDiff,
                    stratDiff: strategyDiff,
                    
                    // 策略
                    s1_strat: s1.strat,
                    s2_strat: s2.strat,
                    
                    // 基本 EV 信息
                    s1_ev: s1.ev,
                    s2_ev: s2.ev,
                    s1_bestAction: ev1.bestAction,
                    s2_bestAction: ev2.bestAction,
                    s1_bestEV: ev1.bestEV,
                    s2_bestEV: ev2.bestEV,
                    s1_allEV: ev1.allEV,
                    s2_allEV: ev2.allEV,
                    
                    // 跨动作 EV 差距（核心指标）
                    diffAction1: crossEV.action1,  // 场景1 倾向的动作
                    diffAction2: crossEV.action2,  // 场景2 倾向的动作
                    s1_crossEvGapPct: crossEV.s1_evGapPct,  // 场景1 在差异动作上的 EV 优势
                    s2_crossEvGapPct: crossEV.s2_evGapPct,  // 场景2 在差异动作上的 EV 优势
                    minCrossEvGapPct: crossEV.minEvGapPct,  // 两者中较小的（用于过滤）
                    crossEvExplanation: crossEV.explanation,
                });
            }
        }
    }
    
    const analysisTime = (Date.now() - startTime) / 1000;
    console.log(`\n分析完成，用时: ${analysisTime.toFixed(1)}s`);
    
    // 筛选策略差异显著的反例
    // 条件：策略差异大 + 两个场景在差异动作上的 EV 差距都足够大（排除策略抖动）
    const significantPairs = pairs.filter(p => 
        p.stratDiff > strategyThreshold && 
        p.s1_crossEvGapPct > evGapThresholdPct && 
        p.s2_crossEvGapPct > evGapThresholdPct
    );
    significantPairs.sort((a, b) => b.stratDiff - a.stratDiff);
    
    // 统计被 EV 差距过滤掉的数量
    const filteredByEV = pairs.filter(p => 
        p.stratDiff > strategyThreshold && 
        (p.s1_crossEvGapPct <= evGapThresholdPct || p.s2_crossEvGapPct <= evGapThresholdPct)
    ).length;
    
    console.log(`\n策略差异>${(strategyThreshold*100).toFixed(0)}%的场景对: ${pairs.filter(p => p.stratDiff > strategyThreshold).length}`);
    console.log(`  - 其中因EV差距小被过滤: ${filteredByEV} (策略抖动)`);
    console.log(`  - 有效反例: ${significantPairs.length}`);
    
    return { pairs, significantPairs, analysisTime, filteredByEV, evGapThresholdPct };
}

/**
 * 生成分析报告
 */
function generateReport(scenarios, pairs, significantPairs, equityThreshold, strategyThreshold, evGapThreshold, filteredByEV) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    
    const stratDiffPairs = pairs.filter(p => p.stratDiff > strategyThreshold).length;
    
    let report = `# 四维度胜率-策略验证分析报告

## 分析时间
${new Date().toISOString()}

## 数据统计
| 指标 | 值 |
|------|-----|
| 总场景数 | ${scenarios.length} |
| 四维度胜率相近对数 | ${pairs.length} |
| 策略差异>${(strategyThreshold*100).toFixed(0)}%的对数 | ${stratDiffPairs} |
| 因EV差距小被过滤（策略抖动） | ${filteredByEV} |
| **有效反例数** | **${significantPairs.length}** |

## 分析参数
| 参数 | 值 | 说明 |
|------|-----|------|
| 四维度胜率阈值 | ${(equityThreshold * 100).toFixed(3)}% | 手牌胜率、手牌平局率、范围胜率、范围平局率的差异都需小于此值 |
| 策略差异阈值 | ${(strategyThreshold * 100).toFixed(1)}% | Check概率差异需大于此值 |
| EV差距阈值 | ${(evGapThreshold * 100).toFixed(1)}% 底池 | 两个场景在差异动作上的EV差距都需大于此值才算有效（排除策略抖动） |

## 关键发现
`;

    if (significantPairs.length > 0) {
        const ratio = stratDiffPairs > 0 ? (significantPairs.length / stratDiffPairs * 100).toFixed(1) : 0;
        report += `
### ⚠️ 发现 ${significantPairs.length} 个有效反例

在 ${stratDiffPairs} 对策略差异显著的场景中：
- ${filteredByEV} 对因EV差距小被过滤（属于策略抖动，不是真正的策略差异）
- **${significantPairs.length} 对是有效反例**（${ratio}%）

**结论：即使四维度胜率精确匹配（差异<${(equityThreshold * 100).toFixed(3)}%），且排除策略抖动后，仍有 ${significantPairs.length} 对场景的最优策略完全不同。**

### 有效反例详情（前10个）
`;
        for (let i = 0; i < Math.min(10, significantPairs.length); i++) {
            const p = significantPairs[i];
            // 获取差异动作的 EV
            const s1_ev_action1 = p.s1_allEV[p.diffAction1] || 0;
            const s1_ev_action2 = p.s1_allEV[p.diffAction2] || 0;
            const s2_ev_action1 = p.s2_allEV[p.diffAction1] || 0;
            const s2_ev_action2 = p.s2_allEV[p.diffAction2] || 0;
            
            report += `
---
#### 反例 ${i + 1} (策略差异: ${(p.stratDiff * 100).toFixed(1)}%)

**策略差异动作:** ${p.diffAction1} vs ${p.diffAction2}

**场景1:** \`${p.s1_board}\` + \`${p.s1_hand}\`
- 文件: \`${p.s1_src}\`, ID: ${p.s1_id}
- 手牌胜率: ${(p.s1_hWin * 100).toFixed(3)}%, 平局率: ${(p.s1_hTie * 100).toFixed(3)}%
- 范围胜率: ${(p.s1_rWin * 100).toFixed(3)}%, 平局率: ${(p.s1_rTie * 100).toFixed(3)}%
- 策略: ${JSON.stringify(p.s1_strat)}
- **EV对比: ${p.diffAction1}=${s1_ev_action1.toFixed(2)}, ${p.diffAction2}=${s1_ev_action2.toFixed(2)}, 差距=${(p.s1_crossEvGapPct * 100).toFixed(1)}%底池**

**场景2:** \`${p.s2_board}\` + \`${p.s2_hand}\`
- 文件: \`${p.s2_src}\`, ID: ${p.s2_id}
- 手牌胜率: ${(p.s2_hWin * 100).toFixed(3)}%, 平局率: ${(p.s2_hTie * 100).toFixed(3)}%
- 范围胜率: ${(p.s2_rWin * 100).toFixed(3)}%, 平局率: ${(p.s2_rTie * 100).toFixed(3)}%
- 策略: ${JSON.stringify(p.s2_strat)}
- **EV对比: ${p.diffAction1}=${s2_ev_action1.toFixed(2)}, ${p.diffAction2}=${s2_ev_action2.toFixed(2)}, 差距=${(p.s2_crossEvGapPct * 100).toFixed(1)}%底池**
`;
        }
    } else if (stratDiffPairs > 0) {
        report += `
### ✅ 未发现有效反例

在 ${stratDiffPairs} 对策略差异显著的场景中，全部 ${filteredByEV} 对都是因为EV差距小导致的策略抖动。

**结论：当排除策略抖动后，四维度胜率相近的场景对的策略也是相近的。这支持了"四维度胜率可以代表策略"的假设。**
`;
    } else if (pairs.length > 0) {
        report += `
### ✅ 未发现策略差异显著的场景对
在 ${pairs.length} 对四维度胜率相近的场景中，没有策略差异超过${(strategyThreshold*100).toFixed(0)}%的。
`;
    } else {
        report += `
### 需要更多数据
在 ${scenarios.length} 个场景中，未找到四维度胜率都相近的场景对。
建议：生成更多数据或放宽阈值。
`;
    }

    // 策略差异分布
    if (pairs.length > 0) {
        report += `
## 策略差异分布（所有四维度相近的场景对）
| 差异范围 | 数量 | 占比 |
|----------|------|------|
`;
        const ranges = [[0, 0.05, '0-5%'], [0.05, 0.10, '5-10%'], [0.10, 0.15, '10-15%'], 
                       [0.15, 0.20, '15-20%'], [0.20, 0.30, '20-30%'], [0.30, 1.0, '>30%']];
        for (const [min, max, label] of ranges) {
            const count = pairs.filter(p => p.stratDiff >= min && p.stratDiff < max).length;
            report += `| ${label} | ${count} | ${(count / pairs.length * 100).toFixed(1)}% |\n`;
        }
    }

    return { report, timestamp };
}

/**
 * 分析已提取的数据
 */
async function runAnalysis(equityThreshold, strategyThreshold, evGapThreshold) {
    console.log('='.repeat(80));
    console.log('数据分析 - 从已提取数据中找反例');
    console.log('='.repeat(80));
    
    if (!existsSync(RESULTS_DIR)) mkdirSync(RESULTS_DIR, { recursive: true });
    
    const scenarios = loadExtractedData();
    
    if (scenarios.length === 0) {
        console.log('\n没有可分析的数据，请先运行: node analyze_validation_data.mjs extract');
        return;
    }
    
    const { pairs, significantPairs, filteredByEV } = analyzeScenarios(scenarios, equityThreshold, strategyThreshold, evGapThreshold);
    
    console.log('\n' + '='.repeat(80));
    console.log('分析结果');
    console.log('='.repeat(80));
    console.log(`四维度胜率相近的场景对: ${pairs.length}`);
    console.log(`策略差异显著且EV差距大的有效反例: ${significantPairs.length}`);
    
    if (significantPairs.length > 0) {
        console.log('\n【有效反例（排除策略抖动）】');
        for (const p of significantPairs.slice(0, 5)) {
            const s1_ev_action1 = p.s1_allEV[p.diffAction1] || 0;
            const s1_ev_action2 = p.s1_allEV[p.diffAction2] || 0;
            const s2_ev_action1 = p.s2_allEV[p.diffAction1] || 0;
            const s2_ev_action2 = p.s2_allEV[p.diffAction2] || 0;
            
            console.log(`\n${'─'.repeat(70)}`);
            console.log(`【策略差异动作】 ${p.diffAction1} vs ${p.diffAction2}`);
            console.log(`【场景1】 ${p.s1_board} + ${p.s1_hand} (${p.s1_src}, ID:${p.s1_id})`);
            console.log(`  四维度: hWin=${(p.s1_hWin*100).toFixed(3)}%, hTie=${(p.s1_hTie*100).toFixed(3)}%, rWin=${(p.s1_rWin*100).toFixed(3)}%, rTie=${(p.s1_rTie*100).toFixed(3)}%`);
            console.log(`  策略: ${JSON.stringify(p.s1_strat)}`);
            console.log(`  EV对比: ${p.diffAction1}=${s1_ev_action1.toFixed(2)}, ${p.diffAction2}=${s1_ev_action2.toFixed(2)}, 差距=${(p.s1_crossEvGapPct*100).toFixed(1)}%底池`);
            console.log(`【场景2】 ${p.s2_board} + ${p.s2_hand} (${p.s2_src}, ID:${p.s2_id})`);
            console.log(`  四维度: hWin=${(p.s2_hWin*100).toFixed(3)}%, hTie=${(p.s2_hTie*100).toFixed(3)}%, rWin=${(p.s2_rWin*100).toFixed(3)}%, rTie=${(p.s2_rTie*100).toFixed(3)}%`);
            console.log(`  策略: ${JSON.stringify(p.s2_strat)}`);
            console.log(`  EV对比: ${p.diffAction1}=${s2_ev_action1.toFixed(2)}, ${p.diffAction2}=${s2_ev_action2.toFixed(2)}, 差距=${(p.s2_crossEvGapPct*100).toFixed(1)}%底池`);
            console.log(`【差异】 策略差异: ${(p.stratDiff*100).toFixed(1)}%`);
        }
    }
    
    const { report, timestamp } = generateReport(scenarios, pairs, significantPairs, equityThreshold, strategyThreshold, evGapThreshold, filteredByEV);
    
    writeFileSync(join(RESULTS_DIR, `analysis_report_${timestamp}.md`), report);
    writeFileSync(join(RESULTS_DIR, 'analysis_report_latest.md'), report);
    console.log(`\n报告已保存到: experiments/results/analysis_report_${timestamp}.md`);
    
    if (significantPairs.length > 0) {
        const counterexamplesData = { timestamp, equityThreshold, strategyThreshold, evGapThreshold, totalScenarios: scenarios.length, totalPairs: pairs.length, filteredByEV, counterexamples: significantPairs };
        writeFileSync(join(RESULTS_DIR, `counterexamples_${timestamp}.json`), JSON.stringify(counterexamplesData, null, 2));
        console.log(`反例数据已保存到: experiments/results/counterexamples_${timestamp}.json`);
    }
}

/**
 * 主函数
 */
async function main() {
    const args = process.argv.slice(2);
    const command = args[0] || 'extract';
    
    if (command === 'extract') {
        console.log('='.repeat(80));
        console.log('数据提取 - 从批次文件提取关键信息');
        console.log('='.repeat(80));
        extractBatchFiles();
    } else if (command === 'analyze') {
        const equityThreshold = parseFloat(args[1]) / 100 || 0.0005;  // 默认 0.05%
        const strategyThreshold = parseFloat(args[2]) / 100 || 0.15;  // 默认 15%
        const evGapThresholdPct = parseFloat(args[3]) / 100 || 0.01;  // 默认 1% 底池
        await runAnalysis(equityThreshold, strategyThreshold, evGapThresholdPct);
    } else {
        console.log('用法:');
        console.log('  提取数据: node analyze_validation_data.mjs extract');
        console.log('  分析数据: node analyze_validation_data.mjs analyze [胜率阈值%] [策略差异阈值%] [EV差距阈值%]');
        console.log('');
        console.log('示例:');
        console.log('  node analyze_validation_data.mjs analyze 0.05 15 1');
        console.log('  - 四维度胜率差异 < 0.05%');
        console.log('  - 策略差异 > 15%');
        console.log('  - Check和Bet的EV差距 > 1%底池（排除策略抖动）');
    }
}

main().catch(console.error);
