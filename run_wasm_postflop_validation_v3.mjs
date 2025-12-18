#!/usr/bin/env node
/**
 * 使用wasm-postflop + OMPEval 进行跨公共牌四维度胜率-策略验证实验 V3
 * 
 * 改进：使用 OMPEval (C++) 计算范围胜率，速度比 poker-odds-calc 快约60倍
 */

import { solveRiver } from './solver_tools/postflop_solver.mjs';
import { parseBoardString } from './solver_tools/equity_calculator.mjs';
import { writeFileSync, readFileSync, existsSync } from 'fs';
import { execSync } from 'child_process';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// OMPEval路径
const OMPEVAL_PATH = join(__dirname, 'solver_tools', 'OMPEval', 'equity_calc');

// 所有可能的手牌类型（用于随机生成范围）
const ALL_HANDS = [];
const RANKS = '23456789TJQKA';

// 初始化所有手牌类型
function initAllHands() {
    // 对子: AA, KK, ..., 22
    for (let r = 12; r >= 0; r--) {
        ALL_HANDS.push(RANKS[r] + RANKS[r]);
    }
    // 同花非对子: AKs, AQs, ..., 32s
    for (let r1 = 12; r1 >= 1; r1--) {
        for (let r2 = r1 - 1; r2 >= 0; r2--) {
            ALL_HANDS.push(RANKS[r1] + RANKS[r2] + 's');
        }
    }
    // 非同花非对子: AKo, AQo, ..., 32o
    for (let r1 = 12; r1 >= 1; r1--) {
        for (let r2 = r1 - 1; r2 >= 0; r2--) {
            ALL_HANDS.push(RANKS[r1] + RANKS[r2] + 'o');
        }
    }
}
initAllHands();

/**
 * 生成随机范围
 * @param {number} minHands - 最少手牌数量
 * @param {number} maxHands - 最多手牌数量
 * @returns {string} - 范围字符串
 */
function generateRandomRange(minHands = 50, maxHands = 150) {
    const numHands = minHands + Math.floor(Math.random() * (maxHands - minHands + 1));
    
    // 随机选择手牌，但倾向于选择更强的手牌（前面的手牌）
    const selectedHands = new Set();
    
    // 首先确保包含一些强牌（前20%的手牌有更高概率被选中）
    const strongHandsCount = Math.floor(numHands * 0.4);
    const strongHandsPool = ALL_HANDS.slice(0, Math.floor(ALL_HANDS.length * 0.3));
    
    while (selectedHands.size < strongHandsCount && strongHandsPool.length > 0) {
        const idx = Math.floor(Math.random() * strongHandsPool.length);
        selectedHands.add(strongHandsPool[idx]);
    }
    
    // 然后从所有手牌中随机选择剩余的
    while (selectedHands.size < numHands) {
        const idx = Math.floor(Math.random() * ALL_HANDS.length);
        selectedHands.add(ALL_HANDS[idx]);
    }
    
    return Array.from(selectedHands).join(',');
}

/**
 * 将范围字符串展开为OMPEval支持的格式
 */
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
                // 对子范围: AA-22
                const startRank = ranks.indexOf(start[0]);
                const endRank = ranks.indexOf(end[0]);
                const minRank = Math.min(startRank, endRank);
                const maxRank = Math.max(startRank, endRank);
                for (let r = minRank; r <= maxRank; r++) {
                    result.push(ranks[r] + ranks[r]);
                }
            } else if (start.endsWith('s')) {
                // 同花范围: AKs-A2s
                const highRank = ranks.indexOf(start[0]);
                const startLow = ranks.indexOf(start[1]);
                const endLow = ranks.indexOf(end[1]);
                const minLow = Math.min(startLow, endLow);
                const maxLow = Math.max(startLow, endLow);
                for (let r = minLow; r <= maxLow; r++) {
                    if (r !== highRank) {
                        result.push(ranks[highRank] + ranks[r] + 's');
                    }
                }
            } else if (start.endsWith('o')) {
                // 不同花范围: AKo-A2o
                const highRank = ranks.indexOf(start[0]);
                const startLow = ranks.indexOf(start[1]);
                const endLow = ranks.indexOf(end[1]);
                const minLow = Math.min(startLow, endLow);
                const maxLow = Math.max(startLow, endLow);
                for (let r = minLow; r <= maxLow; r++) {
                    if (r !== highRank) {
                        result.push(ranks[highRank] + ranks[r] + 'o');
                    }
                }
            }
        } else {
            result.push(p);
        }
    }
    
    return result.join(',');
}


/**
 * 使用OMPEval计算范围对范围的胜率
 */
function calculateEquityOmpeval(range1, range2, board, dead = '') {
    const range1Expanded = expandRangeToOmpeval(range1);
    const range2Expanded = expandRangeToOmpeval(range2);
    
    // 转换公共牌格式: "As Ks 6d Qc Td" -> "AsKs6dQcTd"
    const boardCompact = board.replace(/\s+/g, '');
    
    let cmd = `"${OMPEVAL_PATH}" "${range1Expanded}" "${range2Expanded}" "${boardCompact}"`;
    if (dead) {
        cmd += ` "${dead}"`;
    }
    
    try {
        const output = execSync(cmd, { encoding: 'utf8' });
        // 处理 nan 值
        const cleanOutput = output.replace(/nan/g, 'null');
        const data = JSON.parse(cleanOutput);
        
        const wins = data.wins[0];
        const losses = data.wins[1];
        const ties = data.tieCount;
        const total = wins + losses + ties;
        
        if (total === 0) {
            return { equity: 0.5, winRate: 0.5, tieRate: 0, wins: 0, ties: 0, total: 0 };
        }
        
        const winRate = wins / total;
        const tieRate = ties / total;
        const equity = (wins + ties * 0.5) / total;
        
        return { equity, winRate, tieRate, wins, ties, total };
    } catch (e) {
        throw new Error(`OMPEval error: ${e.message}`);
    }
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
        const cardStr = ranks[rank] + suits[suit];
        
        if (!usedCards.has(cardStr)) {
            usedCards.add(cardStr);
            cards.push(cardStr);
        }
    }
    
    return cards.join(' ');
}

/**
 * 将手牌类型（如AA, AKs, AKo）展开为具体的手牌组合
 * @param {string} handType - 手牌类型
 * @returns {Array} - 具体手牌组合数组 [{hand: 'AcAd', cards: ['Ac', 'Ad']}, ...]
 */
function expandHandType(handType) {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    const result = [];
    
    if (handType.length === 2 && handType[0] === handType[1]) {
        // 对子: AA -> AcAd, AcAh, AcAs, AdAh, AdAs, AhAs
        const r = handType[0];
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = s1 + 1; s2 < 4; s2++) {
                const card1 = r + suits[s1];
                const card2 = r + suits[s2];
                result.push({ hand: card1 + card2, cards: [card1, card2] });
            }
        }
    } else if (handType.endsWith('s')) {
        // 同花: AKs -> AcKc, AdKd, AhKh, AsKs
        const r1 = handType[0];
        const r2 = handType[1];
        for (let s = 0; s < 4; s++) {
            const card1 = r1 + suits[s];
            const card2 = r2 + suits[s];
            result.push({ hand: card1 + card2, cards: [card1, card2] });
        }
    } else if (handType.endsWith('o')) {
        // 非同花: AKo -> AcKd, AcKh, AcKs, AdKc, ...
        const r1 = handType[0];
        const r2 = handType[1];
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = 0; s2 < 4; s2++) {
                if (s1 !== s2) {
                    const card1 = r1 + suits[s1];
                    const card2 = r2 + suits[s2];
                    result.push({ hand: card1 + card2, cards: [card1, card2] });
                }
            }
        }
    } else if (handType.length === 2) {
        // 无后缀的非对子: AK -> 包含同花和非同花
        const r1 = handType[0];
        const r2 = handType[1];
        // 同花
        for (let s = 0; s < 4; s++) {
            const card1 = r1 + suits[s];
            const card2 = r2 + suits[s];
            result.push({ hand: card1 + card2, cards: [card1, card2] });
        }
        // 非同花
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = 0; s2 < 4; s2++) {
                if (s1 !== s2) {
                    const card1 = r1 + suits[s1];
                    const card2 = r2 + suits[s2];
                    result.push({ hand: card1 + card2, cards: [card1, card2] });
                }
            }
        }
    }
    
    return result;
}

/**
 * 从OOP范围中随机选择一个不与公共牌冲突的手牌
 * @param {string} board - 公共牌
 * @param {string} oopRange - OOP范围字符串
 * @returns {Object|null} - {hand: 'AcAd', cards: ['Ac', 'Ad']} 或 null
 */
function selectRandomHeroHand(board, oopRange) {
    const boardCards = new Set(parseBoardString(board));
    
    // 展开OOP范围为具体手牌
    const handTypes = oopRange.split(',').map(h => h.trim());
    const allHands = [];
    
    for (const handType of handTypes) {
        const expanded = expandHandType(handType);
        for (const hand of expanded) {
            // 检查是否与公共牌冲突
            if (!boardCards.has(hand.cards[0]) && !boardCards.has(hand.cards[1])) {
                allHands.push(hand);
            }
        }
    }
    
    if (allHands.length === 0) return null;
    return allHands[Math.floor(Math.random() * allHands.length)];
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

// 从solver结果中提取特定手牌的策略、EV和动作EV
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
    
    // 结果数组布局:
    // [0]: version
    // [1]: num_actions (OOP)
    // [2]: num_actions (IP)
    // weights: oopLen + ipLen
    // normalizer: oopLen + ipLen
    // equity: oopLen + ipLen
    // ev: oopLen + ipLen
    // eqr: oopLen + ipLen
    // strategy: numActions * oopLen (for OOP)
    // action_ev: numActions * oopLen (for OOP) - 紧跟在策略后面
    
    let offset = 3;
    offset += oopLen + ipLen; // weights
    offset += oopLen + ipLen; // normalizer
    
    // 提取equity
    const equityOffset = offset;
    const heroEquitySolver = results[equityOffset + heroIndex];
    offset += oopLen + ipLen; // equity
    
    // 提取EV
    const evOffset = offset;
    const heroEV = results[evOffset + heroIndex];
    offset += oopLen + ipLen; // ev
    
    // 提取EQR
    const eqrOffset = offset;
    const heroEQR = results[eqrOffset + heroIndex];
    offset += oopLen + ipLen; // eqr
    
    // 提取策略
    const strategyOffset = offset;
    const strategy = {};
    for (let i = 0; i < numActions; i++) {
        strategy[actionList[i].name] = results[strategyOffset + i * oopLen + heroIndex];
    }
    offset += numActions * oopLen; // strategy
    
    // 提取动作EV (action_ev紧跟在strategy后面)
    const actionEvOffset = offset;
    const actionEV = {};
    for (let i = 0; i < numActions; i++) {
        actionEV[actionList[i].name] = results[actionEvOffset + i * oopLen + heroIndex];
    }
    
    // 计算加权EV = sum(strategy[action] * actionEV[action])
    let weightedEV = 0;
    for (const actionName of Object.keys(strategy)) {
        const prob = strategy[actionName] || 0;
        const ev = actionEV[actionName] || 0;
        weightedEV += prob * ev;
    }
    
    return { 
        strategy, actions: actionList, ev: heroEV, equity: heroEquitySolver, eqr: heroEQR,
        actionEV, weightedEV
    };
}


// 运行单个场景的求解
async function solveScenario(board, oopRange, ipRange) {
    try {
        return await solveRiver({
            oopRange, ipRange, board,
            startingPot: 100, effectiveStack: 500,  // 有效筹码500
            oopBetSizes: '33,50,75,100,120', ipBetSizes: '33,50,75,100,120',  // 下注尺寸33%,50%,75%,100%,120%
            oopRaiseSizes: '50,100', ipRaiseSizes: '50,100',  // 加注尺寸50%,100%
            targetExploitability: 0.1, maxIterations: 1000,  // 目标可剥削度0.1%
        });
    } catch (e) {
        return null;
    }
}

// 计算策略差异
function calculateStrategyDiff(strategy1, strategy2) {
    const keys = new Set([...Object.keys(strategy1), ...Object.keys(strategy2)]);
    let diff = 0;
    for (const key of keys) {
        diff += Math.abs((strategy1[key] || 0) - (strategy2[key] || 0));
    }
    return diff / keys.size;
}

// 主实验函数
async function runExperiment() {
    console.log('='.repeat(80));
    console.log('使用 wasm-postflop + OMPEval 进行跨公共牌四维度胜率-策略验证实验 V3');
    console.log('='.repeat(80));
    console.log('\n核心改进: 使用 OMPEval (C++) 计算范围胜率，速度比 poker-odds-calc 快约60倍');
    console.log('\n核心问题: 在不同的（公共牌+固定手牌+随机范围）组合下：');
    console.log('  当手牌胜率和范围胜率都相近时，策略是否相同？');
    console.log('\n范围策略: 每100组实验随机生成一次OOP和IP范围');
    
    const NUM_SCENARIOS = 100000;  // 10万次测试
    const RANGE_REFRESH_INTERVAL = 100;  // 每100组实验随机一次范围
    const scenarios = [];
    const usedRanges = [];  // 记录使用过的范围
    const failStats = { noHeroHand: 0, solverFailed: 0, strategyExtractFailed: 0, otherError: 0 };  // 失败统计
    
    console.log(`\n生成 ${NUM_SCENARIOS} 个（公共牌+固定手牌+随机范围）场景...`);
    console.log(`范围刷新间隔: 每 ${RANGE_REFRESH_INTERVAL} 组实验`);
    console.log('使用 OMPEval 计算胜率，预计总时间约 15-20 分钟\n');
    
    const startTime = Date.now();
    
    // 当前使用的范围
    let currentOopRange = generateRandomRange(60, 120);
    let currentIpRange = generateRandomRange(40, 100);
    usedRanges.push({ oopRange: currentOopRange, ipRange: currentIpRange, startIdx: 0 });
    console.log(`初始范围 - OOP: ${currentOopRange.split(',').length}种手牌, IP: ${currentIpRange.split(',').length}种手牌`);
    
    for (let i = 0; i < NUM_SCENARIOS; i++) {
        // 每100组实验随机一次范围
        if (i > 0 && i % RANGE_REFRESH_INTERVAL === 0) {
            currentOopRange = generateRandomRange(60, 120);
            currentIpRange = generateRandomRange(40, 100);
            usedRanges.push({ oopRange: currentOopRange, ipRange: currentIpRange, startIdx: i });
            console.log(`\n[范围刷新 #${usedRanges.length}] OOP: ${currentOopRange.split(',').length}种手牌, IP: ${currentIpRange.split(',').length}种手牌`);
        }
        
        const board = generateRandomBoard();
        const heroHandInfo = selectRandomHeroHand(board, currentOopRange);
        
        if (!heroHandInfo) {
            failStats.noHeroHand++;
            continue;
        }
        
        // 进度输出
        if ((i + 1) % 100 === 0 || i === 0) {
            const elapsed = (Date.now() - startTime) / 1000;
            const rate = elapsed > 0 ? (i + 1) / elapsed : 1;
            const remaining = (NUM_SCENARIOS - i - 1) / rate;
            const successRate = scenarios.length > 0 ? (scenarios.length / (i + 1) * 100).toFixed(1) : '0.0';
            console.log(`进度: ${i+1}/${NUM_SCENARIOS} - 成功: ${scenarios.length} (${successRate}%) - 已用时: ${elapsed.toFixed(0)}s - 预计剩余: ${remaining.toFixed(0)}s`);
        }
        
        // 死牌 = OOP固定手牌
        const deadCards = heroHandInfo.cards.join('');
        
        try {
            // 1. 计算手牌vs IP范围的胜率
            const heroEquity = calculateEquityOmpeval(heroHandInfo.hand, currentIpRange, board, '');
            
            // 2. 计算OOP范围 vs IP范围的胜率（使用死牌=固定手牌）
            const rangeEquity = calculateEquityOmpeval(currentOopRange, currentIpRange, board, deadCards);
            
            // 3. 使用solver获取策略
            const solverResult = await solveScenario(board, currentOopRange, currentIpRange);
            if (!solverResult) {
                failStats.solverFailed++;
                continue;
            }
            
            const strategyData = extractStrategy(solverResult, heroHandInfo.cards);
            if (!strategyData) {
                failStats.strategyExtractFailed++;
                continue;
            }
            
            // 详细日志只在前10个场景输出
            if (i < 10) {
                console.log(`  场景 ${i+1}: ${board} + ${heroHandInfo.hand}`);
                console.log(`    手牌: 胜率=${(heroEquity.winRate*100).toFixed(3)}%, 平局率=${(heroEquity.tieRate*100).toFixed(3)}%`);
                console.log(`    范围: 胜率=${(rangeEquity.winRate*100).toFixed(3)}%, 平局率=${(rangeEquity.tieRate*100).toFixed(3)}%`);
            }
            
            scenarios.push({
                board, heroHand: heroHandInfo.hand, heroCards: heroHandInfo.cards,
                oopRange: currentOopRange, ipRange: currentIpRange,  // 记录使用的范围
                heroEquity, rangeEquity, strategy: strategyData.strategy, actions: strategyData.actions,
                ev: strategyData.ev, solverEquity: strategyData.equity, eqr: strategyData.eqr,
                actionEV: strategyData.actionEV, weightedEV: strategyData.weightedEV,
            });
        } catch (e) {
            failStats.otherError++;
            if (i < 10) console.log(`  错误: ${e.message}`);
        }
    }
    
    const totalTime = (Date.now() - startTime) / 1000;
    console.log(`\n\n成功生成 ${scenarios.length} 个场景，总用时: ${totalTime.toFixed(1)}s`);
    console.log(`使用了 ${usedRanges.length} 组不同的范围`);
    console.log(`\n失败统计:`);
    console.log(`  无有效手牌: ${failStats.noHeroHand}`);
    console.log(`  Solver求解失败: ${failStats.solverFailed}`);
    console.log(`  策略提取失败: ${failStats.strategyExtractFailed}`);
    console.log(`  其他错误: ${failStats.otherError}`);
    console.log(`  总失败: ${failStats.noHeroHand + failStats.solverFailed + failStats.strategyExtractFailed + failStats.otherError}`);
    
    analyzeResults(scenarios, usedRanges);
}


// 分析结果
function analyzeResults(scenarios, usedRanges) {
    console.log('\n' + '='.repeat(80));
    console.log('分析结果：寻找四维度胜率相近的场景对（阈值: 0.05%）');
    console.log('条件：手牌胜率、手牌平局率、范围胜率、范围平局率 都相差不超过0.05%');
    console.log(`使用了 ${usedRanges.length} 组不同的随机范围`);
    console.log('='.repeat(80));
    
    const threshold = 0.0005; // 0.05%
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
                const strategyDiff = calculateStrategyDiff(s1.strategy, s2.strategy);
                pairs.push({ s1, s2, heroWinDiff, heroTieDiff, rangeWinDiff, rangeTieDiff, strategyDiff });
            }
        }
    }
    
    console.log(`\n找到 ${pairs.length} 对四维度胜率相近的场景`);
    
    const significantPairs = pairs.filter(p => p.strategyDiff > 0.15);
    console.log(`其中策略差异显著(>15%)的: ${significantPairs.length} 对`);
    
    if (significantPairs.length > 0) {
        console.log('\n【策略差异显著的反例】');
        for (const p of significantPairs.slice(0, 10)) {
            console.log(`\n${'─'.repeat(70)}`);
            console.log(`【场景1】 ${p.s1.board} + ${p.s1.heroHand}`);
            console.log(`  手牌: 胜率=${(p.s1.heroEquity.winRate*100).toFixed(3)}%, 平局率=${(p.s1.heroEquity.tieRate*100).toFixed(3)}%`);
            console.log(`  范围: 胜率=${(p.s1.rangeEquity.winRate*100).toFixed(3)}%, 平局率=${(p.s1.rangeEquity.tieRate*100).toFixed(3)}%`);
            console.log(`  策略: ${JSON.stringify(p.s1.strategy)}`);
            console.log(`【场景2】 ${p.s2.board} + ${p.s2.heroHand}`);
            console.log(`  手牌: 胜率=${(p.s2.heroEquity.winRate*100).toFixed(3)}%, 平局率=${(p.s2.heroEquity.tieRate*100).toFixed(3)}%`);
            console.log(`  范围: 胜率=${(p.s2.rangeEquity.winRate*100).toFixed(3)}%, 平局率=${(p.s2.rangeEquity.tieRate*100).toFixed(3)}%`);
            console.log(`  策略: ${JSON.stringify(p.s2.strategy)}`);
            console.log(`【对比】 策略差异: ${(p.strategyDiff*100).toFixed(1)}%`);
        }
    }
    
    // 保存结果
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    
    const outputData = {
        method: 'OMPEval + wasm-postflop (随机范围)',
        timestamp,
        usedRanges,
        numScenarios: scenarios.length, numPairs: pairs.length, numSignificantPairs: significantPairs.length,
        scenarios, pairs: pairs.map(p => ({
            board1: p.s1.board, heroHand1: p.s1.heroHand, oopRange1: p.s1.oopRange, ipRange1: p.s1.ipRange,
            board2: p.s2.board, heroHand2: p.s2.heroHand, oopRange2: p.s2.oopRange, ipRange2: p.s2.ipRange,
            strategyDiff: p.strategyDiff,
        })),
    };
    
    // 使用时间戳保存本次数据
    const jsonFilename = `experiments/results/wasm_postflop_validation_v3_${timestamp}.json`;
    writeFileSync(jsonFilename, JSON.stringify(outputData, null, 2));
    console.log(`\n本次结果已保存到: ${jsonFilename}`);
    
    // 同时保存到固定文件名（最新结果）
    writeFileSync('experiments/results/wasm_postflop_validation_v3_latest.json', JSON.stringify(outputData, null, 2));
    console.log('最新结果已保存到: experiments/results/wasm_postflop_validation_v3_latest.json');
    
    // 累计数据：合并历史数据
    const cumulativeFile = 'experiments/results/wasm_postflop_validation_v3_cumulative.json';
    let cumulativeData = {
        method: 'OMPEval + wasm-postflop (随机范围) - 累计数据',
        runs: [],
        totalScenarios: 0,
        totalPairs: 0,
        totalSignificantPairs: 0,
        allSignificantPairs: [],  // 所有反例
    };
    
    // 读取已有的累计数据
    if (existsSync(cumulativeFile)) {
        try {
            const existingData = JSON.parse(readFileSync(cumulativeFile, 'utf8'));
            cumulativeData = existingData;
            console.log(`\n读取已有累计数据: ${cumulativeData.runs.length} 次运行, ${cumulativeData.totalScenarios} 个场景`);
        } catch (e) {
            console.log('读取累计数据失败，将创建新文件');
        }
    }
    
    // 添加本次运行数据
    cumulativeData.runs.push({
        timestamp,
        numScenarios: scenarios.length,
        numPairs: pairs.length,
        numSignificantPairs: significantPairs.length,
    });
    cumulativeData.totalScenarios += scenarios.length;
    cumulativeData.totalPairs += pairs.length;
    cumulativeData.totalSignificantPairs += significantPairs.length;
    
    // 添加本次的反例（带时间戳）
    for (const p of significantPairs) {
        cumulativeData.allSignificantPairs.push({
            timestamp,
            board1: p.s1.board, heroHand1: p.s1.heroHand, 
            oopRange1: p.s1.oopRange, ipRange1: p.s1.ipRange,
            strategy1: p.s1.strategy, actionEV1: p.s1.actionEV,
            heroEquity1: p.s1.heroEquity, rangeEquity1: p.s1.rangeEquity,
            board2: p.s2.board, heroHand2: p.s2.heroHand,
            oopRange2: p.s2.oopRange, ipRange2: p.s2.ipRange,
            strategy2: p.s2.strategy, actionEV2: p.s2.actionEV,
            heroEquity2: p.s2.heroEquity, rangeEquity2: p.s2.rangeEquity,
            strategyDiff: p.strategyDiff,
            heroWinDiff: p.heroWinDiff, heroTieDiff: p.heroTieDiff,
            rangeWinDiff: p.rangeWinDiff, rangeTieDiff: p.rangeTieDiff,
        });
    }
    
    // 保存累计数据
    writeFileSync(cumulativeFile, JSON.stringify(cumulativeData, null, 2));
    console.log(`\n累计数据已更新: ${cumulativeFile}`);
    console.log(`  总运行次数: ${cumulativeData.runs.length}`);
    console.log(`  总场景数: ${cumulativeData.totalScenarios}`);
    console.log(`  总反例数: ${cumulativeData.totalSignificantPairs}`);
    
    generateReport(scenarios, pairs, significantPairs, usedRanges, timestamp, cumulativeData);
}


// 生成报告
function generateReport(scenarios, pairs, significantPairs, usedRanges, timestamp = '', cumulativeData = null) {
    let report = `# 跨公共牌四维度胜率-策略验证实验报告 V3

## 实验时间

${timestamp || new Date().toISOString()}

## 累计统计

| 指标 | 本次 | 累计 |
|------|------|------|
| 运行次数 | 1 | ${cumulativeData ? cumulativeData.runs.length : 1} |
| 场景数 | ${scenarios.length} | ${cumulativeData ? cumulativeData.totalScenarios : scenarios.length} |
| 胜率相近对数 | ${pairs.length} | ${cumulativeData ? cumulativeData.totalPairs : pairs.length} |
| 策略差异显著反例 | ${significantPairs.length} | ${cumulativeData ? cumulativeData.totalSignificantPairs : significantPairs.length} |

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
| 下注尺寸 | 33%, 50%, 75%, 100%, 120% pot |
| 加注尺寸 | 50%, 100% pot |
| 目标可剥削度 | 0.1% |
| 最大迭代次数 | 1000 |

## 范围策略

- **范围刷新间隔**: 每100组实验随机生成一次
- **OOP范围大小**: 60-120种手牌
- **IP范围大小**: 40-100种手牌
- **使用的范围组数**: ${usedRanges.length}

## 实验规模

- 生成场景数: ${scenarios.length}
- 四维度胜率相近的场景对（差异<0.05%）: ${pairs.length}
- 策略差异显著(>15%)的场景对: ${significantPairs.length}

## 关键发现

`;
    
    if (significantPairs.length > 0) {
        const ratio = pairs.length > 0 ? (significantPairs.length / pairs.length * 100).toFixed(1) : 0;
        report += `### ⚠️ 四维度胜率标量不足以决定最优策略

在 ${pairs.length} 对四维度胜率相近的场景中，有 ${significantPairs.length} 对（${ratio}%）的策略差异显著。

**结论：即使手牌胜率、手牌平局率、范围胜率、范围平局率都精确匹配（差异<0.05%），最优策略仍然可能完全不同。**

### 策略差异显著的反例

`;
        // 添加反例详情
        let caseNum = 1;
        for (const p of significantPairs.slice(0, 20)) {
            // 格式化动作EV
            const formatActionEV = (actionEV) => {
                if (!actionEV) return 'N/A';
                return Object.entries(actionEV)
                    .map(([k, v]) => `${k}:${v !== undefined ? v.toFixed(2) : 'N/A'}`)
                    .join(', ');
            };
            
            // 计算加权EV差异
            const weightedEVDiff = Math.abs((p.s1.weightedEV || 0) - (p.s2.weightedEV || 0));
            
            // 检查是否使用相同范围
            const sameRange = p.s1.oopRange === p.s2.oopRange && p.s1.ipRange === p.s2.ipRange;
            
            report += `---

#### 反例 ${caseNum}

**场景1:**
- 公共牌: \`${p.s1.board}\`
- OOP手牌: \`${p.s1.heroHand}\`
- OOP范围 (${p.s1.oopRange ? p.s1.oopRange.split(',').length : 0}种): \`${p.s1.oopRange || 'N/A'}\`
- IP范围 (${p.s1.ipRange ? p.s1.ipRange.split(',').length : 0}种): \`${p.s1.ipRange || 'N/A'}\`
- 手牌: 胜率=${(p.s1.heroEquity.winRate*100).toFixed(3)}%, 平局率=${(p.s1.heroEquity.tieRate*100).toFixed(3)}%
- 范围: 胜率=${(p.s1.rangeEquity.winRate*100).toFixed(3)}%, 平局率=${(p.s1.rangeEquity.tieRate*100).toFixed(3)}%
- EV: ${p.s1.ev !== undefined ? p.s1.ev.toFixed(2) : 'N/A'}, Solver Equity: ${p.s1.solverEquity !== undefined ? (p.s1.solverEquity*100).toFixed(2) + '%' : 'N/A'}
- 动作EV: ${formatActionEV(p.s1.actionEV)}
- 加权EV: ${p.s1.weightedEV !== undefined ? p.s1.weightedEV.toFixed(2) : 'N/A'}
- 策略: \`${JSON.stringify(p.s1.strategy)}\`

**场景2:**
- 公共牌: \`${p.s2.board}\`
- OOP手牌: \`${p.s2.heroHand}\`
- OOP范围 (${p.s2.oopRange ? p.s2.oopRange.split(',').length : 0}种): \`${p.s2.oopRange || 'N/A'}\`
- IP范围 (${p.s2.ipRange ? p.s2.ipRange.split(',').length : 0}种): \`${p.s2.ipRange || 'N/A'}\`
- 手牌: 胜率=${(p.s2.heroEquity.winRate*100).toFixed(3)}%, 平局率=${(p.s2.heroEquity.tieRate*100).toFixed(3)}%
- 范围: 胜率=${(p.s2.rangeEquity.winRate*100).toFixed(3)}%, 平局率=${(p.s2.rangeEquity.tieRate*100).toFixed(3)}%
- EV: ${p.s2.ev !== undefined ? p.s2.ev.toFixed(2) : 'N/A'}, Solver Equity: ${p.s2.solverEquity !== undefined ? (p.s2.solverEquity*100).toFixed(2) + '%' : 'N/A'}
- 动作EV: ${formatActionEV(p.s2.actionEV)}
- 加权EV: ${p.s2.weightedEV !== undefined ? p.s2.weightedEV.toFixed(2) : 'N/A'}
- 策略: \`${JSON.stringify(p.s2.strategy)}\`

**对比:**
- 范围相同: ${sameRange ? '是' : '否'}
- 手牌胜率差异: ${(p.heroWinDiff*100).toFixed(3)}%, 手牌平局率差异: ${(p.heroTieDiff*100).toFixed(3)}%
- 范围胜率差异: ${(p.rangeWinDiff*100).toFixed(3)}%, 范围平局率差异: ${(p.rangeTieDiff*100).toFixed(3)}%
- **策略差异: ${(p.strategyDiff*100).toFixed(1)}%**
- **加权EV差异: ${weightedEVDiff.toFixed(2)}**

`;
            caseNum++;
        }
    } else if (pairs.length > 0) {
        report += `### ✅ 未发现反例

在找到的 ${pairs.length} 对四维度胜率相近的场景中，未发现策略显著不同的反例。
`;
    } else {
        report += `### 需要更多数据或调整阈值

在 ${scenarios.length} 个场景中，未找到四维度胜率都相近（差异<0.05%）的场景对。
`;
    }
    
    // 使用时间戳保存报告
    const reportFilename = timestamp 
        ? `experiments/results/wasm_postflop_validation_v3_report_${timestamp}.md`
        : 'experiments/results/wasm_postflop_validation_v3_report.md';
    writeFileSync(reportFilename, report);
    console.log(`报告已保存到: ${reportFilename}`);
    
    // 同时保存到固定文件名（最新报告）
    writeFileSync('experiments/results/wasm_postflop_validation_v3_report_latest.md', report);
    console.log('最新报告已保存到: experiments/results/wasm_postflop_validation_v3_report_latest.md');
}

// 运行实验
runExperiment().catch(console.error);
