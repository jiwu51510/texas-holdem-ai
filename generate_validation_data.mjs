#!/usr/bin/env node
/**
 * 数据生成脚本 - 使用 wasm-postflop + OMPEval 生成验证数据
 * 
 * 功能：生成（公共牌+固定手牌+随机范围）场景数据，保存到本地 JSON 文件
 * 用法：node generate_validation_data.mjs [场景数量] [范围刷新间隔]
 */

import { solveRiver } from './solver_tools/postflop_solver.mjs';
import { parseBoardString } from './solver_tools/equity_calculator.mjs';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { execSync } from 'child_process';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// OMPEval路径
const OMPEVAL_PATH = join(__dirname, 'solver_tools', 'OMPEval', 'equity_calc');

// 数据存储目录
const DATA_DIR = join(__dirname, 'experiments', 'validation_data');

// 所有可能的手牌类型
const ALL_HANDS = [];
const RANKS = '23456789TJQKA';

// 初始化所有手牌类型
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

/**
 * 生成随机范围
 */
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
                const startRank = ranks.indexOf(start[0]);
                const endRank = ranks.indexOf(end[0]);
                const minRank = Math.min(startRank, endRank);
                const maxRank = Math.max(startRank, endRank);
                for (let r = minRank; r <= maxRank; r++) {
                    result.push(ranks[r] + ranks[r]);
                }
            } else if (start.endsWith('s')) {
                const highRank = ranks.indexOf(start[0]);
                const startLow = ranks.indexOf(start[1]);
                const endLow = ranks.indexOf(end[1]);
                const minLow = Math.min(startLow, endLow);
                const maxLow = Math.max(startLow, endLow);
                for (let r = minLow; r <= maxLow; r++) {
                    if (r !== highRank) result.push(ranks[highRank] + ranks[r] + 's');
                }
            } else if (start.endsWith('o')) {
                const highRank = ranks.indexOf(start[0]);
                const startLow = ranks.indexOf(start[1]);
                const endLow = ranks.indexOf(end[1]);
                const minLow = Math.min(startLow, endLow);
                const maxLow = Math.max(startLow, endLow);
                for (let r = minLow; r <= maxLow; r++) {
                    if (r !== highRank) result.push(ranks[highRank] + ranks[r] + 'o');
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
        
        if (total === 0) {
            return { equity: 0.5, winRate: 0.5, tieRate: 0, wins: 0, ties: 0, total: 0 };
        }
        
        return {
            equity: (wins + ties * 0.5) / total,
            winRate: wins / total,
            tieRate: ties / total,
            wins, ties, total
        };
    } catch (e) {
        throw new Error(`OMPEval error: ${e.message}`);
    }
}

/**
 * 生成随机公共牌
 */
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
 * 将手牌类型展开为具体的手牌组合
 */
function expandHandType(handType) {
    const suits = 'cdhs';
    const result = [];
    
    if (handType.length === 2 && handType[0] === handType[1]) {
        const r = handType[0];
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = s1 + 1; s2 < 4; s2++) {
                const card1 = r + suits[s1];
                const card2 = r + suits[s2];
                result.push({ hand: card1 + card2, cards: [card1, card2] });
            }
        }
    } else if (handType.endsWith('s')) {
        const r1 = handType[0];
        const r2 = handType[1];
        for (let s = 0; s < 4; s++) {
            const card1 = r1 + suits[s];
            const card2 = r2 + suits[s];
            result.push({ hand: card1 + card2, cards: [card1, card2] });
        }
    } else if (handType.endsWith('o')) {
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
        const r1 = handType[0];
        const r2 = handType[1];
        for (let s = 0; s < 4; s++) {
            result.push({ hand: r1 + suits[s] + r2 + suits[s], cards: [r1 + suits[s], r2 + suits[s]] });
        }
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

/**
 * 过滤范围，移除与公共牌或死牌冲突的手牌类型
 * @param {string} rangeStr - 原始范围字符串
 * @param {string} board - 公共牌
 * @param {string[]} deadCards - 额外的死牌（如英雄手牌）
 * @returns {Object} - { range: 过滤后的范围字符串, validCombos: 有效组合数量, totalCombos: 原始组合数量 }
 */
function filterRange(rangeStr, board, deadCards = []) {
    const boardCards = new Set(parseBoardString(board));
    const allDeadCards = new Set([...boardCards, ...deadCards]);
    
    const handTypes = rangeStr.split(',').map(h => h.trim());
    const validHandTypes = [];
    let validCombos = 0;
    let totalCombos = 0;
    
    for (const handType of handTypes) {
        const expanded = expandHandType(handType);
        totalCombos += expanded.length;
        
        // 计算有效组合数量
        const validCount = expanded.filter(hand => 
            !allDeadCards.has(hand.cards[0]) && !allDeadCards.has(hand.cards[1])
        ).length;
        
        validCombos += validCount;
        
        // 如果有任何有效组合，保留该手牌类型
        if (validCount > 0) {
            validHandTypes.push(handType);
        }
    }
    
    return {
        range: validHandTypes.join(','),
        validCombos,
        totalCombos,
        handTypes: validHandTypes.length,
    };
}

/**
 * 展开范围为具体的有效手牌组合列表（过滤掉与死牌冲突的组合）
 * @param {string} rangeStr - 原始范围字符串
 * @param {string} board - 公共牌
 * @param {string[]} deadCards - 额外的死牌（如英雄手牌）
 * @returns {string[]} - 有效的具体手牌组合列表，如 ['AhKh', 'AsKs', ...]
 */
function expandRangeToValidCombos(rangeStr, board, deadCards = []) {
    const boardCards = new Set(parseBoardString(board));
    const allDeadCards = new Set([...boardCards, ...deadCards]);
    
    const handTypes = rangeStr.split(',').map(h => h.trim());
    const validCombos = [];
    
    for (const handType of handTypes) {
        const expanded = expandHandType(handType);
        for (const hand of expanded) {
            // 只保留不与死牌冲突的组合
            if (!allDeadCards.has(hand.cards[0]) && !allDeadCards.has(hand.cards[1])) {
                validCombos.push(hand.hand);
            }
        }
    }
    
    return validCombos;
}

/**
 * 从OOP范围中随机选择一个不与公共牌冲突的手牌
 */
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

/**
 * 在oopCards中找到手牌索引
 */
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

/**
 * 从solver结果中提取特定手牌的策略、EV和动作EV
 */
function extractStrategy(solverResult, heroCards) {
    const { oopCards, ipCards, results, numActions, actions } = solverResult;
    const heroIndex = findHandIndex(oopCards, heroCards);
    if (heroIndex === -1) return null;
    
    // 解析动作列表，保留完整的动作名称（包括尺寸）
    const actionList = actions.split('/').map(a => {
        const match = a.match(/(\w+):(\d+)/);
        if (match) {
            // 带尺寸的动作，如 "Bet:33" -> { name: "Bet:33", type: "Bet", amount: 33 }
            return { name: a, type: match[1], amount: parseInt(match[2]) };
        } else {
            // 不带尺寸的动作，如 "Check" -> { name: "Check", type: "Check", amount: 0 }
            return { name: a, type: a, amount: 0 };
        }
    });
    
    const oopLen = oopCards.length;
    const ipLen = ipCards.length;
    
    let offset = 3;
    offset += oopLen + ipLen; // weights
    offset += oopLen + ipLen; // normalizer
    
    const equityOffset = offset;
    const heroEquitySolver = results[equityOffset + heroIndex];
    offset += oopLen + ipLen;
    
    const evOffset = offset;
    const heroEV = results[evOffset + heroIndex];
    offset += oopLen + ipLen;
    
    const eqrOffset = offset;
    const heroEQR = results[eqrOffset + heroIndex];
    offset += oopLen + ipLen;
    
    // 使用完整的动作名称作为 key（如 "Bet:33", "Bet:50" 等）
    const strategyOffset = offset;
    const strategy = {};
    for (let i = 0; i < numActions; i++) {
        strategy[actionList[i].name] = results[strategyOffset + i * oopLen + heroIndex];
    }
    offset += numActions * oopLen;
    
    const actionEvOffset = offset;
    const actionEV = {};
    for (let i = 0; i < numActions; i++) {
        actionEV[actionList[i].name] = results[actionEvOffset + i * oopLen + heroIndex];
    }
    
    let weightedEV = 0;
    for (const actionName of Object.keys(strategy)) {
        weightedEV += (strategy[actionName] || 0) * (actionEV[actionName] || 0);
    }
    
    return { strategy, actions: actionList, ev: heroEV, equity: heroEquitySolver, eqr: heroEQR, actionEV, weightedEV };
}

/**
 * 运行单个场景的求解
 */
async function solveScenario(board, oopRange, ipRange) {
    try {
        return await solveRiver({
            oopRange, ipRange, board,
            startingPot: 100, effectiveStack: 500,
            oopBetSizes: '33,50,75,100,120', ipBetSizes: '33,50,75,100,120',
            oopRaiseSizes: '50,100', ipRaiseSizes: '50,100',
            targetExploitability: 0.1, maxIterations: 1000,
        });
    } catch (e) {
        return null;
    }
}


/**
 * 保存数据到文件
 */
function saveData(scenarios, usedRanges, failStats, timestamp, startTime, isFinal = false) {
    const totalTime = (Date.now() - startTime) / 1000;
    
    const outputData = {
        metadata: {
            timestamp,
            numScenarios: scenarios.length,
            numRanges: usedRanges.length,
            totalTime,
            failStats: { ...failStats },
            isFinal,
            solverParams: {
                startingPot: 100,
                effectiveStack: 500,
                oopBetSizes: '33,50,75,100,120',
                ipBetSizes: '33,50,75,100,120',
                oopRaiseSizes: '50,100',
                ipRaiseSizes: '50,100',
                targetExploitability: 0.1,
                maxIterations: 1000,
            }
        },
        usedRanges,
        scenarios,
    };
    
    // 保存到带时间戳的文件
    const dataFilename = join(DATA_DIR, `scenarios_${timestamp}.json`);
    writeFileSync(dataFilename, JSON.stringify(outputData, null, 2));
    
    // 同时保存到 latest 文件
    const latestFilename = join(DATA_DIR, 'scenarios_latest.json');
    writeFileSync(latestFilename, JSON.stringify(outputData, null, 2));
    
    console.log(`\n[保存] ${scenarios.length} 个场景已保存到: ${dataFilename}`);
    
    return dataFilename;
}

/**
 * 主数据生成函数
 */
async function generateData() {
    // 解析命令行参数
    const args = process.argv.slice(2);
    const NUM_SCENARIOS = parseInt(args[0]) || 1000;
    const RANGE_REFRESH_INTERVAL = parseInt(args[1]) || 100;
    const SAVE_INTERVAL = parseInt(args[2]) || 1000;  // 每N个场景保存一次，默认1000
    
    console.log('='.repeat(80));
    console.log('数据生成脚本 - 使用 wasm-postflop + OMPEval 生成验证数据');
    console.log('='.repeat(80));
    console.log(`\n场景数量: ${NUM_SCENARIOS}`);
    console.log(`范围刷新间隔: 每 ${RANGE_REFRESH_INTERVAL} 组实验`);
    console.log(`自动保存间隔: 每 ${SAVE_INTERVAL} 个成功场景`);
    
    // 确保数据目录存在
    if (!existsSync(DATA_DIR)) {
        mkdirSync(DATA_DIR, { recursive: true });
    }
    
    const scenarios = [];
    const usedRanges = [];
    const failStats = { noHeroHand: 0, solverFailed: 0, strategyExtractFailed: 0, otherError: 0 };
    
    const startTime = Date.now();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    let lastSaveCount = 0;  // 上次保存时的场景数量
    
    // 当前使用的范围
    let currentOopRange = generateRandomRange(60, 120);
    let currentIpRange = generateRandomRange(40, 100);
    usedRanges.push({ oopRange: currentOopRange, ipRange: currentIpRange, startIdx: 0 });
    console.log(`\n初始范围 - OOP: ${currentOopRange.split(',').length}种手牌, IP: ${currentIpRange.split(',').length}种手牌`);
    
    for (let i = 0; i < NUM_SCENARIOS; i++) {
        // 每N组实验随机一次范围
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
        
        const deadCards = heroHandInfo.cards.join('');
        
        try {
            // 过滤范围：移除与公共牌冲突的手牌类型（用于统计）
            const filteredOop = filterRange(currentOopRange, board);
            const filteredIp = filterRange(currentIpRange, board);
            
            // 展开范围为具体的有效组合列表（脚本自己过滤，不依赖solver/计算器）
            const oopValidCombos = expandRangeToValidCombos(currentOopRange, board);
            const ipValidCombos = expandRangeToValidCombos(currentIpRange, board);
            const ipValidCombosForHero = expandRangeToValidCombos(currentIpRange, board, heroHandInfo.cards);
            
            // 1. 计算手牌vs IP范围的胜率（使用过滤后的具体组合）
            const heroEquity = calculateEquityOmpeval(heroHandInfo.hand, ipValidCombosForHero.join(','), board, '');
            
            // 2. 计算OOP范围 vs IP范围的胜率（使用过滤后的具体组合，死牌=固定手牌）
            const rangeEquity = calculateEquityOmpeval(oopValidCombos.join(','), ipValidCombos.join(','), board, deadCards);
            
            // 3. 使用solver获取策略（使用过滤后的具体组合）
            const solverResult = await solveScenario(board, oopValidCombos.join(','), ipValidCombos.join(','));
            if (!solverResult) {
                failStats.solverFailed++;
                continue;
            }
            
            const strategyData = extractStrategy(solverResult, heroHandInfo.cards);
            if (!strategyData) {
                failStats.strategyExtractFailed++;
                continue;
            }
            
            scenarios.push({
                id: scenarios.length,
                board,
                heroHand: heroHandInfo.hand,
                heroCards: heroHandInfo.cards,
                // OOP范围保持手牌类型格式（不修改）
                oopRange: filteredOop.range,
                oopRangeStats: {
                    handTypes: filteredOop.handTypes,
                    validCombos: filteredOop.validCombos,
                    totalCombos: filteredOop.totalCombos,
                },
                // IP范围保存为逗号分隔的字符串（脚本自己过滤的具体组合）
                ipRange: ipValidCombos.join(','),
                ipRangeStats: {
                    handTypes: filteredIp.handTypes,
                    validCombos: ipValidCombos.length,
                    totalCombos: filteredIp.totalCombos,
                },
                // IP范围排除英雄手牌后的具体组合（逗号分隔字符串）
                ipRangeForHero: ipValidCombosForHero.join(','),
                ipRangeForHeroStats: {
                    validCombos: ipValidCombosForHero.length,
                },
                heroEquity,
                rangeEquity,
                strategy: strategyData.strategy,
                actions: strategyData.actions,
                ev: strategyData.ev,
                solverEquity: strategyData.equity,
                eqr: strategyData.eqr,
                actionEV: strategyData.actionEV,
                weightedEV: strategyData.weightedEV,
            });
            
            // 定期保存：每SAVE_INTERVAL个成功场景保存一次
            if (scenarios.length - lastSaveCount >= SAVE_INTERVAL) {
                saveData(scenarios, usedRanges, failStats, timestamp, startTime, false);
                lastSaveCount = scenarios.length;
            }
        } catch (e) {
            failStats.otherError++;
            if (i < 10) console.log(`  错误: ${e.message}`);
        }
    }
    
    const totalTime = (Date.now() - startTime) / 1000;
    
    console.log(`\n\n${'='.repeat(80)}`);
    console.log(`生成完成！`);
    console.log(`成功生成 ${scenarios.length} 个场景，总用时: ${totalTime.toFixed(1)}s`);
    console.log(`使用了 ${usedRanges.length} 组不同的范围`);
    console.log(`\n失败统计:`);
    console.log(`  无有效手牌: ${failStats.noHeroHand}`);
    console.log(`  Solver求解失败: ${failStats.solverFailed}`);
    console.log(`  策略提取失败: ${failStats.strategyExtractFailed}`);
    console.log(`  其他错误: ${failStats.otherError}`);
    
    // 最终保存
    const dataFilename = saveData(scenarios, usedRanges, failStats, timestamp, startTime, true);
    
    // 更新累计索引
    const indexFile = join(DATA_DIR, 'index.json');
    let indexData = { files: [], totalScenarios: 0 };
    if (existsSync(indexFile)) {
        try {
            indexData = JSON.parse(readFileSync(indexFile, 'utf8'));
        } catch (e) {
            console.log('读取索引文件失败，将创建新索引');
        }
    }
    
    // 检查是否已存在该文件记录（避免重复添加）
    const existingIdx = indexData.files.findIndex(f => f.filename === `scenarios_${timestamp}.json`);
    if (existingIdx === -1) {
        indexData.files.push({
            filename: `scenarios_${timestamp}.json`,
            timestamp,
            numScenarios: scenarios.length,
        });
        indexData.totalScenarios += scenarios.length;
    } else {
        // 更新已有记录
        indexData.totalScenarios = indexData.totalScenarios - indexData.files[existingIdx].numScenarios + scenarios.length;
        indexData.files[existingIdx].numScenarios = scenarios.length;
    }
    indexData.lastUpdated = timestamp;
    
    writeFileSync(indexFile, JSON.stringify(indexData, null, 2));
    console.log(`\n索引已更新: ${indexFile}`);
    console.log(`  总文件数: ${indexData.files.length}`);
    console.log(`  总场景数: ${indexData.totalScenarios}`);
}

// 运行
generateData().catch(console.error);
