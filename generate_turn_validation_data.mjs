#!/usr/bin/env node
/**
 * 转牌数据生成脚本 - 使用 wasm-postflop + OMPEval 生成转牌验证数据
 * 
 * 功能：生成（4张公共牌+固定手牌+随机范围）场景数据，保存到本地 JSON 文件
 * 用法：node generate_turn_validation_data.mjs [场景数量] [范围刷新间隔]
 * 
 * 与河牌数据生成的区别：
 * - 公共牌为4张（翻牌+转牌）
 * - 计算Potential直方图（枚举所有可能的河牌）
 * - Solver配置使用转牌阶段的bet/raise尺寸
 */

import { solveTurn } from './solver_tools/postflop_solver.mjs';
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
const DATA_DIR = join(__dirname, 'experiments', 'turn_validation_data');

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
        
        const validCount = expanded.filter(hand => 
            !allDeadCards.has(hand.cards[0]) && !allDeadCards.has(hand.cards[1])
        ).length;
        
        validCombos += validCount;
        
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
 * 展开范围为具体的有效手牌组合列表
 */
function expandRangeToValidCombos(rangeStr, board, deadCards = []) {
    const boardCards = new Set(parseBoardString(board));
    const allDeadCards = new Set([...boardCards, ...deadCards]);
    
    const handTypes = rangeStr.split(',').map(h => h.trim());
    const validCombos = [];
    
    for (const handType of handTypes) {
        const expanded = expandHandType(handType);
        for (const hand of expanded) {
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
 * 生成随机转牌公共牌（4张）
 */
function generateRandomTurnBoard() {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    const cards = [];
    const usedCards = new Set();
    
    while (cards.length < 4) {
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
 * 枚举所有可能的河牌
 */
function enumerateRiverCards(board, heroCards) {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    const boardCards = new Set(parseBoardString(board));
    const heroSet = new Set(heroCards);
    const riverCards = [];
    
    for (let r = 0; r < 13; r++) {
        for (let s = 0; s < 4; s++) {
            const cardStr = ranks[r] + suits[s];
            if (!boardCards.has(cardStr) && !heroSet.has(cardStr)) {
                riverCards.push(cardStr);
            }
        }
    }
    
    return riverCards;
}

/**
 * 使用OMPEval计算范围对范围的胜率
 */
function calculateEquityOmpeval(range1, range2, board, dead = '') {
    const boardCompact = board.replace(/\s+/g, '');
    
    let cmd = `"${OMPEVAL_PATH}" "${range1}" "${range2}" "${boardCompact}"`;
    if (dead) cmd += ` "${dead}"`;
    
    try {
        const output = execSync(cmd, { encoding: 'utf8', timeout: 30000 });
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
 * 计算直方图特征的辅助函数
 */
function computeHistogramFeatures(histogram, numBins) {
    const binWidth = 1.0 / numBins;
    const binCenters = Array.from({ length: numBins }, (_, i) => (i + 0.5) * binWidth);
    
    let meanEquity = 0;
    for (let i = 0; i < numBins; i++) {
        meanEquity += histogram[i] * binCenters[i];
    }
    
    let variance = 0;
    for (let i = 0; i < numBins; i++) {
        variance += histogram[i] * Math.pow(binCenters[i] - meanEquity, 2);
    }
    
    let entropy = 0;
    for (let i = 0; i < numBins; i++) {
        if (histogram[i] > 0) {
            entropy -= histogram[i] * Math.log2(histogram[i]);
        }
    }
    
    const sparsity = histogram.filter(h => h > 0).length / numBins;
    
    return { meanEquity, variance, std: Math.sqrt(variance), entropy, sparsity };
}

/**
 * 计算手牌vs范围的Potential直方图
 */
function calculatePotentialHistogram(heroHand, board, ipRange, heroCards, numBins = 50) {
    const riverCards = enumerateRiverCards(board, heroCards);
    const equities = [];
    
    for (const riverCard of riverCards) {
        const riverBoard = board + ' ' + riverCard;
        const ipValidCombos = expandRangeToValidCombos(ipRange, riverBoard, heroCards);
        
        if (ipValidCombos.length === 0) {
            equities.push(0.5);
            continue;
        }
        
        try {
            const result = calculateEquityOmpeval(heroHand, ipValidCombos.join(','), riverBoard, '');
            equities.push(result.equity);
        } catch (e) {
            equities.push(0.5);
        }
    }
    
    // 生成直方图
    const histogram = new Array(numBins).fill(0);
    const binWidth = 1.0 / numBins;
    
    for (const eq of equities) {
        const binIdx = Math.min(Math.floor(eq / binWidth), numBins - 1);
        histogram[binIdx]++;
    }
    
    // 归一化
    const total = equities.length;
    if (total > 0) {
        for (let i = 0; i < numBins; i++) {
            histogram[i] /= total;
        }
    }
    
    const features = computeHistogramFeatures(histogram, numBins);
    return { histogram, numRiverCards: riverCards.length, features };
}

/**
 * 计算范围vs范围的Potential直方图
 */
function calculateRangePotentialHistogram(oopRange, ipRange, board, heroCards, numBins = 50) {
    const riverCards = enumerateRiverCards(board, heroCards);
    const equities = [];
    
    for (const riverCard of riverCards) {
        const riverBoard = board + ' ' + riverCard;
        // 获取河牌后的有效范围（排除河牌，但不排除Hero手牌，因为这是范围vs范围）
        const oopValidCombos = expandRangeToValidCombos(oopRange, riverBoard);
        const ipValidCombos = expandRangeToValidCombos(ipRange, riverBoard);
        
        if (oopValidCombos.length === 0 || ipValidCombos.length === 0) {
            equities.push(0.5);
            continue;
        }
        
        try {
            // 范围 vs 范围，使用Hero手牌作为dead cards
            const result = calculateEquityOmpeval(
                oopValidCombos.join(','),
                ipValidCombos.join(','),
                riverBoard,
                heroCards.join('')
            );
            equities.push(result.equity);
        } catch (e) {
            equities.push(0.5);
        }
    }
    
    // 生成直方图
    const histogram = new Array(numBins).fill(0);
    const binWidth = 1.0 / numBins;
    
    for (const eq of equities) {
        const binIdx = Math.min(Math.floor(eq / binWidth), numBins - 1);
        histogram[binIdx]++;
    }
    
    // 归一化
    const total = equities.length;
    if (total > 0) {
        for (let i = 0; i < numBins; i++) {
            histogram[i] /= total;
        }
    }
    
    const features = computeHistogramFeatures(histogram, numBins);
    return { histogram, numRiverCards: riverCards.length, features };
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
    
    // 解析动作列表
    const actionList = actions.split('/').map(a => {
        const match = a.match(/(\w+):(\d+)/);
        if (match) {
            return { name: a, type: match[1], amount: parseInt(match[2]) };
        } else {
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
 * 运行转牌场景的求解
 */
async function solveScenario(board, oopRange, ipRange) {
    try {
        return await solveTurn({
            oopRange, ipRange, board,
            startingPot: 100, effectiveStack: 200,
            oopTurnBetSizes: '50,75', ipTurnBetSizes: '50,75',
            oopTurnRaiseSizes: '', ipTurnRaiseSizes: '',
            oopRiverBetSizes: '50,75', ipRiverBetSizes: '50,75',
            oopRiverRaiseSizes: '', ipRiverRaiseSizes: '',
            targetExploitability: 0.5, maxIterations: 300,
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
            stage: 'turn',
            numScenarios: scenarios.length,
            numRanges: usedRanges.length,
            totalTime,
            failStats: { ...failStats },
            isFinal,
            solverParams: {
                startingPot: 100,
                effectiveStack: 200,
                oopTurnBetSizes: '50,75',
                ipTurnBetSizes: '50,75',
                oopRiverBetSizes: '50,75',
                ipRiverBetSizes: '50,75',
                targetExploitability: 0.5,
                maxIterations: 300,
                numBins: 50,
            }
        },
        usedRanges,
        scenarios,
    };
    
    const dataFilename = join(DATA_DIR, `turn_scenarios_${timestamp}.json`);
    writeFileSync(dataFilename, JSON.stringify(outputData, null, 2));
    
    const latestFilename = join(DATA_DIR, 'turn_scenarios_latest.json');
    writeFileSync(latestFilename, JSON.stringify(outputData, null, 2));
    
    console.log(`\n[保存] ${scenarios.length} 个场景已保存到: ${dataFilename}`);
    
    return dataFilename;
}

/**
 * 主数据生成函数
 */
async function generateData() {
    const args = process.argv.slice(2);
    const NUM_SCENARIOS = parseInt(args[0]) || 1000;
    const RANGE_REFRESH_INTERVAL = parseInt(args[1]) || 100;
    const SAVE_INTERVAL = parseInt(args[2]) || 200;
    
    console.log('='.repeat(80));
    console.log('转牌数据生成脚本 - 使用 wasm-postflop + OMPEval 生成验证数据');
    console.log('='.repeat(80));
    console.log(`\n场景数量: ${NUM_SCENARIOS}`);
    console.log(`范围刷新间隔: 每 ${RANGE_REFRESH_INTERVAL} 组实验`);
    console.log(`自动保存间隔: 每 ${SAVE_INTERVAL} 个成功场景`);
    
    if (!existsSync(DATA_DIR)) {
        mkdirSync(DATA_DIR, { recursive: true });
    }
    
    const scenarios = [];
    const usedRanges = [];
    const failStats = { noHeroHand: 0, solverFailed: 0, strategyExtractFailed: 0, histogramFailed: 0, rangeHistogramFailed: 0, otherError: 0 };
    
    const startTime = Date.now();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    let lastSaveCount = 0;
    
    let currentOopRange = generateRandomRange(60, 120);
    let currentIpRange = generateRandomRange(40, 100);
    usedRanges.push({ oopRange: currentOopRange, ipRange: currentIpRange, startIdx: 0 });
    console.log(`\n初始范围 - OOP: ${currentOopRange.split(',').length}种手牌, IP: ${currentIpRange.split(',').length}种手牌`);
    
    for (let i = 0; i < NUM_SCENARIOS; i++) {
        if (i > 0 && i % RANGE_REFRESH_INTERVAL === 0) {
            currentOopRange = generateRandomRange(60, 120);
            currentIpRange = generateRandomRange(40, 100);
            usedRanges.push({ oopRange: currentOopRange, ipRange: currentIpRange, startIdx: i });
            console.log(`\n[范围刷新 #${usedRanges.length}] OOP: ${currentOopRange.split(',').length}种手牌, IP: ${currentIpRange.split(',').length}种手牌`);
        }
        
        const board = generateRandomTurnBoard();
        const heroHandInfo = selectRandomHeroHand(board, currentOopRange);
        
        if (!heroHandInfo) {
            failStats.noHeroHand++;
            continue;
        }
        
        if ((i + 1) % 20 === 0 || i === 0) {
            const elapsed = (Date.now() - startTime) / 1000;
            const rate = elapsed > 0 ? (i + 1) / elapsed : 1;
            const remaining = (NUM_SCENARIOS - i - 1) / rate;
            const successRate = scenarios.length > 0 ? (scenarios.length / (i + 1) * 100).toFixed(1) : '0.0';
            console.log(`进度: ${i+1}/${NUM_SCENARIOS} - 成功: ${scenarios.length} (${successRate}%) - 已用时: ${elapsed.toFixed(0)}s - 预计剩余: ${remaining.toFixed(0)}s`);
        }
        
        const deadCards = heroHandInfo.cards.join('');
        
        try {
            const filteredOop = filterRange(currentOopRange, board);
            const filteredIp = filterRange(currentIpRange, board);
            const oopValidCombos = expandRangeToValidCombos(currentOopRange, board);
            const ipValidCombos = expandRangeToValidCombos(currentIpRange, board);
            const ipValidCombosForHero = expandRangeToValidCombos(currentIpRange, board, heroHandInfo.cards);
            
            // 1. 计算手牌vs范围的Potential直方图
            let potentialData;
            try {
                potentialData = calculatePotentialHistogram(heroHandInfo.hand, board, currentIpRange, heroHandInfo.cards);
            } catch (e) {
                failStats.histogramFailed++;
                continue;
            }
            
            // 2. 计算范围vs范围的Potential直方图
            let rangePotentialData;
            try {
                rangePotentialData = calculateRangePotentialHistogram(currentOopRange, currentIpRange, board, heroHandInfo.cards);
            } catch (e) {
                failStats.rangeHistogramFailed++;
                continue;
            }
            
            // 3. 使用solver获取策略 - 注意：IP范围需要排除Hero手牌
            const solverResult = await solveScenario(board, oopValidCombos.join(','), ipValidCombosForHero.join(','));
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
                oopRange: filteredOop.range,
                oopRangeStats: { handTypes: filteredOop.handTypes, validCombos: filteredOop.validCombos, totalCombos: filteredOop.totalCombos },
                // ipRange: 未排除Hero手牌的原始IP范围
                ipRange: ipValidCombos.join(','),
                ipRangeStats: { handTypes: filteredIp.handTypes, validCombos: ipValidCombos.length, totalCombos: filteredIp.totalCombos },
                // ipRangeForHero: 排除Hero手牌的IP范围（用于Solver计算）
                ipRangeForHero: ipValidCombosForHero.join(','),
                ipRangeForHeroStats: { validCombos: ipValidCombosForHero.length },
                // 手牌 vs 范围的Potential直方图
                potentialHistogram: potentialData.histogram,
                potentialFeatures: potentialData.features,
                numRiverCards: potentialData.numRiverCards,
                // 范围 vs 范围的Potential直方图
                rangePotentialHistogram: rangePotentialData.histogram,
                rangePotentialFeatures: rangePotentialData.features,
                strategy: strategyData.strategy,
                actions: strategyData.actions,
                ev: strategyData.ev,
                solverEquity: strategyData.equity,
                eqr: strategyData.eqr,
                actionEV: strategyData.actionEV,
                weightedEV: strategyData.weightedEV,
                exploitability: solverResult.exploitability,
            });
            
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
    console.log(`  直方图计算失败: ${failStats.histogramFailed}`);
    console.log(`  范围直方图计算失败: ${failStats.rangeHistogramFailed}`);
    console.log(`  其他错误: ${failStats.otherError}`);
    
    const dataFilename = saveData(scenarios, usedRanges, failStats, timestamp, startTime, true);
    
    const indexFile = join(DATA_DIR, 'index.json');
    let indexData = { files: [], totalScenarios: 0 };
    if (existsSync(indexFile)) {
        try {
            indexData = JSON.parse(readFileSync(indexFile, 'utf8'));
        } catch (e) {
            console.log('读取索引文件失败，将创建新索引');
        }
    }
    
    const existingIdx = indexData.files.findIndex(f => f.filename === `turn_scenarios_${timestamp}.json`);
    if (existingIdx === -1) {
        indexData.files.push({ filename: `turn_scenarios_${timestamp}.json`, timestamp, numScenarios: scenarios.length });
        indexData.totalScenarios += scenarios.length;
    } else {
        indexData.totalScenarios = indexData.totalScenarios - indexData.files[existingIdx].numScenarios + scenarios.length;
        indexData.files[existingIdx].numScenarios = scenarios.length;
    }
    indexData.lastUpdated = timestamp;
    
    writeFileSync(indexFile, JSON.stringify(indexData, null, 2));
    console.log(`\n索引已更新: ${indexFile}`);
    console.log(`  总文件数: ${indexData.files.length}`);
    console.log(`  总场景数: ${indexData.totalScenarios}`);
}

generateData().catch(console.error);
