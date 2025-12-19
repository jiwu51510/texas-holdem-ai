#!/usr/bin/env node
/**
 * 并行数据生成脚本 - 使用 Worker Threads 并行处理
 * 
 * 用法：node generate_validation_data_parallel.mjs [场景数量] [并行数] [保存间隔]
 * 示例：node generate_validation_data_parallel.mjs 100000 12 1000
 */

import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { cpus } from 'os';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// 数据存储目录
const DATA_DIR = join(__dirname, 'experiments', 'validation_data');

// ============== Worker 线程代码 ==============
if (!isMainThread) {
    // Worker 线程：处理单个场景
    const { solveRiver } = await import('./solver_tools/postflop_solver.mjs');
    const { parseBoardString } = await import('./solver_tools/equity_calculator.mjs');
    const { execSync } = await import('child_process');
    
    const OMPEVAL_PATH = join(__dirname, 'solver_tools', 'OMPEval', 'equity_calc');
    const RANKS = '23456789TJQKA';
    const ALL_HANDS = [];
    
    // 初始化手牌类型
    for (let r = 12; r >= 0; r--) ALL_HANDS.push(RANKS[r] + RANKS[r]);
    for (let r1 = 12; r1 >= 1; r1--) {
        for (let r2 = r1 - 1; r2 >= 0; r2--) {
            ALL_HANDS.push(RANKS[r1] + RANKS[r2] + 's');
            ALL_HANDS.push(RANKS[r1] + RANKS[r2] + 'o');
        }
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
            const [r1, r2] = [handType[0], handType[1]];
            for (let s = 0; s < 4; s++) result.push({ hand: r1 + suits[s] + r2 + suits[s], cards: [r1 + suits[s], r2 + suits[s]] });
        } else if (handType.endsWith('o')) {
            const [r1, r2] = [handType[0], handType[1]];
            for (let s1 = 0; s1 < 4; s1++) {
                for (let s2 = 0; s2 < 4; s2++) {
                    if (s1 !== s2) result.push({ hand: r1 + suits[s1] + r2 + suits[s2], cards: [r1 + suits[s1], r2 + suits[s2]] });
                }
            }
        }
        return result;
    }
    
    function expandRangeToValidCombos(rangeStr, board, deadCards = []) {
        const boardCards = new Set(parseBoardString(board));
        const allDeadCards = new Set([...boardCards, ...deadCards]);
        const validCombos = [];
        for (const handType of rangeStr.split(',').map(h => h.trim())) {
            for (const hand of expandHandType(handType)) {
                if (!allDeadCards.has(hand.cards[0]) && !allDeadCards.has(hand.cards[1])) {
                    validCombos.push(hand.hand);
                }
            }
        }
        return validCombos;
    }
    
    function filterRange(rangeStr, board, deadCards = []) {
        const boardCards = new Set(parseBoardString(board));
        const allDeadCards = new Set([...boardCards, ...deadCards]);
        const handTypes = rangeStr.split(',').map(h => h.trim());
        const validHandTypes = [];
        let validCombos = 0, totalCombos = 0;
        for (const handType of handTypes) {
            const expanded = expandHandType(handType);
            totalCombos += expanded.length;
            const validCount = expanded.filter(h => !allDeadCards.has(h.cards[0]) && !allDeadCards.has(h.cards[1])).length;
            validCombos += validCount;
            if (validCount > 0) validHandTypes.push(handType);
        }
        return { range: validHandTypes.join(','), validCombos, totalCombos, handTypes: validHandTypes.length };
    }
    
    function selectRandomHeroHand(board, oopRange) {
        const boardCards = new Set(parseBoardString(board));
        const allHands = [];
        for (const handType of oopRange.split(',').map(h => h.trim())) {
            for (const hand of expandHandType(handType)) {
                if (!boardCards.has(hand.cards[0]) && !boardCards.has(hand.cards[1])) allHands.push(hand);
            }
        }
        return allHands.length > 0 ? allHands[Math.floor(Math.random() * allHands.length)] : null;
    }
    
    function generateRandomBoard() {
        const ranks = '23456789TJQKA', suits = 'cdhs';
        const cards = [], usedCards = new Set();
        while (cards.length < 5) {
            const cardStr = ranks[Math.floor(Math.random() * 13)] + suits[Math.floor(Math.random() * 4)];
            if (!usedCards.has(cardStr)) { usedCards.add(cardStr); cards.push(cardStr); }
        }
        return cards.join(' ');
    }
    
    function calculateEquityOmpeval(range1, range2, board, dead = '') {
        const boardCompact = board.replace(/\s+/g, '');
        let cmd = `"${OMPEVAL_PATH}" "${range1}" "${range2}" "${boardCompact}"`;
        if (dead) cmd += ` "${dead}"`;
        try {
            const output = execSync(cmd, { encoding: 'utf8' });
            const data = JSON.parse(output.replace(/nan/g, 'null'));
            const [wins, losses, ties] = [data.wins[0], data.wins[1], data.tieCount];
            const total = wins + losses + ties;
            if (total === 0) return { equity: 0.5, winRate: 0.5, tieRate: 0, wins: 0, ties: 0, total: 0 };
            return { equity: (wins + ties * 0.5) / total, winRate: wins / total, tieRate: ties / total, wins, ties, total };
        } catch (e) { throw new Error(`OMPEval error: ${e.message}`); }
    }
    
    function findHandIndex(oopCards, heroCards) {
        const ranks = '23456789TJQKA', suits = 'cdhs';
        const heroIndices = heroCards.map(c => ranks.indexOf(c[0]) * 4 + suits.indexOf(c[1])).sort((a, b) => a - b);
        for (let i = 0; i < oopCards.length; i++) {
            const [c1, c2] = [oopCards[i] & 0xFF, (oopCards[i] >> 8) & 0xFF].sort((a, b) => a - b);
            if (c1 === heroIndices[0] && c2 === heroIndices[1]) return i;
        }
        return -1;
    }
    
    function extractStrategy(solverResult, heroCards) {
        const { oopCards, ipCards, results, numActions, actions } = solverResult;
        const heroIndex = findHandIndex(oopCards, heroCards);
        if (heroIndex === -1) return null;
        
        // 解析动作列表，保留完整的动作名称（包括尺寸）
        const actionList = actions.split('/').map(a => {
            const m = a.match(/(\w+):(\d+)/);
            if (m) {
                // 带尺寸的动作，如 "Bet:33" -> { name: "Bet:33", type: "Bet", amount: 33 }
                return { name: a, type: m[1], amount: parseInt(m[2]) };
            } else {
                // 不带尺寸的动作，如 "Check" -> { name: "Check", type: "Check", amount: 0 }
                return { name: a, type: a, amount: 0 };
            }
        });
        
        const [oopLen, ipLen] = [oopCards.length, ipCards.length];
        let offset = 3 + (oopLen + ipLen) * 2;
        const heroEquitySolver = results[offset + heroIndex]; offset += oopLen + ipLen;
        const heroEV = results[offset + heroIndex]; offset += oopLen + ipLen;
        const heroEQR = results[offset + heroIndex]; offset += oopLen + ipLen;
        
        // 使用完整的动作名称作为 key（如 "Bet:33", "Bet:50" 等）
        const strategy = {}, actionEV = {};
        for (let i = 0; i < numActions; i++) {
            strategy[actionList[i].name] = results[offset + i * oopLen + heroIndex];
        }
        offset += numActions * oopLen;
        for (let i = 0; i < numActions; i++) {
            actionEV[actionList[i].name] = results[offset + i * oopLen + heroIndex];
        }
        
        let weightedEV = 0;
        for (const k of Object.keys(strategy)) weightedEV += (strategy[k] || 0) * (actionEV[k] || 0);
        return { strategy, actions: actionList, ev: heroEV, equity: heroEquitySolver, eqr: heroEQR, actionEV, weightedEV };
    }
    
    // Worker 主循环：处理任务
    parentPort.on('message', async (task) => {
        const { taskId, oopRange, ipRange } = task;
        try {
            const board = generateRandomBoard();
            const heroHandInfo = selectRandomHeroHand(board, oopRange);
            if (!heroHandInfo) {
                parentPort.postMessage({ taskId, error: 'noHeroHand' });
                return;
            }
            
            const filteredOop = filterRange(oopRange, board);
            const filteredIp = filterRange(ipRange, board);
            const oopValidCombos = expandRangeToValidCombos(oopRange, board);
            const ipValidCombos = expandRangeToValidCombos(ipRange, board);
            const ipValidCombosForHero = expandRangeToValidCombos(ipRange, board, heroHandInfo.cards);
            
            const heroEquity = calculateEquityOmpeval(heroHandInfo.hand, ipValidCombosForHero.join(','), board, '');
            const rangeEquity = calculateEquityOmpeval(oopValidCombos.join(','), ipValidCombos.join(','), board, heroHandInfo.cards.join(''));
            
            const solverResult = await solveRiver({
                oopRange: oopValidCombos.join(','), ipRange: ipValidCombos.join(','), board,
                startingPot: 100, effectiveStack: 500,
                oopBetSizes: '33,50,75,100,120', ipBetSizes: '33,50,75,100,120',
                oopRaiseSizes: '50,100', ipRaiseSizes: '50,100',
                targetExploitability: 0.1, maxIterations: 1000,
            });
            if (!solverResult) { parentPort.postMessage({ taskId, error: 'solverFailed' }); return; }
            
            const strategyData = extractStrategy(solverResult, heroHandInfo.cards);
            if (!strategyData) { parentPort.postMessage({ taskId, error: 'strategyExtractFailed' }); return; }
            
            parentPort.postMessage({
                taskId,
                result: {
                    board, heroHand: heroHandInfo.hand, heroCards: heroHandInfo.cards,
                    oopRange: filteredOop.range,
                    oopRangeStats: { handTypes: filteredOop.handTypes, validCombos: filteredOop.validCombos, totalCombos: filteredOop.totalCombos },
                    ipRange: ipValidCombos.join(','),
                    ipRangeStats: { handTypes: filteredIp.handTypes, validCombos: ipValidCombos.length, totalCombos: filteredIp.totalCombos },
                    ipRangeForHero: ipValidCombosForHero.join(','),
                    ipRangeForHeroStats: { validCombos: ipValidCombosForHero.length },
                    heroEquity, rangeEquity,
                    strategy: strategyData.strategy, actions: strategyData.actions,
                    ev: strategyData.ev, solverEquity: strategyData.equity, eqr: strategyData.eqr,
                    actionEV: strategyData.actionEV, weightedEV: strategyData.weightedEV,
                }
            });
        } catch (e) {
            parentPort.postMessage({ taskId, error: 'otherError', message: e.message });
        }
    });
}

// ============== 主线程代码 ==============
if (isMainThread) {
    const args = process.argv.slice(2);
    const NUM_SCENARIOS = parseInt(args[0]) || 100000;
    const NUM_WORKERS = parseInt(args[1]) || Math.min(12, cpus().length);
    const SAVE_INTERVAL = parseInt(args[2]) || 1000;
    const RANGE_REFRESH_INTERVAL = 100;
    
    console.log('='.repeat(80));
    console.log('并行数据生成脚本 - 使用 Worker Threads');
    console.log('='.repeat(80));
    console.log(`\nCPU 核心数: ${cpus().length}`);
    console.log(`并行 Worker 数: ${NUM_WORKERS}`);
    console.log(`场景数量: ${NUM_SCENARIOS}`);
    console.log(`自动保存间隔: 每 ${SAVE_INTERVAL} 个成功场景`);
    
    if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });
    
    const scenarios = [];
    const usedRanges = [];
    const failStats = { noHeroHand: 0, solverFailed: 0, strategyExtractFailed: 0, otherError: 0 };
    const startTime = Date.now();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    let lastSaveCount = 0;
    let tasksSent = 0, tasksCompleted = 0;
    
    // 生成随机范围
    const RANKS = '23456789TJQKA';
    const ALL_HANDS = [];
    for (let r = 12; r >= 0; r--) ALL_HANDS.push(RANKS[r] + RANKS[r]);
    for (let r1 = 12; r1 >= 1; r1--) {
        for (let r2 = r1 - 1; r2 >= 0; r2--) {
            ALL_HANDS.push(RANKS[r1] + RANKS[r2] + 's');
            ALL_HANDS.push(RANKS[r1] + RANKS[r2] + 'o');
        }
    }
    
    function generateRandomRange(minHands, maxHands) {
        const numHands = minHands + Math.floor(Math.random() * (maxHands - minHands + 1));
        const selectedHands = new Set();
        const strongHandsPool = ALL_HANDS.slice(0, Math.floor(ALL_HANDS.length * 0.3));
        while (selectedHands.size < Math.floor(numHands * 0.4) && strongHandsPool.length > 0) {
            selectedHands.add(strongHandsPool[Math.floor(Math.random() * strongHandsPool.length)]);
        }
        while (selectedHands.size < numHands) selectedHands.add(ALL_HANDS[Math.floor(Math.random() * ALL_HANDS.length)]);
        return Array.from(selectedHands).join(',');
    }
    
    let batchNumber = 0;
    let batchScenarios = [];
    const savedBatches = [];
    
    function saveBatch(isFinal = false) {
        if (batchScenarios.length === 0 && !isFinal) return;
        
        const totalTime = (Date.now() - startTime) / 1000;
        const batchFilename = `scenarios_${timestamp}_batch${String(batchNumber).padStart(3, '0')}.json`;
        
        const outputData = {
            metadata: { 
                timestamp, 
                batchNumber,
                numScenarios: batchScenarios.length, 
                totalScenariosGenerated: scenarios.length,
                numRanges: usedRanges.length, 
                totalTime, 
                failStats: { ...failStats }, 
                isFinal,
                solverParams: { startingPot: 100, effectiveStack: 500, oopBetSizes: '33,50,75,100,120', ipBetSizes: '33,50,75,100,120', oopRaiseSizes: '50,100', ipRaiseSizes: '50,100', targetExploitability: 0.1, maxIterations: 1000 }
            },
            scenarios: batchScenarios,
        };
        
        writeFileSync(join(DATA_DIR, batchFilename), JSON.stringify(outputData, null, 2));
        savedBatches.push({ filename: batchFilename, numScenarios: batchScenarios.length });
        console.log(`\n[保存] 批次 ${batchNumber}: ${batchScenarios.length} 个场景 -> ${batchFilename}`);
        
        batchScenarios = [];
        batchNumber++;
    }
    
    // 创建 Workers
    const workers = [];
    const pendingTasks = new Map();
    let currentOopRange = generateRandomRange(60, 120);
    let currentIpRange = generateRandomRange(40, 100);
    usedRanges.push({ oopRange: currentOopRange, ipRange: currentIpRange, startIdx: 0 });
    
    function sendTask(worker) {
        if (tasksSent >= NUM_SCENARIOS) return false;
        
        // 范围刷新
        if (tasksSent > 0 && tasksSent % RANGE_REFRESH_INTERVAL === 0) {
            currentOopRange = generateRandomRange(60, 120);
            currentIpRange = generateRandomRange(40, 100);
            usedRanges.push({ oopRange: currentOopRange, ipRange: currentIpRange, startIdx: tasksSent });
        }
        
        const taskId = tasksSent++;
        pendingTasks.set(taskId, Date.now());
        worker.postMessage({ taskId, oopRange: currentOopRange, ipRange: currentIpRange });
        return true;
    }
    
    function handleResult(msg) {
        tasksCompleted++;
        pendingTasks.delete(msg.taskId);
        
        if (msg.error) {
            failStats[msg.error] = (failStats[msg.error] || 0) + 1;
        } else {
            const scenario = { id: scenarios.length, ...msg.result };
            scenarios.push(scenario);
            batchScenarios.push(scenario);
        }
        
        // 进度输出
        if (tasksCompleted % 100 === 0) {
            const elapsed = (Date.now() - startTime) / 1000;
            const rate = elapsed > 0 ? tasksCompleted / elapsed : 1;
            const remaining = (NUM_SCENARIOS - tasksCompleted) / rate;
            console.log(`进度: ${tasksCompleted}/${NUM_SCENARIOS} - 成功: ${scenarios.length} (${(scenarios.length/tasksCompleted*100).toFixed(1)}%) - 速度: ${rate.toFixed(1)}/s - 剩余: ${(remaining/60).toFixed(1)}min`);
        }
        
        // 定期保存批次（每 SAVE_INTERVAL 个场景保存一个批次文件）
        if (batchScenarios.length >= SAVE_INTERVAL) {
            saveBatch(false);
        }
    }
    
    console.log(`\n初始范围 - OOP: ${currentOopRange.split(',').length}种, IP: ${currentIpRange.split(',').length}种`);
    console.log(`\n启动 ${NUM_WORKERS} 个 Worker...`);
    
    for (let i = 0; i < NUM_WORKERS; i++) {
        const worker = new Worker(__filename);
        workers.push(worker);
        
        worker.on('message', (msg) => {
            handleResult(msg);
            if (!sendTask(worker) && tasksCompleted >= NUM_SCENARIOS) {
                // 所有任务完成
                if (workers.every((w, idx) => !pendingTasks.has(idx))) {
                    finishUp();
                }
            }
        });
        
        worker.on('error', (err) => console.error(`Worker error: ${err.message}`));
        
        // 初始发送任务
        sendTask(worker);
    }
    
    function finishUp() {
        // 保存剩余的场景
        if (batchScenarios.length > 0) {
            saveBatch(true);
        }
        
        const totalTime = (Date.now() - startTime) / 1000;
        console.log(`\n${'='.repeat(80)}`);
        console.log(`生成完成！`);
        console.log(`成功: ${scenarios.length}/${NUM_SCENARIOS} (${(scenarios.length/NUM_SCENARIOS*100).toFixed(1)}%)`);
        console.log(`总用时: ${totalTime.toFixed(1)}s (${(totalTime/60).toFixed(1)}min)`);
        console.log(`平均速度: ${(scenarios.length/totalTime).toFixed(1)} 场景/秒`);
        console.log(`保存了 ${savedBatches.length} 个批次文件`);
        console.log(`\n失败统计: noHeroHand=${failStats.noHeroHand}, solverFailed=${failStats.solverFailed}, strategyExtractFailed=${failStats.strategyExtractFailed}, otherError=${failStats.otherError}`);
        
        // 保存汇总索引
        const summaryData = {
            timestamp,
            totalScenarios: scenarios.length,
            numBatches: savedBatches.length,
            totalTime,
            failStats: { ...failStats },
            batches: savedBatches,
            usedRanges,
        };
        writeFileSync(join(DATA_DIR, `scenarios_${timestamp}_summary.json`), JSON.stringify(summaryData, null, 2));
        console.log(`\n汇总文件: scenarios_${timestamp}_summary.json`);
        
        // 更新全局索引
        const indexFile = join(DATA_DIR, 'index.json');
        let indexData = { files: [], totalScenarios: 0 };
        if (existsSync(indexFile)) {
            try { indexData = JSON.parse(readFileSync(indexFile, 'utf8')); } catch (e) {}
        }
        
        // 添加所有批次文件到索引
        for (const batch of savedBatches) {
            if (!indexData.files.find(f => f.filename === batch.filename)) {
                indexData.files.push({ filename: batch.filename, timestamp, numScenarios: batch.numScenarios });
                indexData.totalScenarios += batch.numScenarios;
            }
        }
        indexData.lastUpdated = timestamp;
        writeFileSync(indexFile, JSON.stringify(indexData, null, 2));
        console.log(`\n索引已更新，总场景数: ${indexData.totalScenarios}`);
        
        workers.forEach(w => w.terminate());
        process.exit(0);
    }
}
