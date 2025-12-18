/**
 * Node.js wrapper for wasm-postflop solver
 * 用于调用wasm-postflop求解器的Node.js封装
 */

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// wasm文件路径
const solverWasmPath = join(__dirname, 'wasm-postflop/pkg/solver-st/solver_bg.wasm');
const rangeWasmPath = join(__dirname, 'wasm-postflop/pkg/range/range_bg.wasm');

let solverWasm = null;
let rangeWasm = null;
let RangeManager = null;
let GameManager = null;

/**
 * 初始化range模块
 */
async function initRange() {
    if (RangeManager) return;
    
    // 读取wasm文件
    const wasmBuffer = readFileSync(rangeWasmPath);
    
    // 导入range_bg.js
    const rangeBg = await import('./wasm-postflop/pkg/range/range_bg.js');
    
    // 编译并实例化wasm
    const wasmModule = await WebAssembly.compile(wasmBuffer);
    const wasmInstance = await WebAssembly.instantiate(wasmModule, {
        './range_bg.js': rangeBg,
    });
    
    // 设置wasm实例
    rangeBg.__wbg_set_wasm(wasmInstance.exports);
    
    // 初始化externref表
    if (rangeBg.__wbindgen_init_externref_table) {
        rangeBg.__wbindgen_init_externref_table();
    }
    
    RangeManager = rangeBg.RangeManager;
    rangeWasm = wasmInstance;
}


/**
 * 初始化solver模块
 */
async function initSolver() {
    if (GameManager) return;
    
    // 读取wasm文件
    const wasmBuffer = readFileSync(solverWasmPath);
    
    // 导入solver_bg.js (solver-st使用web target，需要不同的导入方式)
    const solverJs = await import('./wasm-postflop/pkg/solver-st/solver.js');
    
    // 使用initSync初始化
    solverJs.initSync(wasmBuffer);
    
    GameManager = solverJs.GameManager;
    solverWasm = true;
}

/**
 * 将范围字符串转换为Float32Array
 * @param {string} rangeStr - 范围字符串，如 "AA,KK,QQ,AKs"
 * @returns {Float32Array} - 1326长度的权重数组
 */
export async function parseRange(rangeStr) {
    await initRange();
    const manager = RangeManager.new();
    const error = manager.from_string(rangeStr);
    if (error) {
        manager.free();
        throw new Error(`Invalid range: ${error}`);
    }
    // 使用raw_data()获取1326长度的数组，而不是get_weights()的169长度
    const weights = manager.raw_data();
    manager.free();
    return weights;
}

/**
 * 将牌面字符串转换为Uint8Array
 * @param {string} boardStr - 牌面字符串，如 "Ks Td 7c 4h 2s"
 * @returns {Uint8Array} - 牌面数组
 */
export function parseBoard(boardStr) {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    
    const cards = boardStr.trim().split(/\s+/);
    const result = new Uint8Array(cards.length);
    
    for (let i = 0; i < cards.length; i++) {
        const card = cards[i];
        const rank = ranks.indexOf(card[0].toUpperCase());
        const suit = suits.indexOf(card[1].toLowerCase());
        if (rank === -1 || suit === -1) {
            throw new Error(`Invalid card: ${card}`);
        }
        result[i] = rank * 4 + suit;
    }
    
    return result;
}

/**
 * 求解河牌阶段
 * @param {Object} config - 配置对象
 * @returns {Object} - 求解结果
 */
export async function solveRiver(config) {
    const {
        oopRange,      // OOP范围字符串
        ipRange,       // IP范围字符串
        board,         // 牌面字符串 (5张牌)
        startingPot = 100,
        effectiveStack = 100,
        oopBetSizes = '50',    // OOP下注尺寸
        ipBetSizes = '50',     // IP下注尺寸
        oopRaiseSizes = '',    // OOP加注尺寸
        ipRaiseSizes = '',     // IP加注尺寸
        targetExploitability = 0.5,  // 目标可利用度 (%)
        maxIterations = 1000,
    } = config;
    
    await initSolver();
    const game = GameManager.new();
    
    try {
        // 解析范围
        const oopWeights = await parseRange(oopRange);
        const ipWeights = await parseRange(ipRange);
        const boardArray = parseBoard(board);
        
        // 转换bet sizes为正确格式 (需要加%)
        const oopBetStr = oopBetSizes.split(',').map(s => s.trim() + '%').join(',');
        const ipBetStr = ipBetSizes.split(',').map(s => s.trim() + '%').join(',');
        // 转换raise sizes为正确格式
        const oopRaiseStr = oopRaiseSizes ? oopRaiseSizes.split(',').map(s => s.trim() + '%').join(',') : '';
        const ipRaiseStr = ipRaiseSizes ? ipRaiseSizes.split(',').map(s => s.trim() + '%').join(',') : '';
        
        // 初始化游戏
        const error = game.init(
            oopWeights,
            ipWeights,
            boardArray,
            startingPot,
            effectiveStack,
            0,      // rake_rate
            0,      // rake_cap
            false,  // donk_option
            '',     // oop_flop_bet
            '',     // oop_flop_raise
            '',     // oop_turn_bet
            '',     // oop_turn_raise
            '',     // oop_turn_donk
            oopBetStr,  // oop_river_bet
            oopRaiseStr,  // oop_river_raise
            '',     // oop_river_donk
            '',     // ip_flop_bet
            '',     // ip_flop_raise
            '',     // ip_turn_bet
            '',     // ip_turn_raise
            ipBetStr,   // ip_river_bet
            ipRaiseStr,   // ip_river_raise
            1.5,    // add_allin_threshold (150%)
            0.2,    // force_allin_threshold (20%)
            0.1,    // merging_threshold (10%)
            '',     // added_lines
            ''      // removed_lines
        );
        
        if (error) {
            throw new Error(`Init error: ${error}`);
        }
        
        // 分配内存
        game.allocate_memory(false);
        
        // 迭代求解
        let exploitability = Infinity;
        let iteration = 0;
        
        while (iteration < maxIterations && exploitability > targetExploitability) {
            game.solve_step(iteration);
            iteration++;
            
            if (iteration % 100 === 0) {
                exploitability = game.exploitability();
            }
        }
        
        exploitability = game.exploitability();
        
        // 获取结果
        game.finalize();
        
        // 获取OOP和IP的私有牌
        const oopCards = game.private_cards(0);
        const ipCards = game.private_cards(1);
        
        // 获取策略结果
        const results = game.get_results();
        const numActions = game.num_actions();
        const actions = game.actions_after(new Uint32Array(0));
        
        return {
            exploitability,
            iterations: iteration,
            oopCards: Array.from(oopCards),
            ipCards: Array.from(ipCards),
            results: Array.from(results),
            numActions,
            actions,
        };
        
    } finally {
        // 必须手动释放WASM内存，否则会内存泄漏
        try {
            game.free();
        } catch (e) {
            // 忽略释放错误
        }
    }
}


/**
 * 将牌索引转换为可读字符串
 * @param {number} cardIndex - 牌索引 (0-51)
 * @returns {string} - 可读字符串，如 "As"
 */
export function cardIndexToString(cardIndex) {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    const rank = Math.floor(cardIndex / 4);
    const suit = cardIndex % 4;
    return ranks[rank] + suits[suit];
}

/**
 * 将手牌索引转换为可读字符串
 * @param {number} handIndex - 手牌索引 (0-1325)
 * @returns {string} - 可读字符串，如 "AsKs"
 */
export function handIndexToString(handIndex) {
    let idx = 0;
    for (let c1 = 0; c1 < 52; c1++) {
        for (let c2 = c1 + 1; c2 < 52; c2++) {
            if (idx === handIndex) {
                return cardIndexToString(c1) + cardIndexToString(c2);
            }
            idx++;
        }
    }
    return 'Unknown';
}

// 测试函数
async function test() {
    console.log('Testing wasm-postflop solver...');
    
    try {
        const result = await solveRiver({
            oopRange: 'AA,KK,QQ,JJ,TT,99,88,77,AKs,AKo,AQs,KQs',
            ipRange: 'AA,KK,QQ,JJ,TT,99,AKs,AKo,AQs,KQs,QJs',
            board: 'Ks Td 7c 4h 2s',
            startingPot: 100,
            effectiveStack: 100,
            oopBetSizes: '50',
            ipBetSizes: '50',
            targetExploitability: 0.5,
            maxIterations: 500,
        });
        
        console.log('Solver result:');
        console.log(`  Exploitability: ${result.exploitability.toFixed(4)}%`);
        console.log(`  Iterations: ${result.iterations}`);
        console.log(`  Num actions: ${result.numActions}`);
        console.log(`  Actions: ${result.actions}`);
        console.log(`  OOP hands: ${result.oopCards.length}`);
        console.log(`  IP hands: ${result.ipCards.length}`);
        
        if (result.results.length > 0) {
            console.log(`  Results length: ${result.results.length}`);
            console.log(`  First few results: ${result.results.slice(0, 10).map(x => x.toFixed(4)).join(', ')}`);
        }
        
    } catch (e) {
        console.error('Error:', e);
    }
}

// 如果直接运行此文件，执行测试
if (process.argv[1] === fileURLToPath(import.meta.url)) {
    test();
}
