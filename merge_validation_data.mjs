#!/usr/bin/env node
/**
 * 合并历史验证数据到累计文件
 */

import { readFileSync, writeFileSync, existsSync, readdirSync } from 'fs';

const RESULTS_DIR = 'experiments/results';
const CUMULATIVE_FILE = `${RESULTS_DIR}/wasm_postflop_validation_v3_cumulative.json`;

// 初始化累计数据结构
let cumulativeData = {
    method: 'OMPEval + wasm-postflop (随机范围) - 累计数据',
    runs: [],
    totalScenarios: 0,
    totalPairs: 0,
    totalSignificantPairs: 0,
    allSignificantPairs: [],
};

// 读取已有的累计数据
if (existsSync(CUMULATIVE_FILE)) {
    try {
        cumulativeData = JSON.parse(readFileSync(CUMULATIVE_FILE, 'utf8'));
        console.log(`读取已有累计数据: ${cumulativeData.runs.length} 次运行`);
    } catch (e) {
        console.log('读取累计数据失败，将创建新文件');
    }
}

// 获取已记录的时间戳
const existingTimestamps = new Set(cumulativeData.runs.map(r => r.timestamp));

// 查找所有历史数据文件
const files = readdirSync(RESULTS_DIR).filter(f => 
    f.startsWith('wasm_postflop_validation_v3_') && 
    f.endsWith('.json') &&
    !f.includes('cumulative') &&
    !f.includes('latest')
);

console.log(`\n找到 ${files.length} 个历史数据文件`);

let addedCount = 0;
for (const file of files) {
    const filePath = `${RESULTS_DIR}/${file}`;
    try {
        const data = JSON.parse(readFileSync(filePath, 'utf8'));
        
        // 从文件名提取时间戳
        const timestamp = data.timestamp || file.replace('wasm_postflop_validation_v3_', '').replace('.json', '');
        
        // 跳过已存在的
        if (existingTimestamps.has(timestamp)) {
            console.log(`  跳过已存在: ${file}`);
            continue;
        }
        
        // 添加运行记录
        cumulativeData.runs.push({
            timestamp,
            numScenarios: data.numScenarios || data.scenarios?.length || 0,
            numPairs: data.numPairs || data.pairs?.length || 0,
            numSignificantPairs: data.numSignificantPairs || 0,
        });
        
        cumulativeData.totalScenarios += data.numScenarios || data.scenarios?.length || 0;
        cumulativeData.totalPairs += data.numPairs || data.pairs?.length || 0;
        cumulativeData.totalSignificantPairs += data.numSignificantPairs || 0;
        
        // 添加反例（如果有scenarios数据）
        if (data.scenarios && data.pairs) {
            // 找出策略差异显著的对
            for (const pair of data.pairs) {
                if (pair.strategyDiff > 0.15) {
                    // 从scenarios中找到对应的场景
                    const s1 = data.scenarios.find(s => s.board === pair.board1 && s.heroHand === pair.heroHand1);
                    const s2 = data.scenarios.find(s => s.board === pair.board2 && s.heroHand === pair.heroHand2);
                    
                    if (s1 && s2) {
                        cumulativeData.allSignificantPairs.push({
                            timestamp,
                            board1: s1.board, heroHand1: s1.heroHand,
                            oopRange1: s1.oopRange, ipRange1: s1.ipRange,
                            strategy1: s1.strategy, actionEV1: s1.actionEV,
                            heroEquity1: s1.heroEquity, rangeEquity1: s1.rangeEquity,
                            board2: s2.board, heroHand2: s2.heroHand,
                            oopRange2: s2.oopRange, ipRange2: s2.ipRange,
                            strategy2: s2.strategy, actionEV2: s2.actionEV,
                            heroEquity2: s2.heroEquity, rangeEquity2: s2.rangeEquity,
                            strategyDiff: pair.strategyDiff,
                        });
                    }
                }
            }
        }
        
        existingTimestamps.add(timestamp);
        addedCount++;
        console.log(`  添加: ${file} (${data.numScenarios || data.scenarios?.length || 0} 场景)`);
        
    } catch (e) {
        console.log(`  错误: ${file} - ${e.message}`);
    }
}

// 保存更新后的累计数据
writeFileSync(CUMULATIVE_FILE, JSON.stringify(cumulativeData, null, 2));

console.log(`\n合并完成!`);
console.log(`  新增运行: ${addedCount}`);
console.log(`  总运行次数: ${cumulativeData.runs.length}`);
console.log(`  总场景数: ${cumulativeData.totalScenarios}`);
console.log(`  总反例数: ${cumulativeData.totalSignificantPairs}`);
