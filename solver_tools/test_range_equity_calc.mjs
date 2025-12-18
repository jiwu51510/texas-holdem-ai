#!/usr/bin/env node
/**
 * 测试范围胜率计算
 * 验证 calculateRangeVsRangeEquityDirect 函数的正确性
 */

import pkg from 'poker-odds-calc';
const { TexasHoldem } = pkg;
import { expandRange, parseBoardString } from './equity_calculator.mjs';

// OOP范围定义
const OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o';

// IP范围定义
const IP_RANGE = 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo';

/**
 * 直接使用已展开的IP范围计算范围vs范围胜率
 * 这是从 run_wasm_postflop_validation_v2.mjs 复制的函数
 */
async function calculateRangeVsRangeEquityDirect(heroRange, villainHands, board, fixedHeroCards) {
    const boardCards = parseBoardString(board);
    
    // 死牌 = 公共牌 + 固定的OOP手牌
    const deadCards = new Set([...boardCards, ...fixedHeroCards]);
    
    // 展开OOP范围（排除死牌）
    const heroHands = expandRange(heroRange, deadCards);
    
    console.log(`  OOP范围展开后组合数: ${heroHands.length}`);
    console.log(`  IP有效范围组合数: ${villainHands.length}`);
    
    if (heroHands.length === 0) return 0.5;
    
    let totalWins = 0;
    let totalTies = 0;
    let totalCombos = 0;
    
    // 遍历OOP范围中的每个手牌
    for (const heroHand of heroHands) {
        const heroSet = new Set(heroHand);
        
        // 排除与当前OOP手牌冲突的IP组合
        const validVillainHands = villainHands.filter(vh => 
            !heroSet.has(vh[0]) && !heroSet.has(vh[1])
        );
        
        if (validVillainHands.length === 0) continue;
        
        // 遍历IP范围中的每个手牌
        for (const villainHand of validVillainHands) {
            try {
                const table = new TexasHoldem();
                table.addPlayer(heroHand);
                table.addPlayer(villainHand);
                table.setBoard(boardCards);
                
                const result = table.calculate();
                const players = result.getPlayers();
                
                if (players[0].getWins() > 0) {
                    totalWins++;
                } else if (players[0].getTies() > 0) {
                    totalTies++;
                }
                totalCombos++;
            } catch (e) {
                // 跳过无效组合
            }
        }
    }
    
    console.log(`  总组合对数: ${totalCombos}`);
    console.log(`  OOP胜: ${totalWins}, 平局: ${totalTies}, IP胜: ${totalCombos - totalWins - totalTies}`);
    
    return totalCombos > 0 ? (totalWins + totalTies * 0.5) / totalCombos : 0.5;
}

async function main() {
    console.log('=' .repeat(70));
    console.log('测试范围胜率计算');
    console.log('=' .repeat(70));
    
    // 场景1: 公共牌 5h Td Jc 3h 4d, OOP手牌 AdAh
    console.log('\n【场景1】');
    console.log('公共牌: 5h Td Jc 3h 4d');
    console.log('OOP手牌: AdAh');
    
    const board1 = '5h Td Jc 3h 4d';
    const fixedOopHand1 = ['Ad', 'Ah'];
    
    // 计算IP有效范围
    const boardCards1 = parseBoardString(board1);
    const deadCards1 = new Set([...boardCards1, ...fixedOopHand1]);
    const effectiveIPHands1 = expandRange(IP_RANGE, deadCards1);
    
    console.log(`\n计算范围胜率...`);
    const rangeEquity1 = await calculateRangeVsRangeEquityDirect(OOP_RANGE, effectiveIPHands1, board1, fixedOopHand1);
    console.log(`\n范围胜率: ${(rangeEquity1 * 100).toFixed(3)}%`);
    console.log('报告中的范围胜率: 41.669%');
    
    // 场景2: 公共牌 2h 6h 3h Ad 8c, OOP手牌 AcAs
    console.log('\n' + '-'.repeat(70));
    console.log('\n【场景2】');
    console.log('公共牌: 2h 6h 3h Ad 8c');
    console.log('OOP手牌: AcAs');
    
    const board2 = '2h 6h 3h Ad 8c';
    const fixedOopHand2 = ['Ac', 'As'];
    
    // 计算IP有效范围
    const boardCards2 = parseBoardString(board2);
    const deadCards2 = new Set([...boardCards2, ...fixedOopHand2]);
    const effectiveIPHands2 = expandRange(IP_RANGE, deadCards2);
    
    console.log(`\n计算范围胜率...`);
    const rangeEquity2 = await calculateRangeVsRangeEquityDirect(OOP_RANGE, effectiveIPHands2, board2, fixedOopHand2);
    console.log(`\n范围胜率: ${(rangeEquity2 * 100).toFixed(3)}%`);
    console.log('报告中的范围胜率: 40.922%');
}

main().catch(console.error);
