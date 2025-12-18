/**
 * 调试手牌索引查找
 */

import { solveRiver, parseBoard } from './postflop_solver.mjs';

async function debug() {
    const result = await solveRiver({
        oopRange: 'AA,KK,QQ,JJ,TT,99,88,77,AKs,AKo,AQs,KQs',
        ipRange: 'AA,KK,QQ,JJ,TT,99,AKs,AKo,AQs,KQs,QJs',
        board: 'Ks Td 7c 4h 2s',
        startingPot: 100,
        effectiveStack: 100,
        oopBetSizes: '50',
        ipBetSizes: '50',
        targetExploitability: 0.5,
        maxIterations: 200,
    });
    
    console.log('OOP hands count:', result.oopCards.length);
    console.log('First 10 OOP hand indices:', result.oopCards.slice(0, 10));
    
    // 将手牌索引转换为可读格式
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    
    function handIndexToCards(handIdx) {
        // handIdx是 c1 | (c2 << 8) 的编码
        const c1 = handIdx & 0xFF;
        const c2 = (handIdx >> 8) & 0xFF;
        
        const r1 = Math.floor(c1 / 4);
        const s1 = c1 % 4;
        const r2 = Math.floor(c2 / 4);
        const s2 = c2 % 4;
        return `${ranks[r1]}${suits[s1]}${ranks[r2]}${suits[s2]}`;
    }
    
    console.log('\nFirst 20 OOP hands:');
    for (let i = 0; i < Math.min(20, result.oopCards.length); i++) {
        const handIdx = result.oopCards[i];
        const handStr = handIndexToCards(handIdx);
        console.log(`  ${i}: index=${handIdx}, hand=${handStr}`);
    }
    
    // 查找特定手牌
    const targetHand = 'AhKh';
    console.log(`\nLooking for ${targetHand}...`);
    
    // 解析目标手牌
    const r1 = ranks.indexOf(targetHand[0].toUpperCase());
    const s1 = suits.indexOf(targetHand[1].toLowerCase());
    const r2 = ranks.indexOf(targetHand[2].toUpperCase());
    const s2 = suits.indexOf(targetHand[3].toLowerCase());
    
    const c1 = r1 * 4 + s1;
    const c2 = r2 * 4 + s2;
    
    console.log(`  Card 1: ${targetHand.substring(0,2)} -> rank=${r1}, suit=${s1}, index=${c1}`);
    console.log(`  Card 2: ${targetHand.substring(2,4)} -> rank=${r2}, suit=${s2}, index=${c2}`);
    
    // 计算手牌索引
    const [minC, maxC] = c1 < c2 ? [c1, c2] : [c2, c1];
    let targetIdx = 0;
    for (let i = 0; i < minC; i++) {
        targetIdx += 51 - i;
    }
    targetIdx += maxC - minC - 1;
    
    console.log(`  Expected hand index: ${targetIdx}`);
    console.log(`  Hand at that index: ${handIndexToCards(targetIdx)}`);
    
    // 检查是否在oopCards中
    const found = result.oopCards.includes(targetIdx);
    console.log(`  Found in oopCards: ${found}`);
}

debug().catch(console.error);
