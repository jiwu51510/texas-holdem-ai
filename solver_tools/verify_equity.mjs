#!/usr/bin/env node
/**
 * 验证特定场景的胜率计算
 * 
 * 场景：
 * - 公共牌: Jd 7c Qh 6s 7h
 * - 固定手牌: KhKs
 * - IP范围: AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo
 */

import { solveRiver, parseRange, parseBoard, cardIndexToString } from './postflop_solver.mjs';

const OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o';
const IP_RANGE = 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo';

async function verifyEquity() {
    const board = 'Jd 7c Qh 6s 7h';
    const heroHand = 'KhKs';
    
    console.log('='.repeat(60));
    console.log('验证胜率计算');
    console.log('='.repeat(60));
    console.log(`公共牌: ${board}`);
    console.log(`固定手牌: ${heroHand}`);
    console.log(`OOP范围: ${OOP_RANGE}`);
    console.log(`IP范围: ${IP_RANGE}`);
    console.log();
    
    // 解析手牌
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    
    // KhKs -> Kh = K(11) * 4 + h(3) = 47, Ks = K(11) * 4 + s(3) = 47
    // 等等，让我重新计算
    // K = 11 (0-indexed: 2=0, 3=1, ..., K=11, A=12)
    // h = 3, s = 3
    // Kh = 11 * 4 + 3 = 47
    // Ks = 11 * 4 + 3 = 47 -- 这不对，s应该是不同的
    // suits = 'cdhs' -> c=0, d=1, h=2, s=3
    // Kh = 11 * 4 + 2 = 46
    // Ks = 11 * 4 + 3 = 47
    
    const heroCards = [46, 47]; // Kh, Ks
    
    console.log(`手牌索引: ${heroCards} (Kh=${46}, Ks=${47})`);
    
    // 求解
    const result = await solveRiver({
        oopRange: OOP_RANGE,
        ipRange: IP_RANGE,
        board,
        startingPot: 100,
        effectiveStack: 100,
        oopBetSizes: '50',
        ipBetSizes: '50',
        targetExploitability: 0.3,
        maxIterations: 500,
    });
    
    console.log(`\n求解完成，可利用度: ${result.exploitability.toFixed(4)}%`);
    console.log(`OOP手牌数: ${result.oopCards.length}`);
    console.log(`IP手牌数: ${result.ipCards.length}`);
    
    // 找到KhKs在oopCards中的索引
    const hero = heroCards.slice().sort((a, b) => a - b);
    let heroIndex = -1;
    
    for (let i = 0; i < result.oopCards.length; i++) {
        const handIdx = result.oopCards[i];
        const c1 = handIdx & 0xFF;
        const c2 = (handIdx >> 8) & 0xFF;
        const cards = [c1, c2].sort((a, b) => a - b);
        
        if (cards[0] === hero[0] && cards[1] === hero[1]) {
            heroIndex = i;
            console.log(`\n找到KhKs在oopCards中的索引: ${heroIndex}`);
            console.log(`  handIdx = ${handIdx}, c1 = ${c1}, c2 = ${c2}`);
            console.log(`  c1 = ${cardIndexToString(c1)}, c2 = ${cardIndexToString(c2)}`);
            break;
        }
    }
    
    if (heroIndex === -1) {
        console.log('\n未找到KhKs在oopCards中');
        
        // 打印所有OOP手牌
        console.log('\nOOP手牌列表:');
        for (let i = 0; i < Math.min(20, result.oopCards.length); i++) {
            const handIdx = result.oopCards[i];
            const c1 = handIdx & 0xFF;
            const c2 = (handIdx >> 8) & 0xFF;
            console.log(`  ${i}: ${cardIndexToString(c1)}${cardIndexToString(c2)}`);
        }
        return;
    }
    
    // 解析results数组
    const oopLen = result.oopCards.length;
    const ipLen = result.ipCards.length;
    
    let offset = 3; // header
    offset += oopLen + ipLen; // weights
    offset += oopLen + ipLen; // normalizer
    
    // equity
    const equityOop = result.results.slice(offset, offset + oopLen);
    offset += oopLen;
    const equityIp = result.results.slice(offset, offset + ipLen);
    offset += ipLen;
    
    // ev
    const evOop = result.results.slice(offset, offset + oopLen);
    
    const heroEquity = equityOop[heroIndex];
    const heroEv = evOop[heroIndex];
    
    console.log(`\nKhKs的胜率: ${(heroEquity * 100).toFixed(3)}%`);
    console.log(`KhKs的EV: ${heroEv.toFixed(4)}`);
    
    // 计算范围平均胜率
    let totalEquity = 0;
    let count = 0;
    for (let i = 0; i < oopLen; i++) {
        const eq = equityOop[i];
        if (!isNaN(eq) && eq >= 0 && eq <= 1) {
            totalEquity += eq;
            count++;
        }
    }
    const rangeEquity = count > 0 ? totalEquity / count : 0.5;
    console.log(`范围平均胜率: ${(rangeEquity * 100).toFixed(3)}%`);
    
    // 打印一些其他手牌的胜率作为参考
    console.log('\n其他手牌胜率参考:');
    for (let i = 0; i < Math.min(10, oopLen); i++) {
        const handIdx = result.oopCards[i];
        const c1 = handIdx & 0xFF;
        const c2 = (handIdx >> 8) & 0xFF;
        console.log(`  ${cardIndexToString(c1)}${cardIndexToString(c2)}: ${(equityOop[i] * 100).toFixed(3)}%`);
    }
    
    // 手动计算KhKs vs IP范围的胜率
    console.log('\n' + '='.repeat(60));
    console.log('手动验证胜率计算');
    console.log('='.repeat(60));
    
    // 公共牌: Jd 7c Qh 6s 7h
    // KhKs 组成的最佳牌型: KK77Q (两对，K和7，Q踢脚)
    // 
    // 对手可能的牌型:
    // - 四条7: 不可能（公共牌只有两张7）
    // - 葫芦: 77X (X是对子) - 如77JJ, 77QQ, 7766等
    // - 同花: 红心同花 (Qh, 7h + 两张红心)
    // - 顺子: 不可能（公共牌是J,7,Q,6,7，没有顺子可能）
    // - 三条7: 不可能（我们没有7）
    // - 两对: KK77 vs 其他两对
    
    console.log('公共牌: Jd 7c Qh 6s 7h');
    console.log('KhKs 组成的最佳牌型: KK77Q (两对，K和7，Q踢脚)');
    console.log();
    console.log('能击败KhKs的对手牌型:');
    console.log('  - 葫芦: 77X (需要一张7和一对) - 但公共牌已有两张7');
    console.log('  - 三条7: 需要一张7 - 如 7x');
    console.log('  - 更大的两对: AA77, QQ77, JJ77');
    console.log('  - 同花: 红心同花 (需要两张红心，且比KK77Q大)');
    console.log();
    
    // 列出IP范围中能击败KhKs的手牌
    console.log('IP范围中能击败KhKs的手牌:');
    
    // 死牌: Jd(37), 7c(20), Qh(42), 6s(19), 7h(22), Kh(46), Ks(47)
    const deadCards = new Set([37, 20, 42, 19, 22, 46, 47]);
    
    // 解析IP范围
    const ipWeights = await parseRange(IP_RANGE);
    
    let winCount = 0;
    let loseCount = 0;
    let tieCount = 0;
    let totalCombos = 0;
    
    // 遍历所有可能的IP手牌
    let idx = 0;
    for (let c1 = 0; c1 < 52; c1++) {
        for (let c2 = c1 + 1; c2 < 52; c2++) {
            const weight = ipWeights[idx];
            idx++;
            
            if (weight === 0) continue;
            if (deadCards.has(c1) || deadCards.has(c2)) continue;
            
            totalCombos++;
            
            // 判断这手牌是否能击败KhKs
            const card1 = cardIndexToString(c1);
            const card2 = cardIndexToString(c2);
            
            // 检查是否有7（三条7）
            const has7 = (Math.floor(c1 / 4) === 5) || (Math.floor(c2 / 4) === 5);
            
            // 检查是否是AA, QQ, JJ（更大的两对）
            const r1 = Math.floor(c1 / 4);
            const r2 = Math.floor(c2 / 4);
            const isPair = r1 === r2;
            const isAA = isPair && r1 === 12;
            const isQQ = isPair && r1 === 10;
            const isJJ = isPair && r1 === 9;
            
            // 检查是否是红心同花（需要两张红心）
            const s1 = c1 % 4;
            const s2 = c2 % 4;
            const isHeartFlush = s1 === 2 && s2 === 2; // h = 2
            
            let result = 'lose'; // KhKs输
            
            if (has7) {
                // 对手有三条7，KhKs输
                result = 'lose';
                loseCount++;
                if (loseCount <= 10) {
                    console.log(`  ${card1}${card2}: 三条7 (KhKs输)`);
                }
            } else if (isAA || isQQ || isJJ) {
                // 对手有更大的两对
                result = 'lose';
                loseCount++;
                if (loseCount <= 10) {
                    console.log(`  ${card1}${card2}: 更大的两对 (KhKs输)`);
                }
            } else if (isHeartFlush) {
                // 检查是否能组成同花
                // 公共牌红心: Qh(42), 7h(22)
                // 需要检查是否能组成5张红心
                result = 'lose';
                loseCount++;
                if (loseCount <= 10) {
                    console.log(`  ${card1}${card2}: 红心同花 (KhKs输)`);
                }
            } else {
                // KhKs赢
                result = 'win';
                winCount++;
            }
        }
    }
    
    console.log(`\n统计结果:`);
    console.log(`  总组合数: ${totalCombos}`);
    console.log(`  KhKs赢: ${winCount}`);
    console.log(`  KhKs输: ${loseCount}`);
    console.log(`  平局: ${tieCount}`);
    console.log(`  手动计算胜率: ${(winCount / totalCombos * 100).toFixed(3)}%`);
    console.log(`  Solver报告胜率: ${(heroEquity * 100).toFixed(3)}%`);
}

verifyEquity().catch(console.error);
