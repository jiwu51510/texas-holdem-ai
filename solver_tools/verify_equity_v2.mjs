#!/usr/bin/env node
/**
 * 更详细地验证特定场景的胜率计算
 */

import { solveRiver, parseRange, parseBoard, cardIndexToString } from './postflop_solver.mjs';

const IP_RANGE = 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo';

// 手牌评估函数
function evaluateHand(holeCards, boardCards) {
    // 合并所有牌
    const allCards = [...holeCards, ...boardCards];
    
    // 转换为rank和suit
    const cards = allCards.map(c => ({
        rank: Math.floor(c / 4),  // 0=2, 1=3, ..., 12=A
        suit: c % 4,
        index: c
    }));
    
    // 检查同花
    const suitCounts = [0, 0, 0, 0];
    cards.forEach(c => suitCounts[c.suit]++);
    const flushSuit = suitCounts.findIndex(count => count >= 5);
    const hasFlush = flushSuit !== -1;
    
    // 检查顺子
    const rankCounts = new Array(13).fill(0);
    cards.forEach(c => rankCounts[c.rank]++);
    
    // 检查顺子（包括A-2-3-4-5）
    let hasStraight = false;
    let straightHighCard = -1;
    
    // 检查普通顺子
    for (let i = 12; i >= 4; i--) {
        if (rankCounts[i] > 0 && rankCounts[i-1] > 0 && rankCounts[i-2] > 0 && 
            rankCounts[i-3] > 0 && rankCounts[i-4] > 0) {
            hasStraight = true;
            straightHighCard = i;
            break;
        }
    }
    
    // 检查A-2-3-4-5
    if (!hasStraight && rankCounts[12] > 0 && rankCounts[0] > 0 && rankCounts[1] > 0 && 
        rankCounts[2] > 0 && rankCounts[3] > 0) {
        hasStraight = true;
        straightHighCard = 3; // 5 high
    }
    
    // 检查同花顺
    let hasStraightFlush = false;
    let straightFlushHighCard = -1;
    if (hasFlush) {
        const flushCards = cards.filter(c => c.suit === flushSuit);
        const flushRankCounts = new Array(13).fill(0);
        flushCards.forEach(c => flushRankCounts[c.rank]++);
        
        for (let i = 12; i >= 4; i--) {
            if (flushRankCounts[i] > 0 && flushRankCounts[i-1] > 0 && flushRankCounts[i-2] > 0 && 
                flushRankCounts[i-3] > 0 && flushRankCounts[i-4] > 0) {
                hasStraightFlush = true;
                straightFlushHighCard = i;
                break;
            }
        }
        
        if (!hasStraightFlush && flushRankCounts[12] > 0 && flushRankCounts[0] > 0 && 
            flushRankCounts[1] > 0 && flushRankCounts[2] > 0 && flushRankCounts[3] > 0) {
            hasStraightFlush = true;
            straightFlushHighCard = 3;
        }
    }
    
    // 检查四条、葫芦、三条、两对、一对
    const pairs = [];
    const trips = [];
    const quads = [];
    
    for (let r = 0; r < 13; r++) {
        if (rankCounts[r] === 4) quads.push(r);
        else if (rankCounts[r] === 3) trips.push(r);
        else if (rankCounts[r] === 2) pairs.push(r);
    }
    
    // 排序（从大到小）
    pairs.sort((a, b) => b - a);
    trips.sort((a, b) => b - a);
    quads.sort((a, b) => b - a);
    
    // 计算手牌强度值
    // 格式: [类型, 主要牌, 次要牌, 踢脚...]
    // 类型: 9=同花顺, 8=四条, 7=葫芦, 6=同花, 5=顺子, 4=三条, 3=两对, 2=一对, 1=高牌
    
    if (hasStraightFlush) {
        return { type: 9, ranks: [straightFlushHighCard], name: '同花顺' };
    }
    
    if (quads.length > 0) {
        const kicker = cards.filter(c => c.rank !== quads[0]).sort((a, b) => b.rank - a.rank)[0].rank;
        return { type: 8, ranks: [quads[0], kicker], name: '四条' };
    }
    
    if (trips.length > 0 && (pairs.length > 0 || trips.length > 1)) {
        const tripRank = trips[0];
        const pairRank = trips.length > 1 ? trips[1] : pairs[0];
        return { type: 7, ranks: [tripRank, pairRank], name: '葫芦' };
    }
    
    if (hasFlush) {
        const flushCards = cards.filter(c => c.suit === flushSuit).sort((a, b) => b.rank - a.rank);
        return { type: 6, ranks: flushCards.slice(0, 5).map(c => c.rank), name: '同花' };
    }
    
    if (hasStraight) {
        return { type: 5, ranks: [straightHighCard], name: '顺子' };
    }
    
    if (trips.length > 0) {
        const kickers = cards.filter(c => c.rank !== trips[0]).sort((a, b) => b.rank - a.rank).slice(0, 2);
        return { type: 4, ranks: [trips[0], ...kickers.map(c => c.rank)], name: '三条' };
    }
    
    if (pairs.length >= 2) {
        const kicker = cards.filter(c => c.rank !== pairs[0] && c.rank !== pairs[1])
            .sort((a, b) => b.rank - a.rank)[0].rank;
        return { type: 3, ranks: [pairs[0], pairs[1], kicker], name: '两对' };
    }
    
    if (pairs.length === 1) {
        const kickers = cards.filter(c => c.rank !== pairs[0]).sort((a, b) => b.rank - a.rank).slice(0, 3);
        return { type: 2, ranks: [pairs[0], ...kickers.map(c => c.rank)], name: '一对' };
    }
    
    const highCards = cards.sort((a, b) => b.rank - a.rank).slice(0, 5);
    return { type: 1, ranks: highCards.map(c => c.rank), name: '高牌' };
}

// 比较两手牌
function compareHands(hand1, hand2) {
    if (hand1.type !== hand2.type) {
        return hand1.type - hand2.type;
    }
    
    for (let i = 0; i < Math.min(hand1.ranks.length, hand2.ranks.length); i++) {
        if (hand1.ranks[i] !== hand2.ranks[i]) {
            return hand1.ranks[i] - hand2.ranks[i];
        }
    }
    
    return 0;
}

async function verifyEquity() {
    const board = 'Jd 7c Qh 6s 7h';
    const heroHand = 'KhKs';
    
    console.log('='.repeat(60));
    console.log('详细验证胜率计算');
    console.log('='.repeat(60));
    console.log(`公共牌: ${board}`);
    console.log(`固定手牌: ${heroHand}`);
    console.log(`IP范围: ${IP_RANGE}`);
    console.log();
    
    // 解析公共牌
    const boardCards = parseBoard(board);
    console.log(`公共牌索引: ${Array.from(boardCards)}`);
    
    // 解析手牌 KhKs
    // K = 11, h = 2, s = 3
    const heroCards = [46, 47]; // Kh, Ks
    console.log(`手牌索引: ${heroCards}`);
    
    // 评估KhKs的牌型
    const heroEval = evaluateHand(heroCards, Array.from(boardCards));
    console.log(`\nKhKs的牌型: ${heroEval.name}`);
    console.log(`  类型: ${heroEval.type}`);
    console.log(`  等级: ${heroEval.ranks.map(r => '23456789TJQKA'[r]).join(', ')}`);
    
    // 死牌
    const deadCards = new Set([...boardCards, ...heroCards]);
    console.log(`\n死牌: ${Array.from(deadCards).map(c => cardIndexToString(c)).join(', ')}`);
    
    // 解析IP范围
    const ipWeights = await parseRange(IP_RANGE);
    
    let winCount = 0;
    let loseCount = 0;
    let tieCount = 0;
    let totalCombos = 0;
    
    const losers = [];
    const winners = [];
    const ties = [];
    
    // 遍历所有可能的IP手牌
    let idx = 0;
    for (let c1 = 0; c1 < 52; c1++) {
        for (let c2 = c1 + 1; c2 < 52; c2++) {
            const weight = ipWeights[idx];
            idx++;
            
            if (weight === 0) continue;
            if (deadCards.has(c1) || deadCards.has(c2)) continue;
            
            totalCombos++;
            
            // 评估对手牌型
            const villainCards = [c1, c2];
            const villainEval = evaluateHand(villainCards, Array.from(boardCards));
            
            // 比较
            const cmp = compareHands(heroEval, villainEval);
            
            const handStr = `${cardIndexToString(c1)}${cardIndexToString(c2)}`;
            
            if (cmp > 0) {
                winCount++;
                winners.push({ hand: handStr, eval: villainEval });
            } else if (cmp < 0) {
                loseCount++;
                losers.push({ hand: handStr, eval: villainEval });
            } else {
                tieCount++;
                ties.push({ hand: handStr, eval: villainEval });
            }
        }
    }
    
    console.log(`\n统计结果:`);
    console.log(`  总组合数: ${totalCombos}`);
    console.log(`  KhKs赢: ${winCount}`);
    console.log(`  KhKs输: ${loseCount}`);
    console.log(`  平局: ${tieCount}`);
    
    const equity = (winCount + tieCount * 0.5) / totalCombos;
    console.log(`\n手动计算胜率: ${(equity * 100).toFixed(3)}%`);
    
    // 打印输的手牌
    console.log(`\nKhKs输给的手牌 (${losers.length}个):`);
    for (const l of losers) {
        console.log(`  ${l.hand}: ${l.eval.name} (${l.eval.ranks.map(r => '23456789TJQKA'[r]).join(', ')})`);
    }
    
    // 打印平局的手牌
    if (ties.length > 0) {
        console.log(`\n平局的手牌 (${ties.length}个):`);
        for (const t of ties) {
            console.log(`  ${t.hand}: ${t.eval.name}`);
        }
    }
    
    // 使用solver验证
    console.log('\n' + '='.repeat(60));
    console.log('Solver验证');
    console.log('='.repeat(60));
    
    const OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o';
    
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
    
    // 找到KhKs
    const hero = heroCards.slice().sort((a, b) => a - b);
    let heroIndex = -1;
    
    for (let i = 0; i < result.oopCards.length; i++) {
        const handIdx = result.oopCards[i];
        const c1 = handIdx & 0xFF;
        const c2 = (handIdx >> 8) & 0xFF;
        const cards = [c1, c2].sort((a, b) => a - b);
        
        if (cards[0] === hero[0] && cards[1] === hero[1]) {
            heroIndex = i;
            break;
        }
    }
    
    const oopLen = result.oopCards.length;
    const ipLen = result.ipCards.length;
    
    let offset = 3;
    offset += oopLen + ipLen;
    offset += oopLen + ipLen;
    
    const equityOop = result.results.slice(offset, offset + oopLen);
    
    console.log(`Solver报告的KhKs胜率: ${(equityOop[heroIndex] * 100).toFixed(3)}%`);
    console.log(`手动计算的KhKs胜率: ${(equity * 100).toFixed(3)}%`);
    console.log(`差异: ${((equityOop[heroIndex] - equity) * 100).toFixed(3)}%`);
}

verifyEquity().catch(console.error);
