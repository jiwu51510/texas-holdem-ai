#!/usr/bin/env node
/**
 * 检查范围展开是否正确
 */

// 范围定义
const OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o';
const IP_RANGE = 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo';

function expandRange(rangeStr, deadCards) {
    const hands = [];
    const ranks = '23456789TJQKA';
    const suits = ['c', 'd', 'h', 's'];
    
    const parts = rangeStr.split(',');
    
    for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed) continue;
        
        if (trimmed.includes('-')) {
            const [start, end] = trimmed.split('-');
            const expandedHands = expandRangeNotation(start, end, ranks, suits, deadCards);
            hands.push(...expandedHands);
        } else {
            const expandedHands = expandSingleHand(trimmed, ranks, suits, deadCards);
            hands.push(...expandedHands);
        }
    }
    
    return hands;
}

function expandRangeNotation(start, end, ranks, suits, deadCards) {
    const hands = [];
    
    if (start.length === 2 && start[0] === start[1]) {
        const startRank = ranks.indexOf(start[0]);
        const endRank = ranks.indexOf(end[0]);
        const minRank = Math.min(startRank, endRank);
        const maxRank = Math.max(startRank, endRank);
        
        for (let r = minRank; r <= maxRank; r++) {
            const pairHands = expandSingleHand(ranks[r] + ranks[r], ranks, suits, deadCards);
            hands.push(...pairHands);
        }
    } else if (start.endsWith('s')) {
        const highRank = ranks.indexOf(start[0]);
        const startLowRank = ranks.indexOf(start[1]);
        const endLowRank = ranks.indexOf(end[1]);
        const minLowRank = Math.min(startLowRank, endLowRank);
        const maxLowRank = Math.max(startLowRank, endLowRank);
        
        for (let r = minLowRank; r <= maxLowRank; r++) {
            if (r !== highRank) {
                const suitedHands = expandSingleHand(ranks[highRank] + ranks[r] + 's', ranks, suits, deadCards);
                hands.push(...suitedHands);
            }
        }
    } else if (start.endsWith('o')) {
        const highRank = ranks.indexOf(start[0]);
        const startLowRank = ranks.indexOf(start[1]);
        const endLowRank = ranks.indexOf(end[1]);
        const minLowRank = Math.min(startLowRank, endLowRank);
        const maxLowRank = Math.max(startLowRank, endLowRank);
        
        for (let r = minLowRank; r <= maxLowRank; r++) {
            if (r !== highRank) {
                const offsuitHands = expandSingleHand(ranks[highRank] + ranks[r] + 'o', ranks, suits, deadCards);
                hands.push(...offsuitHands);
            }
        }
    }
    
    return hands;
}

function expandSingleHand(handType, ranks, suits, deadCards) {
    const hands = [];
    
    if (handType.length === 2 && handType[0] === handType[1]) {
        const rank = handType[0];
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = s1 + 1; s2 < 4; s2++) {
                const card1 = rank + suits[s1];
                const card2 = rank + suits[s2];
                if (!deadCards.has(card1) && !deadCards.has(card2)) {
                    hands.push([card1, card2]);
                }
            }
        }
    } else if (handType.endsWith('s')) {
        const rank1 = handType[0];
        const rank2 = handType[1];
        for (let s = 0; s < 4; s++) {
            const card1 = rank1 + suits[s];
            const card2 = rank2 + suits[s];
            if (!deadCards.has(card1) && !deadCards.has(card2)) {
                hands.push([card1, card2]);
            }
        }
    } else if (handType.endsWith('o')) {
        const rank1 = handType[0];
        const rank2 = handType[1];
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = 0; s2 < 4; s2++) {
                if (s1 !== s2) {
                    const card1 = rank1 + suits[s1];
                    const card2 = rank2 + suits[s2];
                    if (!deadCards.has(card1) && !deadCards.has(card2)) {
                        hands.push([card1, card2]);
                    }
                }
            }
        }
    } else if (handType.length === 2) {
        const rank = handType[0];
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = s1 + 1; s2 < 4; s2++) {
                const card1 = rank + suits[s1];
                const card2 = rank + suits[s2];
                if (!deadCards.has(card1) && !deadCards.has(card2)) {
                    hands.push([card1, card2]);
                }
            }
        }
    }
    
    return hands;
}

// 计算无死牌时的组合数
const noDeadCards = new Set();
const oopHandsNoDead = expandRange(OOP_RANGE, noDeadCards);
const ipHandsNoDead = expandRange(IP_RANGE, noDeadCards);

console.log('无死牌时的组合数:');
console.log(`OOP范围: ${oopHandsNoDead.length} 个组合`);
console.log(`IP范围: ${ipHandsNoDead.length} 个组合`);

// 统计OOP范围中各类型手牌的数量
const oopPairs = oopHandsNoDead.filter(h => h[0][0] === h[1][0]).length;
const oopSuited = oopHandsNoDead.filter(h => h[0][0] !== h[1][0] && h[0][1] === h[1][1]).length;
const oopOffsuit = oopHandsNoDead.filter(h => h[0][0] !== h[1][0] && h[0][1] !== h[1][1]).length;

console.log(`\nOOP范围组成:`);
console.log(`  对子: ${oopPairs} 个组合 (13种 × 6 = 78)`);
console.log(`  同花: ${oopSuited} 个组合`);
console.log(`  不同花: ${oopOffsuit} 个组合`);

// 统计IP范围中各类型手牌的数量
const ipPairs = ipHandsNoDead.filter(h => h[0][0] === h[1][0]).length;
const ipSuited = ipHandsNoDead.filter(h => h[0][0] !== h[1][0] && h[0][1] === h[1][1]).length;
const ipOffsuit = ipHandsNoDead.filter(h => h[0][0] !== h[1][0] && h[0][1] !== h[1][1]).length;

console.log(`\nIP范围组成:`);
console.log(`  对子: ${ipPairs} 个组合 (13种 × 6 = 78)`);
console.log(`  同花: ${ipSuited} 个组合`);
console.log(`  不同花: ${ipOffsuit} 个组合`);

// 输出OOP范围的手牌类型
console.log(`\n=== OOP范围手牌类型 ===`);
const oopHandTypes = new Set();
for (const hand of oopHandsNoDead) {
    const r1 = hand[0][0];
    const r2 = hand[1][0];
    const s1 = hand[0][1];
    const s2 = hand[1][1];
    
    const ranks = '23456789TJQKA';
    const r1Idx = ranks.indexOf(r1);
    const r2Idx = ranks.indexOf(r2);
    
    let type;
    if (r1 === r2) {
        type = r1 + r2;
    } else if (s1 === s2) {
        type = (r1Idx > r2Idx ? r1 + r2 : r2 + r1) + 's';
    } else {
        type = (r1Idx > r2Idx ? r1 + r2 : r2 + r1) + 'o';
    }
    oopHandTypes.add(type);
}
console.log(`OOP范围包含 ${oopHandTypes.size} 种手牌类型:`);
console.log([...oopHandTypes].sort().join(', '));

// 输出IP范围的手牌类型
console.log(`\n=== IP范围手牌类型 ===`);
const ipHandTypes = new Set();
for (const hand of ipHandsNoDead) {
    const r1 = hand[0][0];
    const r2 = hand[1][0];
    const s1 = hand[0][1];
    const s2 = hand[1][1];
    
    const ranks = '23456789TJQKA';
    const r1Idx = ranks.indexOf(r1);
    const r2Idx = ranks.indexOf(r2);
    
    let type;
    if (r1 === r2) {
        type = r1 + r2;
    } else if (s1 === s2) {
        type = (r1Idx > r2Idx ? r1 + r2 : r2 + r1) + 's';
    } else {
        type = (r1Idx > r2Idx ? r1 + r2 : r2 + r1) + 'o';
    }
    ipHandTypes.add(type);
}
console.log(`IP范围包含 ${ipHandTypes.size} 种手牌类型:`);
console.log([...ipHandTypes].sort().join(', '));
