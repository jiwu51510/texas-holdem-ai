#!/usr/bin/env node
/**
 * Case 2 详细验证
 * 公共牌: 2h 8d 2c Jc Ts
 * 固定OOP手牌: KdKs
 */

import pkg from 'poker-odds-calc';
const { TexasHoldem } = pkg;

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

function handsConflict(hand1, hand2) {
    const cards1 = new Set(hand1);
    for (const card of hand2) {
        if (cards1.has(card)) return true;
    }
    return false;
}

// Case 2: 公共牌 2h 8d 2c Jc Ts, 固定OOP手牌 KdKs
const board = ['2h', '8d', '2c', 'Jc', 'Ts'];
const fixedOOPHand = ['Kd', 'Ks'];

console.log('Case 2 详细验证');
console.log(`公共牌: ${board.join(' ')}`);
console.log(`固定OOP手牌: ${fixedOOPHand.join('')}`);

// 方案1: 死牌 = 公共牌 + 固定OOP手牌
const deadCards1 = new Set([...board, ...fixedOOPHand]);
console.log(`\n方案1: 死牌 = 公共牌 + 固定OOP手牌`);
console.log(`死牌: ${[...deadCards1].join(', ')}`);

const oopHands1 = expandRange(OOP_RANGE, deadCards1);
const ipHands1 = expandRange(IP_RANGE, deadCards1);
console.log(`OOP范围有效组合数: ${oopHands1.length}`);
console.log(`IP范围有效组合数: ${ipHands1.length}`);

// 方案2: 死牌 = 只有公共牌
const deadCards2 = new Set(board);
console.log(`\n方案2: 死牌 = 只有公共牌`);
console.log(`死牌: ${[...deadCards2].join(', ')}`);

const oopHands2 = expandRange(OOP_RANGE, deadCards2);
const ipHands2 = expandRange(IP_RANGE, deadCards2);
console.log(`OOP范围有效组合数: ${oopHands2.length}`);
console.log(`IP范围有效组合数: ${ipHands2.length}`);

// 计算有效组合对数量
let validCombos1 = 0;
let conflicts1 = 0;
for (const oopHand of oopHands1) {
    for (const ipHand of ipHands1) {
        if (handsConflict(oopHand, ipHand)) {
            conflicts1++;
        } else {
            validCombos1++;
        }
    }
}
console.log(`\n方案1: 有效组合对数量: ${validCombos1}, 冲突: ${conflicts1}`);

let validCombos2 = 0;
let conflicts2 = 0;
for (const oopHand of oopHands2) {
    for (const ipHand of ipHands2) {
        if (handsConflict(oopHand, ipHand)) {
            conflicts2++;
        } else {
            validCombos2++;
        }
    }
}
console.log(`方案2: 有效组合对数量: ${validCombos2}, 冲突: ${conflicts2}`);

// 输出IP范围的所有组合（方案1）
console.log(`\n方案1 IP范围所有组合 (${ipHands1.length}个):`);
console.log(ipHands1.map(h => h.join('')).join(', '));
