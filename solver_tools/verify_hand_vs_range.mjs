#!/usr/bin/env node
/**
 * 验证固定手牌 vs 范围的胜率计算
 * 枚举IP范围内所有组合，计算固定OOP手牌的胜率
 */

import pkg from 'poker-odds-calc';
const { TexasHoldem } = pkg;

// 范围定义
const IP_RANGE = 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo';

function parseBoardString(boardStr) {
    return boardStr.trim().split(/\s+/);
}

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

/**
 * 检查两个手牌是否有冲突（共享同一张牌）
 */
function handsConflict(hand1, hand2) {
    const cards1 = new Set(hand1);
    for (const card of hand2) {
        if (cards1.has(card)) return true;
    }
    return false;
}

/**
 * 计算固定手牌 vs 范围的胜率
 */
async function calculateHandVsRangeEquity(board, fixedOOPHand) {
    const boardCards = parseBoardString(board);
    const oopCards = fixedOOPHand.match(/.{2}/g);
    
    // 死牌 = 公共牌 + 固定的OOP手牌
    const deadCards = new Set([...boardCards, ...oopCards]);
    
    console.log(`\n${'='.repeat(70)}`);
    console.log(`公共牌: ${board}`);
    console.log(`固定OOP手牌: ${fixedOOPHand}`);
    console.log(`死牌: ${[...deadCards].join(', ')}`);
    
    // 展开IP范围（排除死牌）
    const ipHands = expandRange(IP_RANGE, deadCards);
    
    console.log(`\nIP范围有效组合数: ${ipHands.length}`);
    
    // 枚举所有IP组合
    let totalWins = 0;
    let totalTies = 0;
    let totalLosses = 0;
    let totalCombos = 0;
    
    console.log(`\n开始枚举所有组合...`);
    
    for (const ipHand of ipHands) {
        // 检查OOP和IP手牌是否冲突
        if (handsConflict(oopCards, ipHand)) {
            continue;
        }
        
        try {
            const table = new TexasHoldem();
            table.addPlayer(oopCards);
            table.addPlayer(ipHand);
            table.setBoard(boardCards);
            
            const result = table.calculate();
            const players = result.getPlayers();
            
            if (players[0].getWins() > 0) {
                totalWins++;
            } else if (players[0].getTies() > 0) {
                totalTies++;
            } else {
                totalLosses++;
            }
            totalCombos++;
        } catch (e) {
            console.log(`错误: ${fixedOOPHand} vs ${ipHand.join('')}: ${e.message}`);
        }
    }
    
    console.log(`\n${'─'.repeat(50)}`);
    console.log(`统计结果:`);
    console.log(`  有效组合总数: ${totalCombos}`);
    console.log(`  OOP胜: ${totalWins}`);
    console.log(`  平局: ${totalTies}`);
    console.log(`  OOP负: ${totalLosses}`);
    
    const winRate = totalWins / totalCombos;
    const tieRate = totalTies / totalCombos;
    const lossRate = totalLosses / totalCombos;
    const equity = (totalWins + totalTies * 0.5) / totalCombos;
    
    console.log(`\n胜率统计:`);
    console.log(`  OOP胜率: ${(winRate * 100).toFixed(3)}%`);
    console.log(`  平局率: ${(tieRate * 100).toFixed(3)}%`);
    console.log(`  OOP负率: ${(lossRate * 100).toFixed(3)}%`);
    console.log(`\n  手牌胜率 (胜 + 平/2): ${(equity * 100).toFixed(3)}%`);
    
    return { equity, totalWins, totalTies, totalLosses, totalCombos };
}

// 运行验证
async function main() {
    console.log('验证固定手牌 vs 范围的胜率计算');
    console.log(`\nIP范围: ${IP_RANGE}`);
    
    // Case 2: 公共牌 2h 8d 2c Jc Ts, OOP手牌 KdKs
    await calculateHandVsRangeEquity('2h 8d 2c Jc Ts', 'KdKs');
}

main().catch(console.error);
