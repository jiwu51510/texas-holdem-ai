#!/usr/bin/env node
/**
 * 调试版本：验证范围vs范围胜率计算
 * 只计算Case 2，输出详细信息
 */

import pkg from 'poker-odds-calc';
const { TexasHoldem } = pkg;

// 范围定义
const OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o';
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
        // 对子范围: AA-22
        const startRank = ranks.indexOf(start[0]);
        const endRank = ranks.indexOf(end[0]);
        const minRank = Math.min(startRank, endRank);
        const maxRank = Math.max(startRank, endRank);
        
        for (let r = minRank; r <= maxRank; r++) {
            const pairHands = expandSingleHand(ranks[r] + ranks[r], ranks, suits, deadCards);
            hands.push(...pairHands);
        }
    } else if (start.endsWith('s')) {
        // 同花范围: AKs-A2s
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
        // 不同花范围: AKo-ATo
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
        // 对子: AA, KK, etc.
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
        // 同花: AKs, AQs, etc.
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
        // 不同花: AKo, AQo, etc.
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
        // 对子简写: AA
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
    } else if (handType.length === 3) {
        // 单个同花或不同花手牌: AKs, AKo
        if (handType.endsWith('s')) {
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
 * 计算范围vs范围的胜率（完整枚举所有组合对）
 */
async function calculateRangeVsRangeEquity(board, fixedOOPHand) {
    const boardCards = parseBoardString(board);
    const oopCards = fixedOOPHand.match(/.{2}/g);
    
    // 死牌 = 公共牌 + 固定的OOP手牌
    const deadCards = new Set([...boardCards, ...oopCards]);
    
    console.log(`\n${'='.repeat(70)}`);
    console.log(`公共牌: ${board}`);
    console.log(`固定OOP手牌: ${fixedOOPHand}`);
    console.log(`死牌: ${[...deadCards].join(', ')}`);
    
    // 展开OOP范围（排除死牌）
    const oopHands = expandRange(OOP_RANGE, deadCards);
    
    // 展开IP范围（排除死牌）
    const ipHands = expandRange(IP_RANGE, deadCards);
    
    console.log(`\nOOP范围有效组合数: ${oopHands.length}`);
    console.log(`IP范围有效组合数: ${ipHands.length}`);
    
    // 输出IP范围的所有组合
    console.log(`\nIP范围所有组合:`);
    const ipHandsStr = ipHands.map(h => h.join('')).join(', ');
    console.log(ipHandsStr);
    
    // 枚举所有OOP vs IP组合对
    let totalWins = 0;
    let totalTies = 0;
    let totalLosses = 0;
    let totalCombos = 0;
    let skippedConflicts = 0;
    
    console.log(`\n开始枚举所有组合对...`);
    
    // 记录一些样本对比
    const sampleResults = [];
    
    for (let i = 0; i < oopHands.length; i++) {
        const oopHand = oopHands[i];
        
        // 每100个OOP手牌输出一次进度
        if ((i + 1) % 100 === 0) {
            console.log(`  进度: ${i+1}/${oopHands.length} OOP手牌`);
        }
        
        for (const ipHand of ipHands) {
            // 检查OOP和IP手牌是否冲突
            if (handsConflict(oopHand, ipHand)) {
                skippedConflicts++;
                continue;
            }
            
            try {
                const table = new TexasHoldem();
                table.addPlayer(oopHand);
                table.addPlayer(ipHand);
                table.setBoard(boardCards);
                
                const result = table.calculate();
                const players = result.getPlayers();
                
                const oopWins = players[0].getWins();
                const oopTies = players[0].getTies();
                
                if (oopWins > 0) {
                    totalWins++;
                } else if (oopTies > 0) {
                    totalTies++;
                } else {
                    totalLosses++;
                }
                totalCombos++;
                
                // 记录前10个样本
                if (sampleResults.length < 10) {
                    sampleResults.push({
                        oop: oopHand.join(''),
                        ip: ipHand.join(''),
                        wins: oopWins,
                        ties: oopTies,
                        result: oopWins > 0 ? 'WIN' : (oopTies > 0 ? 'TIE' : 'LOSS')
                    });
                }
            } catch (e) {
                console.log(`错误: OOP=${oopHand.join('')} vs IP=${ipHand.join('')}: ${e.message}`);
            }
        }
    }
    
    console.log(`\n${'─'.repeat(50)}`);
    console.log(`统计结果:`);
    console.log(`  跳过的冲突组合: ${skippedConflicts}`);
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
    console.log(`\n  范围胜率 (胜 + 平/2): ${(equity * 100).toFixed(3)}%`);
    
    console.log(`\n前10个样本结果:`);
    for (const s of sampleResults) {
        console.log(`  ${s.oop} vs ${s.ip}: ${s.result} (wins=${s.wins}, ties=${s.ties})`);
    }
    
    return { equity, totalWins, totalTies, totalLosses, totalCombos };
}

// 运行验证 - 只计算Case 2
async function main() {
    console.log('调试版本：验证范围vs范围胜率计算');
    console.log(`\nOOP范围: ${OOP_RANGE}`);
    console.log(`IP范围: ${IP_RANGE}`);
    
    // Case 2: 公共牌 2h 8d 2c Jc Ts, OOP手牌 KdKs
    await calculateRangeVsRangeEquity('2h 8d 2c Jc Ts', 'KdKs');
}

main().catch(console.error);
