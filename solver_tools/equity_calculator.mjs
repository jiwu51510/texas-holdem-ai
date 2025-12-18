#!/usr/bin/env node
/**
 * 使用 poker-odds-calc 库计算胜率
 * https://github.com/siavashg87/poker-odds-calc
 */

import pkg from 'poker-odds-calc';
const { TexasHoldem } = pkg;

/**
 * 将牌索引(0-51)转换为poker-odds-calc格式的字符串
 * 索引格式: rank * 4 + suit, rank: 0=2, 1=3, ..., 12=A, suit: 0=c, 1=d, 2=h, 3=s
 * poker-odds-calc格式: "Ah", "Kd", etc.
 */
export function cardIndexToPokerOddsFormat(cardIndex) {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    const rank = Math.floor(cardIndex / 4);
    const suit = cardIndex % 4;
    return ranks[rank] + suits[suit];
}

/**
 * 将牌字符串转换为poker-odds-calc格式
 * 输入: "Kh" 或 "KhKs" (两张牌)
 * 输出: ["Kh"] 或 ["Kh", "Ks"]
 */
export function parseCardString(cardStr) {
    const cards = [];
    for (let i = 0; i < cardStr.length; i += 2) {
        cards.push(cardStr.substring(i, i + 2));
    }
    return cards;
}

/**
 * 将公共牌字符串转换为数组
 * 输入: "Jd 7c Qh 6s 7h"
 * 输出: ["Jd", "7c", "Qh", "6s", "7h"]
 */
export function parseBoardString(boardStr) {
    return boardStr.trim().split(/\s+/);
}

/**
 * 计算手牌vs对手范围的胜率
 * @param {string} heroHand - 英雄手牌，如 "KhKs"
 * @param {string} villainRange - 对手范围字符串
 * @param {string} board - 公共牌，如 "Jd 7c Qh 6s 7h"
 * @returns {number} - 胜率 (0-1)
 */
export async function calculateHandVsRangeEquity(heroHand, villainRange, board) {
    const heroCards = parseCardString(heroHand);
    const boardCards = parseBoardString(board);
    
    // 创建死牌集合（公共牌 + 英雄手牌）
    const deadCards = new Set([...boardCards, ...heroCards]);
    
    // 展开对手范围为具体手牌组合
    const villainHands = expandRange(villainRange, deadCards);
    
    if (villainHands.length === 0) {
        return 0.5; // 没有有效对手组合
    }
    
    // 对每个对手手牌计算胜率
    let totalWins = 0;
    let totalTies = 0;
    let totalCombos = villainHands.length;
    
    for (const villainHand of villainHands) {
        try {
            const table = new TexasHoldem();
            table.addPlayer(heroCards);
            table.addPlayer(villainHand);
            table.setBoard(boardCards);
            
            const result = table.calculate();
            const players = result.getPlayers();
            
            // 获取英雄的胜率和平局率
            const heroWins = players[0].getWins();
            const heroTies = players[0].getTies();
            const iterations = result.getIterations();
            
            // 河牌阶段只有1次迭代
            if (heroWins > 0) {
                totalWins++;
            } else if (heroTies > 0) {
                totalTies++;
            }
        } catch (e) {
            // 跳过无效组合
            totalCombos--;
        }
    }
    
    if (totalCombos === 0) {
        return 0.5;
    }
    
    // 胜率 = (赢的次数 + 平局次数 * 0.5) / 总组合数
    return (totalWins + totalTies * 0.5) / totalCombos;
}

/**
 * 展开范围字符串为具体手牌组合
 * @param {string} rangeStr - 范围字符串，如 "AA,KK,AKs,AKo"
 * @param {Set} deadCards - 死牌集合
 * @returns {string[][]} - 手牌组合数组，每个元素是两张牌的数组
 */
export function expandRange(rangeStr, deadCards) {
    const hands = [];
    const ranks = '23456789TJQKA';
    const suits = ['c', 'd', 'h', 's'];
    
    // 解析范围字符串
    const parts = rangeStr.split(',');
    
    for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed) continue;
        
        // 检查是否是范围表示法 (如 AA-22, AKs-A2s)
        if (trimmed.includes('-')) {
            const [start, end] = trimmed.split('-');
            const expandedHands = expandRangeNotation(start, end, ranks, suits, deadCards);
            hands.push(...expandedHands);
        } else {
            // 单个手牌类型
            const expandedHands = expandSingleHand(trimmed, ranks, suits, deadCards);
            hands.push(...expandedHands);
        }
    }
    
    return hands;
}

/**
 * 展开范围表示法
 */
function expandRangeNotation(start, end, ranks, suits, deadCards) {
    const hands = [];
    
    // 判断类型：对子范围、同花范围、非同花范围
    if (start.length === 2 && start[0] === start[1]) {
        // 对子范围 (如 AA-22)
        const startRank = ranks.indexOf(start[0]);
        const endRank = ranks.indexOf(end[0]);
        const minRank = Math.min(startRank, endRank);
        const maxRank = Math.max(startRank, endRank);
        
        for (let r = minRank; r <= maxRank; r++) {
            const pairHands = expandSingleHand(ranks[r] + ranks[r], ranks, suits, deadCards);
            hands.push(...pairHands);
        }
    } else if (start.endsWith('s')) {
        // 同花范围 (如 AKs-A2s)
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
        // 非同花范围 (如 AKo-ATo)
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

/**
 * 展开单个手牌类型为所有具体组合
 */
function expandSingleHand(handType, ranks, suits, deadCards) {
    const hands = [];
    
    if (handType.length === 2) {
        // 对子 (如 AA)
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
        // 同花 (如 AKs)
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
        // 非同花 (如 AKo)
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
        // 没有后缀的两张不同牌 - 包括同花和非同花
        const rank1 = handType[0];
        const rank2 = handType[1];
        // 同花
        for (let s = 0; s < 4; s++) {
            const card1 = rank1 + suits[s];
            const card2 = rank2 + suits[s];
            if (!deadCards.has(card1) && !deadCards.has(card2)) {
                hands.push([card1, card2]);
            }
        }
        // 非同花
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
    
    return hands;
}

/**
 * 计算范围vs范围的平均胜率
 * @param {string} heroRange - 英雄范围字符串
 * @param {string} villainRange - 对手范围字符串
 * @param {string} board - 公共牌
 * @returns {number} - 平均胜率 (0-1)
 */
export async function calculateRangeVsRangeEquity(heroRange, villainRange, board) {
    const boardCards = parseBoardString(board);
    const deadCards = new Set(boardCards);
    
    // 展开英雄范围
    const heroHands = expandRange(heroRange, deadCards);
    
    if (heroHands.length === 0) {
        return 0.5;
    }
    
    let totalEquity = 0;
    let validHands = 0;
    
    for (const heroHand of heroHands) {
        const heroHandStr = heroHand.join('');
        const heroDeadCards = new Set([...deadCards, ...heroHand]);
        
        // 展开对手范围（排除英雄手牌）
        const villainHands = expandRange(villainRange, heroDeadCards);
        
        if (villainHands.length === 0) continue;
        
        let wins = 0;
        let ties = 0;
        let combos = villainHands.length;
        
        for (const villainHand of villainHands) {
            try {
                const table = new TexasHoldem();
                table.addPlayer(heroHand);
                table.addPlayer(villainHand);
                table.setBoard(boardCards);
                
                const result = table.calculate();
                const players = result.getPlayers();
                
                if (players[0].getWins() > 0) {
                    wins++;
                } else if (players[0].getTies() > 0) {
                    ties++;
                }
            } catch (e) {
                combos--;
            }
        }
        
        if (combos > 0) {
            const equity = (wins + ties * 0.5) / combos;
            totalEquity += equity;
            validHands++;
        }
    }
    
    return validHands > 0 ? totalEquity / validHands : 0.5;
}

// 测试
async function test() {
    console.log('测试 poker-odds-calc 胜率计算...\n');
    
    const board = 'Jd 7c Qh 6s 7h';
    const heroHand = 'KhKs';
    const villainRange = 'AA,KK,QQ,JJ,TT,99,88,77,AKs,AKo';
    
    console.log(`公共牌: ${board}`);
    console.log(`英雄手牌: ${heroHand}`);
    console.log(`对手范围: ${villainRange}`);
    
    const equity = await calculateHandVsRangeEquity(heroHand, villainRange, board);
    console.log(`\n手牌胜率: ${(equity * 100).toFixed(3)}%`);
}

// 如果直接运行此文件，执行测试
import { fileURLToPath } from 'url';
if (process.argv[1] === fileURLToPath(import.meta.url)) {
    test().catch(console.error);
}
