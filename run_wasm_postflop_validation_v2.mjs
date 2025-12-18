#!/usr/bin/env node
/**
 * 使用wasm-postflop进行跨公共牌双维度胜率-策略验证实验 V2
 * 
 * 改进：使用 poker-odds-calc 独立计算胜率，而不是依赖 solver 的结果
 * 
 * 核心问题：在不同的（公共牌+固定手牌）组合下：
 * 当以下两个条件同时满足时，策略是否相同？
 * 1. 固定手牌vs对手范围的胜率相近（差异<0.1%）
 * 2. 自己范围vs对手范围的胜率相近（差异<0.1%）
 */

import { solveRiver, parseBoard } from './solver_tools/postflop_solver.mjs';
import { calculateHandVsRangeEquity, calculateRangeVsRangeEquity, expandRange, parseBoardString } from './solver_tools/equity_calculator.mjs';
import { writeFileSync } from 'fs';

// 范围定义
const OOP_RANGE = 'AA-22,AKs-A2s,KQs-K2s,QJs-Q2s,JTs-J6s,T9s-T6s,98s-96s,87s-86s,76s-75s,65s-64s,54s,AKo-A2o,KQo-K9o,QJo-Q9o,JTo-J9o,T9o';
const IP_RANGE = 'AA-22,AKs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,65s,AKo-ATo,KQo-KJo,QJo';

// 生成随机公共牌
function generateRandomBoard() {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    const cards = [];
    const usedCards = new Set();
    
    while (cards.length < 5) {
        const rank = Math.floor(Math.random() * 13);
        const suit = Math.floor(Math.random() * 4);
        const cardStr = ranks[rank] + suits[suit];
        
        if (!usedCards.has(cardStr)) {
            usedCards.add(cardStr);
            cards.push(cardStr);
        }
    }
    
    return cards.join(' ');
}

// 为给定公共牌随机选择一个不冲突的手牌
function selectRandomHeroHand(board) {
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    const boardCards = new Set(parseBoardString(board));
    
    // 生成所有可能的手牌组合（在OOP范围内且不与公共牌冲突）
    const validHands = [];
    
    // 口袋对
    for (let r = 0; r < 13; r++) {
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = s1 + 1; s2 < 4; s2++) {
                const card1 = ranks[r] + suits[s1];
                const card2 = ranks[r] + suits[s2];
                if (!boardCards.has(card1) && !boardCards.has(card2)) {
                    validHands.push({
                        hand: card1 + card2,
                        cards: [card1, card2]
                    });
                }
            }
        }
    }
    
    // 同花连接和高牌
    const suitedCombos = [
        [12, 11], [11, 10], [10, 9], [9, 8], [8, 7], [7, 6],
        [12, 10], [12, 9], [11, 9],
    ];
    
    for (const [r1, r2] of suitedCombos) {
        for (let s = 0; s < 4; s++) {
            const card1 = ranks[r1] + suits[s];
            const card2 = ranks[r2] + suits[s];
            if (!boardCards.has(card1) && !boardCards.has(card2)) {
                validHands.push({
                    hand: card1 + card2,
                    cards: [card1, card2]
                });
            }
        }
    }
    
    // 非同花高牌
    const offsuitCombos = [
        [12, 11], [12, 10], [12, 9], [11, 10], [11, 9],
    ];
    
    for (const [r1, r2] of offsuitCombos) {
        for (let s1 = 0; s1 < 4; s1++) {
            for (let s2 = 0; s2 < 4; s2++) {
                if (s1 !== s2) {
                    const card1 = ranks[r1] + suits[s1];
                    const card2 = ranks[r2] + suits[s2];
                    if (!boardCards.has(card1) && !boardCards.has(card2)) {
                        validHands.push({
                            hand: card1 + card2,
                            cards: [card1, card2]
                        });
                    }
                }
            }
        }
    }
    
    if (validHands.length === 0) return null;
    
    const idx = Math.floor(Math.random() * validHands.length);
    return validHands[idx];
}

/**
 * 在oopCards中找到手牌索引
 */
function findHandIndex(oopCards, heroCards) {
    // 将字符串格式转换为索引格式
    const ranks = '23456789TJQKA';
    const suits = 'cdhs';
    
    const heroIndices = heroCards.map(card => {
        const rank = ranks.indexOf(card[0]);
        const suit = suits.indexOf(card[1]);
        return rank * 4 + suit;
    }).sort((a, b) => a - b);
    
    for (let i = 0; i < oopCards.length; i++) {
        const handIdx = oopCards[i];
        const c1 = handIdx & 0xFF;
        const c2 = (handIdx >> 8) & 0xFF;
        const cards = [c1, c2].sort((a, b) => a - b);
        
        if (cards[0] === heroIndices[0] && cards[1] === heroIndices[1]) {
            return i;
        }
    }
    
    return -1;
}

/**
 * 从solver结果中提取特定手牌的策略
 */
function extractStrategy(solverResult, heroCards) {
    const { oopCards, ipCards, results, numActions, actions } = solverResult;
    
    const heroIndex = findHandIndex(oopCards, heroCards);
    if (heroIndex === -1) {
        return null;
    }
    
    const actionList = actions.split('/').map(a => {
        const match = a.match(/(\w+):(\d+)/);
        if (match) {
            return { name: match[1], amount: parseInt(match[2]) };
        }
        return { name: a, amount: 0 };
    });
    
    const oopLen = oopCards.length;
    const ipLen = ipCards.length;
    
    // 跳过header, weights, normalizer, equity, ev, eqr
    let offset = 3;
    offset += oopLen + ipLen; // weights
    offset += oopLen + ipLen; // normalizer
    offset += oopLen + ipLen; // equity
    offset += oopLen + ipLen; // ev
    offset += oopLen + ipLen; // eqr
    
    // strategy
    const strategyData = results.slice(offset, offset + numActions * oopLen);
    
    // 提取策略
    const strategy = {};
    for (let i = 0; i < numActions; i++) {
        strategy[actionList[i].name] = strategyData[i * oopLen + heroIndex];
    }
    
    return {
        strategy,
        actions: actionList,
    };
}

/**
 * 运行单个场景的求解
 */
async function solveScenario(board, oopRange, ipRange) {
    try {
        const result = await solveRiver({
            oopRange,
            ipRange,
            board,
            startingPot: 100,
            effectiveStack: 100,
            oopBetSizes: '50',
            ipBetSizes: '50',
            targetExploitability: 0.3,
            maxIterations: 500,
        });
        
        return result;
    } catch (e) {
        console.error(`Error solving ${board}:`, e.message);
        return null;
    }
}

/**
 * 计算策略差异
 */
function calculateStrategyDiff(strategy1, strategy2) {
    const keys = new Set([...Object.keys(strategy1), ...Object.keys(strategy2)]);
    let diff = 0;
    for (const key of keys) {
        diff += Math.abs((strategy1[key] || 0) - (strategy2[key] || 0));
    }
    return diff / keys.size;
}

/**
 * 主实验函数
 */
async function runExperiment() {
    console.log('='.repeat(80));
    console.log('使用wasm-postflop + poker-odds-calc 进行跨公共牌双维度胜率-策略验证实验 V2');
    console.log('='.repeat(80));
    console.log('\n核心改进: 使用 poker-odds-calc 独立计算胜率，不依赖 solver 结果');
    console.log('\n核心问题: 在不同的（公共牌+固定手牌）组合下：');
    console.log('  当手牌胜率和范围胜率都相近时，策略是否相同？');
    console.log('\n关键约束: 每个公共牌场景只选择一个固定手牌');
    console.log('\n范围定义:');
    console.log(`  OOP: ${OOP_RANGE}`);
    console.log(`  IP: ${IP_RANGE}`);
    
    // 生成场景
    const NUM_SCENARIOS = 10000; // 10000个场景
    const scenarios = [];
    
    console.log(`\n生成 ${NUM_SCENARIOS} 个（公共牌+固定手牌）场景...`);
    console.log('注意：使用 poker-odds-calc 完整枚举计算胜率，速度较慢\n');
    
    const startTime = Date.now();
    
    for (let i = 0; i < NUM_SCENARIOS; i++) {
        const board = generateRandomBoard();
        const heroHandInfo = selectRandomHeroHand(board);
        
        if (!heroHandInfo) {
            continue;
        }
        
        // 每10个场景输出一次进度（因为范围vs范围计算很慢）
        if ((i + 1) % 10 === 0 || i === 0) {
            const elapsed = (Date.now() - startTime) / 1000;
            const rate = elapsed > 0 ? (i + 1) / elapsed : 1;
            const remaining = (NUM_SCENARIOS - i - 1) / rate;
            console.log(`进度: ${i+1}/${NUM_SCENARIOS} - 已用时: ${elapsed.toFixed(0)}s - 预计剩余: ${remaining.toFixed(0)}s`);
        }
        
        // 0. 计算IP有效范围 = IP_RANGE - 公共牌 - OOP固定手牌
        const boardCards = parseBoardString(board);
        const deadCards = new Set([...boardCards, ...heroHandInfo.cards]);
        const effectiveIPHands = expandRange(IP_RANGE, deadCards);
        
        if (effectiveIPHands.length === 0) continue;
        
        // 将有效IP范围转换为字符串格式（用于solver和胜率计算）
        const effectiveIPRangeStr = effectiveIPHands.map(h => h.join('')).join(',');
        
        // 1. 计算手牌vs有效IP范围的胜率
        const heroEquityResult = await calculateHandVsRangeEquityDirect(
            heroHandInfo.cards,
            effectiveIPHands,
            boardCards
        );
        const heroEquity = heroEquityResult.equity;
        
        // 2. 计算OOP范围vs有效IP范围的胜率（OOP范围保持原始范围不变）
        const rangeEquityResult = await calculateRangeVsRangeEquityDirect(OOP_RANGE, effectiveIPHands, board, heroHandInfo.cards);
        const rangeEquity = rangeEquityResult.equity;
        
        // 3. 使用solver获取策略（使用有效IP范围）
        const solverResult = await solveScenario(board, OOP_RANGE, effectiveIPRangeStr);
        if (!solverResult) continue;
        
        const strategyData = extractStrategy(solverResult, heroHandInfo.cards);
        if (!strategyData) continue;
        
        // 详细日志只在前10个场景输出
        if (i < 10) {
            console.log(`  场景 ${i+1}: ${board} + ${heroHandInfo.hand} -> 手牌胜率: ${(heroEquity * 100).toFixed(3)}%, 范围胜率: ${(rangeEquity * 100).toFixed(3)}%`);
        }
        
        // 保存有效IP范围信息
        const effectiveIPRangeInfo = {
            count: effectiveIPHands.length,
            hands: effectiveIPHands.map(h => h.join('')).sort(),
        };
        
        scenarios.push({
            board,
            heroHand: heroHandInfo.hand,
            heroCards: heroHandInfo.cards,
            heroEquity,
            heroEquityDetail: {
                wins: heroEquityResult.wins,
                ties: heroEquityResult.ties,
                combos: heroEquityResult.combos,
                winRate: heroEquityResult.winRate,
                tieRate: heroEquityResult.tieRate,
            },
            rangeEquity,
            rangeEquityDetail: {
                wins: rangeEquityResult.wins,
                ties: rangeEquityResult.ties,
                combos: rangeEquityResult.combos,
                oopHandCount: rangeEquityResult.oopHandCount,
                ipHandCount: rangeEquityResult.ipHandCount,
                winRate: rangeEquityResult.combos > 0 ? rangeEquityResult.wins / rangeEquityResult.combos : 0,
                tieRate: rangeEquityResult.combos > 0 ? rangeEquityResult.ties / rangeEquityResult.combos : 0,
            },
            strategy: strategyData.strategy,
            actions: strategyData.actions,
            effectiveIPRange: effectiveIPRangeInfo,
        });
    }
    
    const totalTime = (Date.now() - startTime) / 1000;
    console.log(`\n\n成功生成 ${scenarios.length} 个场景，总用时: ${totalTime.toFixed(1)}s`);
    
    // 分析结果
    analyzeResults(scenarios);
}

/**
 * 直接使用已展开的IP范围计算手牌胜率
 * @param {string[]} heroCards - OOP手牌 [card1, card2]
 * @param {string[][]} villainHands - 已展开的IP范围
 * @param {string[]} boardCards - 公共牌数组
 * @returns {object} - 包含胜率、胜场、平局、总组合数的对象
 */
async function calculateHandVsRangeEquityDirect(heroCards, villainHands, boardCards) {
    const { TexasHoldem } = (await import('poker-odds-calc')).default;
    
    // 排除与OOP手牌冲突的IP组合
    const heroSet = new Set(heroCards);
    const validVillainHands = villainHands.filter(vh => 
        !heroSet.has(vh[0]) && !heroSet.has(vh[1])
    );
    
    if (validVillainHands.length === 0) {
        return { equity: 0.5, wins: 0, ties: 0, combos: 0, winRate: 0, tieRate: 0 };
    }
    
    let wins = 0;
    let ties = 0;
    let combos = validVillainHands.length;
    
    for (const villainHand of validVillainHands) {
        try {
            const table = new TexasHoldem();
            table.addPlayer(heroCards);
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
    
    const equity = combos > 0 ? (wins + ties * 0.5) / combos : 0.5;
    const winRate = combos > 0 ? wins / combos : 0;
    const tieRate = combos > 0 ? ties / combos : 0;
    
    return { equity, wins, ties, combos, winRate, tieRate };
}

/**
 * 直接使用已展开的IP范围计算范围vs范围胜率
 * 
 * 重要：OOP范围保持原始范围不变（只排除公共牌），不排除固定OOP手牌
 * IP有效范围已经在调用前排除了公共牌和固定OOP手牌
 * 
 * @param {string} heroRange - OOP范围字符串
 * @param {string[][]} villainHands - 已展开的IP范围（已排除公共牌和固定OOP手牌）
 * @param {string} board - 公共牌
 * @param {string[]} fixedHeroCards - 固定的OOP手牌（仅用于记录，不用于排除OOP范围）
 * @returns {object} - 包含胜率、胜场、平局、总组合数的对象
 */
async function calculateRangeVsRangeEquityDirect(heroRange, villainHands, board, fixedHeroCards) {
    const { TexasHoldem } = (await import('poker-odds-calc')).default;
    
    const boardCards = parseBoardString(board);
    
    // 死牌 = 只有公共牌（OOP范围保持原始范围不变）
    const deadCards = new Set(boardCards);
    
    // 展开OOP范围（只排除公共牌，不排除固定OOP手牌）
    const heroHands = expandRange(heroRange, deadCards);
    
    if (heroHands.length === 0) {
        return { equity: 0.5, wins: 0, ties: 0, combos: 0 };
    }
    
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
    
    const equity = totalCombos > 0 ? (totalWins + totalTies * 0.5) / totalCombos : 0.5;
    
    return {
        equity,
        wins: totalWins,
        ties: totalTies,
        combos: totalCombos,
        oopHandCount: heroHands.length,
        ipHandCount: villainHands.length
    };
}

/**
 * 完整枚举计算范围vs范围胜率（考虑额外死牌）- 旧版本，保留兼容
 * 
 * 重要：当我们固定了一个OOP手牌后，IP范围中与这个手牌冲突的组合应该被排除
 * 
 * @param {string} heroRange - OOP范围
 * @param {string} villainRange - IP范围
 * @param {string} board - 公共牌
 * @param {string[]} fixedHeroCards - 固定的OOP手牌（作为死牌）
 */
async function calculateRangeVsRangeEquityWithDeadCards(heroRange, villainRange, board, fixedHeroCards = null) {
    const { TexasHoldem } = (await import('poker-odds-calc')).default;
    
    const boardCards = parseBoardString(board);
    
    // 死牌 = 公共牌 + 固定的OOP手牌
    const deadCards = new Set(boardCards);
    if (fixedHeroCards) {
        fixedHeroCards.forEach(card => deadCards.add(card));
    }
    
    // 展开OOP范围（排除死牌）
    const heroHands = expandRange(heroRange, deadCards);
    
    if (heroHands.length === 0) return 0.5;
    
    let totalEquity = 0;
    let validHands = 0;
    
    // 遍历OOP范围中的每个手牌
    for (const heroHand of heroHands) {
        // 对于每个OOP手牌，死牌 = 公共牌 + 固定OOP手牌 + 当前OOP手牌
        const heroDeadCards = new Set([...deadCards, ...heroHand]);
        
        // 展开IP范围（排除死牌）
        const villainHands = expandRange(villainRange, heroDeadCards);
        
        if (villainHands.length === 0) continue;
        
        let wins = 0;
        let ties = 0;
        let combos = villainHands.length;
        
        // 遍历IP范围中的每个手牌
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

/**
 * 计算手牌vs对手范围的胜率（考虑额外死牌）
 */
async function calculateHandVsRangeEquityWithDeadCards(heroHand, villainRange, board, extraDeadCards = null) {
    const { TexasHoldem } = (await import('poker-odds-calc')).default;
    
    const heroCards = heroHand.match(/.{2}/g);
    const boardCards = parseBoardString(board);
    
    // 创建死牌集合（公共牌 + 英雄手牌 + 额外死牌）
    const deadCards = new Set([...boardCards, ...heroCards]);
    if (extraDeadCards) {
        extraDeadCards.forEach(card => deadCards.add(card));
    }
    
    // 展开对手范围为具体手牌组合
    const villainHands = expandRange(villainRange, deadCards);
    
    if (villainHands.length === 0) {
        return 0.5;
    }
    
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
            
            if (players[0].getWins() > 0) {
                totalWins++;
            } else if (players[0].getTies() > 0) {
                totalTies++;
            }
        } catch (e) {
            totalCombos--;
        }
    }
    
    if (totalCombos === 0) {
        return 0.5;
    }
    
    return (totalWins + totalTies * 0.5) / totalCombos;
}

/**
 * 获取IP范围去掉死牌后的真实范围
 * @param {string} board - 公共牌
 * @param {string[]} heroCards - OOP手牌
 * @returns {object} - 真实IP范围的详细信息
 */
function getEffectiveIPRange(board, heroCards) {
    const boardCards = new Set(parseBoardString(board));
    const deadCards = new Set([...boardCards, ...heroCards]);
    
    // 展开IP范围
    const ipHands = expandRange(IP_RANGE, deadCards);
    
    // 将手牌组合转换为可读格式
    const handStrings = ipHands.map(hand => hand.join('')).sort();
    
    // 生成简化的范围表示（按手牌类型分组）
    const summary = summarizeRange(ipHands);
    
    return {
        count: ipHands.length,
        hands: handStrings,
        summary
    };
}

/**
 * 将手牌组合列表转换为简化的范围表示
 */
function summarizeRange(hands) {
    const ranks = '23456789TJQKA';
    const handTypes = {};
    
    for (const hand of hands) {
        const card1 = hand[0];
        const card2 = hand[1];
        const rank1 = ranks.indexOf(card1[0]);
        const rank2 = ranks.indexOf(card2[0]);
        const suit1 = card1[1];
        const suit2 = card2[1];
        
        let handType;
        if (rank1 === rank2) {
            // 对子
            handType = card1[0] + card2[0];
        } else {
            // 非对子
            const highRank = Math.max(rank1, rank2);
            const lowRank = Math.min(rank1, rank2);
            const isSuited = suit1 === suit2;
            handType = ranks[highRank] + ranks[lowRank] + (isSuited ? 's' : 'o');
        }
        
        if (!handTypes[handType]) {
            handTypes[handType] = 0;
        }
        handTypes[handType]++;
    }
    
    // 按类型排序输出
    const sortedTypes = Object.keys(handTypes).sort((a, b) => {
        const aRank1 = ranks.indexOf(a[0]);
        const bRank1 = ranks.indexOf(b[0]);
        if (aRank1 !== bRank1) return bRank1 - aRank1;
        const aRank2 = ranks.indexOf(a[1]);
        const bRank2 = ranks.indexOf(b[1]);
        return bRank2 - aRank2;
    });
    
    return {
        text: sortedTypes.map(t => `${t}(${handTypes[t]})`).join(', '),
        types: handTypes,
        sortedTypes
    };
}

/**
 * 生成IP范围的完整枚举格式（用于markdown报告）
 * 输出所有具体的手牌组合，如 KsJd, KsJh
 */
function formatRangeAsTable(rangeInfo) {
    // 直接使用hands数组，它包含所有具体的手牌组合
    const hands = rangeInfo.hands;
    
    // 将手牌组合格式化为 "XxYy" 格式（如 KsJd）
    // hands已经是 ["KsJd", "KsJh", ...] 格式
    const formattedHands = hands.map(h => {
        // 确保格式正确：如果是4字符格式如"KsJd"，直接使用
        // 如果是其他格式，需要转换
        if (h.length === 4) {
            return h;
        }
        return h;
    });
    
    // 按照牌力排序（高牌在前）
    const ranks = '23456789TJQKA';
    formattedHands.sort((a, b) => {
        const aRank1 = ranks.indexOf(a[0]);
        const bRank1 = ranks.indexOf(b[0]);
        if (aRank1 !== bRank1) return bRank1 - aRank1;
        const aRank2 = ranks.indexOf(a[2]);
        const bRank2 = ranks.indexOf(b[2]);
        return bRank2 - aRank2;
    });
    
    // 输出所有组合，用逗号分隔
    return `  \`${formattedHands.join(', ')}\`\n`;
}

/**
 * 分析结果
 */
function analyzeResults(scenarios) {
    console.log('\n' + '='.repeat(80));
    console.log('分析结果：寻找四维度胜率相近的场景对（阈值: 0.1%）');
    console.log('条件：手牌胜率、手牌平局率、范围胜率、范围平局率 都相差不超过0.1%');
    console.log('='.repeat(80));
    
    // 使用0.1%的阈值
    const threshold = 0.001; // 0.1%
    const pairs = [];
    
    for (let i = 0; i < scenarios.length; i++) {
        for (let j = i + 1; j < scenarios.length; j++) {
            const s1 = scenarios[i];
            const s2 = scenarios[j];
            
            // 手牌胜率和平局率
            const heroWinDiff = Math.abs(s1.heroEquityDetail.winRate - s2.heroEquityDetail.winRate);
            const heroTieDiff = Math.abs(s1.heroEquityDetail.tieRate - s2.heroEquityDetail.tieRate);
            
            // 范围胜率和平局率
            const rangeWinDiff = Math.abs(s1.rangeEquityDetail.winRate - s2.rangeEquityDetail.winRate);
            const rangeTieDiff = Math.abs(s1.rangeEquityDetail.tieRate - s2.rangeEquityDetail.tieRate);
            
            // 四个条件都要满足
            if (heroWinDiff < threshold && heroTieDiff < threshold && 
                rangeWinDiff < threshold && rangeTieDiff < threshold) {
                const strategyDiff = calculateStrategyDiff(s1.strategy, s2.strategy);
                
                // 使用已保存的有效IP范围
                const ipRange1 = s1.effectiveIPRange || getEffectiveIPRange(s1.board, s1.heroCards);
                const ipRange2 = s2.effectiveIPRange || getEffectiveIPRange(s2.board, s2.heroCards);
                
                pairs.push({
                    scenario1: s1,
                    scenario2: s2,
                    heroWinDiff,
                    heroTieDiff,
                    rangeWinDiff,
                    rangeTieDiff,
                    strategyDiff,
                    ipRange1,
                    ipRange2,
                });
            }
        }
    }
    
    console.log(`\n找到 ${pairs.length} 对四维度胜率相近的场景`);
    
    const significantPairs = pairs.filter(p => p.strategyDiff > 0.15);
    console.log(`其中策略差异显著(>15%)的: ${significantPairs.length} 对`);
    
    // 输出反例
    if (significantPairs.length > 0) {
        console.log('\n【策略差异显著的反例】');
        for (const p of significantPairs.slice(0, 10)) {
            const hd1 = p.scenario1.heroEquityDetail;
            const hd2 = p.scenario2.heroEquityDetail;
            const rd1 = p.scenario1.rangeEquityDetail;
            const rd2 = p.scenario2.rangeEquityDetail;
            
            console.log(`\n${'─'.repeat(70)}`);
            console.log(`【场景1】`);
            console.log(`  公共牌: ${p.scenario1.board}`);
            console.log(`  OOP手牌: ${p.scenario1.heroHand}`);
            console.log(`  手牌: 胜率=${(hd1.winRate*100).toFixed(3)}%, 平局率=${(hd1.tieRate*100).toFixed(3)}%`);
            console.log(`  范围: 胜率=${(rd1.winRate*100).toFixed(3)}%, 平局率=${(rd1.tieRate*100).toFixed(3)}%`);
            console.log(`  策略: ${JSON.stringify(p.scenario1.strategy)}`);
            
            console.log(`\n【场景2】`);
            console.log(`  公共牌: ${p.scenario2.board}`);
            console.log(`  OOP手牌: ${p.scenario2.heroHand}`);
            console.log(`  手牌: 胜率=${(hd2.winRate*100).toFixed(3)}%, 平局率=${(hd2.tieRate*100).toFixed(3)}%`);
            console.log(`  范围: 胜率=${(rd2.winRate*100).toFixed(3)}%, 平局率=${(rd2.tieRate*100).toFixed(3)}%`);
            console.log(`  策略: ${JSON.stringify(p.scenario2.strategy)}`);
            
            console.log(`\n【对比】`);
            console.log(`  手牌胜率差异: ${(p.heroWinDiff*100).toFixed(3)}%, 手牌平局率差异: ${(p.heroTieDiff*100).toFixed(3)}%`);
            console.log(`  范围胜率差异: ${(p.rangeWinDiff*100).toFixed(3)}%, 范围平局率差异: ${(p.rangeTieDiff*100).toFixed(3)}%`);
            console.log(`  策略差异: ${(p.strategyDiff*100).toFixed(1)}%`);
        }
    }
    
    // 保存结果
    const outputData = {
        method: 'poker-odds-calc独立计算胜率',
        oopRange: OOP_RANGE,
        ipRange: IP_RANGE,
        numScenarios: scenarios.length,
        scenarios,
        pairs: pairs.map(p => ({
            board1: p.scenario1.board,
            heroHand1: p.scenario1.heroHand,
            board2: p.scenario2.board,
            heroHand2: p.scenario2.heroHand,
            heroEq1: p.scenario1.heroEquity,
            heroEq2: p.scenario2.heroEquity,
            rangeEq1: p.scenario1.rangeEquity,
            rangeEq2: p.scenario2.rangeEquity,
            strategy1: p.scenario1.strategy,
            strategy2: p.scenario2.strategy,
            strategyDiff: p.strategyDiff,
            ipRange1: p.ipRange1,
            ipRange2: p.ipRange2,
        })),
    };
    
    const outputPath = 'experiments/results/wasm_postflop_validation_v2.json';
    writeFileSync(outputPath, JSON.stringify(outputData, null, 2));
    console.log(`\n结果已保存到: ${outputPath}`);
    
    // 生成报告
    generateReport(scenarios, pairs, significantPairs);
}

/**
 * 生成实验报告
 */
function generateReport(scenarios, pairs, significantPairs) {
    let report = `# 跨公共牌四维度胜率-策略验证实验报告 V2

## 实验改进

**本版本使用 poker-odds-calc 库独立计算胜率，而不是依赖 solver 的结果。**

这确保了胜率计算的独立性和准确性。

## 实验目的

验证：**在不同的（公共牌+固定手牌）组合下：**
当以下四个条件同时满足时，策略是否相同？
1. 固定手牌vs对手范围的胜率相近（差异<0.1%）
2. 固定手牌vs对手范围的平局率相近（差异<0.1%）
3. 自己范围vs对手范围的胜率相近（差异<0.1%）
4. 自己范围vs对手范围的平局率相近（差异<0.1%）

## 实验方法

1. 随机生成公共牌场景
2. 为每个公共牌场景随机选择一个固定手牌
3. **使用 poker-odds-calc 完整枚举计算胜率**：
   - 手牌胜率/平局率：遍历对手范围内所有有效组合，分别计算胜率和平局率
   - 范围胜率/平局率：完整枚举OOP范围和IP范围的所有组合对，分别计算胜率和平局率
4. 使用 wasm-postflop solver 获取最优策略
5. 比较四维度胜率都相近的场景对的策略差异

## 范围定义

- **OOP范围**: ${OOP_RANGE}
- **IP范围**: ${IP_RANGE}

## 实验规模

- 生成场景数: ${scenarios.length}
- 四维度胜率相近的场景对（差异<0.1%）: ${pairs.length}
- 策略差异显著(>15%)的场景对: ${significantPairs.length}

## 关键发现

`;

    if (significantPairs.length > 0) {
        report += `### 发现四维度胜率相近但策略不同的反例

以下是策略差异显著的反例详情（包含IP去掉死牌后的真实范围）：

`;
        
        let caseNum = 1;
        for (const p of significantPairs.slice(0, 20)) {
            // 格式化手牌胜率详情
            const hd1 = p.scenario1.heroEquityDetail;
            const hd2 = p.scenario2.heroEquityDetail;
            const rd1 = p.scenario1.rangeEquityDetail;
            const rd2 = p.scenario2.rangeEquityDetail;
            
            const heroEquityStr1 = `胜率: ${(hd1.winRate*100).toFixed(3)}%, 平局率: ${(hd1.tieRate*100).toFixed(3)}%`;
            const heroEquityStr2 = `胜率: ${(hd2.winRate*100).toFixed(3)}%, 平局率: ${(hd2.tieRate*100).toFixed(3)}%`;
            const rangeEquityStr1 = `胜率: ${(rd1.winRate*100).toFixed(3)}%, 平局率: ${(rd1.tieRate*100).toFixed(3)}%`;
            const rangeEquityStr2 = `胜率: ${(rd2.winRate*100).toFixed(3)}%, 平局率: ${(rd2.tieRate*100).toFixed(3)}%`;
            
            report += `---

#### 反例 ${caseNum}

**场景1:**
- 公共牌: \`${p.scenario1.board}\`
- OOP手牌: \`${p.scenario1.heroHand}\`
- 手牌: ${heroEquityStr1}
- 范围: ${rangeEquityStr1}
- 策略: \`${JSON.stringify(p.scenario1.strategy)}\`
- IP有效范围 (${p.ipRange1.count}个组合):
${formatRangeAsTable(p.ipRange1)}
**场景2:**
- 公共牌: \`${p.scenario2.board}\`
- OOP手牌: \`${p.scenario2.heroHand}\`
- 手牌: ${heroEquityStr2}
- 范围: ${rangeEquityStr2}
- 策略: \`${JSON.stringify(p.scenario2.strategy)}\`
- IP有效范围 (${p.ipRange2.count}个组合):
${formatRangeAsTable(p.ipRange2)}
**对比:**
- 手牌胜率差异: ${(p.heroWinDiff*100).toFixed(3)}%, 手牌平局率差异: ${(p.heroTieDiff*100).toFixed(3)}%
- 范围胜率差异: ${(p.rangeWinDiff*100).toFixed(3)}%, 范围平局率差异: ${(p.rangeTieDiff*100).toFixed(3)}%
- **策略差异: ${(p.strategyDiff*100).toFixed(1)}%**

`;
            caseNum++;
        }
    } else if (pairs.length > 0) {
        report += `在找到的 ${pairs.length} 对四维度胜率相近的场景中，未发现策略显著不同的反例。\n`;
    } else {
        report += `在 ${scenarios.length} 个场景中，未找到四维度胜率都相近（差异<0.1%）的场景对。\n`;
        report += `这可能是因为样本量较小或阈值过于严格。\n`;
    }
    
    report += `\n## 结论

`;
    
    if (significantPairs.length > 0) {
        const ratio = (significantPairs.length / pairs.length * 100).toFixed(1);
        report += `### ⚠️ 四维度胜率标量不足以决定最优策略

使用独立的胜率计算方法验证后，实验发现：在 ${pairs.length} 对四维度胜率相近的场景中，有 ${significantPairs.length} 对（${ratio}%）的策略差异显著。

**结论：即使手牌胜率、手牌平局率、范围胜率、范围平局率都精确匹配（差异<0.1%），最优策略仍然可能完全不同。**

### 分析

从反例中可以看出，即使两个场景的四个胜率维度都非常接近，但由于：
1. IP的有效范围组合不同（死牌不同导致）
2. 公共牌结构不同（顺子/同花可能性不同）
3. 手牌与公共牌的互动不同

这些因素导致最优策略可能完全不同。这证明了**仅靠四个胜率标量无法替代完整的博弈论求解**。
`;
    } else if (pairs.length > 0) {
        report += `### ✅ 未发现反例

在找到的 ${pairs.length} 对四维度胜率相近的场景中，未发现策略显著不同的反例。

这可能意味着四维度胜率（手牌胜率、手牌平局率、范围胜率、范围平局率）在0.1%阈值下能够较好地预测策略。
`;
    } else {
        report += `### 需要更多数据或调整阈值

在 ${scenarios.length} 个场景中，未找到四维度胜率都相近（差异<0.1%）的场景对。

这可能是因为：
1. 0.1%的阈值过于严格
2. 需要更大的样本量
`;
    }
    
    const reportPath = 'experiments/results/wasm_postflop_validation_v2_report.md';
    writeFileSync(reportPath, report);
    console.log(`\n报告已保存到: ${reportPath}`);
}

// 运行实验
runExperiment().catch(console.error);
