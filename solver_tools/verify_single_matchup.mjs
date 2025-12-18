#!/usr/bin/env node
/**
 * 验证单个对局的胜率计算
 */

import pkg from 'poker-odds-calc';
const { TexasHoldem } = pkg;

// 测试几个具体的对局
const testCases = [
    // Case 2的公共牌: 2h 8d 2c Jc Ts
    { board: ['2h', '8d', '2c', 'Jc', 'Ts'], oop: ['Kd', 'Ks'], ip: ['Ac', 'Ad'] },
    { board: ['2h', '8d', '2c', 'Jc', 'Ts'], oop: ['Kd', 'Ks'], ip: ['Jd', 'Jh'] },
    { board: ['2h', '8d', '2c', 'Jc', 'Ts'], oop: ['Kd', 'Ks'], ip: ['Tc', 'Td'] },
    { board: ['2h', '8d', '2c', 'Jc', 'Ts'], oop: ['Kd', 'Ks'], ip: ['2d', '2s'] },
    { board: ['2h', '8d', '2c', 'Jc', 'Ts'], oop: ['Kd', 'Ks'], ip: ['Qc', 'Qd'] },
    { board: ['2h', '8d', '2c', 'Jc', 'Ts'], oop: ['Kd', 'Ks'], ip: ['9c', '9d'] },
    // 测试平局情况
    { board: ['2h', '8d', '2c', 'Jc', 'Ts'], oop: ['Ac', 'Kc'], ip: ['Ad', 'Kh'] },
];

for (const tc of testCases) {
    try {
        const table = new TexasHoldem();
        table.addPlayer(tc.oop);
        table.addPlayer(tc.ip);
        table.setBoard(tc.board);
        
        const result = table.calculate();
        const players = result.getPlayers();
        
        console.log(`\n公共牌: ${tc.board.join(' ')}`);
        console.log(`OOP: ${tc.oop.join('')} vs IP: ${tc.ip.join('')}`);
        console.log(`  OOP胜: ${players[0].getWins()}`);
        console.log(`  OOP平: ${players[0].getTies()}`);
        console.log(`  IP胜: ${players[1].getWins()}`);
        console.log(`  IP平: ${players[1].getTies()}`);
        
        // 输出手牌强度
        console.log(`  OOP手牌: ${players[0].getHand()}`);
        console.log(`  IP手牌: ${players[1].getHand()}`);
    } catch (e) {
        console.log(`错误: ${e.message}`);
    }
}
