#!/usr/bin/env node
import pkg from 'poker-odds-calc';
const { TexasHoldem } = pkg;

// Case 2: 公共牌 2h 8d 2c Jc Ts, OOP手牌 KdKs vs IP手牌 AcAd
const board = ['2h', '8d', '2c', 'Jc', 'Ts'];
const oop = ['Kd', 'Ks'];
const ip = ['Ac', 'Ad'];

const table = new TexasHoldem();
table.addPlayer(oop);
table.addPlayer(ip);
table.setBoard(board);

const result = table.calculate();
const players = result.getPlayers();

console.log(`公共牌: ${board.join(' ')}`);
console.log(`OOP: ${oop.join('')} vs IP: ${ip.join('')}`);
console.log(`迭代次数: ${result.getIterations()}`);
console.log(`OOP胜: ${players[0].getWins()}, 平: ${players[0].getTies()}`);
console.log(`IP胜: ${players[1].getWins()}, 平: ${players[1].getTies()}`);
console.log(`OOP胜率: ${players[0].getWinsPercentageString()}`);
