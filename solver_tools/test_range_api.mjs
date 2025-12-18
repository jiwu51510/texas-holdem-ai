#!/usr/bin/env node
/**
 * 测试 poker-odds-calc 的范围API
 */

import pkg from 'poker-odds-calc';
console.log('poker-odds-calc 导出的内容:');
console.log(Object.keys(pkg));

const { TexasHoldem, CardGroup } = pkg;

// 测试是否支持范围字符串
const table = new TexasHoldem();

// 尝试直接传入范围字符串
try {
    table.addPlayer('AA');
    console.log('\n可以直接传入范围字符串 "AA"');
} catch (e) {
    console.log('\n不能直接传入范围字符串:', e.message);
}

// 查看 TexasHoldem 的方法
console.log('\nTexasHoldem 实例的方法:');
console.log(Object.getOwnPropertyNames(Object.getPrototypeOf(table)));

// 查看 CardGroup
if (CardGroup) {
    console.log('\nCardGroup 存在');
    console.log('CardGroup 方法:', Object.keys(CardGroup));
}
