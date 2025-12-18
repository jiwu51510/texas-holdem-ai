/**
 * 测试range解析
 */

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const rangeWasmPath = join(__dirname, 'wasm-postflop/pkg/range/range_bg.wasm');

async function test() {
    console.log('Testing range parsing...');
    
    // 读取wasm文件
    const wasmBuffer = readFileSync(rangeWasmPath);
    
    // 导入range_bg.js
    const rangeBg = await import('./wasm-postflop/pkg/range/range_bg.js');
    
    // 编译并实例化wasm
    const wasmModule = await WebAssembly.compile(wasmBuffer);
    const wasmInstance = await WebAssembly.instantiate(wasmModule, {
        './range_bg.js': rangeBg,
    });
    
    // 设置wasm实例
    rangeBg.__wbg_set_wasm(wasmInstance.exports);
    
    // 初始化externref表
    if (rangeBg.__wbindgen_init_externref_table) {
        rangeBg.__wbindgen_init_externref_table();
    }
    
    const RangeManager = rangeBg.RangeManager;
    
    // 测试解析范围
    const manager = RangeManager.new();
    const rangeStr = 'AA,KK,QQ,JJ,TT';
    console.log(`Parsing range: ${rangeStr}`);
    
    const error = manager.from_string(rangeStr);
    if (error) {
        console.log(`Error: ${error}`);
    } else {
        console.log('Range parsed successfully!');
        const weights = manager.get_weights();
        console.log(`Weights length: ${weights.length}`);
        
        // 统计非零权重
        let nonZero = 0;
        for (let i = 0; i < weights.length; i++) {
            if (weights[i] > 0) nonZero++;
        }
        console.log(`Non-zero weights: ${nonZero}`);
        
        // 打印前几个非零权重
        console.log('First few non-zero weights:');
        let count = 0;
        for (let i = 0; i < weights.length && count < 10; i++) {
            if (weights[i] > 0) {
                console.log(`  Index ${i}: ${weights[i]}`);
                count++;
            }
        }
    }
    
    manager.free();
}

test().catch(console.error);
