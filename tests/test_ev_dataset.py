"""
EV数据集测试模块

包含属性测试和单元测试，验证数据加载和解析的正确性。
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from training.ev_dataset import EVDataset, create_scenario_json, DEFAULT_ACTIONS


class TestEVDatasetProperties:
    """属性测试类"""
    
    @given(
        hero_equity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        range_equity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        solver_equity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        eqr=st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        ev=st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_data_parsing_roundtrip(
        self, hero_equity: float, range_equity: float,
        solver_equity: float, eqr: float, ev: float
    ):
        """
        **Feature: ev-prediction-network, Property 4: 数据解析往返一致性**
        
        对于任意有效的场景数据，序列化为JSON后再解析，
        提取的特征和目标值应该与原始数据一致
        
        **Validates: Requirements 2.1**
        """
        # 生成随机策略（确保和为1）
        raw_probs = np.random.random(5)
        strategy_values = (raw_probs / raw_probs.sum()).tolist()
        
        # 生成随机动作EV
        action_ev_values = [float(np.random.uniform(-100, 100)) for _ in range(5)]
        
        # 创建场景数据
        action_ev = {name: val for name, val in zip(DEFAULT_ACTIONS, action_ev_values)}
        strategy = {name: val for name, val in zip(DEFAULT_ACTIONS, strategy_values)}
        
        scenario = create_scenario_json(
            hero_equity=hero_equity,
            range_equity=range_equity,
            solver_equity=solver_equity,
            eqr=eqr,
            ev=ev,
            action_ev=action_ev,
            strategy=strategy
        )
        
        # 创建临时目录和文件
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test_batch_data.json"
            with open(json_path, 'w') as f:
                json.dump({"scenarios": [scenario]}, f)
            
            # 加载数据集
            dataset = EVDataset(tmpdir)
            
            # 验证数据集大小
            assert len(dataset) == 1
            
            # 获取样本
            features, ev_tensor, action_ev_tensor, strategy_tensor = dataset[0]
            
            # 验证特征往返一致性
            assert torch.isclose(features[0], torch.tensor(hero_equity), atol=1e-5)
            assert torch.isclose(features[1], torch.tensor(range_equity), atol=1e-5)
            assert torch.isclose(features[2], torch.tensor(solver_equity), atol=1e-5)
            assert torch.isclose(features[3], torch.tensor(eqr), atol=1e-5)
            
            # 验证EV往返一致性
            assert torch.isclose(ev_tensor[0], torch.tensor(ev), atol=1e-5)
            
            # 验证动作EV往返一致性
            for i, val in enumerate(action_ev_values):
                assert torch.isclose(action_ev_tensor[i], torch.tensor(val), atol=1e-5)
            
            # 验证策略往返一致性
            for i, val in enumerate(strategy_values):
                assert torch.isclose(strategy_tensor[i], torch.tensor(val), atol=1e-5)


class TestEVDatasetUnit:
    """单元测试类"""
    
    def test_empty_directory(self):
        """测试空目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = EVDataset(tmpdir)
            assert len(dataset) == 0
    
    def test_nonexistent_directory(self):
        """测试不存在的目录"""
        with pytest.raises(FileNotFoundError):
            EVDataset("/nonexistent/path/to/data")
    
    def test_skip_invalid_nan_values(self):
        """测试跳过包含NaN的样本"""
        scenario_valid = create_scenario_json(
            hero_equity=0.5, range_equity=0.5, solver_equity=0.5, eqr=1.0,
            ev=10.0,
            action_ev={name: 5.0 for name in DEFAULT_ACTIONS},
            strategy={name: 0.2 for name in DEFAULT_ACTIONS}
        )
        
        scenario_invalid = create_scenario_json(
            hero_equity=float('nan'), range_equity=0.5, solver_equity=0.5, eqr=1.0,
            ev=10.0,
            action_ev={name: 5.0 for name in DEFAULT_ACTIONS},
            strategy={name: 0.2 for name in DEFAULT_ACTIONS}
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test_batch_data.json"
            with open(json_path, 'w') as f:
                json.dump({"scenarios": [scenario_valid, scenario_invalid]}, f)
            
            dataset = EVDataset(tmpdir)
            
            # 只有有效样本被加载
            assert len(dataset) == 1
            assert dataset.skipped_samples == 1
    
    def test_skip_invalid_inf_values(self):
        """测试跳过包含Inf的样本"""
        scenario_valid = create_scenario_json(
            hero_equity=0.5, range_equity=0.5, solver_equity=0.5, eqr=1.0,
            ev=10.0,
            action_ev={name: 5.0 for name in DEFAULT_ACTIONS},
            strategy={name: 0.2 for name in DEFAULT_ACTIONS}
        )
        
        scenario_invalid = create_scenario_json(
            hero_equity=0.5, range_equity=0.5, solver_equity=0.5, eqr=float('inf'),
            ev=10.0,
            action_ev={name: 5.0 for name in DEFAULT_ACTIONS},
            strategy={name: 0.2 for name in DEFAULT_ACTIONS}
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test_batch_data.json"
            with open(json_path, 'w') as f:
                json.dump({"scenarios": [scenario_valid, scenario_invalid]}, f)
            
            dataset = EVDataset(tmpdir)
            
            assert len(dataset) == 1
            assert dataset.skipped_samples == 1
    
    def test_multiple_files(self):
        """测试加载多个文件"""
        scenario1 = create_scenario_json(
            hero_equity=0.3, range_equity=0.4, solver_equity=0.35, eqr=0.9,
            ev=5.0,
            action_ev={name: 3.0 for name in DEFAULT_ACTIONS},
            strategy={name: 0.2 for name in DEFAULT_ACTIONS}
        )
        
        scenario2 = create_scenario_json(
            hero_equity=0.6, range_equity=0.5, solver_equity=0.55, eqr=1.1,
            ev=15.0,
            action_ev={name: 10.0 for name in DEFAULT_ACTIONS},
            strategy={name: 0.2 for name in DEFAULT_ACTIONS}
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建两个文件
            with open(Path(tmpdir) / "batch_file1.json", 'w') as f:
                json.dump({"scenarios": [scenario1]}, f)
            with open(Path(tmpdir) / "batch_file2.json", 'w') as f:
                json.dump({"scenarios": [scenario2]}, f)
            
            dataset = EVDataset(tmpdir)
            
            assert len(dataset) == 2
            assert dataset.loaded_files == 2
    
    def test_max_files_limit(self):
        """测试最大文件数限制"""
        scenario = create_scenario_json(
            hero_equity=0.5, range_equity=0.5, solver_equity=0.5, eqr=1.0,
            ev=10.0,
            action_ev={name: 5.0 for name in DEFAULT_ACTIONS},
            strategy={name: 0.2 for name in DEFAULT_ACTIONS}
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建3个文件
            for i in range(3):
                with open(Path(tmpdir) / f"batch_file{i}.json", 'w') as f:
                    json.dump({"scenarios": [scenario]}, f)
            
            # 只加载1个文件
            dataset = EVDataset(tmpdir, max_files=1)
            
            assert dataset.loaded_files == 1
            assert len(dataset) == 1
    
    def test_output_tensor_types(self):
        """测试输出张量类型"""
        scenario = create_scenario_json(
            hero_equity=0.5, range_equity=0.5, solver_equity=0.5, eqr=1.0,
            ev=10.0,
            action_ev={name: 5.0 for name in DEFAULT_ACTIONS},
            strategy={name: 0.2 for name in DEFAULT_ACTIONS}
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test_batch_data.json"
            with open(json_path, 'w') as f:
                json.dump({"scenarios": [scenario]}, f)
            
            dataset = EVDataset(tmpdir)
            features, ev, action_ev, strategy = dataset[0]
            
            # 验证类型
            assert features.dtype == torch.float32
            assert ev.dtype == torch.float32
            assert action_ev.dtype == torch.float32
            assert strategy.dtype == torch.float32
            
            # 验证形状
            assert features.shape == (4,)
            assert ev.shape == (1,)
            assert action_ev.shape == (5,)
            assert strategy.shape == (5,)
    
    def test_get_statistics(self):
        """测试统计信息获取"""
        scenarios = [
            create_scenario_json(
                hero_equity=0.3, range_equity=0.4, solver_equity=0.35, eqr=0.9,
                ev=5.0,
                action_ev={name: 3.0 for name in DEFAULT_ACTIONS},
                strategy={name: 0.2 for name in DEFAULT_ACTIONS}
            ),
            create_scenario_json(
                hero_equity=0.7, range_equity=0.6, solver_equity=0.65, eqr=1.1,
                ev=15.0,
                action_ev={name: 10.0 for name in DEFAULT_ACTIONS},
                strategy={name: 0.2 for name in DEFAULT_ACTIONS}
            )
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test_batch_data.json"
            with open(json_path, 'w') as f:
                json.dump({"scenarios": scenarios}, f)
            
            dataset = EVDataset(tmpdir)
            stats = dataset.get_statistics()
            
            assert stats["num_samples"] == 2
            assert "features" in stats
            assert "ev" in stats
            assert "action_ev" in stats
            assert "strategy" in stats
            
            # 验证EV统计
            assert stats["ev"]["mean"] == 10.0  # (5 + 15) / 2
            assert stats["ev"]["min"] == 5.0
            assert stats["ev"]["max"] == 15.0
