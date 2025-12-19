"""
EV预测数据集模块

该模块实现了从验证数据JSON文件加载训练数据的功能。
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
import numpy as np

# 配置日志
logger = logging.getLogger(__name__)


# 默认动作顺序
DEFAULT_ACTIONS = ["Check:0", "Bet:33", "Bet:50", "Bet:75", "Bet:120"]


class EVDataset(Dataset):
    """
    EV预测数据集
    
    从JSON格式的验证数据文件加载场景，提取：
    - 输入特征：hero_equity, range_equity, solver_equity, eqr
    - 目标值：ev, action_ev, strategy
    """
    
    def __init__(
        self, 
        data_dir: str, 
        max_files: Optional[int] = None,
        action_names: Optional[List[str]] = None,
        extracted_file: Optional[str] = None
    ):
        """
        初始化数据集
        
        参数:
            data_dir: 验证数据目录路径
            max_files: 最大加载文件数（用于调试）
            action_names: 动作名称列表（用于确定动作顺序）
            extracted_file: 预提取的数据文件路径（如果提供，直接从此文件加载）
        """
        self.data_dir = Path(data_dir)
        self.max_files = max_files
        self.action_names = action_names or DEFAULT_ACTIONS
        self.num_actions = len(self.action_names)
        self.extracted_file = extracted_file
        
        # 存储所有样本
        self.samples: List[Dict[str, Any]] = []
        
        # 统计信息
        self.skipped_samples = 0
        self.loaded_files = 0
        
        # 加载数据
        if extracted_file and Path(extracted_file).exists():
            self._load_extracted_data(extracted_file)
        else:
            self._load_data()
    
    def _load_extracted_data(self, extracted_file: str):
        """从预提取的数据文件加载"""
        logger.info(f"从预提取文件加载数据: {extracted_file}")
        
        with open(extracted_file, 'r') as f:
            data = json.load(f)
        
        raw_data = data.get("data", [])
        
        for item in raw_data:
            sample = self._parse_extracted_item(item)
            if sample is not None:
                self.samples.append(sample)
            else:
                self.skipped_samples += 1
        
        self.loaded_files = 1
        logger.info(
            f"数据加载完成: {len(self.samples)} 个样本, "
            f"{self.skipped_samples} 个跳过"
        )
    
    def _parse_extracted_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解析预提取的数据项"""
        try:
            features = [
                item["hero_equity"],
                item["range_equity"],
                item["solver_equity"],
                item["eqr"]
            ]
            
            action_ev = [item["action_ev"].get(name, 0.0) for name in self.action_names]
            strategy = [item["strategy"].get(name, 0.0) for name in self.action_names]
            
            # 验证数据有效性
            all_values = features + [item["ev"]] + action_ev + strategy
            for val in all_values:
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    return None
            
            return {
                "features": features,
                "ev": item["ev"],
                "action_ev": action_ev,
                "strategy": strategy,
                "scenario_id": item.get("id", -1)
            }
        except Exception:
            return None
    
    def _load_data(self):
        """加载所有验证数据文件"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 查找所有场景JSON文件（排除index.json和summary.json）
        json_files = sorted([
            f for f in self.data_dir.glob("*.json")
            if "batch" in f.name  # 只加载包含batch的文件
        ])
        
        if self.max_files is not None:
            json_files = json_files[:self.max_files]
        
        for json_file in json_files:
            try:
                self._load_file(json_file)
                self.loaded_files += 1
            except Exception as e:
                logger.warning(f"加载文件失败 {json_file}: {e}")
        
        logger.info(
            f"数据加载完成: {len(self.samples)} 个样本, "
            f"{self.loaded_files} 个文件, "
            f"{self.skipped_samples} 个跳过"
        )
    
    def _load_file(self, file_path: Path):
        """加载单个JSON文件"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        scenarios = data.get("scenarios", [])
        
        for scenario in scenarios:
            sample = self._parse_scenario(scenario)
            if sample is not None:
                self.samples.append(sample)
    
    def _parse_scenario(self, scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        解析单个场景数据
        
        返回:
            解析后的样本字典，如果数据无效则返回None
        """
        try:
            # 提取输入特征
            hero_equity_data = scenario.get("heroEquity", {})
            range_equity_data = scenario.get("rangeEquity", {})
            
            hero_equity = hero_equity_data.get("equity", 0.0)
            range_equity = range_equity_data.get("equity", 0.0)
            solver_equity = scenario.get("solverEquity", 0.0)
            eqr = scenario.get("eqr", 0.0)
            
            # 提取目标值
            ev = scenario.get("ev", 0.0)
            action_ev_dict = scenario.get("actionEV", {})
            strategy_dict = scenario.get("strategy", {})
            
            # 转换为有序数组
            action_ev = [action_ev_dict.get(name, 0.0) for name in self.action_names]
            strategy = [strategy_dict.get(name, 0.0) for name in self.action_names]
            
            # 验证数据有效性
            features = [hero_equity, range_equity, solver_equity, eqr]
            all_values = features + [ev] + action_ev + strategy
            
            for val in all_values:
                if not isinstance(val, (int, float)):
                    logger.warning(f"非数值类型: {type(val)}")
                    self.skipped_samples += 1
                    return None
                if np.isnan(val) or np.isinf(val):
                    logger.warning(f"无效值 (NaN/Inf): {val}")
                    self.skipped_samples += 1
                    return None
            
            return {
                "features": features,
                "ev": ev,
                "action_ev": action_ev,
                "strategy": strategy,
                "scenario_id": scenario.get("id", -1)
            }
            
        except Exception as e:
            logger.warning(f"解析场景失败: {e}")
            self.skipped_samples += 1
            return None
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        返回:
            features: 输入特征 [4]
            ev: 目标EV [1]
            action_ev: 目标动作EV [num_actions]
            strategy: 目标策略 [num_actions]
        """
        sample = self.samples[idx]
        
        features = torch.tensor(sample["features"], dtype=torch.float32)
        ev = torch.tensor([sample["ev"]], dtype=torch.float32)
        action_ev = torch.tensor(sample["action_ev"], dtype=torch.float32)
        strategy = torch.tensor(sample["strategy"], dtype=torch.float32)
        
        return features, ev, action_ev, strategy
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        返回:
            包含各项统计指标的字典
        """
        if len(self.samples) == 0:
            return {"error": "数据集为空"}
        
        # 收集所有数值
        features_list = [s["features"] for s in self.samples]
        ev_list = [s["ev"] for s in self.samples]
        action_ev_list = [s["action_ev"] for s in self.samples]
        strategy_list = [s["strategy"] for s in self.samples]
        
        features_arr = np.array(features_list)
        ev_arr = np.array(ev_list)
        action_ev_arr = np.array(action_ev_list)
        strategy_arr = np.array(strategy_list)
        
        return {
            "num_samples": len(self.samples),
            "num_files": self.loaded_files,
            "skipped_samples": self.skipped_samples,
            "features": {
                "mean": features_arr.mean(axis=0).tolist(),
                "std": features_arr.std(axis=0).tolist(),
                "min": features_arr.min(axis=0).tolist(),
                "max": features_arr.max(axis=0).tolist()
            },
            "ev": {
                "mean": float(ev_arr.mean()),
                "std": float(ev_arr.std()),
                "min": float(ev_arr.min()),
                "max": float(ev_arr.max())
            },
            "action_ev": {
                "mean": action_ev_arr.mean(axis=0).tolist(),
                "std": action_ev_arr.std(axis=0).tolist()
            },
            "strategy": {
                "mean": strategy_arr.mean(axis=0).tolist(),
                "std": strategy_arr.std(axis=0).tolist()
            }
        }


def create_scenario_json(
    hero_equity: float,
    range_equity: float,
    solver_equity: float,
    eqr: float,
    ev: float,
    action_ev: Dict[str, float],
    strategy: Dict[str, float],
    scenario_id: int = 0
) -> Dict[str, Any]:
    """
    创建场景JSON数据（用于测试）
    
    参数:
        hero_equity: 英雄权益
        range_equity: 范围权益
        solver_equity: Solver权益
        eqr: 权益实现率
        ev: 期望值
        action_ev: 动作EV字典
        strategy: 策略字典
        scenario_id: 场景ID
    
    返回:
        场景数据字典
    """
    return {
        "id": scenario_id,
        "heroEquity": {"equity": hero_equity},
        "rangeEquity": {"equity": range_equity},
        "solverEquity": solver_equity,
        "eqr": eqr,
        "ev": ev,
        "actionEV": action_ev,
        "strategy": strategy
    }
