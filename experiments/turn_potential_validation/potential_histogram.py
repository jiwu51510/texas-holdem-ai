"""Potential直方图计算器模块。

本模块实现了转牌阶段的Potential直方图计算功能。
Potential直方图捕获了手牌在河牌阶段的潜在强度分布。
"""

from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
import numpy as np

from models.core import Card
from environment.hand_evaluator import compare_hands
from experiments.turn_potential_validation.river_enumerator import RiverCardEnumerator
from experiments.equity_solver_validation.equity_calculator_wrapper import (
    DeadCardRemover,
    RangeVsRangeCalculator,
)


class PotentialHistogramCalculator:
    """Potential直方图计算器。
    
    计算转牌阶段手牌在所有可能河牌下的Equity分布直方图。
    """
    
    def __init__(self, num_bins: int = 50, num_workers: int = 0):
        """初始化计算器。
        
        Args:
            num_bins: 直方图区间数量（默认50，每个区间宽度0.02）
            num_workers: 并行工作进程数（0=使用所有CPU核心）
        """
        self.num_bins = num_bins
        self.num_workers = num_workers if num_workers > 0 else cpu_count()
        self.river_enumerator = RiverCardEnumerator()
        self.dead_card_remover = DeadCardRemover()
        self.range_calculator = RangeVsRangeCalculator(num_workers=num_workers)
    
    def calculate_potential_histogram(
        self,
        hole_cards: Tuple[Card, Card],
        turn_community: List[Card],
        opponent_range: Dict[str, float]
    ) -> np.ndarray:
        """计算单个手牌的Potential直方图。
        
        枚举所有可能的河牌（46张），计算每种情况下的Equity，
        然后生成Equity分布直方图。
        
        Args:
            hole_cards: 我的手牌
            turn_community: 4张公共牌（翻牌+转牌）
            opponent_range: 对手范围
            
        Returns:
            归一化的Potential直方图，形状为 (num_bins,)
        """
        if len(turn_community) != 4:
            raise ValueError(f"转牌阶段必须有4张公共牌，当前：{len(turn_community)}")
        
        # 枚举所有可能的河牌
        river_cards = self.river_enumerator.enumerate_river_cards(
            hole_cards, turn_community
        )
        
        # 计算每张河牌下的Equity
        equities = []
        
        for river_card in river_cards:
            # 构建完整的5张公共牌
            river_community = turn_community + [river_card]
            
            # 计算手牌VS范围的Equity
            equity = self._calculate_river_equity(
                hole_cards, river_community, opponent_range
            )
            equities.append(equity)
        
        # 生成直方图
        histogram = self._create_histogram(equities)
        
        return histogram
    
    def _calculate_river_equity(
        self,
        hole_cards: Tuple[Card, Card],
        river_community: List[Card],
        opponent_range: Dict[str, float]
    ) -> float:
        """计算河牌阶段的Equity。
        
        Args:
            hole_cards: 我的手牌
            river_community: 5张公共牌
            opponent_range: 对手范围
            
        Returns:
            Equity值 [0, 1]
        """
        # 移除死牌
        dead_cards = list(hole_cards) + river_community
        clean_range = self.dead_card_remover.remove_dead_cards(opponent_range, dead_cards)
        
        # 展开抽象手牌
        expanded_range = self.range_calculator._expand_range(clean_range)
        
        if not expanded_range:
            return 0.5  # 对手范围为空，返回中性值
        
        # 计算加权胜率
        total_weight = 0.0
        weighted_equity = 0.0
        
        for opp_hand_str, weight in expanded_range.items():
            opp_cards = self.dead_card_remover._parse_hand_string(opp_hand_str)
            if opp_cards is None:
                continue
            
            # 再次检查死牌
            opp_dead = set((c.rank, c.suit) for c in dead_cards)
            if any((c.rank, c.suit) in opp_dead for c in opp_cards):
                continue
            
            # 比较手牌
            result = compare_hands(list(hole_cards), list(opp_cards), river_community)
            
            if result == 0:  # 我方胜
                equity = 1.0
            elif result == -1:  # 平局
                equity = 0.5
            else:  # 对方胜
                equity = 0.0
            
            weighted_equity += equity * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_equity / total_weight
    
    def _create_histogram(self, equities: List[float]) -> np.ndarray:
        """从Equity列表创建归一化直方图。
        
        Args:
            equities: Equity值列表
            
        Returns:
            归一化的直方图数组
        """
        if not equities:
            # 返回均匀分布
            return np.ones(self.num_bins) / self.num_bins
        
        # 创建直方图区间
        bins = np.linspace(0, 1, self.num_bins + 1)
        
        # 计算直方图
        counts, _ = np.histogram(equities, bins=bins)
        
        # 归一化
        total = np.sum(counts)
        if total > 0:
            histogram = counts.astype(float) / total
        else:
            histogram = np.ones(self.num_bins) / self.num_bins
        
        return histogram
    
    def calculate_range_potential_histograms(
        self,
        my_range: Dict[str, float],
        opponent_range: Dict[str, float],
        turn_community: List[Card]
    ) -> Dict[str, np.ndarray]:
        """计算我的范围中每个手牌的Potential直方图。
        
        Args:
            my_range: 我的范围
            opponent_range: 对手范围
            turn_community: 4张公共牌
            
        Returns:
            每个手牌组合的Potential直方图映射
        """
        if len(turn_community) != 4:
            raise ValueError(f"转牌阶段必须有4张公共牌，当前：{len(turn_community)}")
        
        # 展开我的范围
        expanded_my_range = self.range_calculator._expand_range(my_range)
        
        # 移除与公共牌冲突的手牌
        board_dead = turn_community
        clean_my_range = self.dead_card_remover.remove_dead_cards(expanded_my_range, board_dead)
        
        result = {}
        
        for hand_str in clean_my_range.keys():
            cards = self.dead_card_remover._parse_hand_string(hand_str)
            if cards is None:
                continue
            
            # 计算Potential直方图
            histogram = self.calculate_potential_histogram(
                cards, turn_community, opponent_range
            )
            result[hand_str] = histogram
        
        return result
    
    def get_histogram_features(self, histogram: np.ndarray) -> Dict[str, float]:
        """从直方图提取特征。
        
        Args:
            histogram: 归一化的直方图
            
        Returns:
            特征字典，包含均值、方差、熵等
        """
        # 计算区间中点
        bin_centers = np.linspace(0.01, 0.99, self.num_bins)
        
        # 均值（期望Equity）
        mean_equity = np.sum(histogram * bin_centers)
        
        # 方差
        variance = np.sum(histogram * (bin_centers - mean_equity) ** 2)
        
        # 熵
        # 避免log(0)
        nonzero_hist = histogram[histogram > 0]
        entropy = -np.sum(nonzero_hist * np.log2(nonzero_hist)) if len(nonzero_hist) > 0 else 0.0
        
        # 稀疏度（非零区间比例）
        sparsity = np.sum(histogram > 0) / self.num_bins
        
        # 峰度（分布的尖锐程度）
        if variance > 0:
            kurtosis = np.sum(histogram * ((bin_centers - mean_equity) / np.sqrt(variance)) ** 4) - 3
        else:
            kurtosis = 0.0
        
        return {
            'mean_equity': mean_equity,
            'variance': variance,
            'std': np.sqrt(variance),
            'entropy': entropy,
            'sparsity': sparsity,
            'kurtosis': kurtosis,
        }
    
    def is_normalized(self, histogram: np.ndarray, tolerance: float = 1e-6) -> bool:
        """检查直方图是否归一化。
        
        Args:
            histogram: 直方图数组
            tolerance: 允许的误差
            
        Returns:
            如果归一化返回True
        """
        return abs(np.sum(histogram) - 1.0) < tolerance
    
    def validate_histogram(self, histogram: np.ndarray) -> Tuple[bool, str]:
        """验证直方图的有效性。
        
        Args:
            histogram: 直方图数组
            
        Returns:
            (是否有效, 错误信息) 元组
        """
        # 检查形状
        if len(histogram) != self.num_bins:
            return False, f"直方图长度不正确：期望{self.num_bins}，实际{len(histogram)}"
        
        # 检查非负性
        if np.any(histogram < 0):
            return False, "直方图包含负值"
        
        # 检查归一化
        if not self.is_normalized(histogram):
            return False, f"直方图未归一化：和为{np.sum(histogram)}"
        
        # 检查值范围
        if np.any(histogram > 1):
            return False, "直方图包含大于1的值"
        
        return True, ""
