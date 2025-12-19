"""Potential直方图验证器模块。

本模块实现了Potential直方图计算的验证功能，通过手动枚举方法
验证PotentialHistogramCalculator的计算正确性。

验证方法：
1. 手动枚举所有可能的河牌（46张）
2. 对每张河牌计算Equity
3. 生成直方图并与PotentialHistogramCalculator的结果对比
4. 报告计算误差
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

from models.core import Card
from environment.hand_evaluator import compare_hands
from experiments.turn_potential_validation.river_enumerator import RiverCardEnumerator
from experiments.equity_solver_validation.equity_calculator_wrapper import DeadCardRemover


@dataclass
class ValidationResult:
    """验证结果数据类。
    
    存储手动枚举验证的结果。
    
    Attributes:
        is_valid: 验证是否通过
        max_error: 最大误差
        mean_error: 平均误差
        error_histogram: 每个区间的误差
        computed_histogram: 计算器生成的直方图
        manual_histogram: 手动枚举生成的直方图
        error_message: 错误信息（如果有）
    """
    is_valid: bool = True
    max_error: float = 0.0
    mean_error: float = 0.0
    error_histogram: np.ndarray = field(default_factory=lambda: np.array([]))
    computed_histogram: np.ndarray = field(default_factory=lambda: np.array([]))
    manual_histogram: np.ndarray = field(default_factory=lambda: np.array([]))
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """将结果转换为字典格式。"""
        return {
            'is_valid': self.is_valid,
            'max_error': self.max_error,
            'mean_error': self.mean_error,
            'error_histogram': self.error_histogram.tolist() if len(self.error_histogram) > 0 else [],
            'computed_histogram': self.computed_histogram.tolist() if len(self.computed_histogram) > 0 else [],
            'manual_histogram': self.manual_histogram.tolist() if len(self.manual_histogram) > 0 else [],
            'error_message': self.error_message,
        }


@dataclass
class BatchValidationResult:
    """批量验证结果数据类。
    
    存储多个手牌的验证结果。
    
    Attributes:
        results: 每个手牌的验证结果
        all_valid: 所有验证是否都通过
        overall_max_error: 所有手牌中的最大误差
        overall_mean_error: 所有手牌的平均误差
        num_validated: 验证的手牌数量
        num_passed: 通过验证的手牌数量
    """
    results: Dict[str, ValidationResult] = field(default_factory=dict)
    all_valid: bool = True
    overall_max_error: float = 0.0
    overall_mean_error: float = 0.0
    num_validated: int = 0
    num_passed: int = 0
    
    def to_dict(self) -> Dict:
        """将结果转换为字典格式。"""
        return {
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'all_valid': self.all_valid,
            'overall_max_error': self.overall_max_error,
            'overall_mean_error': self.overall_mean_error,
            'num_validated': self.num_validated,
            'num_passed': self.num_passed,
        }


class ManualHistogramCalculator:
    """手动枚举直方图计算器。
    
    通过手动枚举所有河牌来计算Potential直方图，
    用于验证PotentialHistogramCalculator的正确性。
    """
    
    def __init__(self, num_bins: int = 50):
        """初始化计算器。
        
        Args:
            num_bins: 直方图区间数量
        """
        self.num_bins = num_bins
        self.river_enumerator = RiverCardEnumerator()
        self.dead_card_remover = DeadCardRemover()
    
    def calculate_histogram_manually(
        self,
        hole_cards: Tuple[Card, Card],
        turn_community: List[Card],
        opponent_range: Dict[str, float]
    ) -> np.ndarray:
        """手动枚举计算Potential直方图。
        
        这是一个独立的实现，用于验证PotentialHistogramCalculator。
        
        Args:
            hole_cards: 我的手牌
            turn_community: 4张公共牌
            opponent_range: 对手范围
            
        Returns:
            归一化的Potential直方图
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
            
            # 手动计算Equity
            equity = self._calculate_equity_manually(
                hole_cards, river_community, opponent_range
            )
            equities.append(equity)
        
        # 生成直方图
        histogram = self._create_histogram_manually(equities)
        
        return histogram
    
    def _calculate_equity_manually(
        self,
        hole_cards: Tuple[Card, Card],
        river_community: List[Card],
        opponent_range: Dict[str, float]
    ) -> float:
        """手动计算河牌阶段的Equity。
        
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
        expanded_range = self._expand_range_manually(clean_range)
        
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
    
    def _expand_range_manually(self, range_dict: Dict[str, float]) -> Dict[str, float]:
        """手动展开范围中的抽象手牌表示。
        
        Args:
            range_dict: 可能包含抽象表示的范围
            
        Returns:
            只包含具体手牌的范围
        """
        expanded = {}
        
        for hand_str, weight in range_dict.items():
            concrete_hands = self.dead_card_remover.expand_abstract_hand(hand_str)
            
            # 分配权重
            per_hand_weight = weight / len(concrete_hands) if concrete_hands else 0
            
            for concrete in concrete_hands:
                if concrete in expanded:
                    expanded[concrete] += per_hand_weight
                else:
                    expanded[concrete] = per_hand_weight
        
        return expanded
    
    def _create_histogram_manually(self, equities: List[float]) -> np.ndarray:
        """手动从Equity列表创建归一化直方图。
        
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


class HistogramValidator:
    """Potential直方图验证器。
    
    验证PotentialHistogramCalculator的计算结果与手动枚举结果的一致性。
    """
    
    def __init__(self, num_bins: int = 50, tolerance: float = 1e-6):
        """初始化验证器。
        
        Args:
            num_bins: 直方图区间数量
            tolerance: 允许的误差容限
        """
        self.num_bins = num_bins
        self.tolerance = tolerance
        self.manual_calculator = ManualHistogramCalculator(num_bins=num_bins)
        
        # 延迟导入以避免循环依赖
        from experiments.turn_potential_validation.potential_histogram import (
            PotentialHistogramCalculator
        )
        self.potential_calculator = PotentialHistogramCalculator(num_bins=num_bins)
    
    def validate_histogram(
        self,
        hole_cards: Tuple[Card, Card],
        turn_community: List[Card],
        opponent_range: Dict[str, float]
    ) -> ValidationResult:
        """验证单个手牌的Potential直方图计算。
        
        Args:
            hole_cards: 我的手牌
            turn_community: 4张公共牌
            opponent_range: 对手范围
            
        Returns:
            验证结果
        """
        try:
            # 使用PotentialHistogramCalculator计算
            computed_histogram = self.potential_calculator.calculate_potential_histogram(
                hole_cards, turn_community, opponent_range
            )
            
            # 使用手动枚举计算
            manual_histogram = self.manual_calculator.calculate_histogram_manually(
                hole_cards, turn_community, opponent_range
            )
            
            # 计算误差
            error_histogram = np.abs(computed_histogram - manual_histogram)
            max_error = float(np.max(error_histogram))
            mean_error = float(np.mean(error_histogram))
            
            # 判断是否通过验证
            is_valid = max_error < self.tolerance
            
            return ValidationResult(
                is_valid=is_valid,
                max_error=max_error,
                mean_error=mean_error,
                error_histogram=error_histogram,
                computed_histogram=computed_histogram,
                manual_histogram=manual_histogram,
                error_message=None if is_valid else f"最大误差{max_error}超过容限{self.tolerance}"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=str(e)
            )
    
    def validate_range_histograms(
        self,
        my_range: Dict[str, float],
        opponent_range: Dict[str, float],
        turn_community: List[Card]
    ) -> BatchValidationResult:
        """验证范围内所有手牌的Potential直方图计算。
        
        Args:
            my_range: 我的范围
            opponent_range: 对手范围
            turn_community: 4张公共牌
            
        Returns:
            批量验证结果
        """
        # 使用PotentialHistogramCalculator计算范围直方图
        computed_histograms = self.potential_calculator.calculate_range_potential_histograms(
            my_range, opponent_range, turn_community
        )
        
        results = {}
        all_errors = []
        num_passed = 0
        
        for hand_str, computed_histogram in computed_histograms.items():
            # 解析手牌
            cards = self.manual_calculator.dead_card_remover._parse_hand_string(hand_str)
            if cards is None:
                continue
            
            # 手动计算
            manual_histogram = self.manual_calculator.calculate_histogram_manually(
                cards, turn_community, opponent_range
            )
            
            # 计算误差
            error_histogram = np.abs(computed_histogram - manual_histogram)
            max_error = float(np.max(error_histogram))
            mean_error = float(np.mean(error_histogram))
            
            is_valid = max_error < self.tolerance
            if is_valid:
                num_passed += 1
            
            all_errors.append(max_error)
            
            results[hand_str] = ValidationResult(
                is_valid=is_valid,
                max_error=max_error,
                mean_error=mean_error,
                error_histogram=error_histogram,
                computed_histogram=computed_histogram,
                manual_histogram=manual_histogram,
                error_message=None if is_valid else f"最大误差{max_error}超过容限{self.tolerance}"
            )
        
        # 计算总体统计
        all_valid = all(r.is_valid for r in results.values())
        overall_max_error = max(all_errors) if all_errors else 0.0
        overall_mean_error = float(np.mean(all_errors)) if all_errors else 0.0
        
        return BatchValidationResult(
            results=results,
            all_valid=all_valid,
            overall_max_error=overall_max_error,
            overall_mean_error=overall_mean_error,
            num_validated=len(results),
            num_passed=num_passed,
        )
    
    def generate_validation_report(
        self,
        result: ValidationResult,
        hand_str: str = ""
    ) -> str:
        """生成单个手牌的验证报告。
        
        Args:
            result: 验证结果
            hand_str: 手牌字符串
            
        Returns:
            报告字符串
        """
        lines = []
        lines.append(f"=== Potential直方图验证报告 ===")
        if hand_str:
            lines.append(f"手牌: {hand_str}")
        lines.append(f"验证结果: {'通过' if result.is_valid else '失败'}")
        lines.append(f"最大误差: {result.max_error:.2e}")
        lines.append(f"平均误差: {result.mean_error:.2e}")
        lines.append(f"容限: {self.tolerance:.2e}")
        
        if result.error_message:
            lines.append(f"错误信息: {result.error_message}")
        
        return "\n".join(lines)
    
    def generate_batch_validation_report(
        self,
        result: BatchValidationResult
    ) -> str:
        """生成批量验证报告。
        
        Args:
            result: 批量验证结果
            
        Returns:
            报告字符串
        """
        lines = []
        lines.append(f"=== Potential直方图批量验证报告 ===")
        lines.append(f"验证手牌数: {result.num_validated}")
        lines.append(f"通过数量: {result.num_passed}")
        lines.append(f"失败数量: {result.num_validated - result.num_passed}")
        lines.append(f"总体结果: {'全部通过' if result.all_valid else '存在失败'}")
        lines.append(f"总体最大误差: {result.overall_max_error:.2e}")
        lines.append(f"总体平均误差: {result.overall_mean_error:.2e}")
        lines.append(f"容限: {self.tolerance:.2e}")
        
        if not result.all_valid:
            lines.append("\n失败的手牌:")
            for hand_str, r in result.results.items():
                if not r.is_valid:
                    lines.append(f"  - {hand_str}: 最大误差={r.max_error:.2e}")
        
        return "\n".join(lines)
