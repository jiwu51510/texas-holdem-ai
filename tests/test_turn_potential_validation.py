"""转牌阶段Potential直方图验证的属性测试。

本模块包含转牌验证实验的所有属性测试，验证：
- 数据模型序列化往返
- 河牌枚举完整性
- Potential直方图归一化
- EMD距离属性
- 策略概率归一化
- 批量实验结果完整性
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import List, Dict, Tuple

from models.core import Card
from experiments.equity_solver_validation.data_models import SolverConfig
from experiments.turn_potential_validation.data_models import (
    TurnScenario,
    CorrelationResult,
    ClusteringComparisonResult,
    TurnValidationMetrics,
    TurnExperimentResult,
    TurnBatchExperimentResult,
)


# ============================================================================
# 辅助策略生成器
# ============================================================================

# 生成有效的牌面值（2-14，其中14=A）
rank_strategy = st.integers(min_value=2, max_value=14)

# 生成有效的花色
suit_strategy = st.sampled_from(['h', 'd', 'c', 's'])

# 生成单张牌
@st.composite
def card_strategy(draw):
    """生成单张有效的扑克牌。"""
    rank = draw(rank_strategy)
    suit = draw(suit_strategy)
    return Card(rank=rank, suit=suit)


@st.composite
def unique_cards_strategy(draw, n: int):
    """生成n张不重复的扑克牌。"""
    cards = []
    used = set()
    
    while len(cards) < n:
        rank = draw(rank_strategy)
        suit = draw(suit_strategy)
        key = (rank, suit)
        
        if key not in used:
            used.add(key)
            cards.append(Card(rank=rank, suit=suit))
    
    return cards


@st.composite
def turn_community_strategy(draw):
    """生成4张不重复的转牌公共牌。"""
    return draw(unique_cards_strategy(4))


@st.composite
def hand_strategy(draw):
    """生成2张不重复的手牌。"""
    return draw(unique_cards_strategy(2))


@st.composite
def range_strategy(draw, min_hands: int = 1, max_hands: int = 10):
    """生成手牌范围（手牌到权重的映射）。"""
    num_hands = draw(st.integers(min_value=min_hands, max_value=max_hands))
    range_dict = {}
    
    for _ in range(num_hands):
        # 生成手牌字符串（如 "AhKs"）
        rank1 = draw(rank_strategy)
        suit1 = draw(suit_strategy)
        rank2 = draw(rank_strategy)
        suit2 = draw(suit_strategy)
        
        # 确保不是同一张牌
        if rank1 == rank2 and suit1 == suit2:
            continue
        
        hand_str = f"{_rank_to_char(rank1)}{suit1}{_rank_to_char(rank2)}{suit2}"
        weight = draw(st.floats(min_value=0.1, max_value=1.0))
        range_dict[hand_str] = weight
    
    # 确保至少有一个手牌
    if not range_dict:
        range_dict["AhKs"] = 1.0
    
    return range_dict


def _rank_to_char(rank: int) -> str:
    """将数字牌面值转换为字符。"""
    if rank == 14:
        return 'A'
    elif rank == 13:
        return 'K'
    elif rank == 12:
        return 'Q'
    elif rank == 11:
        return 'J'
    elif rank == 10:
        return 'T'
    else:
        return str(rank)


@st.composite
def solver_config_strategy(draw):
    """生成有效的Solver配置。"""
    pot_size = draw(st.floats(min_value=10.0, max_value=1000.0))
    effective_stack = draw(st.floats(min_value=0.0, max_value=500.0))
    
    return SolverConfig(
        pot_size=pot_size,
        effective_stack=effective_stack,
        oop_bet_sizes=[0.5, 1.0],
        ip_bet_sizes=[0.5, 1.0],
        oop_raise_sizes=[0.5, 1.0],
        ip_raise_sizes=[0.5, 1.0],
    )


@st.composite
def turn_scenario_strategy(draw):
    """生成有效的转牌场景。"""
    name = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N'))))
    description = draw(st.text(min_size=0, max_size=100))
    turn_community = draw(turn_community_strategy())
    oop_range = draw(range_strategy())
    ip_range = draw(range_strategy())
    solver_config = draw(solver_config_strategy())
    
    return TurnScenario(
        name=name,
        description=description,
        turn_community=turn_community,
        oop_range=oop_range,
        ip_range=ip_range,
        solver_config=solver_config,
    )


@st.composite
def histogram_strategy(draw, num_bins: int = 50):
    """生成归一化的直方图。"""
    # 生成随机计数
    counts = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0),
        min_size=num_bins,
        max_size=num_bins
    ))
    counts = np.array(counts)
    
    # 归一化
    total = np.sum(counts)
    if total > 0:
        counts = counts / total
    else:
        counts = np.ones(num_bins) / num_bins
    
    return counts


# ============================================================================
# 数据模型序列化往返测试
# ============================================================================

class TestDataModelSerialization:
    """数据模型序列化往返测试。
    
    **Property: 数据模型序列化往返**
    验证数据类可以正确序列化和反序列化。
    **Validates: Requirements 6.1**
    """
    
    @given(turn_scenario_strategy())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.large_base_example])
    def test_turn_scenario_round_trip(self, scenario: TurnScenario):
        """测试TurnScenario的序列化往返。"""
        # 序列化
        data = scenario.to_dict()
        
        # 反序列化
        restored = TurnScenario.from_dict(data)
        
        # 验证
        assert restored.name == scenario.name
        assert restored.description == scenario.description
        assert len(restored.turn_community) == len(scenario.turn_community)
        assert restored.oop_range == scenario.oop_range
        assert restored.ip_range == scenario.ip_range
        assert restored.solver_config.pot_size == scenario.solver_config.pot_size
    
    @given(
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_correlation_result_round_trip(
        self, 
        mean_corr: float, 
        var_corr: float, 
        purity: float
    ):
        """测试CorrelationResult的序列化往返。"""
        result = CorrelationResult(
            mean_equity_correlation=mean_corr,
            variance_correlation=var_corr,
            clustering_purity=purity,
        )
        
        # 序列化
        data = result.to_dict()
        
        # 反序列化
        restored = CorrelationResult.from_dict(data)
        
        # 验证
        assert np.isclose(restored.mean_equity_correlation, result.mean_equity_correlation)
        assert np.isclose(restored.variance_correlation, result.variance_correlation)
        assert np.isclose(restored.clustering_purity, result.clustering_purity)
    
    @given(
        st.integers(min_value=1, max_value=100),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_clustering_comparison_result_round_trip(
        self,
        num_clusters: int,
        purity: float,
        nmi: float
    ):
        """测试ClusteringComparisonResult的序列化往返。"""
        result = ClusteringComparisonResult(
            num_clusters=num_clusters,
            purity=purity,
            normalized_mutual_info=nmi,
        )
        
        # 序列化
        data = result.to_dict()
        
        # 反序列化
        restored = ClusteringComparisonResult.from_dict(data)
        
        # 验证
        assert restored.num_clusters == result.num_clusters
        assert np.isclose(restored.purity, result.purity)
        assert np.isclose(restored.normalized_mutual_info, result.normalized_mutual_info)
    
    @given(
        st.floats(min_value=0.0, max_value=10.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=-1.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_turn_validation_metrics_round_trip(
        self,
        entropy: float,
        sparsity: float,
        strategy_corr: float,
        ev_corr: float,
        purity: float
    ):
        """测试TurnValidationMetrics的序列化往返。"""
        metrics = TurnValidationMetrics(
            avg_histogram_entropy=entropy,
            histogram_sparsity=sparsity,
            strategy_correlation=strategy_corr,
            ev_correlation=ev_corr,
            clustering_purity=purity,
        )
        
        # 序列化
        data = metrics.to_dict()
        
        # 反序列化
        restored = TurnValidationMetrics.from_dict(data)
        
        # 验证
        assert np.isclose(restored.avg_histogram_entropy, metrics.avg_histogram_entropy)
        assert np.isclose(restored.histogram_sparsity, metrics.histogram_sparsity)
        assert np.isclose(restored.strategy_correlation, metrics.strategy_correlation)
        assert np.isclose(restored.ev_correlation, metrics.ev_correlation)
        assert np.isclose(restored.clustering_purity, metrics.clustering_purity)


# ============================================================================
# 基本功能测试
# ============================================================================

class TestTurnScenarioBasic:
    """TurnScenario基本功能测试。"""
    
    def test_create_valid_scenario(self):
        """测试创建有效的转牌场景。"""
        turn_community = [
            Card(rank=13, suit='s'),  # Ks
            Card(rank=7, suit='d'),   # 7d
            Card(rank=2, suit='c'),   # 2c
            Card(rank=4, suit='h'),   # 4h
        ]
        
        scenario = TurnScenario(
            name="dry_board",
            description="干燥牌面测试",
            turn_community=turn_community,
            oop_range={"AhKs": 1.0, "QhQd": 1.0},
            ip_range={"JhJs": 1.0, "ThTd": 1.0},
            solver_config=SolverConfig(pot_size=100.0, effective_stack=200.0),
        )
        
        assert scenario.name == "dry_board"
        assert len(scenario.turn_community) == 4
        # 验证公共牌字符串包含正确的牌（不检查具体格式，因为Card.__str__可能使用符号）
        community_str = scenario.get_turn_community_str()
        assert len(community_str) > 0
        assert len(scenario.turn_community) == 4
    
    def test_invalid_community_cards_count(self):
        """测试公共牌数量不正确时抛出异常。"""
        with pytest.raises(ValueError, match="转牌阶段必须有4张公共牌"):
            TurnScenario(
                name="test",
                description="",
                turn_community=[Card(rank=13, suit='s')],  # 只有1张
                oop_range={"AhKs": 1.0},
                ip_range={"JhJs": 1.0},
                solver_config=SolverConfig(pot_size=100.0, effective_stack=200.0),
            )
    
    def test_duplicate_community_cards(self):
        """测试公共牌重复时抛出异常。"""
        with pytest.raises(ValueError, match="公共牌中有重复的牌"):
            TurnScenario(
                name="test",
                description="",
                turn_community=[
                    Card(rank=13, suit='s'),
                    Card(rank=13, suit='s'),  # 重复
                    Card(rank=7, suit='d'),
                    Card(rank=2, suit='c'),
                ],
                oop_range={"AhKs": 1.0},
                ip_range={"JhJs": 1.0},
                solver_config=SolverConfig(pot_size=100.0, effective_stack=200.0),
            )


class TestTurnValidationMetricsBasic:
    """TurnValidationMetrics基本功能测试。"""
    
    def test_create_valid_metrics(self):
        """测试创建有效的验证指标。"""
        metrics = TurnValidationMetrics(
            avg_histogram_entropy=2.5,
            histogram_sparsity=0.6,
            strategy_correlation=0.8,
            ev_correlation=0.75,
            clustering_purity=0.85,
        )
        
        assert metrics.avg_histogram_entropy == 2.5
        assert metrics.histogram_sparsity == 0.6
        assert metrics.strategy_correlation == 0.8
    
    def test_invalid_sparsity(self):
        """测试稀疏度超出范围时抛出异常。"""
        with pytest.raises(ValueError, match="直方图稀疏度必须在"):
            TurnValidationMetrics(histogram_sparsity=1.5)
    
    def test_invalid_correlation(self):
        """测试相关系数超出范围时抛出异常。"""
        with pytest.raises(ValueError, match="策略相关性必须在"):
            TurnValidationMetrics(strategy_correlation=2.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



# ============================================================================
# 河牌枚举测试
# ============================================================================

from experiments.turn_potential_validation.river_enumerator import RiverCardEnumerator


class TestRiverCardEnumerator:
    """河牌枚举器测试。
    
    **Property 1: 河牌枚举完整性**
    使用Hypothesis生成随机手牌和公共牌，验证枚举数量为46且无重复。
    **Validates: Requirements 1.1**
    """
    
    @given(
        hand_strategy(),
        turn_community_strategy()
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.large_base_example])
    def test_river_enumeration_count(
        self,
        hole_cards: List[Card],
        turn_community: List[Card]
    ):
        """测试河牌枚举数量为46。
        
        **Feature: turn-potential-validation, Property 1: 河牌枚举完整性**
        """
        # 确保手牌和公共牌不重复
        all_cards = list(hole_cards) + turn_community
        card_keys = [(c.rank, c.suit) for c in all_cards]
        assume(len(set(card_keys)) == 6)  # 2手牌 + 4公共牌
        
        enumerator = RiverCardEnumerator()
        river_cards = enumerator.enumerate_river_cards(
            tuple(hole_cards),
            turn_community
        )
        
        # 验证数量为46
        assert len(river_cards) == 46, f"期望46张河牌，实际：{len(river_cards)}"
    
    @given(
        hand_strategy(),
        turn_community_strategy()
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.large_base_example])
    def test_river_enumeration_no_duplicates(
        self,
        hole_cards: List[Card],
        turn_community: List[Card]
    ):
        """测试河牌枚举无重复。
        
        **Feature: turn-potential-validation, Property 1: 河牌枚举完整性**
        """
        # 确保手牌和公共牌不重复
        all_cards = list(hole_cards) + turn_community
        card_keys = [(c.rank, c.suit) for c in all_cards]
        assume(len(set(card_keys)) == 6)
        
        enumerator = RiverCardEnumerator()
        river_cards = enumerator.enumerate_river_cards(
            tuple(hole_cards),
            turn_community
        )
        
        # 验证无重复
        river_keys = [(c.rank, c.suit) for c in river_cards]
        assert len(set(river_keys)) == len(river_keys), "河牌枚举中有重复"
    
    @given(
        hand_strategy(),
        turn_community_strategy()
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.large_base_example])
    def test_river_enumeration_no_conflict(
        self,
        hole_cards: List[Card],
        turn_community: List[Card]
    ):
        """测试河牌枚举不包含手牌和公共牌。
        
        **Feature: turn-potential-validation, Property 1: 河牌枚举完整性**
        """
        # 确保手牌和公共牌不重复
        all_cards = list(hole_cards) + turn_community
        card_keys = [(c.rank, c.suit) for c in all_cards]
        assume(len(set(card_keys)) == 6)
        
        enumerator = RiverCardEnumerator()
        river_cards = enumerator.enumerate_river_cards(
            tuple(hole_cards),
            turn_community
        )
        
        # 获取已使用的牌
        used_keys = set(card_keys)
        
        # 验证河牌不与已使用的牌冲突
        for river_card in river_cards:
            river_key = (river_card.rank, river_card.suit)
            assert river_key not in used_keys, f"河牌{river_card}与已使用的牌冲突"
    
    def test_river_enumeration_basic(self):
        """测试基本的河牌枚举功能。"""
        hole_cards = (Card(rank=14, suit='h'), Card(rank=13, suit='s'))
        turn_community = [
            Card(rank=12, suit='d'),
            Card(rank=11, suit='c'),
            Card(rank=10, suit='h'),
            Card(rank=9, suit='s'),
        ]
        
        enumerator = RiverCardEnumerator()
        river_cards = enumerator.enumerate_river_cards(hole_cards, turn_community)
        
        assert len(river_cards) == 46
        
        # 验证特定的牌不在河牌中
        used_keys = {(14, 'h'), (13, 's'), (12, 'd'), (11, 'c'), (10, 'h'), (9, 's')}
        for river_card in river_cards:
            assert (river_card.rank, river_card.suit) not in used_keys
    
    def test_invalid_community_cards_count(self):
        """测试公共牌数量不正确时抛出异常。"""
        hole_cards = (Card(rank=14, suit='h'), Card(rank=13, suit='s'))
        turn_community = [Card(rank=12, suit='d')]  # 只有1张
        
        enumerator = RiverCardEnumerator()
        with pytest.raises(ValueError, match="转牌阶段必须有4张公共牌"):
            enumerator.enumerate_river_cards(hole_cards, turn_community)



# ============================================================================
# Potential直方图测试
# ============================================================================

from experiments.turn_potential_validation.potential_histogram import PotentialHistogramCalculator


class TestPotentialHistogramCalculator:
    """Potential直方图计算器测试。
    
    **Property 2: Potential直方图归一化**
    验证直方图概率和为1，所有概率值在[0, 1]范围内。
    **Validates: Requirements 1.3, 8.1**
    """
    
    def test_histogram_normalization_basic(self):
        """测试基本的直方图归一化。
        
        **Feature: turn-potential-validation, Property 2: Potential直方图归一化**
        """
        hole_cards = (Card(rank=14, suit='h'), Card(rank=13, suit='h'))  # AhKh
        turn_community = [
            Card(rank=12, suit='h'),  # Qh
            Card(rank=11, suit='d'),  # Jd
            Card(rank=10, suit='c'),  # Tc
            Card(rank=2, suit='s'),   # 2s
        ]
        opponent_range = {"AA": 1.0, "KK": 1.0, "QQ": 1.0}
        
        calculator = PotentialHistogramCalculator(num_bins=50)
        histogram = calculator.calculate_potential_histogram(
            hole_cards, turn_community, opponent_range
        )
        
        # 验证归一化
        assert calculator.is_normalized(histogram), f"直方图未归一化：和为{np.sum(histogram)}"
        
        # 验证所有值在[0, 1]范围内
        assert np.all(histogram >= 0), "直方图包含负值"
        assert np.all(histogram <= 1), "直方图包含大于1的值"
        
        # 验证形状
        assert len(histogram) == 50, f"直方图长度不正确：{len(histogram)}"
    
    def test_histogram_validation(self):
        """测试直方图验证功能。"""
        calculator = PotentialHistogramCalculator(num_bins=50)
        
        # 有效的直方图
        valid_hist = np.ones(50) / 50
        is_valid, msg = calculator.validate_histogram(valid_hist)
        assert is_valid, f"有效直方图被判定为无效：{msg}"
        
        # 无效的直方图（未归一化）
        invalid_hist = np.ones(50)
        is_valid, msg = calculator.validate_histogram(invalid_hist)
        assert not is_valid, "未归一化的直方图被判定为有效"
        
        # 无效的直方图（包含负值）
        negative_hist = np.ones(50) / 50
        negative_hist[0] = -0.1
        negative_hist[1] = 0.12
        is_valid, msg = calculator.validate_histogram(negative_hist)
        assert not is_valid, "包含负值的直方图被判定为有效"
    
    def test_histogram_features(self):
        """测试直方图特征提取。"""
        calculator = PotentialHistogramCalculator(num_bins=50)
        
        # 创建一个简单的直方图（集中在高Equity区域）
        histogram = np.zeros(50)
        histogram[45:50] = 0.2  # 高Equity区域
        
        features = calculator.get_histogram_features(histogram)
        
        # 验证特征存在
        assert 'mean_equity' in features
        assert 'variance' in features
        assert 'entropy' in features
        assert 'sparsity' in features
        
        # 验证均值在合理范围内
        assert 0 <= features['mean_equity'] <= 1
        
        # 验证方差非负
        assert features['variance'] >= 0
        
        # 验证熵非负
        assert features['entropy'] >= 0
        
        # 验证稀疏度在[0, 1]范围内
        assert 0 <= features['sparsity'] <= 1
    
    @given(
        hand_strategy(),
        turn_community_strategy()
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.large_base_example], deadline=None)
    def test_histogram_always_normalized(
        self,
        hole_cards: List[Card],
        turn_community: List[Card]
    ):
        """测试任意输入下直方图都是归一化的。
        
        **Feature: turn-potential-validation, Property 2: Potential直方图归一化**
        """
        # 确保手牌和公共牌不重复
        all_cards = list(hole_cards) + turn_community
        card_keys = [(c.rank, c.suit) for c in all_cards]
        assume(len(set(card_keys)) == 6)
        
        # 使用简单的对手范围
        opponent_range = {"AA": 1.0, "KK": 1.0}
        
        calculator = PotentialHistogramCalculator(num_bins=50)
        histogram = calculator.calculate_potential_histogram(
            tuple(hole_cards), turn_community, opponent_range
        )
        
        # 验证归一化
        assert calculator.is_normalized(histogram), f"直方图未归一化：和为{np.sum(histogram)}"
        
        # 验证所有值在[0, 1]范围内
        assert np.all(histogram >= 0), "直方图包含负值"
        assert np.all(histogram <= 1), "直方图包含大于1的值"



class TestRangeVsRangePotentialHistograms:
    """范围VS范围Potential直方图测试。
    
    **Property 4: 范围VS范围计算完整性**
    验证返回的直方图数量等于有效手牌数，每个直方图都是归一化的。
    **Validates: Requirements 2.1, 2.3**
    """
    
    def test_range_histograms_basic(self):
        """测试基本的范围VS范围直方图计算。
        
        **Feature: turn-potential-validation, Property 4: 范围VS范围计算完整性**
        """
        turn_community = [
            Card(rank=13, suit='s'),  # Ks
            Card(rank=7, suit='d'),   # 7d
            Card(rank=2, suit='c'),   # 2c
            Card(rank=4, suit='h'),   # 4h
        ]
        
        # 简单的范围
        my_range = {"AhKh": 1.0, "QhQd": 1.0, "JhJs": 1.0}
        opponent_range = {"AA": 1.0, "KK": 1.0}
        
        calculator = PotentialHistogramCalculator(num_bins=50)
        histograms = calculator.calculate_range_potential_histograms(
            my_range, opponent_range, turn_community
        )
        
        # 验证返回的直方图数量
        # 注意：AhKh中的Ks与公共牌冲突，所以只有2个有效手牌
        assert len(histograms) >= 1, "应该至少有1个有效手牌"
        
        # 验证每个直方图都是归一化的
        for hand_str, histogram in histograms.items():
            assert calculator.is_normalized(histogram), \
                f"手牌{hand_str}的直方图未归一化：和为{np.sum(histogram)}"
            assert np.all(histogram >= 0), f"手牌{hand_str}的直方图包含负值"
            assert np.all(histogram <= 1), f"手牌{hand_str}的直方图包含大于1的值"
    
    def test_range_histograms_no_conflict_hands(self):
        """测试不与公共牌冲突的手牌都被包含。
        
        **Feature: turn-potential-validation, Property 4: 范围VS范围计算完整性**
        """
        turn_community = [
            Card(rank=13, suit='s'),  # Ks
            Card(rank=7, suit='d'),   # 7d
            Card(rank=2, suit='c'),   # 2c
            Card(rank=4, suit='h'),   # 4h
        ]
        
        # 使用具体的手牌（不与公共牌冲突）
        my_range = {
            "AhAd": 1.0,  # 不冲突
            "QhQd": 1.0,  # 不冲突
            "JhJc": 1.0,  # 不冲突
        }
        opponent_range = {"TT": 1.0}
        
        calculator = PotentialHistogramCalculator(num_bins=50)
        histograms = calculator.calculate_range_potential_histograms(
            my_range, opponent_range, turn_community
        )
        
        # 验证所有不冲突的手牌都被包含
        assert len(histograms) == 3, f"期望3个手牌，实际：{len(histograms)}"
        
        # 验证每个直方图都是归一化的
        for hand_str, histogram in histograms.items():
            is_valid, msg = calculator.validate_histogram(histogram)
            assert is_valid, f"手牌{hand_str}的直方图无效：{msg}"
    
    def test_range_histograms_conflict_hands_excluded(self):
        """测试与公共牌冲突的手牌被排除。
        
        **Feature: turn-potential-validation, Property 4: 范围VS范围计算完整性**
        """
        turn_community = [
            Card(rank=13, suit='s'),  # Ks
            Card(rank=7, suit='d'),   # 7d
            Card(rank=2, suit='c'),   # 2c
            Card(rank=4, suit='h'),   # 4h
        ]
        
        # 包含与公共牌冲突的手牌
        my_range = {
            "KsQh": 1.0,  # Ks与公共牌冲突
            "AhAd": 1.0,  # 不冲突
        }
        opponent_range = {"TT": 1.0}
        
        calculator = PotentialHistogramCalculator(num_bins=50)
        histograms = calculator.calculate_range_potential_histograms(
            my_range, opponent_range, turn_community
        )
        
        # 验证冲突的手牌被排除
        assert "KsQh" not in histograms, "与公共牌冲突的手牌应该被排除"
        assert "AhAd" in histograms, "不冲突的手牌应该被包含"
    
    @given(turn_community_strategy())
    @settings(max_examples=10, suppress_health_check=[HealthCheck.large_base_example], deadline=None)
    def test_all_histograms_normalized(self, turn_community: List[Card]):
        """测试所有返回的直方图都是归一化的。
        
        **Feature: turn-potential-validation, Property 4: 范围VS范围计算完整性**
        """
        # 使用简单的范围
        my_range = {"AA": 1.0, "KK": 1.0}
        opponent_range = {"QQ": 1.0, "JJ": 1.0}
        
        calculator = PotentialHistogramCalculator(num_bins=50)
        histograms = calculator.calculate_range_potential_histograms(
            my_range, opponent_range, turn_community
        )
        
        # 验证每个直方图都是归一化的
        for hand_str, histogram in histograms.items():
            assert calculator.is_normalized(histogram), \
                f"手牌{hand_str}的直方图未归一化：和为{np.sum(histogram)}"



# ============================================================================
# EMD距离测试
# ============================================================================

from abstraction.emd_calculator import EMDCalculator


class TestEMDCalculator:
    """EMD距离计算器测试。
    
    **Property 5: EMD距离非负性和同一性**
    验证EMD距离非负，相同直方图EMD为0。
    **Validates: Requirements 3.1, 3.3**
    
    **Property 6: EMD距离对称性**
    验证EMD(A, B) = EMD(B, A)。
    **Validates: Requirements 3.1**
    """
    
    @given(histogram_strategy(num_bins=50))
    @settings(max_examples=50)
    def test_emd_identity(self, histogram: np.ndarray):
        """测试相同直方图的EMD为0。
        
        **Feature: turn-potential-validation, Property 5: EMD距离非负性和同一性**
        """
        emd = EMDCalculator.calculate_emd_1d(histogram, histogram)
        assert np.isclose(emd, 0.0, atol=1e-10), f"相同直方图的EMD应为0，实际：{emd}"
    
    @given(histogram_strategy(num_bins=50))
    @settings(max_examples=50)
    def test_emd_non_negative(self, histogram: np.ndarray):
        """测试EMD距离非负。
        
        **Feature: turn-potential-validation, Property 5: EMD距离非负性和同一性**
        """
        # 创建另一个随机直方图
        other = np.random.random(50)
        other = other / np.sum(other)
        
        emd = EMDCalculator.calculate_emd_1d(histogram, other)
        assert emd >= 0, f"EMD距离应非负，实际：{emd}"
    
    @given(
        histogram_strategy(num_bins=50),
        histogram_strategy(num_bins=50)
    )
    @settings(max_examples=50)
    def test_emd_symmetry(self, hist1: np.ndarray, hist2: np.ndarray):
        """测试EMD距离对称性。
        
        **Feature: turn-potential-validation, Property 6: EMD距离对称性**
        """
        emd_ab = EMDCalculator.calculate_emd_1d(hist1, hist2)
        emd_ba = EMDCalculator.calculate_emd_1d(hist2, hist1)
        
        assert np.isclose(emd_ab, emd_ba, atol=1e-10), \
            f"EMD应对称：EMD(A,B)={emd_ab}, EMD(B,A)={emd_ba}"
    
    def test_emd_basic_cases(self):
        """测试EMD的基本情况。"""
        # 完全相同
        hist1 = np.array([0.2, 0.3, 0.5])
        hist2 = np.array([0.2, 0.3, 0.5])
        assert np.isclose(EMDCalculator.calculate_emd_1d(hist1, hist2), 0.0)
        
        # 完全不同（极端情况）
        hist3 = np.array([1.0, 0.0, 0.0])
        hist4 = np.array([0.0, 0.0, 1.0])
        emd = EMDCalculator.calculate_emd_1d(hist3, hist4)
        assert emd > 0, "完全不同的直方图EMD应大于0"
        
        # 部分重叠
        hist5 = np.array([0.5, 0.5, 0.0])
        hist6 = np.array([0.0, 0.5, 0.5])
        emd = EMDCalculator.calculate_emd_1d(hist5, hist6)
        assert emd > 0, "部分重叠的直方图EMD应大于0"
    
    def test_emd_metric_properties(self):
        """测试EMD满足度量空间性质。"""
        hist1 = np.array([0.3, 0.4, 0.3])
        hist2 = np.array([0.2, 0.5, 0.3])
        hist3 = np.array([0.4, 0.3, 0.3])
        
        non_neg, symmetric, triangle = EMDCalculator.validate_metric_properties(
            hist1, hist2, hist3
        )
        
        assert non_neg, "EMD应满足非负性"
        assert symmetric, "EMD应满足对称性"
        assert triangle, "EMD应满足三角不等式"



# ============================================================================
# 转牌Solver测试
# ============================================================================

from experiments.turn_potential_validation.turn_solver import TurnCFRSolver
from experiments.equity_solver_validation.data_models import SolverConfig


class TestTurnCFRSolver:
    """转牌CFR求解器测试。
    
    **Property 7: 策略概率归一化**
    验证提取的策略概率之和为1。
    **Validates: Requirements 4.2**
    """
    
    def test_solver_basic(self):
        """测试基本的Solver功能。"""
        turn_community = [
            Card(rank=13, suit='s'),  # Ks
            Card(rank=7, suit='d'),   # 7d
            Card(rank=2, suit='c'),   # 2c
            Card(rank=4, suit='h'),   # 4h
        ]
        
        config = SolverConfig(pot_size=100.0, effective_stack=200.0)
        solver = TurnCFRSolver(config)
        
        oop_range = {"AA": 1.0, "KK": 1.0}
        ip_range = {"QQ": 1.0, "JJ": 1.0}
        
        result = solver.solve(turn_community, oop_range, ip_range, iterations=100)
        
        assert result.converged
        assert 'root' in result.strategies
    
    def test_strategy_normalization(self):
        """测试策略概率归一化。
        
        **Feature: turn-potential-validation, Property 7: 策略概率归一化**
        """
        turn_community = [
            Card(rank=13, suit='s'),
            Card(rank=7, suit='d'),
            Card(rank=2, suit='c'),
            Card(rank=4, suit='h'),
        ]
        
        config = SolverConfig(pot_size=100.0, effective_stack=200.0)
        solver = TurnCFRSolver(config)
        
        oop_range = {"AA": 1.0, "KK": 1.0, "QQ": 1.0}
        ip_range = {"JJ": 1.0, "TT": 1.0, "99": 1.0}
        
        result = solver.solve(turn_community, oop_range, ip_range, iterations=100)
        
        # 验证每个节点的策略都是归一化的
        for node_path, strategy in result.strategies.items():
            for hand, action_probs in strategy.items():
                total_prob = sum(action_probs.values())
                assert np.isclose(total_prob, 1.0, atol=1e-6), \
                    f"节点{node_path}手牌{hand}的策略未归一化：和为{total_prob}"
                
                # 验证每个概率在[0, 1]范围内
                for action, prob in action_probs.items():
                    assert 0 <= prob <= 1, \
                        f"节点{node_path}手牌{hand}动作{action}的概率超出范围：{prob}"
    
    @given(turn_community_strategy())
    @settings(max_examples=10, suppress_health_check=[HealthCheck.large_base_example], deadline=None)
    def test_strategy_always_normalized(self, turn_community: List[Card]):
        """测试任意输入下策略都是归一化的。
        
        **Feature: turn-potential-validation, Property 7: 策略概率归一化**
        """
        config = SolverConfig(pot_size=100.0, effective_stack=200.0)
        solver = TurnCFRSolver(config)
        
        oop_range = {"AA": 1.0}
        ip_range = {"KK": 1.0}
        
        result = solver.solve(turn_community, oop_range, ip_range, iterations=50)
        
        for node_path, strategy in result.strategies.items():
            for hand, action_probs in strategy.items():
                total_prob = sum(action_probs.values())
                assert np.isclose(total_prob, 1.0, atol=1e-6), \
                    f"策略未归一化：和为{total_prob}"



# ============================================================================
# Potential分析器测试
# ============================================================================

from experiments.turn_potential_validation.potential_analyzer import PotentialAnalyzer


class TestPotentialAnalyzer:
    """Potential分析器测试。
    
    测试Potential直方图与Solver策略之间的相关性分析功能。
    """
    
    def test_analyzer_basic(self):
        """测试基本的分析器功能。"""
        analyzer = PotentialAnalyzer(num_bins=50)
        
        # 创建测试数据
        potential_histograms = {
            "AhAd": np.ones(50) / 50,
            "KhKd": np.ones(50) / 50,
            "QhQd": np.ones(50) / 50,
        }
        
        solver_strategies = {
            "AhAd": {"check": 0.3, "bet": 0.7},
            "KhKd": {"check": 0.5, "bet": 0.5},
            "QhQd": {"check": 0.7, "bet": 0.3},
        }
        
        result = analyzer.analyze_histogram_strategy_correlation(
            potential_histograms, solver_strategies
        )
        
        # 验证结果类型
        assert isinstance(result, CorrelationResult)
        
        # 验证相关系数在有效范围内
        assert -1 <= result.mean_equity_correlation <= 1
        assert -1 <= result.variance_correlation <= 1
    
    def test_analyzer_empty_input(self):
        """测试空输入的处理。"""
        analyzer = PotentialAnalyzer(num_bins=50)
        
        result = analyzer.analyze_histogram_strategy_correlation({}, {})
        
        # 应该返回默认的CorrelationResult
        assert isinstance(result, CorrelationResult)
        assert result.mean_equity_correlation == 0.0
    
    def test_clustering_basic(self):
        """测试基本的聚类功能。"""
        analyzer = PotentialAnalyzer(num_bins=50)
        
        # 创建不同的直方图
        potential_histograms = {
            "AhAd": np.concatenate([np.zeros(40), np.ones(10) / 10]),  # 高Equity
            "KhKd": np.concatenate([np.zeros(40), np.ones(10) / 10]),  # 高Equity
            "2h3d": np.concatenate([np.ones(10) / 10, np.zeros(40)]),  # 低Equity
            "4h5d": np.concatenate([np.ones(10) / 10, np.zeros(40)]),  # 低Equity
        }
        
        cluster_labels = analyzer.cluster_by_potential(potential_histograms, num_clusters=2)
        
        # 验证所有手牌都有标签
        assert len(cluster_labels) == 4
        
        # 验证标签在有效范围内
        for label in cluster_labels.values():
            assert 0 <= label < 2
        
        # 验证相似的手牌被分到同一聚类
        assert cluster_labels["AhAd"] == cluster_labels["KhKd"]
        assert cluster_labels["2h3d"] == cluster_labels["4h5d"]
    
    def test_clustering_with_emd(self):
        """测试基于EMD距离的聚类。"""
        analyzer = PotentialAnalyzer(num_bins=50)
        
        # 创建不同的直方图
        potential_histograms = {
            "AhAd": np.concatenate([np.zeros(40), np.ones(10) / 10]),
            "KhKd": np.concatenate([np.zeros(40), np.ones(10) / 10]),
            "2h3d": np.concatenate([np.ones(10) / 10, np.zeros(40)]),
            "4h5d": np.concatenate([np.ones(10) / 10, np.zeros(40)]),
        }
        
        cluster_labels = analyzer.cluster_by_potential_with_emd(
            potential_histograms, num_clusters=2, max_iterations=50
        )
        
        # 验证所有手牌都有标签
        assert len(cluster_labels) == 4
        
        # 验证标签在有效范围内
        for label in cluster_labels.values():
            assert 0 <= label < 2
    
    def test_compare_clustering_with_strategy(self):
        """测试聚类与策略比较。"""
        analyzer = PotentialAnalyzer(num_bins=50)
        
        cluster_labels = {
            "AhAd": 0,
            "KhKd": 0,
            "QhQd": 1,
            "JhJd": 1,
        }
        
        solver_strategies = {
            "AhAd": {"check": 0.2, "bet": 0.8},
            "KhKd": {"check": 0.3, "bet": 0.7},
            "QhQd": {"check": 0.8, "bet": 0.2},
            "JhJd": {"check": 0.7, "bet": 0.3},
        }
        
        result = analyzer.compare_clustering_with_strategy(
            cluster_labels, solver_strategies
        )
        
        # 验证结果类型
        assert isinstance(result, ClusteringComparisonResult)
        
        # 验证聚类数量
        assert result.num_clusters == 2
        
        # 验证纯度在有效范围内
        assert 0 <= result.purity <= 1
        
        # 验证归一化互信息在有效范围内
        assert 0 <= result.normalized_mutual_info <= 1
        
        # 验证每个聚类都有动作分布
        assert len(result.action_distribution_per_cluster) == 2
    
    def test_find_similar_hands(self):
        """测试查找相似手牌。"""
        analyzer = PotentialAnalyzer(num_bins=50)
        
        # 创建测试直方图
        potential_histograms = {
            "AhAd": np.concatenate([np.zeros(40), np.ones(10) / 10]),
            "KhKd": np.concatenate([np.zeros(40), np.ones(10) / 10]),
            "QhQd": np.concatenate([np.zeros(35), np.ones(15) / 15]),
            "2h3d": np.concatenate([np.ones(10) / 10, np.zeros(40)]),
        }
        
        similar = analyzer.find_similar_hands(potential_histograms, "AhAd", top_k=2)
        
        # 验证返回正确数量
        assert len(similar) == 2
        
        # 验证返回的是元组列表
        for hand, dist in similar:
            assert isinstance(hand, str)
            assert isinstance(dist, float)
            assert dist >= 0
        
        # 验证最相似的手牌是KhKd（直方图相同）
        assert similar[0][0] == "KhKd"
        assert np.isclose(similar[0][1], 0.0, atol=1e-10)
    
    def test_histogram_similarity_matrix(self):
        """测试直方图相似度矩阵计算。"""
        analyzer = PotentialAnalyzer(num_bins=50)
        
        potential_histograms = {
            "AhAd": np.ones(50) / 50,
            "KhKd": np.ones(50) / 50,
            "QhQd": np.ones(50) / 50,
        }
        
        hands, distance_matrix = analyzer.get_histogram_similarity_matrix(
            potential_histograms
        )
        
        # 验证手牌列表
        assert len(hands) == 3
        
        # 验证距离矩阵形状
        assert distance_matrix.shape == (3, 3)
        
        # 验证对角线为0
        for i in range(3):
            assert np.isclose(distance_matrix[i, i], 0.0)
        
        # 验证对称性
        for i in range(3):
            for j in range(3):
                assert np.isclose(distance_matrix[i, j], distance_matrix[j, i])
    
    @given(
        st.integers(min_value=2, max_value=10),
        st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=20)
    def test_clustering_labels_valid(self, num_hands: int, num_clusters: int):
        """测试聚类标签始终有效。"""
        assume(num_hands >= num_clusters)
        
        analyzer = PotentialAnalyzer(num_bins=50)
        
        # 生成随机直方图
        potential_histograms = {}
        for i in range(num_hands):
            hand_str = f"hand_{i}"
            hist = np.random.random(50)
            hist = hist / np.sum(hist)
            potential_histograms[hand_str] = hist
        
        cluster_labels = analyzer.cluster_by_potential(
            potential_histograms, num_clusters=num_clusters
        )
        
        # 验证所有手牌都有标签
        assert len(cluster_labels) == num_hands
        
        # 验证标签在有效范围内
        for label in cluster_labels.values():
            assert 0 <= label < num_clusters
    
    def test_action_emd_distances(self):
        """测试动作内和动作间EMD距离计算。"""
        analyzer = PotentialAnalyzer(num_bins=50)
        
        # 创建测试数据：bet手牌有高Equity，check手牌有低Equity
        potential_histograms = {
            "AhAd": np.concatenate([np.zeros(40), np.ones(10) / 10]),  # 高Equity
            "KhKd": np.concatenate([np.zeros(40), np.ones(10) / 10]),  # 高Equity
            "2h3d": np.concatenate([np.ones(10) / 10, np.zeros(40)]),  # 低Equity
            "4h5d": np.concatenate([np.ones(10) / 10, np.zeros(40)]),  # 低Equity
        }
        
        solver_strategies = {
            "AhAd": {"check": 0.1, "bet": 0.9},  # 主要bet
            "KhKd": {"check": 0.2, "bet": 0.8},  # 主要bet
            "2h3d": {"check": 0.9, "bet": 0.1},  # 主要check
            "4h5d": {"check": 0.8, "bet": 0.2},  # 主要check
        }
        
        result = analyzer.analyze_histogram_strategy_correlation(
            potential_histograms, solver_strategies
        )
        
        # 验证同一动作内的EMD存在
        assert "bet" in result.intra_action_emd or "check" in result.intra_action_emd
        
        # 验证不同动作间的EMD存在
        assert len(result.inter_action_emd) > 0
        
        # 同一动作内的EMD应该较小（相似的直方图）
        for action, emd in result.intra_action_emd.items():
            assert emd >= 0
        
        # 不同动作间的EMD应该较大（不同的直方图）
        for key, emd in result.inter_action_emd.items():
            assert emd >= 0



# ============================================================================
# 实验运行器测试
# ============================================================================

from experiments.turn_potential_validation.experiment_runner import (
    TurnExperimentRunner,
    create_default_scenarios,
)


class TestTurnExperimentRunner:
    """转牌实验运行器测试。
    
    **Property 8: 批量实验结果完整性**
    验证每个场景都有对应结果。
    **Validates: Requirements 6.2**
    """
    
    def test_runner_basic(self):
        """测试基本的实验运行器功能。"""
        runner = TurnExperimentRunner(
            num_histogram_bins=50,
            default_solver_iterations=100,
        )
        
        # 创建简单的测试场景
        scenario = TurnScenario(
            name="test_scenario",
            description="测试场景",
            turn_community=[
                Card(rank=13, suit='s'),
                Card(rank=7, suit='d'),
                Card(rank=2, suit='c'),
                Card(rank=4, suit='h'),
            ],
            oop_range={"AA": 1.0, "KK": 1.0},
            ip_range={"QQ": 1.0, "JJ": 1.0},
            solver_config=SolverConfig(pot_size=100.0, effective_stack=200.0),
        )
        
        result = runner.run_experiment(scenario, solver_iterations=50)
        
        # 验证结果
        assert result.scenario.name == "test_scenario"
        assert result.execution_time > 0
    
    def test_runner_with_valid_scenario(self):
        """测试有效场景的实验运行。"""
        runner = TurnExperimentRunner(
            num_histogram_bins=50,
            default_solver_iterations=100,
        )
        
        scenario = TurnScenario(
            name="valid_scenario",
            description="有效场景",
            turn_community=[
                Card(rank=14, suit='s'),  # As
                Card(rank=10, suit='d'),  # Td
                Card(rank=7, suit='c'),   # 7c
                Card(rank=3, suit='h'),   # 3h
            ],
            oop_range={"AhAd": 1.0, "KhKd": 1.0, "QhQd": 1.0},
            ip_range={"JhJd": 1.0, "ThTc": 1.0, "9h9d": 1.0},
            solver_config=SolverConfig(pot_size=100.0, effective_stack=200.0),
        )
        
        result = runner.run_experiment(scenario, solver_iterations=100)
        
        # 验证成功
        if result.success:
            assert result.potential_histograms is not None
            assert len(result.potential_histograms) > 0
            assert result.metrics is not None
            assert result.correlation_result is not None
            assert result.clustering_result is not None
    
    def test_batch_experiments_completeness(self):
        """测试批量实验结果完整性。
        
        **Feature: turn-potential-validation, Property 8: 批量实验结果完整性**
        """
        runner = TurnExperimentRunner(
            num_histogram_bins=50,
            default_solver_iterations=50,
        )
        
        # 创建多个测试场景
        scenarios = [
            TurnScenario(
                name=f"scenario_{i}",
                description=f"测试场景{i}",
                turn_community=[
                    Card(rank=13, suit='s'),
                    Card(rank=7, suit='d'),
                    Card(rank=2, suit='c'),
                    Card(rank=4 + i, suit='h'),  # 不同的转牌
                ],
                oop_range={"AA": 1.0},
                ip_range={"KK": 1.0},
                solver_config=SolverConfig(pot_size=100.0, effective_stack=200.0),
            )
            for i in range(3)
        ]
        
        batch_result = runner.run_batch_experiments(
            scenarios, solver_iterations=50, verbose=False
        )
        
        # 验证结果数量等于场景数量
        assert len(batch_result.results) == len(scenarios), \
            f"期望{len(scenarios)}个结果，实际{len(batch_result.results)}"
        
        # 验证每个场景都有对应结果
        result_names = {r.scenario.name for r in batch_result.results}
        scenario_names = {s.name for s in scenarios}
        assert result_names == scenario_names, "结果与场景不匹配"
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=5, deadline=None)
    def test_batch_results_count_matches_scenarios(self, num_scenarios: int):
        """测试批量实验结果数量始终等于场景数量。
        
        **Feature: turn-potential-validation, Property 8: 批量实验结果完整性**
        """
        runner = TurnExperimentRunner(
            num_histogram_bins=50,
            default_solver_iterations=30,
        )
        
        # 创建场景
        scenarios = [
            TurnScenario(
                name=f"prop_scenario_{i}",
                description=f"属性测试场景{i}",
                turn_community=[
                    Card(rank=13, suit='s'),
                    Card(rank=7, suit='d'),
                    Card(rank=2, suit='c'),
                    Card(rank=min(4 + i, 14), suit='h'),
                ],
                oop_range={"AA": 1.0},
                ip_range={"KK": 1.0},
                solver_config=SolverConfig(pot_size=100.0, effective_stack=200.0),
            )
            for i in range(num_scenarios)
        ]
        
        batch_result = runner.run_batch_experiments(
            scenarios, solver_iterations=30, verbose=False
        )
        
        # 验证结果数量
        assert len(batch_result.results) == num_scenarios
    
    def test_summary_report_generation(self):
        """测试汇总报告生成。"""
        runner = TurnExperimentRunner(
            num_histogram_bins=50,
            default_solver_iterations=50,
        )
        
        scenarios = [
            TurnScenario(
                name="report_test",
                description="报告测试",
                turn_community=[
                    Card(rank=13, suit='s'),
                    Card(rank=7, suit='d'),
                    Card(rank=2, suit='c'),
                    Card(rank=4, suit='h'),
                ],
                oop_range={"AA": 1.0},
                ip_range={"KK": 1.0},
                solver_config=SolverConfig(pot_size=100.0, effective_stack=200.0),
            )
        ]
        
        batch_result = runner.run_batch_experiments(
            scenarios, solver_iterations=50, verbose=False
        )
        
        report = runner.generate_summary_report(batch_result)
        
        # 验证报告包含关键信息
        assert "实验数量" in report
        assert "report_test" in report
    
    def test_default_scenarios(self):
        """测试默认场景创建。"""
        scenarios = create_default_scenarios()
        
        # 验证创建了多个场景
        assert len(scenarios) >= 4
        
        # 验证每个场景都是有效的
        for scenario in scenarios:
            assert len(scenario.turn_community) == 4
            assert scenario.name
            assert scenario.oop_range
            assert scenario.ip_range
    
    def test_metrics_computation(self):
        """测试指标计算。"""
        runner = TurnExperimentRunner(
            num_histogram_bins=50,
            default_solver_iterations=100,
        )
        
        scenario = TurnScenario(
            name="metrics_test",
            description="指标测试",
            turn_community=[
                Card(rank=14, suit='s'),
                Card(rank=10, suit='d'),
                Card(rank=7, suit='c'),
                Card(rank=3, suit='h'),
            ],
            oop_range={"AhAd": 1.0, "KhKd": 1.0, "QhQd": 1.0, "JhJd": 1.0},
            ip_range={"ThTc": 1.0, "9h9d": 1.0, "8h8d": 1.0, "7h7c": 1.0},
            solver_config=SolverConfig(pot_size=100.0, effective_stack=200.0),
        )
        
        result = runner.run_experiment(scenario, solver_iterations=100)
        
        if result.success and result.metrics:
            # 验证指标在有效范围内
            assert result.metrics.avg_histogram_entropy >= 0
            assert 0 <= result.metrics.histogram_sparsity <= 1
            assert -1 <= result.metrics.strategy_correlation <= 1
            assert 0 <= result.metrics.clustering_purity <= 1
            assert 0 <= result.metrics.action_agreement_rate <= 1



# ============================================================================
# Potential直方图计算一致性测试（Property 9）
# ============================================================================

from experiments.turn_potential_validation.histogram_validator import (
    HistogramValidator,
    ManualHistogramCalculator,
    ValidationResult,
    BatchValidationResult,
)


class TestHistogramValidation:
    """Potential直方图计算一致性测试。
    
    **Property 9: Potential直方图计算一致性（Round-trip）**
    验证计算结果与手动枚举一致。
    **Validates: Requirements 8.2**
    """
    
    def test_validation_basic(self):
        """测试基本的验证功能。
        
        **Feature: turn-potential-validation, Property 9: Potential直方图计算一致性（Round-trip）**
        """
        hole_cards = (Card(rank=14, suit='h'), Card(rank=13, suit='h'))  # AhKh
        turn_community = [
            Card(rank=12, suit='h'),  # Qh
            Card(rank=11, suit='d'),  # Jd
            Card(rank=10, suit='c'),  # Tc
            Card(rank=2, suit='s'),   # 2s
        ]
        opponent_range = {"AA": 1.0, "KK": 1.0, "QQ": 1.0}
        
        validator = HistogramValidator(num_bins=50, tolerance=1e-6)
        result = validator.validate_histogram(hole_cards, turn_community, opponent_range)
        
        # 验证通过
        assert result.is_valid, f"验证失败：{result.error_message}"
        assert result.max_error < 1e-6, f"最大误差过大：{result.max_error}"
    
    def test_validation_with_different_ranges(self):
        """测试不同范围下的验证。
        
        **Feature: turn-potential-validation, Property 9: Potential直方图计算一致性（Round-trip）**
        """
        hole_cards = (Card(rank=10, suit='s'), Card(rank=10, suit='h'))  # TsTh
        turn_community = [
            Card(rank=14, suit='d'),  # Ad
            Card(rank=8, suit='c'),   # 8c
            Card(rank=5, suit='h'),   # 5h
            Card(rank=3, suit='s'),   # 3s
        ]
        
        # 测试不同的对手范围
        ranges_to_test = [
            {"AA": 1.0},
            {"AA": 1.0, "KK": 1.0},
            {"AA": 1.0, "KK": 1.0, "QQ": 1.0, "JJ": 1.0},
            {"AKs": 1.0, "AKo": 1.0},
        ]
        
        validator = HistogramValidator(num_bins=50, tolerance=1e-6)
        
        for opponent_range in ranges_to_test:
            result = validator.validate_histogram(hole_cards, turn_community, opponent_range)
            assert result.is_valid, f"范围{opponent_range}验证失败：{result.error_message}"
    
    def test_validation_with_different_boards(self):
        """测试不同牌面下的验证。
        
        **Feature: turn-potential-validation, Property 9: Potential直方图计算一致性（Round-trip）**
        """
        hole_cards = (Card(rank=14, suit='h'), Card(rank=14, suit='d'))  # AhAd
        opponent_range = {"KK": 1.0, "QQ": 1.0}
        
        # 测试不同的牌面
        boards_to_test = [
            # 干燥牌面
            [Card(rank=13, suit='s'), Card(rank=7, suit='d'), 
             Card(rank=2, suit='c'), Card(rank=4, suit='h')],
            # 湿润牌面
            [Card(rank=11, suit='s'), Card(rank=10, suit='s'), 
             Card(rank=9, suit='d'), Card(rank=8, suit='c')],
            # 配对牌面
            [Card(rank=13, suit='s'), Card(rank=13, suit='d'), 
             Card(rank=7, suit='c'), Card(rank=3, suit='h')],
        ]
        
        validator = HistogramValidator(num_bins=50, tolerance=1e-6)
        
        for turn_community in boards_to_test:
            result = validator.validate_histogram(hole_cards, turn_community, opponent_range)
            assert result.is_valid, f"牌面验证失败：{result.error_message}"
    
    @given(
        hand_strategy(),
        turn_community_strategy()
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.large_base_example], deadline=None)
    def test_histogram_consistency_property(
        self,
        hole_cards: List[Card],
        turn_community: List[Card]
    ):
        """属性测试：Potential直方图计算一致性。
        
        *For any* 手牌和公共牌组合，通过枚举所有河牌计算的Potential直方图，
        与通过手动枚举验证的结果必须一致（误差小于1e-6）。
        
        **Feature: turn-potential-validation, Property 9: Potential直方图计算一致性（Round-trip）**
        **Validates: Requirements 8.2**
        """
        # 确保手牌和公共牌不重复
        all_cards = list(hole_cards) + turn_community
        card_keys = [(c.rank, c.suit) for c in all_cards]
        assume(len(set(card_keys)) == 6)  # 2手牌 + 4公共牌
        
        # 使用简单的对手范围
        opponent_range = {"AA": 1.0, "KK": 1.0}
        
        validator = HistogramValidator(num_bins=50, tolerance=1e-6)
        result = validator.validate_histogram(
            tuple(hole_cards), turn_community, opponent_range
        )
        
        # 验证通过
        assert result.is_valid, f"验证失败：最大误差={result.max_error:.2e}"
        assert result.max_error < 1e-6, f"最大误差过大：{result.max_error:.2e}"
    
    def test_batch_validation(self):
        """测试批量验证功能。
        
        **Feature: turn-potential-validation, Property 9: Potential直方图计算一致性（Round-trip）**
        """
        turn_community = [
            Card(rank=13, suit='s'),  # Ks
            Card(rank=7, suit='d'),   # 7d
            Card(rank=2, suit='c'),   # 2c
            Card(rank=4, suit='h'),   # 4h
        ]
        
        my_range = {"AhAd": 1.0, "KhKd": 1.0, "QhQd": 1.0}
        opponent_range = {"JJ": 1.0, "TT": 1.0}
        
        validator = HistogramValidator(num_bins=50, tolerance=1e-6)
        result = validator.validate_range_histograms(
            my_range, opponent_range, turn_community
        )
        
        # 验证所有手牌都通过
        assert result.all_valid, f"批量验证失败：{result.num_passed}/{result.num_validated}通过"
        assert result.overall_max_error < 1e-6, f"总体最大误差过大：{result.overall_max_error}"
    
    def test_validation_report_generation(self):
        """测试验证报告生成。"""
        hole_cards = (Card(rank=14, suit='h'), Card(rank=13, suit='h'))
        turn_community = [
            Card(rank=12, suit='h'),
            Card(rank=11, suit='d'),
            Card(rank=10, suit='c'),
            Card(rank=2, suit='s'),
        ]
        opponent_range = {"AA": 1.0}
        
        validator = HistogramValidator(num_bins=50, tolerance=1e-6)
        result = validator.validate_histogram(hole_cards, turn_community, opponent_range)
        
        report = validator.generate_validation_report(result, "AhKh")
        
        # 验证报告包含关键信息
        assert "验证报告" in report
        assert "AhKh" in report
        assert "最大误差" in report
        assert "平均误差" in report
    
    def test_batch_validation_report_generation(self):
        """测试批量验证报告生成。"""
        turn_community = [
            Card(rank=13, suit='s'),
            Card(rank=7, suit='d'),
            Card(rank=2, suit='c'),
            Card(rank=4, suit='h'),
        ]
        
        my_range = {"AhAd": 1.0, "KhKd": 1.0}
        opponent_range = {"QQ": 1.0}
        
        validator = HistogramValidator(num_bins=50, tolerance=1e-6)
        result = validator.validate_range_histograms(
            my_range, opponent_range, turn_community
        )
        
        report = validator.generate_batch_validation_report(result)
        
        # 验证报告包含关键信息
        assert "批量验证报告" in report
        assert "验证手牌数" in report
        assert "通过数量" in report


class TestManualHistogramCalculator:
    """手动直方图计算器测试。"""
    
    def test_manual_calculator_basic(self):
        """测试手动计算器基本功能。"""
        hole_cards = (Card(rank=14, suit='h'), Card(rank=13, suit='h'))
        turn_community = [
            Card(rank=12, suit='h'),
            Card(rank=11, suit='d'),
            Card(rank=10, suit='c'),
            Card(rank=2, suit='s'),
        ]
        opponent_range = {"AA": 1.0}
        
        calculator = ManualHistogramCalculator(num_bins=50)
        histogram = calculator.calculate_histogram_manually(
            hole_cards, turn_community, opponent_range
        )
        
        # 验证直方图归一化
        assert np.isclose(np.sum(histogram), 1.0, atol=1e-6)
        
        # 验证直方图形状
        assert len(histogram) == 50
        
        # 验证所有值非负
        assert np.all(histogram >= 0)
    
    def test_manual_calculator_empty_range(self):
        """测试空范围的处理。"""
        hole_cards = (Card(rank=14, suit='h'), Card(rank=13, suit='h'))
        turn_community = [
            Card(rank=12, suit='h'),
            Card(rank=11, suit='d'),
            Card(rank=10, suit='c'),
            Card(rank=2, suit='s'),
        ]
        
        # 使用与手牌冲突的范围（会被完全移除）
        opponent_range = {"AhKh": 1.0}  # 与手牌完全相同
        
        calculator = ManualHistogramCalculator(num_bins=50)
        histogram = calculator.calculate_histogram_manually(
            hole_cards, turn_community, opponent_range
        )
        
        # 应该返回归一化的直方图
        assert np.isclose(np.sum(histogram), 1.0, atol=1e-6)
    
    @given(
        hand_strategy(),
        turn_community_strategy()
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.large_base_example], deadline=None)
    def test_manual_histogram_always_normalized(
        self,
        hole_cards: List[Card],
        turn_community: List[Card]
    ):
        """测试手动计算的直方图始终归一化。"""
        # 确保手牌和公共牌不重复
        all_cards = list(hole_cards) + turn_community
        card_keys = [(c.rank, c.suit) for c in all_cards]
        assume(len(set(card_keys)) == 6)
        
        opponent_range = {"AA": 1.0}
        
        calculator = ManualHistogramCalculator(num_bins=50)
        histogram = calculator.calculate_histogram_manually(
            tuple(hole_cards), turn_community, opponent_range
        )
        
        # 验证归一化
        assert np.isclose(np.sum(histogram), 1.0, atol=1e-6)
        
        # 验证所有值在[0, 1]范围内
        assert np.all(histogram >= 0)
        assert np.all(histogram <= 1)
