"""转牌阶段Potential直方图验证实验的预定义场景。

本模块定义了各种转牌实验场景，用于验证Potential直方图方法与Solver方法的差异。
场景类型包括：
- 干燥牌面（Dry Board）：连接性低，抽牌少
- 湿润牌面（Wet Board）：连接性高，多种成牌可能
- 听牌牌面（Draw Board）：同花或顺子听牌可能
- 配对牌面（Paired Board）：葫芦可能

Requirements: 6.1
"""

from typing import List, Dict

from models.core import Card
from experiments.equity_solver_validation.data_models import SolverConfig
from experiments.turn_potential_validation.data_models import TurnScenario


# ============================================================================
# 标准范围配置
# ============================================================================

def get_standard_wide_range() -> Dict[str, float]:
    """获取标准宽范围配置。
    
    包含高对、中对、高牌组合，适用于大多数场景。
    复用河牌验证的范围配置，并根据转牌特点进行调整。
    """
    return {
        # 高对
        'AsAd': 1.0, 'AhAc': 1.0, 'AsAh': 1.0, 'AdAc': 1.0,
        'KsKd': 1.0, 'KhKc': 1.0, 'KsKh': 1.0, 'KdKc': 1.0,
        'QsQd': 1.0, 'QhQc': 1.0, 'QsQh': 1.0, 'QdQc': 1.0,
        'JsJd': 1.0, 'JhJc': 1.0, 'JsJh': 1.0, 'JdJc': 1.0,
        'TsTd': 1.0, 'ThTc': 1.0, 'TsTh': 1.0, 'TdTc': 1.0,
        # 中对
        '9s9d': 1.0, '9h9c': 1.0,
        '8s8d': 1.0, '8h8c': 1.0,
        '7s7d': 1.0, '7h7c': 1.0,
        # 高牌组合
        'AhKh': 1.0, 'AdKd': 1.0, 'AsKs': 1.0, 'AcKc': 1.0,
        'AhQh': 1.0, 'AdQd': 1.0, 'AsQs': 1.0, 'AcQc': 1.0,
        'AhJh': 1.0, 'AdJd': 1.0,
        'KhQh': 1.0, 'KdQd': 1.0, 'KsQs': 1.0, 'KcQc': 1.0,
    }


def get_standard_tight_range() -> Dict[str, float]:
    """获取标准紧范围配置。
    
    仅包含高对和顶级高牌组合，适用于3bet场景。
    """
    return {
        # 高对
        'AsAd': 1.0, 'AhAc': 1.0, 'AsAh': 1.0, 'AdAc': 1.0,
        'KsKd': 1.0, 'KhKc': 1.0, 'KsKh': 1.0, 'KdKc': 1.0,
        'QsQd': 1.0, 'QhQc': 1.0, 'QsQh': 1.0, 'QdQc': 1.0,
        'JsJd': 1.0, 'JhJc': 1.0,
        # 顶级高牌
        'AhKh': 1.0, 'AdKd': 1.0, 'AsKs': 1.0, 'AcKc': 1.0,
        'AhKd': 1.0, 'AdKh': 1.0, 'AsKc': 1.0, 'AcKs': 1.0,
    }


def get_polarized_range() -> Dict[str, float]:
    """获取极化范围配置。
    
    包含坚果牌和空气牌，用于测试极化范围场景。
    """
    return {
        # 坚果牌
        'AsAd': 1.0, 'AhAc': 1.0,
        'KsKd': 1.0, 'KhKc': 1.0,
        # 空气牌（低牌组合）
        '7h6h': 1.0, '6h5h': 1.0, '5h4h': 1.0,
        '7d6d': 1.0, '6d5d': 1.0, '5d4d': 1.0,
        '4h3h': 1.0, '3h2h': 1.0,
    }


def get_condensed_range() -> Dict[str, float]:
    """获取凝聚范围配置。
    
    包含中等强度牌，用于测试凝聚范围场景。
    """
    return {
        # 中等对子
        'TsTd': 1.0, 'ThTc': 1.0, 'TsTh': 1.0, 'TdTc': 1.0,
        '9s9d': 1.0, '9h9c': 1.0, '9s9h': 1.0, '9d9c': 1.0,
        '8s8d': 1.0, '8h8c': 1.0, '8s8h': 1.0, '8d8c': 1.0,
        '7s7d': 1.0, '7h7c': 1.0,
        '6s6d': 1.0, '6h6c': 1.0,
    }


# ============================================================================
# 转牌特定范围配置
# ============================================================================

def get_flush_draw_range() -> Dict[str, float]:
    """获取同花听牌范围配置。
    
    包含各种同花听牌组合，用于测试听牌牌面场景。
    主要包含黑桃和红桃的同花组合。
    """
    return {
        # 黑桃同花组合
        'AsKs': 1.0, 'AsQs': 1.0, 'AsJs': 1.0, 'AsTs': 1.0,
        'KsQs': 1.0, 'KsJs': 1.0, 'KsTs': 1.0,
        'QsJs': 1.0, 'QsTs': 1.0, 'Qs9s': 1.0,
        'JsTs': 1.0, 'Js9s': 1.0, 'Js8s': 1.0,
        'Ts9s': 1.0, 'Ts8s': 1.0,
        # 红桃同花组合
        'AhKh': 1.0, 'AhQh': 1.0, 'AhJh': 1.0, 'AhTh': 1.0,
        'KhQh': 1.0, 'KhJh': 1.0, 'KhTh': 1.0,
        'QhJh': 1.0, 'QhTh': 1.0, 'Qh9h': 1.0,
        'JhTh': 1.0, 'Jh9h': 1.0, 'Jh8h': 1.0,
    }


def get_straight_draw_range() -> Dict[str, float]:
    """获取顺子听牌范围配置。
    
    包含各种顺子听牌组合，用于测试连接牌面场景。
    """
    return {
        # 高牌连接组合
        'AhKd': 1.0, 'AdKh': 1.0, 'AsKc': 1.0, 'AcKs': 1.0,
        'KhQd': 1.0, 'KdQh': 1.0, 'KsQc': 1.0, 'KcQs': 1.0,
        'QhJd': 1.0, 'QdJh': 1.0, 'QsJc': 1.0, 'QcJs': 1.0,
        'JhTd': 1.0, 'JdTh': 1.0, 'JsTc': 1.0, 'JcTs': 1.0,
        # 中等连接组合
        'Th9d': 1.0, 'Td9h': 1.0, 'Ts9c': 1.0, 'Tc9s': 1.0,
        '9h8d': 1.0, '9d8h': 1.0, '9s8c': 1.0, '9c8s': 1.0,
        '8h7d': 1.0, '8d7h': 1.0, '8s7c': 1.0, '8c7s': 1.0,
        '7h6d': 1.0, '7d6h': 1.0, '7s6c': 1.0, '7c6s': 1.0,
    }


def get_made_hand_range() -> Dict[str, float]:
    """获取成牌范围配置。
    
    包含各种已成牌的组合，用于测试成牌vs听牌场景。
    """
    return {
        # 高对
        'AsAd': 1.0, 'AhAc': 1.0,
        'KsKd': 1.0, 'KhKc': 1.0,
        'QsQd': 1.0, 'QhQc': 1.0,
        # 中对
        'JsJd': 1.0, 'JhJc': 1.0,
        'TsTd': 1.0, 'ThTc': 1.0,
        '9s9d': 1.0, '9h9c': 1.0,
        # 低对
        '8s8d': 1.0, '8h8c': 1.0,
        '7s7d': 1.0, '7h7c': 1.0,
        '6s6d': 1.0, '6h6c': 1.0,
    }


def get_mixed_range() -> Dict[str, float]:
    """获取混合范围配置。
    
    包含成牌和听牌的混合，用于测试复杂场景。
    """
    return {
        # 高对（成牌）
        'AsAd': 1.0, 'AhAc': 1.0,
        'KsKd': 1.0, 'KhKc': 1.0,
        # 同花听牌
        'AsKs': 1.0, 'AhKh': 1.0, 'AdKd': 1.0, 'AcKc': 1.0,
        'AsQs': 1.0, 'AhQh': 1.0, 'AdQd': 1.0, 'AcQc': 1.0,
        # 顺子听牌
        'JhTd': 1.0, 'JdTh': 1.0,
        'Th9d': 1.0, 'Td9h': 1.0,
        # 中对
        '9s9d': 1.0, '9h9c': 1.0,
        '8s8d': 1.0, '8h8c': 1.0,
    }


# ============================================================================
# Solver配置
# ============================================================================

def create_default_turn_solver_config() -> SolverConfig:
    """创建默认转牌Solver配置。"""
    return SolverConfig(
        pot_size=100.0,
        effective_stack=200.0,
        oop_bet_sizes=[0.5, 1.0],
        ip_bet_sizes=[0.5, 1.0],
        oop_raise_sizes=[0.5, 1.0],
        ip_raise_sizes=[0.5, 1.0],
        max_iterations=500,
    )


def create_deep_stack_solver_config() -> SolverConfig:
    """创建深筹码Solver配置。"""
    return SolverConfig(
        pot_size=100.0,
        effective_stack=400.0,
        oop_bet_sizes=[0.33, 0.67, 1.0],
        ip_bet_sizes=[0.33, 0.67, 1.0],
        oop_raise_sizes=[0.5, 1.0, 2.0],
        ip_raise_sizes=[0.5, 1.0, 2.0],
        max_iterations=500,
    )


# ============================================================================
# 干燥牌面场景（Dry Board）
# ============================================================================

def create_dry_board_scenario_k72() -> TurnScenario:
    """创建干燥牌面场景：K♠7♦2♣4♥
    
    特点：
    - 连接性低，几乎没有顺子听牌
    - 彩虹牌面，没有同花听牌
    - 高牌K占主导
    """
    return TurnScenario(
        name="干燥牌面_K724",
        description="K♠7♦2♣4♥ - 典型的干燥牌面，连接性低，彩虹",
        turn_community=[
            Card(rank=13, suit='s'),  # K♠
            Card(rank=7, suit='d'),   # 7♦
            Card(rank=2, suit='c'),   # 2♣
            Card(rank=4, suit='h'),   # 4♥
        ],
        oop_range=get_standard_wide_range(),
        ip_range=get_standard_wide_range(),
        solver_config=create_default_turn_solver_config(),
        tags=['dry_board', 'rainbow', 'high_card'],
    )


def create_dry_board_scenario_a83() -> TurnScenario:
    """创建干燥牌面场景：A♠8♦3♣5♥
    
    特点：
    - A高牌面
    - 连接性低
    - 彩虹牌面
    """
    return TurnScenario(
        name="干燥牌面_A835",
        description="A♠8♦3♣5♥ - A高干燥牌面，彩虹",
        turn_community=[
            Card(rank=14, suit='s'),  # A♠
            Card(rank=8, suit='d'),   # 8♦
            Card(rank=3, suit='c'),   # 3♣
            Card(rank=5, suit='h'),   # 5♥
        ],
        oop_range=get_standard_wide_range(),
        ip_range=get_standard_wide_range(),
        solver_config=create_default_turn_solver_config(),
        tags=['dry_board', 'rainbow', 'ace_high'],
    )


# ============================================================================
# 湿润牌面场景（Wet Board）
# ============================================================================

def create_wet_board_scenario_jt98() -> TurnScenario:
    """创建湿润牌面场景：J♠T♠9♦8♣
    
    特点：
    - 高度连接，多种顺子可能
    - 两张同花，有同花听牌
    - 中等牌面
    """
    return TurnScenario(
        name="湿润牌面_JT98",
        description="J♠T♠9♦8♣ - 高度连接的湿润牌面，顺子和同花听牌",
        turn_community=[
            Card(rank=11, suit='s'),  # J♠
            Card(rank=10, suit='s'),  # T♠
            Card(rank=9, suit='d'),   # 9♦
            Card(rank=8, suit='c'),   # 8♣
        ],
        oop_range=get_standard_wide_range(),
        ip_range=get_standard_wide_range(),
        solver_config=create_default_turn_solver_config(),
        tags=['wet_board', 'connected', 'two_tone'],
    )


def create_wet_board_scenario_qjt9() -> TurnScenario:
    """创建湿润牌面场景：Q♥J♦T♣9♠
    
    特点：
    - 四张连牌，顺子已成或听牌
    - 彩虹牌面
    - 高牌面
    """
    return TurnScenario(
        name="湿润牌面_QJT9",
        description="Q♥J♦T♣9♠ - 四张连牌，顺子牌面",
        turn_community=[
            Card(rank=12, suit='h'),  # Q♥
            Card(rank=11, suit='d'),  # J♦
            Card(rank=10, suit='c'),  # T♣
            Card(rank=9, suit='s'),   # 9♠
        ],
        oop_range=get_standard_wide_range(),
        ip_range=get_standard_wide_range(),
        solver_config=create_default_turn_solver_config(),
        tags=['wet_board', 'connected', 'straight_board', 'rainbow'],
    )


# ============================================================================
# 听牌牌面场景（Draw Board）
# ============================================================================

def create_flush_draw_scenario_aks7() -> TurnScenario:
    """创建同花听牌场景：A♠K♠7♠3♦
    
    特点：
    - 三张同花，强同花听牌
    - A高牌面
    - 连接性低
    """
    return TurnScenario(
        name="同花听牌_AKs73",
        description="A♠K♠7♠3♦ - 三张黑桃，强同花听牌",
        turn_community=[
            Card(rank=14, suit='s'),  # A♠
            Card(rank=13, suit='s'),  # K♠
            Card(rank=7, suit='s'),   # 7♠
            Card(rank=3, suit='d'),   # 3♦
        ],
        oop_range=get_standard_wide_range(),
        ip_range=get_standard_wide_range(),
        solver_config=create_default_turn_solver_config(),
        tags=['draw_board', 'flush_draw', 'three_flush', 'ace_high'],
    )


def create_straight_draw_scenario_t987() -> TurnScenario:
    """创建顺子听牌场景：T♥9♦8♣7♠
    
    特点：
    - 四张连牌，开放式顺子听牌
    - 彩虹牌面
    - 中等牌面
    """
    return TurnScenario(
        name="顺子听牌_T987",
        description="T♥9♦8♣7♠ - 四张连牌，开放式顺子听牌",
        turn_community=[
            Card(rank=10, suit='h'),  # T♥
            Card(rank=9, suit='d'),   # 9♦
            Card(rank=8, suit='c'),   # 8♣
            Card(rank=7, suit='s'),   # 7♠
        ],
        oop_range=get_standard_wide_range(),
        ip_range=get_standard_wide_range(),
        solver_config=create_default_turn_solver_config(),
        tags=['draw_board', 'straight_draw', 'connected', 'rainbow'],
    )


def create_combo_draw_scenario_jt9h() -> TurnScenario:
    """创建组合听牌场景：J♥T♥9♥3♦
    
    特点：
    - 三张同花+连牌
    - 同花听牌+顺子听牌
    - 高度动态牌面
    """
    return TurnScenario(
        name="组合听牌_JT9h3",
        description="J♥T♥9♥3♦ - 同花听牌+顺子听牌的组合",
        turn_community=[
            Card(rank=11, suit='h'),  # J♥
            Card(rank=10, suit='h'),  # T♥
            Card(rank=9, suit='h'),   # 9♥
            Card(rank=3, suit='d'),   # 3♦
        ],
        oop_range=get_standard_wide_range(),
        ip_range=get_standard_wide_range(),
        solver_config=create_default_turn_solver_config(),
        tags=['draw_board', 'flush_draw', 'straight_draw', 'combo_draw'],
    )


# ============================================================================
# 配对牌面场景（Paired Board）
# ============================================================================

def create_paired_board_scenario_kk73() -> TurnScenario:
    """创建配对牌面场景：K♠K♦7♣3♥
    
    特点：
    - 顶对配对，葫芦可能
    - 连接性低
    - 彩虹牌面
    """
    return TurnScenario(
        name="配对牌面_KK73",
        description="K♠K♦7♣3♥ - K配对牌面，葫芦可能",
        turn_community=[
            Card(rank=13, suit='s'),  # K♠
            Card(rank=13, suit='d'),  # K♦
            Card(rank=7, suit='c'),   # 7♣
            Card(rank=3, suit='h'),   # 3♥
        ],
        oop_range=get_standard_wide_range(),
        ip_range=get_standard_wide_range(),
        solver_config=create_default_turn_solver_config(),
        tags=['paired_board', 'high_pair', 'rainbow'],
    )


def create_paired_board_scenario_7732() -> TurnScenario:
    """创建配对牌面场景：7♠7♦3♣2♥
    
    特点：
    - 低对配对
    - 连接性低
    - 彩虹牌面
    """
    return TurnScenario(
        name="配对牌面_7732",
        description="7♠7♦3♣2♥ - 低对配对牌面",
        turn_community=[
            Card(rank=7, suit='s'),   # 7♠
            Card(rank=7, suit='d'),   # 7♦
            Card(rank=3, suit='c'),   # 3♣
            Card(rank=2, suit='h'),   # 2♥
        ],
        oop_range=get_standard_wide_range(),
        ip_range=get_standard_wide_range(),
        solver_config=create_default_turn_solver_config(),
        tags=['paired_board', 'low_pair', 'rainbow'],
    )


def create_paired_board_scenario_aa52() -> TurnScenario:
    """创建配对牌面场景：A♠A♦5♣2♥
    
    特点：
    - A配对，最高配对牌面
    - 葫芦和四条可能
    - 彩虹牌面
    """
    return TurnScenario(
        name="配对牌面_AA52",
        description="A♠A♦5♣2♥ - A配对牌面，葫芦可能",
        turn_community=[
            Card(rank=14, suit='s'),  # A♠
            Card(rank=14, suit='d'),  # A♦
            Card(rank=5, suit='c'),   # 5♣
            Card(rank=2, suit='h'),   # 2♥
        ],
        oop_range=get_standard_tight_range(),
        ip_range=get_standard_tight_range(),
        solver_config=create_default_turn_solver_config(),
        tags=['paired_board', 'ace_pair', 'rainbow'],
    )


# ============================================================================
# 特殊场景
# ============================================================================

def create_polarized_vs_condensed_scenario() -> TurnScenario:
    """创建极化范围vs凝聚范围场景。
    
    测试极化范围对抗凝聚范围时Potential直方图的表现。
    """
    return TurnScenario(
        name="极化vs凝聚",
        description="A♥8♦3♣5♠ - 测试极化范围对抗凝聚范围",
        turn_community=[
            Card(rank=14, suit='h'),  # A♥
            Card(rank=8, suit='d'),   # 8♦
            Card(rank=3, suit='c'),   # 3♣
            Card(rank=5, suit='s'),   # 5♠
        ],
        oop_range=get_polarized_range(),
        ip_range=get_condensed_range(),
        solver_config=create_default_turn_solver_config(),
        tags=['polarized', 'condensed', 'special'],
    )


def create_deep_stack_scenario() -> TurnScenario:
    """创建深筹码场景。
    
    测试深筹码情况下Potential直方图的表现。
    """
    return TurnScenario(
        name="深筹码场景",
        description="K♠Q♦J♣9♥ - 深筹码连接牌面",
        turn_community=[
            Card(rank=13, suit='s'),  # K♠
            Card(rank=12, suit='d'),  # Q♦
            Card(rank=11, suit='c'),  # J♣
            Card(rank=9, suit='h'),   # 9♥
        ],
        oop_range=get_standard_wide_range(),
        ip_range=get_standard_wide_range(),
        solver_config=create_deep_stack_solver_config(),
        tags=['deep_stack', 'connected', 'special'],
    )


def create_monotone_board_scenario() -> TurnScenario:
    """创建单色牌面场景。
    
    四张同花牌面，同花已成。
    """
    return TurnScenario(
        name="单色牌面_4hearts",
        description="A♥K♥7♥3♥ - 四张红桃，同花已成",
        turn_community=[
            Card(rank=14, suit='h'),  # A♥
            Card(rank=13, suit='h'),  # K♥
            Card(rank=7, suit='h'),   # 7♥
            Card(rank=3, suit='h'),   # 3♥
        ],
        oop_range={
            # 有红桃的手牌
            'QhJh': 1.0, 'QhTh': 1.0, 'Jh9h': 1.0,
            'Th9h': 1.0, '9h8h': 1.0, '8h6h': 1.0,
            # 无红桃的高对
            'AsAd': 1.0, 'AsAc': 1.0,
            'KsKd': 1.0, 'KsKc': 1.0,
            'QsQd': 1.0, 'QsQc': 1.0,
        },
        ip_range={
            # 有红桃的手牌
            'QhJh': 1.0, 'QhTh': 1.0, 'Jh9h': 1.0,
            'Th8h': 1.0, '9h8h': 1.0, '6h5h': 1.0,
            # 无红桃的高对
            'AsAd': 1.0, 'AsAc': 1.0,
            'KsKd': 1.0, 'KsKc': 1.0,
            'JsJd': 1.0, 'JsJc': 1.0,
        },
        solver_config=create_default_turn_solver_config(),
        tags=['monotone', 'flush_complete', 'special'],
    )


# ============================================================================
# 场景获取函数
# ============================================================================

def get_all_turn_scenarios() -> List[TurnScenario]:
    """获取所有预定义的转牌场景。
    
    Returns:
        所有转牌场景的列表
    """
    return [
        # 干燥牌面
        create_dry_board_scenario_k72(),
        create_dry_board_scenario_a83(),
        # 湿润牌面
        create_wet_board_scenario_jt98(),
        create_wet_board_scenario_qjt9(),
        # 听牌牌面
        create_flush_draw_scenario_aks7(),
        create_straight_draw_scenario_t987(),
        create_combo_draw_scenario_jt9h(),
        # 配对牌面
        create_paired_board_scenario_kk73(),
        create_paired_board_scenario_7732(),
        create_paired_board_scenario_aa52(),
        # 特殊场景
        create_polarized_vs_condensed_scenario(),
        create_deep_stack_scenario(),
        create_monotone_board_scenario(),
    ]


def get_turn_scenarios_by_tag(tag: str) -> List[TurnScenario]:
    """根据标签获取转牌场景。
    
    Args:
        tag: 场景标签（如 "dry_board", "wet_board", "draw_board", "paired_board"）
        
    Returns:
        匹配标签的场景列表
    """
    return [s for s in get_all_turn_scenarios() if tag in s.tags]


def get_dry_board_scenarios() -> List[TurnScenario]:
    """获取所有干燥牌面场景。"""
    return get_turn_scenarios_by_tag('dry_board')


def get_wet_board_scenarios() -> List[TurnScenario]:
    """获取所有湿润牌面场景。"""
    return get_turn_scenarios_by_tag('wet_board')


def get_draw_board_scenarios() -> List[TurnScenario]:
    """获取所有听牌牌面场景。"""
    return get_turn_scenarios_by_tag('draw_board')


def get_paired_board_scenarios() -> List[TurnScenario]:
    """获取所有配对牌面场景。"""
    return get_turn_scenarios_by_tag('paired_board')


def get_special_scenarios() -> List[TurnScenario]:
    """获取所有特殊场景。"""
    return get_turn_scenarios_by_tag('special')


def get_scenario_by_name(name: str) -> TurnScenario:
    """根据名称获取场景。
    
    Args:
        name: 场景名称
        
    Returns:
        匹配名称的场景
        
    Raises:
        ValueError: 如果找不到匹配的场景
    """
    for scenario in get_all_turn_scenarios():
        if scenario.name == name:
            return scenario
    raise ValueError(f"找不到名称为 '{name}' 的场景")


def list_all_scenario_names() -> List[str]:
    """列出所有场景名称。"""
    return [s.name for s in get_all_turn_scenarios()]


def list_all_tags() -> List[str]:
    """列出所有可用的标签。"""
    tags = set()
    for scenario in get_all_turn_scenarios():
        tags.update(scenario.tags)
    return sorted(list(tags))
