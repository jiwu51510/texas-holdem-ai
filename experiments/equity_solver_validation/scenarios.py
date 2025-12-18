"""预定义实验场景。

本模块定义了各种实验场景，用于验证胜率方法与Solver方法的差异。
"""

from typing import List
from models.core import Card
from experiments.equity_solver_validation.data_models import (
    SolverConfig,
    ExperimentScenario,
)


def create_default_solver_config() -> SolverConfig:
    """创建默认Solver配置。"""
    return SolverConfig(
        pot_size=100.0,
        effective_stack=200.0,
        oop_bet_sizes=[0.5, 1.0],
        ip_bet_sizes=[0.5, 1.0],
        oop_raise_sizes=[0.5, 1.0],
        ip_raise_sizes=[0.5, 1.0],
        max_iterations=500,
    )


def create_dry_board_scenario() -> ExperimentScenario:
    """创建干燥牌面场景。
    
    干燥牌面特点：连接性低，抽牌少。
    """
    return ExperimentScenario(
        name="干燥牌面_K72r",
        description="K♠7♦2♣4♥9♠ - 典型的干燥牌面，连接性低",
        community_cards=[
            Card(rank=13, suit='s'),  # K♠
            Card(rank=7, suit='d'),   # 7♦
            Card(rank=2, suit='c'),   # 2♣
            Card(rank=4, suit='h'),   # 4♥
            Card(rank=9, suit='s'),   # 9♠
        ],
        oop_range={
            'AsAd': 1.0, 'AhAc': 1.0,
            'KsKd': 1.0, 'KhKc': 1.0,
            'QsQd': 1.0, 'QhQc': 1.0,
            'JsJd': 1.0, 'JhJc': 1.0,
            'AhKh': 1.0, 'AdKd': 1.0,
            'AhQh': 1.0, 'AdQd': 1.0,
        },
        ip_range={
            'AsAd': 1.0, 'AhAc': 1.0,
            'KsKd': 1.0, 'KhKc': 1.0,
            'QsQd': 1.0, 'QhQc': 1.0,
            'JsJd': 1.0, 'JhJc': 1.0,
            'TsTd': 1.0, 'ThTc': 1.0,
            'AhKh': 1.0, 'AdKd': 1.0,
        },
        solver_config=create_default_solver_config(),
        tags=['dry_board', 'rainbow'],
    )


def create_wet_board_scenario() -> ExperimentScenario:
    """创建湿润牌面场景。
    
    湿润牌面特点：连接性高，多种成牌可能。
    """
    return ExperimentScenario(
        name="湿润牌面_JT9",
        description="J♠T♠9♦8♣7♥ - 连接性高的湿润牌面",
        community_cards=[
            Card(rank=11, suit='s'),  # J♠
            Card(rank=10, suit='s'),  # T♠
            Card(rank=9, suit='d'),   # 9♦
            Card(rank=8, suit='c'),   # 8♣
            Card(rank=7, suit='h'),   # 7♥
        ],
        oop_range={
            'AsAd': 1.0, 'AhAc': 1.0,
            'KsKd': 1.0, 'KhKc': 1.0,
            'QsQd': 1.0, 'QhQc': 1.0,
            'AhKh': 1.0, 'AdKd': 1.0,
            'AcKc': 1.0,
        },
        ip_range={
            'AsAd': 1.0, 'AhAc': 1.0,
            'KsKd': 1.0, 'KhKc': 1.0,
            'QsQd': 1.0, 'QhQc': 1.0,
            'AhKh': 1.0, 'AdKd': 1.0,
            'AcKc': 1.0,
        },
        solver_config=create_default_solver_config(),
        tags=['wet_board', 'connected'],
    )


def create_paired_board_scenario() -> ExperimentScenario:
    """创建配对牌面场景。
    
    配对牌面特点：葫芦和四条可能。
    """
    return ExperimentScenario(
        name="配对牌面_KK7",
        description="K♠K♦7♣3♥2♠ - 配对牌面，葫芦可能",
        community_cards=[
            Card(rank=13, suit='s'),  # K♠
            Card(rank=13, suit='d'),  # K♦
            Card(rank=7, suit='c'),   # 7♣
            Card(rank=3, suit='h'),   # 3♥
            Card(rank=2, suit='s'),   # 2♠
        ],
        oop_range={
            'AsAd': 1.0, 'AhAc': 1.0,
            'QsQd': 1.0, 'QhQc': 1.0,
            'JsJd': 1.0, 'JhJc': 1.0,
            'AhKh': 1.0, 'AcKc': 1.0,
            'AhQh': 1.0, 'AdQd': 1.0,
        },
        ip_range={
            'AsAd': 1.0, 'AhAc': 1.0,
            'QsQd': 1.0, 'QhQc': 1.0,
            'JsJd': 1.0, 'JhJc': 1.0,
            'TsTd': 1.0, 'ThTc': 1.0,
            'AhKh': 1.0, 'AcKc': 1.0,
        },
        solver_config=create_default_solver_config(),
        tags=['paired_board'],
    )


def create_flush_board_scenario() -> ExperimentScenario:
    """创建同花牌面场景。
    
    同花牌面特点：同花已成。
    """
    return ExperimentScenario(
        name="同花牌面_4flush",
        description="A♠K♠7♠3♠2♦ - 四张同花牌面",
        community_cards=[
            Card(rank=14, suit='s'),  # A♠
            Card(rank=13, suit='s'),  # K♠
            Card(rank=7, suit='s'),   # 7♠
            Card(rank=3, suit='s'),   # 3♠
            Card(rank=2, suit='d'),   # 2♦
        ],
        oop_range={
            'AdAh': 1.0, 'AdAc': 1.0,
            'KdKh': 1.0, 'KdKc': 1.0,
            'QdQh': 1.0, 'QdQc': 1.0,
            'JdJh': 1.0, 'JdJc': 1.0,
            'QsJs': 1.0,  # 同花
            'TsJs': 1.0,  # 同花
        },
        ip_range={
            'AdAh': 1.0, 'AdAc': 1.0,
            'KdKh': 1.0, 'KdKc': 1.0,
            'QdQh': 1.0, 'QdQc': 1.0,
            'JdJh': 1.0, 'JdJc': 1.0,
            'QsTs': 1.0,  # 同花
            '9s8s': 1.0,  # 同花
        },
        solver_config=create_default_solver_config(),
        tags=['flush_board', 'monotone'],
    )


def create_polarized_vs_condensed_scenario() -> ExperimentScenario:
    """创建极化范围vs凝聚范围场景。"""
    return ExperimentScenario(
        name="极化vs凝聚",
        description="测试极化范围对抗凝聚范围",
        community_cards=[
            Card(rank=14, suit='h'),  # A♥
            Card(rank=8, suit='d'),   # 8♦
            Card(rank=3, suit='c'),   # 3♣
            Card(rank=5, suit='s'),   # 5♠
            Card(rank=9, suit='h'),   # 9♥
        ],
        oop_range={
            # 极化范围：坚果牌和空气牌
            'AsAd': 1.0, 'AcAd': 1.0,  # 坚果
            '7h6h': 1.0, '6h5h': 1.0,  # 空气
            '4h3h': 1.0, '2h3h': 1.0,
        },
        ip_range={
            # 凝聚范围：中等强度牌
            'KsKd': 1.0, 'KhKc': 1.0,
            'QsQd': 1.0, 'QhQc': 1.0,
            'JsJd': 1.0, 'JhJc': 1.0,
            'TsTd': 1.0, 'ThTc': 1.0,
        },
        solver_config=create_default_solver_config(),
        tags=['polarized', 'condensed'],
    )


def get_all_scenarios() -> List[ExperimentScenario]:
    """获取所有预定义场景。"""
    return [
        create_dry_board_scenario(),
        create_wet_board_scenario(),
        create_paired_board_scenario(),
        create_flush_board_scenario(),
        create_polarized_vs_condensed_scenario(),
    ]


def get_scenarios_by_tag(tag: str) -> List[ExperimentScenario]:
    """根据标签获取场景。"""
    return [s for s in get_all_scenarios() if tag in s.tags]
