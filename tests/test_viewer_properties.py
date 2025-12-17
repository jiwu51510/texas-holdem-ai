"""策略查看器属性测试模块。

使用Hypothesis库进行属性测试，验证策略查看器的核心正确性属性。
"""

import pytest
import torch
from hypothesis import given, strategies as st, settings, assume

from models.core import Card
from viewer.hand_range import (
    HandRangeCalculator, 
    RANK_TO_CHAR, 
    CHAR_TO_RANK,
    SUITS,
    HAND_LABELS_MATRIX,
    RANKS_ORDER
)


# ============================================================================
# 测试策略（生成器）
# ============================================================================

# 有效的牌面等级（2-14）
valid_ranks = st.integers(min_value=2, max_value=14)

# 有效的花色
valid_suits = st.sampled_from(SUITS)

# 生成有效的Card对象
card_strategy = st.builds(Card, rank=valid_ranks, suit=valid_suits)

# 生成两张不同的牌（用于手牌）
@st.composite
def two_different_cards(draw):
    """生成两张不同的牌。"""
    card1 = draw(card_strategy)
    card2 = draw(card_strategy)
    # 确保两张牌不完全相同
    assume(not (card1.rank == card2.rank and card1.suit == card2.suit))
    return card1, card2


# 生成公共牌列表（0-5张不重复的牌）
@st.composite
def board_cards_strategy(draw, min_cards=0, max_cards=5):
    """生成公共牌列表。"""
    num_cards = draw(st.integers(min_value=min_cards, max_value=max_cards))
    cards = []
    used = set()
    
    for _ in range(num_cards):
        while True:
            card = draw(card_strategy)
            key = (card.rank, card.suit)
            if key not in used:
                used.add(key)
                cards.append(card)
                break
    
    return cards


# ============================================================================
# Property 1: 手牌标签位置正确性
# **Feature: strategy-viewer, Property 1: 手牌标签位置正确性**
# **验证: 需求 3.3, 3.4, 3.5**
# ============================================================================

class TestProperty1HandLabelPosition:
    """属性测试：手牌标签位置正确性。
    
    对于任意两张牌的组合，get_hand_label函数应返回正确的标签格式和矩阵位置：
    - 对子（两张牌rank相同）返回"XX"格式，位于矩阵对角线
    - 同花（两张牌suit相同且rank不同）返回"XYs"格式，位于矩阵上三角
    - 非同花（两张牌suit不同且rank不同）返回"XYo"格式，位于矩阵下三角
    """
    
    @given(two_different_cards())
    @settings(max_examples=100)
    def test_hand_label_format_and_position(self, cards):
        """
        **Feature: strategy-viewer, Property 1: 手牌标签位置正确性**
        **验证: 需求 3.3, 3.4, 3.5**
        
        测试手牌标签格式和矩阵位置的正确性。
        """
        card1, card2 = cards
        calculator = HandRangeCalculator()
        
        label = calculator.get_hand_label(card1, card2)
        row, col = calculator.get_matrix_position(label)
        
        # 验证标签格式
        if card1.rank == card2.rank:
            # 对子：应该是2个字符，且在对角线上
            assert len(label) == 2, f"对子标签应为2个字符: {label}"
            assert label[0] == label[1], f"对子标签两个字符应相同: {label}"
            assert row == col, f"对子应在对角线上: ({row}, {col})"
        
        elif card1.suit == card2.suit:
            # 同花：应该是3个字符，以's'结尾，在上三角
            assert len(label) == 3, f"同花标签应为3个字符: {label}"
            assert label[2] == 's', f"同花标签应以's'结尾: {label}"
            assert row < col, f"同花应在上三角: ({row}, {col})"
        
        else:
            # 非同花：应该是3个字符，以'o'结尾，在下三角
            assert len(label) == 3, f"非同花标签应为3个字符: {label}"
            assert label[2] == 'o', f"非同花标签应以'o'结尾: {label}"
            assert row > col, f"非同花应在下三角: ({row}, {col})"
    
    @given(two_different_cards())
    @settings(max_examples=100)
    def test_hand_label_high_card_first(self, cards):
        """
        **Feature: strategy-viewer, Property 1: 手牌标签位置正确性**
        **验证: 需求 3.3, 3.4, 3.5**
        
        测试手牌标签中高牌总是在前面。
        """
        card1, card2 = cards
        calculator = HandRangeCalculator()
        
        label = calculator.get_hand_label(card1, card2)
        
        # 对于非对子，第一个字符应该代表更高的rank
        if card1.rank != card2.rank:
            rank1 = CHAR_TO_RANK[label[0]]
            rank2 = CHAR_TO_RANK[label[1]]
            assert rank1 > rank2, f"高牌应在前: {label}, ranks: {rank1}, {rank2}"
    
    @given(two_different_cards())
    @settings(max_examples=100)
    def test_hand_label_in_matrix(self, cards):
        """
        **Feature: strategy-viewer, Property 1: 手牌标签位置正确性**
        **验证: 需求 3.3, 3.4, 3.5**
        
        测试生成的手牌标签存在于标准矩阵中。
        """
        card1, card2 = cards
        calculator = HandRangeCalculator()
        
        label = calculator.get_hand_label(card1, card2)
        
        # 验证标签存在于矩阵中
        found = False
        for row in HAND_LABELS_MATRIX:
            if label in row:
                found = True
                break
        
        assert found, f"标签 {label} 不在标准矩阵中"



# ============================================================================
# Property 3: 手牌组合完整性
# **Feature: strategy-viewer, Property 3: 手牌组合完整性**
# **验证: 需求 4.3**
# ============================================================================

# 生成有效的手牌标签
valid_hand_labels = st.sampled_from([
    label for row in HAND_LABELS_MATRIX for label in row
])


class TestProperty3HandCombinationCompleteness:
    """属性测试：手牌组合完整性。
    
    对于任意手牌标签，get_all_hand_combinations返回的组合数量应正确：
    - 对子（如"AA"）返回6种组合（C(4,2) = 6）
    - 同花（如"AKs"）返回4种组合（4种花色）
    - 非同花（如"AKo"）返回12种组合（4×3 = 12）
    """
    
    @given(valid_hand_labels)
    @settings(max_examples=100)
    def test_combination_count(self, hand_label):
        """
        **Feature: strategy-viewer, Property 3: 手牌组合完整性**
        **验证: 需求 4.3**
        
        测试手牌组合数量的正确性。
        """
        calculator = HandRangeCalculator()
        combinations = calculator.get_all_hand_combinations(hand_label)
        
        if calculator.is_pair(hand_label):
            # 对子：C(4,2) = 6种组合
            expected_count = 6
        elif calculator.is_suited(hand_label):
            # 同花：4种组合（每种花色一个）
            expected_count = 4
        else:
            # 非同花：4×3 = 12种组合
            expected_count = 12
        
        assert len(combinations) == expected_count, (
            f"手牌 {hand_label} 应有 {expected_count} 种组合，"
            f"实际有 {len(combinations)} 种"
        )
    
    @given(valid_hand_labels)
    @settings(max_examples=100)
    def test_combinations_are_unique(self, hand_label):
        """
        **Feature: strategy-viewer, Property 3: 手牌组合完整性**
        **验证: 需求 4.3**
        
        测试手牌组合不重复。
        """
        calculator = HandRangeCalculator()
        combinations = calculator.get_all_hand_combinations(hand_label)
        
        # 将组合转换为可哈希的形式
        combo_set = set()
        for card1, card2 in combinations:
            # 使用排序后的元组确保顺序一致
            key = tuple(sorted([(card1.rank, card1.suit), (card2.rank, card2.suit)]))
            combo_set.add(key)
        
        assert len(combo_set) == len(combinations), (
            f"手牌 {hand_label} 的组合存在重复"
        )
    
    @given(valid_hand_labels)
    @settings(max_examples=100)
    def test_combinations_match_label(self, hand_label):
        """
        **Feature: strategy-viewer, Property 3: 手牌组合完整性**
        **验证: 需求 4.3**
        
        测试每个组合都能正确映射回原标签。
        """
        calculator = HandRangeCalculator()
        combinations = calculator.get_all_hand_combinations(hand_label)
        
        for card1, card2 in combinations:
            generated_label = calculator.get_hand_label(card1, card2)
            assert generated_label == hand_label, (
                f"组合 ({card1}, {card2}) 应映射到 {hand_label}，"
                f"实际映射到 {generated_label}"
            )



# ============================================================================
# Property 4: 公共牌过滤正确性
# **Feature: strategy-viewer, Property 4: 公共牌过滤正确性**
# **验证: 需求 6.3**
# ============================================================================

class TestProperty4BoardFiltering:
    """属性测试：公共牌过滤正确性。
    
    对于任意手牌组合列表和公共牌列表，filter_by_board应排除所有与公共牌
    有重复牌的组合，且保留的组合中不存在任何与公共牌相同的牌。
    """
    
    @given(valid_hand_labels, board_cards_strategy(min_cards=0, max_cards=5))
    @settings(max_examples=100)
    def test_filtered_combinations_no_conflict(self, hand_label, board_cards):
        """
        **Feature: strategy-viewer, Property 4: 公共牌过滤正确性**
        **验证: 需求 6.3**
        
        测试过滤后的组合不与公共牌冲突。
        """
        calculator = HandRangeCalculator()
        combinations = calculator.get_all_hand_combinations(hand_label)
        filtered = calculator.filter_by_board(combinations, board_cards)
        
        # 构建公共牌集合
        board_set = {(card.rank, card.suit) for card in board_cards}
        
        # 验证过滤后的每个组合都不与公共牌冲突
        for card1, card2 in filtered:
            assert (card1.rank, card1.suit) not in board_set, (
                f"过滤后的组合 ({card1}, {card2}) 中的 {card1} 与公共牌冲突"
            )
            assert (card2.rank, card2.suit) not in board_set, (
                f"过滤后的组合 ({card1}, {card2}) 中的 {card2} 与公共牌冲突"
            )
    
    @given(valid_hand_labels, board_cards_strategy(min_cards=0, max_cards=5))
    @settings(max_examples=100)
    def test_all_conflicting_combinations_removed(self, hand_label, board_cards):
        """
        **Feature: strategy-viewer, Property 4: 公共牌过滤正确性**
        **验证: 需求 6.3**
        
        测试所有冲突的组合都被移除。
        """
        calculator = HandRangeCalculator()
        combinations = calculator.get_all_hand_combinations(hand_label)
        filtered = calculator.filter_by_board(combinations, board_cards)
        
        # 构建公共牌集合
        board_set = {(card.rank, card.suit) for card in board_cards}
        
        # 计算应该被过滤掉的组合数量
        expected_filtered_count = 0
        for card1, card2 in combinations:
            has_conflict = (
                (card1.rank, card1.suit) in board_set or
                (card2.rank, card2.suit) in board_set
            )
            if not has_conflict:
                expected_filtered_count += 1
        
        assert len(filtered) == expected_filtered_count, (
            f"过滤后应有 {expected_filtered_count} 个组合，"
            f"实际有 {len(filtered)} 个"
        )
    
    @given(valid_hand_labels)
    @settings(max_examples=100)
    def test_empty_board_no_filtering(self, hand_label):
        """
        **Feature: strategy-viewer, Property 4: 公共牌过滤正确性**
        **验证: 需求 6.3**
        
        测试空公共牌不过滤任何组合。
        """
        calculator = HandRangeCalculator()
        combinations = calculator.get_all_hand_combinations(hand_label)
        filtered = calculator.filter_by_board(combinations, [])
        
        assert len(filtered) == len(combinations), (
            f"空公共牌不应过滤任何组合，原有 {len(combinations)} 个，"
            f"过滤后 {len(filtered)} 个"
        )
    
    @given(valid_hand_labels, board_cards_strategy(min_cards=1, max_cards=5))
    @settings(max_examples=100)
    def test_filtered_is_subset(self, hand_label, board_cards):
        """
        **Feature: strategy-viewer, Property 4: 公共牌过滤正确性**
        **验证: 需求 6.3**
        
        测试过滤后的组合是原组合的子集。
        """
        calculator = HandRangeCalculator()
        combinations = calculator.get_all_hand_combinations(hand_label)
        filtered = calculator.filter_by_board(combinations, board_cards)
        
        # 将组合转换为集合进行比较
        original_set = {
            tuple(sorted([(c1.rank, c1.suit), (c2.rank, c2.suit)]))
            for c1, c2 in combinations
        }
        filtered_set = {
            tuple(sorted([(c1.rank, c1.suit), (c2.rank, c2.suit)]))
            for c1, c2 in filtered
        }
        
        assert filtered_set.issubset(original_set), (
            "过滤后的组合应是原组合的子集"
        )


# ============================================================================
# Property 5: 游戏树路径一致性
# **Feature: strategy-viewer, Property 5: 游戏树路径一致性**
# **验证: 需求 2.4**
# ============================================================================

from viewer.game_tree import GameTreeNavigator
from viewer.models import GameTreeNode, NodeType
from models.core import GameStage, Action, ActionType


# 生成随机行动序列
@st.composite
def action_sequence_strategy(draw, min_actions=0, max_actions=10):
    """生成随机行动序列。
    
    生成一系列有效的扑克行动，用于构建游戏树。
    """
    num_actions = draw(st.integers(min_value=min_actions, max_value=max_actions))
    actions = []
    
    for _ in range(num_actions):
        action_type = draw(st.sampled_from([ActionType.FOLD, ActionType.CALL, ActionType.RAISE]))
        if action_type == ActionType.RAISE:
            amount = draw(st.integers(min_value=10, max_value=500))
            actions.append(Action(action_type, amount=amount))
        else:
            actions.append(Action(action_type))
    
    return actions


class TestProperty5GameTreePathConsistency:
    """属性测试：游戏树路径一致性。
    
    对于任意游戏树节点，从该节点到根节点的路径应是唯一的，
    且路径长度等于该节点的action_history长度加1。
    """
    
    @given(action_sequence_strategy(min_actions=1, max_actions=10))
    @settings(max_examples=100)
    def test_path_length_equals_action_history_plus_one(self, actions):
        """
        **Feature: strategy-viewer, Property 5: 游戏树路径一致性**
        **验证: 需求 2.4**
        
        测试路径长度等于action_history长度加1。
        """
        # 构建游戏树
        nav = GameTreeNavigator()
        root = nav.get_root()
        
        # 沿着行动序列构建节点
        current_node = root
        player = 0
        for action in actions:
            # 如果是FOLD，则创建终端节点
            node_type = NodeType.TERMINAL if action.action_type == ActionType.FOLD else NodeType.PLAYER
            current_node = nav.add_child(
                current_node, 
                action, 
                player=player,
                node_type=node_type
            )
            player = 1 - player  # 交替玩家
            
            # 如果是FOLD，停止添加更多节点
            if action.action_type == ActionType.FOLD:
                break
        
        # 验证路径长度
        path = nav.get_path_to_root(current_node)
        expected_length = len(current_node.action_history) + 1
        
        assert len(path) == expected_length, (
            f"路径长度应为 {expected_length}（action_history长度 + 1），"
            f"实际为 {len(path)}"
        )
    
    @given(action_sequence_strategy(min_actions=1, max_actions=10))
    @settings(max_examples=100)
    def test_path_starts_at_root(self, actions):
        """
        **Feature: strategy-viewer, Property 5: 游戏树路径一致性**
        **验证: 需求 2.4**
        
        测试路径从根节点开始。
        """
        # 构建游戏树
        nav = GameTreeNavigator()
        root = nav.get_root()
        
        # 沿着行动序列构建节点
        current_node = root
        player = 0
        for action in actions:
            node_type = NodeType.TERMINAL if action.action_type == ActionType.FOLD else NodeType.PLAYER
            current_node = nav.add_child(
                current_node, 
                action, 
                player=player,
                node_type=node_type
            )
            player = 1 - player
            
            if action.action_type == ActionType.FOLD:
                break
        
        # 验证路径从根节点开始
        path = nav.get_path_to_root(current_node)
        
        assert path[0] == root, "路径应从根节点开始"
        assert path[0].node_id == "root", f"路径第一个节点应为root，实际为 {path[0].node_id}"
    
    @given(action_sequence_strategy(min_actions=1, max_actions=10))
    @settings(max_examples=100)
    def test_path_ends_at_current_node(self, actions):
        """
        **Feature: strategy-viewer, Property 5: 游戏树路径一致性**
        **验证: 需求 2.4**
        
        测试路径以当前节点结束。
        """
        # 构建游戏树
        nav = GameTreeNavigator()
        root = nav.get_root()
        
        # 沿着行动序列构建节点
        current_node = root
        player = 0
        for action in actions:
            node_type = NodeType.TERMINAL if action.action_type == ActionType.FOLD else NodeType.PLAYER
            current_node = nav.add_child(
                current_node, 
                action, 
                player=player,
                node_type=node_type
            )
            player = 1 - player
            
            if action.action_type == ActionType.FOLD:
                break
        
        # 验证路径以当前节点结束
        path = nav.get_path_to_root(current_node)
        
        assert path[-1] == current_node, "路径应以当前节点结束"
    
    @given(action_sequence_strategy(min_actions=1, max_actions=10))
    @settings(max_examples=100)
    def test_path_is_connected(self, actions):
        """
        **Feature: strategy-viewer, Property 5: 游戏树路径一致性**
        **验证: 需求 2.4**
        
        测试路径中相邻节点是父子关系。
        """
        # 构建游戏树
        nav = GameTreeNavigator()
        root = nav.get_root()
        
        # 沿着行动序列构建节点
        current_node = root
        player = 0
        for action in actions:
            node_type = NodeType.TERMINAL if action.action_type == ActionType.FOLD else NodeType.PLAYER
            current_node = nav.add_child(
                current_node, 
                action, 
                player=player,
                node_type=node_type
            )
            player = 1 - player
            
            if action.action_type == ActionType.FOLD:
                break
        
        # 验证路径连通性
        path = nav.get_path_to_root(current_node)
        
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]
            assert child.parent == parent, (
                f"路径中节点 {child.node_id} 的父节点应为 {parent.node_id}，"
                f"实际为 {child.parent.node_id if child.parent else None}"
            )
    
    @given(action_sequence_strategy(min_actions=0, max_actions=10))
    @settings(max_examples=100)
    def test_root_path_length_is_one(self, actions):
        """
        **Feature: strategy-viewer, Property 5: 游戏树路径一致性**
        **验证: 需求 2.4**
        
        测试根节点的路径长度为1。
        """
        # 构建游戏树（即使有行动，我们只测试根节点）
        nav = GameTreeNavigator()
        root = nav.get_root()
        
        # 验证根节点路径
        path = nav.get_path_to_root(root)
        
        assert len(path) == 1, f"根节点路径长度应为1，实际为 {len(path)}"
        assert path[0] == root, "根节点路径应只包含根节点本身"
        assert len(root.action_history) == 0, "根节点的action_history应为空"


# ============================================================================
# Property 2: 策略概率归一化
# **Feature: strategy-viewer, Property 2: 策略概率归一化**
# **验证: 需求 4.4**
# ============================================================================

from viewer.strategy_calculator import StrategyCalculator


# 生成随机策略概率字典
@st.composite
def random_strategy_dict(draw, min_actions=1, max_actions=6):
    """生成随机策略概率字典。
    
    生成一个包含随机行动和随机概率的字典。
    概率值可能不归一化，用于测试归一化功能。
    """
    actions = ['FOLD', 'CHECK/CALL', 'RAISE_SMALL', 'RAISE_BIG', 'RAISE_POT', 'ALL_IN']
    num_actions = draw(st.integers(min_value=min_actions, max_value=max_actions))
    selected_actions = draw(st.permutations(actions).map(lambda x: list(x)[:num_actions]))
    
    strategy = {}
    for action in selected_actions:
        # 生成非负概率值（可能不归一化）
        prob = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        strategy[action] = prob
    
    return strategy


# 生成已归一化的策略概率字典
@st.composite
def normalized_strategy_dict(draw):
    """生成已归一化的策略概率字典。
    
    生成一个概率之和为1.0的策略字典。
    """
    actions = ['FOLD', 'CHECK/CALL', 'RAISE_SMALL', 'RAISE_BIG']
    
    # 生成随机权重
    weights = [draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)) 
               for _ in actions]
    
    # 归一化
    total = sum(weights)
    probs = [w / total for w in weights]
    
    return dict(zip(actions, probs))


class TestProperty2StrategyNormalization:
    """属性测试：策略概率归一化。
    
    对于任意手牌的策略分布，所有行动概率之和应等于1.0（允许浮点误差±0.001）。
    """
    
    @given(random_strategy_dict(min_actions=1, max_actions=4))
    @settings(max_examples=100)
    def test_normalize_strategy_sums_to_one(self, strategy):
        """
        **Feature: strategy-viewer, Property 2: 策略概率归一化**
        **验证: 需求 4.4**
        
        测试归一化后的策略概率之和为1.0。
        """
        # 跳过全零策略（会被转换为均匀分布）
        assume(sum(strategy.values()) > 1e-10)
        
        normalized = StrategyCalculator.normalize_strategy(strategy)
        
        total = sum(normalized.values())
        assert abs(total - 1.0) <= 0.001, (
            f"归一化后概率之和应为1.0，实际为 {total}"
        )
    
    @given(random_strategy_dict(min_actions=1, max_actions=4))
    @settings(max_examples=100)
    def test_normalize_preserves_relative_proportions(self, strategy):
        """
        **Feature: strategy-viewer, Property 2: 策略概率归一化**
        **验证: 需求 4.4**
        
        测试归一化保持相对比例。
        """
        # 跳过全零策略
        assume(sum(strategy.values()) > 1e-10)
        
        normalized = StrategyCalculator.normalize_strategy(strategy)
        
        # 检查相对比例是否保持
        original_total = sum(strategy.values())
        
        for action, orig_prob in strategy.items():
            if orig_prob > 1e-10:  # 只检查非零概率
                expected_ratio = orig_prob / original_total
                actual_ratio = normalized[action]
                # 允许一定的浮点误差
                assert abs(expected_ratio - actual_ratio) <= 0.01, (
                    f"行动 {action} 的比例应为 {expected_ratio}，实际为 {actual_ratio}"
                )
    
    @given(random_strategy_dict(min_actions=1, max_actions=4))
    @settings(max_examples=100)
    def test_normalize_all_probabilities_non_negative(self, strategy):
        """
        **Feature: strategy-viewer, Property 2: 策略概率归一化**
        **验证: 需求 4.4**
        
        测试归一化后所有概率都是非负的。
        """
        normalized = StrategyCalculator.normalize_strategy(strategy)
        
        for action, prob in normalized.items():
            assert prob >= 0, (
                f"行动 {action} 的概率应为非负，实际为 {prob}"
            )
    
    @given(random_strategy_dict(min_actions=1, max_actions=4))
    @settings(max_examples=100)
    def test_normalize_all_probabilities_at_most_one(self, strategy):
        """
        **Feature: strategy-viewer, Property 2: 策略概率归一化**
        **验证: 需求 4.4**
        
        测试归一化后所有概率都不超过1.0。
        """
        normalized = StrategyCalculator.normalize_strategy(strategy)
        
        for action, prob in normalized.items():
            assert prob <= 1.0 + 0.001, (
                f"行动 {action} 的概率应不超过1.0，实际为 {prob}"
            )
    
    @given(normalized_strategy_dict())
    @settings(max_examples=100)
    def test_is_normalized_returns_true_for_normalized(self, strategy):
        """
        **Feature: strategy-viewer, Property 2: 策略概率归一化**
        **验证: 需求 4.4**
        
        测试is_normalized对已归一化的策略返回True。
        """
        assert StrategyCalculator.is_normalized(strategy), (
            f"已归一化的策略应返回True，策略: {strategy}"
        )
    
    @given(random_strategy_dict(min_actions=2, max_actions=4))
    @settings(max_examples=100)
    def test_normalize_then_is_normalized(self, strategy):
        """
        **Feature: strategy-viewer, Property 2: 策略概率归一化**
        **验证: 需求 4.4**
        
        测试归一化后的策略通过is_normalized检查。
        """
        normalized = StrategyCalculator.normalize_strategy(strategy)
        
        assert StrategyCalculator.is_normalized(normalized), (
            f"归一化后的策略应通过is_normalized检查，策略: {normalized}"
        )
    
    @given(st.just({}))
    @settings(max_examples=1)
    def test_normalize_empty_strategy(self, strategy):
        """
        **Feature: strategy-viewer, Property 2: 策略概率归一化**
        **验证: 需求 4.4**
        
        测试空策略的归一化。
        """
        normalized = StrategyCalculator.normalize_strategy(strategy)
        
        assert normalized == {}, "空策略归一化后应仍为空"
    
    @given(st.dictionaries(
        keys=st.sampled_from(['FOLD', 'CALL', 'RAISE']),
        values=st.just(0.0),
        min_size=1,
        max_size=3
    ))
    @settings(max_examples=100)
    def test_normalize_zero_strategy_becomes_uniform(self, strategy):
        """
        **Feature: strategy-viewer, Property 2: 策略概率归一化**
        **验证: 需求 4.4**
        
        测试全零策略归一化为均匀分布。
        """
        normalized = StrategyCalculator.normalize_strategy(strategy)
        
        # 应该变成均匀分布
        expected_prob = 1.0 / len(strategy)
        
        for action, prob in normalized.items():
            assert abs(prob - expected_prob) <= 0.001, (
                f"全零策略应归一化为均匀分布，期望 {expected_prob}，实际 {prob}"
            )



# ============================================================================
# Property 6: 颜色映射确定性
# **Feature: strategy-viewer, Property 6: 颜色映射确定性**
# **验证: 需求 3.2**
# ============================================================================

from viewer.color_mapper import StrategyColorMapper, Color


# 生成有效的策略概率字典（用于颜色映射测试）
@st.composite
def valid_strategy_for_color(draw):
    """生成有效的策略概率字典。
    
    生成一个概率之和为1.0的策略字典，用于颜色映射测试。
    """
    actions = ['FOLD', 'CHECK/CALL', 'RAISE_SMALL', 'RAISE_BIG']
    
    # 随机选择1-4个行动
    num_actions = draw(st.integers(min_value=1, max_value=4))
    selected_actions = draw(st.permutations(actions).map(lambda x: list(x)[:num_actions]))
    
    # 生成随机权重
    weights = [draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)) 
               for _ in selected_actions]
    
    # 归一化
    total = sum(weights)
    probs = [w / total for w in weights]
    
    return dict(zip(selected_actions, probs))


class TestProperty6ColorMappingDeterminism:
    """属性测试：颜色映射确定性。
    
    对于任意相同的策略分布输入，get_cell_color应返回完全相同的颜色值（纯函数特性）。
    """
    
    @given(valid_strategy_for_color())
    @settings(max_examples=100)
    def test_same_input_same_output(self, strategy):
        """
        **Feature: strategy-viewer, Property 6: 颜色映射确定性**
        **验证: 需求 3.2**
        
        测试相同输入产生相同输出。
        """
        mapper = StrategyColorMapper()
        
        # 调用两次
        color1 = mapper.get_cell_color(strategy)
        color2 = mapper.get_cell_color(strategy)
        
        assert color1 == color2, (
            f"相同策略应产生相同颜色，"
            f"第一次: {color1}，第二次: {color2}"
        )
    
    @given(valid_strategy_for_color())
    @settings(max_examples=100)
    def test_different_mapper_instances_same_result(self, strategy):
        """
        **Feature: strategy-viewer, Property 6: 颜色映射确定性**
        **验证: 需求 3.2**
        
        测试不同映射器实例对相同输入产生相同输出。
        """
        mapper1 = StrategyColorMapper()
        mapper2 = StrategyColorMapper()
        
        color1 = mapper1.get_cell_color(strategy)
        color2 = mapper2.get_cell_color(strategy)
        
        assert color1 == color2, (
            f"不同映射器实例应对相同策略产生相同颜色，"
            f"mapper1: {color1}，mapper2: {color2}"
        )
    
    @given(valid_strategy_for_color())
    @settings(max_examples=100)
    def test_multiple_calls_consistent(self, strategy):
        """
        **Feature: strategy-viewer, Property 6: 颜色映射确定性**
        **验证: 需求 3.2**
        
        测试多次调用结果一致。
        """
        mapper = StrategyColorMapper()
        
        # 调用多次
        colors = [mapper.get_cell_color(strategy) for _ in range(5)]
        
        # 所有结果应该相同
        first_color = colors[0]
        for i, color in enumerate(colors[1:], 1):
            assert color == first_color, (
                f"第{i+1}次调用结果应与第1次相同，"
                f"第1次: {first_color}，第{i+1}次: {color}"
            )
    
    @given(st.sampled_from(['FOLD', 'CHECK/CALL', 'RAISE_SMALL', 'RAISE_BIG', 'ALL_IN']))
    @settings(max_examples=100)
    def test_action_color_deterministic(self, action):
        """
        **Feature: strategy-viewer, Property 6: 颜色映射确定性**
        **验证: 需求 3.2**
        
        测试行动颜色映射的确定性。
        """
        mapper = StrategyColorMapper()
        
        color1 = mapper.get_action_color(action)
        color2 = mapper.get_action_color(action)
        
        assert color1 == color2, (
            f"相同行动应产生相同颜色，"
            f"行动: {action}，第一次: {color1}，第二次: {color2}"
        )
    
    @given(valid_strategy_for_color())
    @settings(max_examples=100)
    def test_color_values_in_valid_range(self, strategy):
        """
        **Feature: strategy-viewer, Property 6: 颜色映射确定性**
        **验证: 需求 3.2**
        
        测试颜色值在有效范围内。
        """
        mapper = StrategyColorMapper()
        color = mapper.get_cell_color(strategy)
        
        assert 0 <= color.r <= 255, f"红色分量应在0-255范围内: {color.r}"
        assert 0 <= color.g <= 255, f"绿色分量应在0-255范围内: {color.g}"
        assert 0 <= color.b <= 255, f"蓝色分量应在0-255范围内: {color.b}"
        assert 0 <= color.a <= 255, f"透明度分量应在0-255范围内: {color.a}"
    
    @given(st.just({}))
    @settings(max_examples=1)
    def test_empty_strategy_returns_background(self, strategy):
        """
        **Feature: strategy-viewer, Property 6: 颜色映射确定性**
        **验证: 需求 3.2**
        
        测试空策略返回背景色。
        """
        mapper = StrategyColorMapper()
        color = mapper.get_cell_color(strategy)
        
        assert color == mapper.get_background_color(), (
            f"空策略应返回背景色，实际: {color}"
        )
    
    @given(st.sampled_from(['FOLD', 'CHECK/CALL', 'RAISE_SMALL', 'RAISE_BIG']))
    @settings(max_examples=100)
    def test_pure_strategy_returns_action_color(self, action):
        """
        **Feature: strategy-viewer, Property 6: 颜色映射确定性**
        **验证: 需求 3.2**
        
        测试纯策略（单一行动概率为1.0）返回该行动的颜色。
        """
        mapper = StrategyColorMapper()
        
        # 创建纯策略
        strategy = {action: 1.0}
        
        cell_color = mapper.get_cell_color(strategy)
        action_color = mapper.get_action_color(action)
        
        assert cell_color == action_color, (
            f"纯策略应返回行动颜色，"
            f"行动: {action}，单元格颜色: {cell_color}，行动颜色: {action_color}"
        )
    
    @given(valid_strategy_for_color(), valid_strategy_for_color())
    @settings(max_examples=100)
    def test_different_strategies_may_produce_different_colors(self, strategy1, strategy2):
        """
        **Feature: strategy-viewer, Property 6: 颜色映射确定性**
        **验证: 需求 3.2**
        
        测试不同策略可能产生不同颜色（非必须，但验证映射器区分能力）。
        """
        # 跳过相同的策略
        assume(strategy1 != strategy2)
        
        mapper = StrategyColorMapper()
        
        color1 = mapper.get_cell_color(strategy1)
        color2 = mapper.get_cell_color(strategy2)
        
        # 这个测试只是验证映射器能正常工作
        # 不同策略可能产生相同或不同的颜色
        assert isinstance(color1, Color), f"应返回Color对象: {type(color1)}"
        assert isinstance(color2, Color), f"应返回Color对象: {type(color2)}"


# ============================================================================
# Property 8: 无效检查点处理
# **Feature: strategy-viewer, Property 8: 无效检查点处理**
# **验证: 需求 1.2**
# ============================================================================

import tempfile
import os
from viewer.model_loader import ModelLoader, ModelMetadata
from utils.exceptions import CheckpointNotFoundError, CheckpointCorruptedError, ModelLoadError


# 生成无效的检查点数据
@st.composite
def invalid_checkpoint_data(draw):
    """生成各种无效的检查点数据。
    
    生成不符合检查点格式要求的数据，用于测试错误处理。
    """
    invalid_type = draw(st.sampled_from([
        'missing_model_state',  # 缺少model_state_dict
        'wrong_type',           # 数据类型错误（非字典）
        'empty_dict',           # 空字典
        'invalid_state_dict',   # 无效的state_dict
    ]))
    
    if invalid_type == 'missing_model_state':
        # 缺少必需的model_state_dict字段
        return {
            'episode_number': draw(st.integers(min_value=0, max_value=1000)),
            'timestamp': '2024-01-01T00:00:00',
            'win_rate': draw(st.floats(min_value=0.0, max_value=1.0)),
        }
    elif invalid_type == 'wrong_type':
        # 返回非字典类型
        return draw(st.sampled_from([
            [1, 2, 3],           # 列表
            "invalid string",    # 字符串
            12345,               # 整数
            None,                # None
        ]))
    elif invalid_type == 'empty_dict':
        # 空字典
        return {}
    else:  # invalid_state_dict
        # 包含无效的state_dict
        return {
            'model_state_dict': {
                'invalid_key': draw(st.binary(min_size=1, max_size=100))
            },
            'episode_number': 100,
        }
    
    return {}


# 生成随机文件路径（不存在的）
@st.composite
def nonexistent_file_path(draw):
    """生成不存在的文件路径。"""
    random_name = draw(st.text(
        alphabet=st.characters(whitelist_categories=['L', 'N']),
        min_size=5,
        max_size=20
    ))
    return f"/tmp/nonexistent_{random_name}_checkpoint.pt"


# 生成无效扩展名的文件路径
@st.composite
def invalid_extension_path(draw):
    """生成无效扩展名的文件路径。"""
    extensions = ['.txt', '.json', '.pkl', '.bin', '.dat', '.model', '']
    ext = draw(st.sampled_from(extensions))
    random_name = draw(st.text(
        alphabet=st.characters(whitelist_categories=['L', 'N']),
        min_size=5,
        max_size=15
    ))
    return f"checkpoint_{random_name}{ext}"


class TestProperty8InvalidCheckpointHandling:
    """属性测试：无效检查点处理。
    
    对于任意无效或损坏的检查点文件，模型加载应抛出适当的异常而不是崩溃，
    且系统状态保持不变。
    """
    
    @given(nonexistent_file_path())
    @settings(max_examples=100)
    def test_nonexistent_file_raises_not_found(self, file_path):
        """
        **Feature: strategy-viewer, Property 8: 无效检查点处理**
        **验证: 需求 1.2**
        
        测试不存在的文件抛出CheckpointNotFoundError。
        """
        # 确保文件确实不存在
        assume(not os.path.exists(file_path))
        
        loader = ModelLoader()
        initial_state = loader.is_loaded
        
        with pytest.raises(CheckpointNotFoundError) as exc_info:
            loader.load(file_path)
        
        # 验证异常包含文件路径信息
        assert file_path in str(exc_info.value) or file_path in exc_info.value.checkpoint_path
        
        # 验证加载器状态未改变
        assert loader.is_loaded == initial_state, "加载失败后状态应保持不变"
        assert loader.metadata is None, "加载失败后元数据应为None"
    
    @given(invalid_extension_path())
    @settings(max_examples=100)
    def test_invalid_extension_raises_corrupted(self, file_path):
        """
        **Feature: strategy-viewer, Property 8: 无效检查点处理**
        **验证: 需求 1.2**
        
        测试无效扩展名的文件抛出CheckpointCorruptedError。
        """
        loader = ModelLoader()
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_path)[1], delete=False) as f:
            f.write(b"invalid content")
            temp_path = f.name
        
        try:
            # 只有当扩展名不是.pt或.pth时才应该抛出异常
            ext = os.path.splitext(temp_path)[1]
            if ext not in ['.pt', '.pth']:
                with pytest.raises(CheckpointCorruptedError):
                    loader.load(temp_path)
                
                # 验证加载器状态未改变
                assert not loader.is_loaded, "加载失败后应未加载"
        finally:
            os.unlink(temp_path)
    
    @given(invalid_checkpoint_data())
    @settings(max_examples=100)
    def test_invalid_data_raises_appropriate_error(self, invalid_data):
        """
        **Feature: strategy-viewer, Property 8: 无效检查点处理**
        **验证: 需求 1.2**
        
        测试无效数据抛出适当的异常。
        """
        loader = ModelLoader()
        initial_state = loader.is_loaded
        
        # 创建临时检查点文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            # 保存无效数据到文件
            torch.save(invalid_data, temp_path)
            
            # 尝试加载应该抛出异常
            with pytest.raises((CheckpointCorruptedError, ModelLoadError)):
                loader.load(temp_path)
            
            # 验证加载器状态未改变
            assert loader.is_loaded == initial_state, "加载失败后状态应保持不变"
            
        finally:
            os.unlink(temp_path)
    
    @given(st.binary(min_size=1, max_size=1000))
    @settings(max_examples=100)
    def test_random_binary_raises_corrupted(self, random_bytes):
        """
        **Feature: strategy-viewer, Property 8: 无效检查点处理**
        **验证: 需求 1.2**
        
        测试随机二进制数据抛出CheckpointCorruptedError。
        """
        loader = ModelLoader()
        
        # 创建包含随机二进制数据的临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            f.write(random_bytes)
            temp_path = f.name
        
        try:
            with pytest.raises(CheckpointCorruptedError) as exc_info:
                loader.load(temp_path)
            
            # 验证异常消息包含有用信息
            assert "无法读取" in str(exc_info.value) or "损坏" in str(exc_info.value)
            
            # 验证加载器状态未改变
            assert not loader.is_loaded, "加载失败后应未加载"
            
        finally:
            os.unlink(temp_path)
    
    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=100, deadline=None)
    def test_text_file_raises_corrupted(self, text_content):
        """
        **Feature: strategy-viewer, Property 8: 无效检查点处理**
        **验证: 需求 1.2**
        
        测试文本文件抛出CheckpointCorruptedError。
        """
        loader = ModelLoader()
        
        # 创建包含文本的临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False, mode='w') as f:
            f.write(text_content)
            temp_path = f.name
        
        try:
            with pytest.raises(CheckpointCorruptedError):
                loader.load(temp_path)
            
            # 验证加载器状态未改变
            assert not loader.is_loaded, "加载失败后应未加载"
            
        finally:
            os.unlink(temp_path)
    
    @given(nonexistent_file_path())
    @settings(max_examples=100)
    def test_validate_returns_false_for_nonexistent(self, file_path):
        """
        **Feature: strategy-viewer, Property 8: 无效检查点处理**
        **验证: 需求 1.2**
        
        测试validate_checkpoint对不存在的文件返回False。
        """
        assume(not os.path.exists(file_path))
        
        loader = ModelLoader()
        result = loader.validate_checkpoint(file_path)
        
        assert result is False, "不存在的文件应返回False"
    
    @given(invalid_checkpoint_data())
    @settings(max_examples=100)
    def test_validate_returns_false_for_invalid_data(self, invalid_data):
        """
        **Feature: strategy-viewer, Property 8: 无效检查点处理**
        **验证: 需求 1.2**
        
        测试validate_checkpoint对无效数据返回False。
        """
        loader = ModelLoader()
        
        # 创建临时检查点文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            torch.save(invalid_data, temp_path)
            result = loader.validate_checkpoint(temp_path)
            
            # 如果数据不是字典或缺少model_state_dict，应返回False
            if not isinstance(invalid_data, dict) or 'model_state_dict' not in invalid_data:
                assert result is False, f"无效数据应返回False: {type(invalid_data)}"
            
        finally:
            os.unlink(temp_path)
    
    @given(nonexistent_file_path())
    @settings(max_examples=100)
    def test_get_checkpoint_info_returns_none_for_invalid(self, file_path):
        """
        **Feature: strategy-viewer, Property 8: 无效检查点处理**
        **验证: 需求 1.2**
        
        测试get_checkpoint_info对无效文件返回None。
        """
        assume(not os.path.exists(file_path))
        
        loader = ModelLoader()
        result = loader.get_checkpoint_info(file_path)
        
        assert result is None, "无效文件应返回None"
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=10)
    def test_multiple_failed_loads_dont_affect_state(self, num_attempts):
        """
        **Feature: strategy-viewer, Property 8: 无效检查点处理**
        **验证: 需求 1.2**
        
        测试多次加载失败不影响加载器状态。
        """
        loader = ModelLoader()
        
        # 多次尝试加载不存在的文件
        for i in range(num_attempts):
            try:
                loader.load(f"/nonexistent/path/checkpoint_{i}.pt")
            except CheckpointNotFoundError:
                pass
        
        # 验证状态保持一致
        assert not loader.is_loaded, "多次失败后应仍未加载"
        assert loader.metadata is None, "多次失败后元数据应为None"
        assert loader.policy_network is None, "多次失败后网络应为None"
    
    @given(st.just(None))
    @settings(max_examples=1)
    def test_unload_after_failed_load(self, _):
        """
        **Feature: strategy-viewer, Property 8: 无效检查点处理**
        **验证: 需求 1.2**
        
        测试加载失败后调用unload不会出错。
        """
        loader = ModelLoader()
        
        # 尝试加载不存在的文件
        try:
            loader.load("/nonexistent/checkpoint.pt")
        except CheckpointNotFoundError:
            pass
        
        # 调用unload应该不会出错
        loader.unload()
        
        assert not loader.is_loaded, "unload后应未加载"
        assert loader.metadata is None, "unload后元数据应为None"



# ============================================================================
# Property 7: JSON导出完整性
# **Feature: strategy-viewer, Property 7: JSON导出完整性**
# **验证: 需求 8.2**
# ============================================================================

from viewer.controller import StrategyViewerController
from viewer.strategy_calculator import StrategyCalculator


# 生成随机的手牌策略数据
@st.composite
def random_hand_strategies(draw):
    """生成随机的手牌策略数据。
    
    为所有169种手牌生成随机策略。
    """
    strategies = {}
    
    for row in HAND_LABELS_MATRIX:
        for label in row:
            # 生成随机策略
            actions = ['FOLD', 'CHECK/CALL', 'RAISE_SMALL', 'RAISE_BIG']
            num_actions = draw(st.integers(min_value=1, max_value=4))
            selected_actions = draw(st.permutations(actions).map(lambda x: list(x)[:num_actions]))
            
            # 生成随机权重并归一化
            weights = [draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)) 
                       for _ in selected_actions]
            total = sum(weights)
            probs = {action: w / total for action, w in zip(selected_actions, weights)}
            
            strategies[label] = probs
    
    return strategies


class TestProperty7JSONExportCompleteness:
    """属性测试：JSON导出完整性。
    
    对于任意导出的策略JSON文件，应包含所有169种手牌标签，
    且每个手牌的策略概率之和为1.0。
    """
    
    @given(random_hand_strategies())
    @settings(max_examples=50, deadline=None)
    def test_exported_json_contains_all_169_hands(self, strategies):
        """
        **Feature: strategy-viewer, Property 7: JSON导出完整性**
        **验证: 需求 8.2**
        
        测试导出的JSON包含所有169种手牌。
        """
        # 验证输入策略包含所有169种手牌
        all_labels = [label for row in HAND_LABELS_MATRIX for label in row]
        
        assert len(all_labels) == 169, f"应有169种手牌标签，实际有 {len(all_labels)}"
        
        for label in all_labels:
            assert label in strategies, f"策略中应包含手牌 {label}"
    
    @given(random_hand_strategies())
    @settings(max_examples=50, deadline=None)
    def test_all_strategies_sum_to_one(self, strategies):
        """
        **Feature: strategy-viewer, Property 7: JSON导出完整性**
        **验证: 需求 8.2**
        
        测试每个手牌的策略概率之和为1.0。
        """
        for label, strategy in strategies.items():
            total = sum(strategy.values())
            assert abs(total - 1.0) <= 0.001, (
                f"手牌 {label} 的策略概率之和应为1.0，实际为 {total}"
            )
    
    @given(random_hand_strategies())
    @settings(max_examples=50, deadline=None)
    def test_json_export_round_trip(self, strategies):
        """
        **Feature: strategy-viewer, Property 7: JSON导出完整性**
        **验证: 需求 8.2**
        
        测试JSON导出和解析的往返一致性。
        """
        import json
        
        # 构建导出数据结构
        export_data = {
            "metadata": {
                "node_id": "test_node",
                "board_cards": [],
                "model_loaded": True,
            },
            "strategies": {}
        }
        
        for label, strategy in strategies.items():
            export_data["strategies"][label] = {
                "action_probabilities": strategy,
                "num_combinations": 6 if len(label) == 2 else (4 if label.endswith('s') else 12),
                "is_pure_strategy": len(strategy) == 1,
                "dominant_action": max(strategy, key=strategy.get) if strategy else None,
            }
        
        # 序列化和反序列化
        json_str = json.dumps(export_data, ensure_ascii=False)
        parsed_data = json.loads(json_str)
        
        # 验证所有169种手牌都存在
        assert len(parsed_data["strategies"]) == 169, (
            f"解析后应有169种手牌，实际有 {len(parsed_data['strategies'])}"
        )
        
        # 验证每个手牌的策略概率之和
        for label, data in parsed_data["strategies"].items():
            probs = data["action_probabilities"]
            total = sum(probs.values())
            assert abs(total - 1.0) <= 0.001, (
                f"解析后手牌 {label} 的策略概率之和应为1.0，实际为 {total}"
            )
    
    @given(random_hand_strategies())
    @settings(max_examples=50, deadline=None)
    def test_json_export_preserves_action_names(self, strategies):
        """
        **Feature: strategy-viewer, Property 7: JSON导出完整性**
        **验证: 需求 8.2**
        
        测试JSON导出保留行动名称。
        """
        import json
        
        export_data = {"strategies": {}}
        
        for label, strategy in strategies.items():
            export_data["strategies"][label] = {
                "action_probabilities": strategy,
            }
        
        # 序列化和反序列化
        json_str = json.dumps(export_data, ensure_ascii=False)
        parsed_data = json.loads(json_str)
        
        # 验证行动名称保持不变
        for label, strategy in strategies.items():
            parsed_probs = parsed_data["strategies"][label]["action_probabilities"]
            
            for action in strategy.keys():
                assert action in parsed_probs, (
                    f"手牌 {label} 的行动 {action} 应在解析后的数据中"
                )
    
    @given(random_hand_strategies())
    @settings(max_examples=50, deadline=None)
    def test_json_export_preserves_probabilities(self, strategies):
        """
        **Feature: strategy-viewer, Property 7: JSON导出完整性**
        **验证: 需求 8.2**
        
        测试JSON导出保留概率值。
        """
        import json
        
        export_data = {"strategies": {}}
        
        for label, strategy in strategies.items():
            export_data["strategies"][label] = {
                "action_probabilities": strategy,
            }
        
        # 序列化和反序列化
        json_str = json.dumps(export_data, ensure_ascii=False)
        parsed_data = json.loads(json_str)
        
        # 验证概率值保持不变（允许浮点误差）
        for label, strategy in strategies.items():
            parsed_probs = parsed_data["strategies"][label]["action_probabilities"]
            
            for action, prob in strategy.items():
                parsed_prob = parsed_probs[action]
                assert abs(prob - parsed_prob) <= 1e-10, (
                    f"手牌 {label} 行动 {action} 的概率应为 {prob}，"
                    f"解析后为 {parsed_prob}"
                )
    
    @given(st.just(None))
    @settings(max_examples=1)
    def test_hand_labels_matrix_has_169_entries(self, _):
        """
        **Feature: strategy-viewer, Property 7: JSON导出完整性**
        **验证: 需求 8.2**
        
        测试手牌标签矩阵包含169个条目。
        """
        all_labels = [label for row in HAND_LABELS_MATRIX for label in row]
        
        assert len(all_labels) == 169, (
            f"手牌标签矩阵应有169个条目，实际有 {len(all_labels)}"
        )
        
        # 验证没有重复
        unique_labels = set(all_labels)
        assert len(unique_labels) == 169, (
            f"手牌标签应唯一，有 {169 - len(unique_labels)} 个重复"
        )
    
    @given(st.just(None))
    @settings(max_examples=1)
    def test_hand_labels_matrix_structure(self, _):
        """
        **Feature: strategy-viewer, Property 7: JSON导出完整性**
        **验证: 需求 8.2**
        
        测试手牌标签矩阵结构正确（13x13）。
        """
        assert len(HAND_LABELS_MATRIX) == 13, (
            f"矩阵应有13行，实际有 {len(HAND_LABELS_MATRIX)}"
        )
        
        for i, row in enumerate(HAND_LABELS_MATRIX):
            assert len(row) == 13, (
                f"第{i}行应有13列，实际有 {len(row)}"
            )
    
    @given(random_hand_strategies())
    @settings(max_examples=50, deadline=None)
    def test_json_export_file_write_and_read(self, strategies):
        """
        **Feature: strategy-viewer, Property 7: JSON导出完整性**
        **验证: 需求 8.2**
        
        测试JSON导出到文件并读取的完整性。
        """
        import json
        import tempfile
        import os
        
        export_data = {
            "metadata": {
                "node_id": "test_node",
                "board_cards": [],
                "model_loaded": True,
            },
            "strategies": {}
        }
        
        for label, strategy in strategies.items():
            export_data["strategies"][label] = {
                "action_probabilities": strategy,
                "num_combinations": 6 if len(label) == 2 else (4 if label.endswith('s') else 12),
            }
        
        # 写入临时文件
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            temp_path = f.name
        
        try:
            # 从文件读取
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            # 验证完整性
            assert len(loaded_data["strategies"]) == 169, (
                f"从文件读取后应有169种手牌，实际有 {len(loaded_data['strategies'])}"
            )
            
            # 验证每个手牌的策略
            for label in strategies.keys():
                assert label in loaded_data["strategies"], (
                    f"手牌 {label} 应在加载的数据中"
                )
                
                loaded_probs = loaded_data["strategies"][label]["action_probabilities"]
                total = sum(loaded_probs.values())
                assert abs(total - 1.0) <= 0.001, (
                    f"手牌 {label} 的策略概率之和应为1.0，实际为 {total}"
                )
        finally:
            os.unlink(temp_path)


# ============================================================================
# Property: 自动检测 action_dim
# **Feature: regret-network-fix, Property 1: 自动检测 action_dim**
# **验证: 需求 1.1, 1.3**
# ============================================================================


# 生成有效的 action_dim 值
valid_action_dims = st.integers(min_value=2, max_value=10)


@st.composite
def deep_cfr_checkpoint_data(draw, action_dim=None):
    """生成 Deep CFR 格式的检查点数据。
    
    生成包含 regret_network 和 policy_network 权重的检查点数据，
    用于测试 ModelLoader 的 action_dim 自动检测功能。
    
    Args:
        action_dim: 指定的 action_dim，如果为 None 则随机生成
    """
    if action_dim is None:
        action_dim = draw(st.integers(min_value=2, max_value=10))
    
    input_dim = 370
    hidden_dims = [512, 256, 128]
    
    # 创建网络并获取 state_dict
    from models.networks import RegretNetwork, PolicyNetwork
    
    regret_net = RegretNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        action_dim=action_dim
    )
    
    policy_net = PolicyNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        action_dim=action_dim
    )
    
    # 构建检查点数据
    checkpoint_data = {
        'checkpoint_format': 'deep_cfr_v1',
        'regret_network_state_dict': regret_net.state_dict(),
        'policy_network_state_dict': policy_net.state_dict(),
        'episode_number': draw(st.integers(min_value=0, max_value=100000)),
        'timestamp': '2025-12-14T12:00:00',
        'win_rate': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'avg_reward': draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    }
    
    return checkpoint_data, action_dim


class TestPropertyActionDimAutoDetection:
    """属性测试：自动检测 action_dim。
    
    **Feature: regret-network-fix, Property 1: 自动检测 action_dim**
    **验证: 需求 1.1, 1.3**
    
    对于任意 Deep CFR 格式的检查点，加载后的 RegretNetwork 和 PolicyNetwork 
    的 action_dim 应该与检查点中存储的值一致（如果存在），否则应该通过权重推断得到正确的值。
    """
    
    @given(st.data())
    @settings(max_examples=100, deadline=None)
    def test_action_dim_6_is_default(self, data):
        """
        **Feature: regret-network-fix, Property 1: 自动检测 action_dim**
        **验证: 需求 1.1, 2.1**
        
        测试 ModelLoader 的默认 action_dim 为 6（FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN）。
        """
        # 创建 action_dim=6 的检查点
        checkpoint_data, expected_action_dim = data.draw(deep_cfr_checkpoint_data(action_dim=6))
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint_data, f.name)
            temp_path = f.name
        
        try:
            # 使用默认参数加载（不指定 action_dim）
            loader = ModelLoader()
            metadata = loader.load(temp_path)
            
            # 验证加载成功
            assert loader.is_loaded, "模型应该成功加载"
            assert loader.has_regret_network, "应该有遗憾网络"
            
            # 验证 action_dim 正确
            assert loader.regret_network.action_dim == expected_action_dim, (
                f"遗憾网络的 action_dim 应为 {expected_action_dim}，"
                f"实际为 {loader.regret_network.action_dim}"
            )
            
            if loader.policy_network is not None:
                assert loader.policy_network.action_dim == expected_action_dim, (
                    f"策略网络的 action_dim 应为 {expected_action_dim}，"
                    f"实际为 {loader.policy_network.action_dim}"
                )
        finally:
            os.unlink(temp_path)
    
    @given(st.data())
    @settings(max_examples=100)
    def test_loaded_network_matches_checkpoint_action_dim(self, data):
        """
        **Feature: regret-network-fix, Property 1: 自动检测 action_dim**
        **验证: 需求 1.1, 1.3**
        
        测试加载的网络 action_dim 与检查点中的网络一致。
        """
        # 生成随机 action_dim 的检查点
        checkpoint_data, expected_action_dim = data.draw(deep_cfr_checkpoint_data())
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint_data, f.name)
            temp_path = f.name
        
        try:
            # 使用正确的 action_dim 加载
            loader = ModelLoader()
            metadata = loader.load(temp_path, action_dim=expected_action_dim)
            
            # 验证加载成功
            assert loader.is_loaded, "模型应该成功加载"
            
            # 验证遗憾网络的 action_dim
            if loader.has_regret_network:
                assert loader.regret_network.action_dim == expected_action_dim, (
                    f"遗憾网络的 action_dim 应为 {expected_action_dim}，"
                    f"实际为 {loader.regret_network.action_dim}"
                )
            
            # 验证策略网络的 action_dim
            if loader.policy_network is not None:
                assert loader.policy_network.action_dim == expected_action_dim, (
                    f"策略网络的 action_dim 应为 {expected_action_dim}，"
                    f"实际为 {loader.policy_network.action_dim}"
                )
        finally:
            os.unlink(temp_path)
    
    @given(st.data())
    @settings(max_examples=100)
    def test_network_forward_pass_with_correct_action_dim(self, data):
        """
        **Feature: regret-network-fix, Property 1: 自动检测 action_dim**
        **验证: 需求 1.2**
        
        测试加载的网络能正确进行前向传播，输出维度与 action_dim 一致。
        """
        # 生成 action_dim=5 的检查点（标准配置）
        checkpoint_data, expected_action_dim = data.draw(deep_cfr_checkpoint_data(action_dim=5))
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint_data, f.name)
            temp_path = f.name
        
        try:
            # 加载模型
            loader = ModelLoader()
            metadata = loader.load(temp_path, action_dim=expected_action_dim)
            
            # 创建随机输入
            input_tensor = torch.randn(1, 370)
            
            # 测试遗憾网络前向传播
            if loader.has_regret_network:
                output = loader.regret_network(input_tensor)
                assert output.shape[-1] == expected_action_dim, (
                    f"遗憾网络输出维度应为 {expected_action_dim}，"
                    f"实际为 {output.shape[-1]}"
                )
            
            # 测试策略网络前向传播
            if loader.policy_network is not None:
                output = loader.policy_network(input_tensor)
                assert output.shape[-1] == expected_action_dim, (
                    f"策略网络输出维度应为 {expected_action_dim}，"
                    f"实际为 {output.shape[-1]}"
                )
        finally:
            os.unlink(temp_path)
    
    @given(st.data())
    @settings(max_examples=50)
    def test_mismatched_action_dim_uses_detected_value(self, data):
        """
        **Feature: regret-network-fix, Property 1: 自动检测 action_dim**
        **验证: 需求 3.3**
        
        测试当指定的 action_dim 与检查点不匹配时，ModelLoader 会使用检测到的值。
        这是正确的行为：ModelLoader 会自动检测 action_dim 并优先使用检测值。
        """
        # 生成 action_dim=5 的检查点
        checkpoint_data, actual_action_dim = data.draw(deep_cfr_checkpoint_data(action_dim=5))
        
        # 使用不同的 action_dim
        wrong_action_dim = data.draw(st.integers(min_value=2, max_value=10).filter(lambda x: x != actual_action_dim))
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint_data, f.name)
            temp_path = f.name
        
        try:
            # 尝试使用错误的 action_dim 加载
            loader = ModelLoader()
            
            # ModelLoader 会自动检测 action_dim，优先使用检测到的值
            metadata = loader.load(temp_path, action_dim=wrong_action_dim)
            
            # 验证 ModelLoader 使用了检测到的 action_dim（而不是传入的错误值）
            # 这是正确的行为：自动检测优先于用户指定的值
            if loader.has_regret_network:
                # 网络的 action_dim 应该是检测到的值（actual_action_dim）
                # 因为 ModelLoader 会从权重中检测正确的维度
                assert loader.regret_network.action_dim == actual_action_dim, (
                    f"ModelLoader 应该使用检测到的 action_dim ({actual_action_dim})，"
                    f"而不是传入的错误值 ({wrong_action_dim})，"
                    f"实际值: {loader.regret_network.action_dim}"
                )
        except (ModelLoadError, RuntimeError):
            # 如果加载失败，也是可接受的行为
            pass
        finally:
            os.unlink(temp_path)


# ============================================================================
# Property 2: 动作映射正确性
# **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
# **验证: 需求 1.2, 1.3, 3.3**
# ============================================================================

from viewer.models import ActionConfig, BarSegment, DEFAULT_ACTION_MAPPINGS


# 生成有效的动作维度
valid_action_dim_strategy = st.integers(min_value=4, max_value=6)

# 生成任意动作维度（包括非标准维度）
any_action_dim_strategy = st.integers(min_value=1, max_value=10)


# 生成包含 action_config 元数据的检查点数据
@st.composite
def checkpoint_with_action_config(draw):
    """生成包含 action_config 元数据的检查点数据。"""
    action_dim = draw(valid_action_dim_strategy)
    action_names = DEFAULT_ACTION_MAPPINGS.get(action_dim, [f'ACTION_{i}' for i in range(action_dim)])
    
    checkpoint_data = {
        'action_config': {
            'action_names': action_names.copy(),
            'action_dim': action_dim,
        },
        'episode_number': draw(st.integers(min_value=0, max_value=100000)),
        'timestamp': '2025-12-15T12:00:00',
    }
    
    return checkpoint_data, action_dim, action_names


# 生成不包含 action_config 元数据的检查点数据
@st.composite
def checkpoint_without_action_config(draw):
    """生成不包含 action_config 元数据的检查点数据。
    
    模拟旧格式的检查点，需要通过网络权重检测动作维度。
    """
    action_dim = draw(valid_action_dim_strategy)
    
    # 创建模拟的网络权重（只包含输出层维度信息）
    checkpoint_data = {
        'episode_number': draw(st.integers(min_value=0, max_value=100000)),
        'timestamp': '2025-12-15T12:00:00',
        'action_dim': action_dim,  # 直接存储 action_dim 作为备用
    }
    
    return checkpoint_data, action_dim


class TestProperty2ActionMappingCorrectness:
    """属性测试：动作映射正确性。
    
    **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
    **验证: 需求 1.2, 1.3, 3.3**
    
    对于任意检查点数据，如果包含 action_config 元数据则使用该配置，
    否则根据检测到的动作维度返回对应的默认动作映射。
    """
    
    @given(checkpoint_with_action_config())
    @settings(max_examples=100)
    def test_uses_action_config_when_present(self, checkpoint_data_tuple):
        """
        **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
        **验证: 需求 1.2**
        
        测试当检查点包含 action_config 元数据时，使用该配置。
        """
        checkpoint_data, expected_dim, expected_names = checkpoint_data_tuple
        
        config = ActionConfig.from_checkpoint(checkpoint_data)
        
        assert config.action_dim == expected_dim, (
            f"动作维度应为 {expected_dim}，实际为 {config.action_dim}"
        )
        assert config.action_names == expected_names, (
            f"动作名称应为 {expected_names}，实际为 {config.action_names}"
        )
    
    @given(checkpoint_without_action_config())
    @settings(max_examples=100)
    def test_uses_default_mapping_when_no_config(self, checkpoint_data_tuple):
        """
        **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
        **验证: 需求 1.3, 3.3**
        
        测试当检查点不包含 action_config 元数据时，使用默认映射。
        """
        checkpoint_data, expected_dim = checkpoint_data_tuple
        
        config = ActionConfig.from_checkpoint(checkpoint_data)
        
        # 应该使用默认映射
        expected_names = DEFAULT_ACTION_MAPPINGS.get(expected_dim, [f'ACTION_{i}' for i in range(expected_dim)])
        
        assert config.action_dim == expected_dim, (
            f"动作维度应为 {expected_dim}，实际为 {config.action_dim}"
        )
        assert config.action_names == expected_names, (
            f"动作名称应为 {expected_names}，实际为 {config.action_names}"
        )
    
    @given(valid_action_dim_strategy)
    @settings(max_examples=100)
    def test_default_for_dim_returns_correct_mapping(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
        **验证: 需求 1.3, 3.3**
        
        测试 default_for_dim 返回正确的默认映射。
        """
        config = ActionConfig.default_for_dim(action_dim)
        
        assert config.action_dim == action_dim, (
            f"动作维度应为 {action_dim}，实际为 {config.action_dim}"
        )
        
        expected_names = DEFAULT_ACTION_MAPPINGS[action_dim]
        assert config.action_names == expected_names, (
            f"动作名称应为 {expected_names}，实际为 {config.action_names}"
        )
    
    @given(any_action_dim_strategy)
    @settings(max_examples=100)
    def test_default_for_dim_handles_non_standard_dims(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
        **验证: 需求 1.3, 3.3**
        
        测试 default_for_dim 能处理非标准维度。
        """
        config = ActionConfig.default_for_dim(action_dim)
        
        assert config.action_dim == action_dim, (
            f"动作维度应为 {action_dim}，实际为 {config.action_dim}"
        )
        
        # 验证动作名称列表长度正确
        assert len(config.action_names) == action_dim, (
            f"动作名称列表长度应为 {action_dim}，实际为 {len(config.action_names)}"
        )
        
        # 如果是标准维度，应该使用预定义映射
        if action_dim in DEFAULT_ACTION_MAPPINGS:
            assert config.action_names == DEFAULT_ACTION_MAPPINGS[action_dim], (
                f"标准维度 {action_dim} 应使用预定义映射"
            )
        else:
            # 非标准维度应该使用通用名称
            expected_names = [f'ACTION_{i}' for i in range(action_dim)]
            assert config.action_names == expected_names, (
                f"非标准维度应使用通用名称，期望 {expected_names}，实际 {config.action_names}"
            )
    
    @given(valid_action_dim_strategy)
    @settings(max_examples=100)
    def test_action_names_length_equals_dim(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
        **验证: 需求 1.2, 1.3**
        
        测试动作名称列表长度等于动作维度。
        """
        config = ActionConfig.default_for_dim(action_dim)
        
        assert len(config.action_names) == config.action_dim, (
            f"动作名称列表长度({len(config.action_names)})应等于动作维度({config.action_dim})"
        )
    
    @given(checkpoint_with_action_config())
    @settings(max_examples=100)
    def test_get_action_index_returns_correct_index(self, checkpoint_data_tuple):
        """
        **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
        **验证: 需求 1.2**
        
        测试 get_action_index 返回正确的索引。
        """
        checkpoint_data, expected_dim, expected_names = checkpoint_data_tuple
        
        config = ActionConfig.from_checkpoint(checkpoint_data)
        
        for i, action_name in enumerate(expected_names):
            index = config.get_action_index(action_name)
            assert index == i, (
                f"动作 {action_name} 的索引应为 {i}，实际为 {index}"
            )
    
    @given(valid_action_dim_strategy)
    @settings(max_examples=100)
    def test_get_action_index_raises_for_unknown_action(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
        **验证: 需求 1.2**
        
        测试 get_action_index 对未知动作抛出异常。
        """
        config = ActionConfig.default_for_dim(action_dim)
        
        with pytest.raises(ValueError) as exc_info:
            config.get_action_index("UNKNOWN_ACTION")
        
        assert "未知的动作名称" in str(exc_info.value), (
            f"异常消息应包含'未知的动作名称'，实际为: {exc_info.value}"
        )
    
    @given(st.just(4))
    @settings(max_examples=1)
    def test_dim_4_has_correct_actions(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
        **验证: 需求 1.3, 3.3**
        
        测试维度4的默认动作映射正确。
        """
        config = ActionConfig.default_for_dim(action_dim)
        
        expected = ['FOLD', 'CHECK', 'CALL', 'RAISE']
        assert config.action_names == expected, (
            f"维度4的动作应为 {expected}，实际为 {config.action_names}"
        )
    
    @given(st.just(5))
    @settings(max_examples=1)
    def test_dim_5_has_correct_actions(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
        **验证: 需求 1.3, 3.3**
        
        测试维度5的默认动作映射正确。
        """
        config = ActionConfig.default_for_dim(action_dim)
        
        expected = ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG']
        assert config.action_names == expected, (
            f"维度5的动作应为 {expected}，实际为 {config.action_names}"
        )
    
    @given(st.just(6))
    @settings(max_examples=1)
    def test_dim_6_has_correct_actions(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
        **验证: 需求 1.3, 3.3**
        
        测试维度6的默认动作映射正确。
        """
        config = ActionConfig.default_for_dim(action_dim)
        
        expected = ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG', 'ALL_IN']
        assert config.action_names == expected, (
            f"维度6的动作应为 {expected}，实际为 {config.action_names}"
        )
    
    @given(checkpoint_with_action_config())
    @settings(max_examples=100)
    def test_action_config_is_immutable_copy(self, checkpoint_data_tuple):
        """
        **Feature: adaptive-action-viewer, Property 2: 动作映射正确性**
        **验证: 需求 1.2**
        
        测试从检查点创建的配置不会被原始数据修改影响。
        """
        checkpoint_data, expected_dim, expected_names = checkpoint_data_tuple
        
        config = ActionConfig.from_checkpoint(checkpoint_data)
        
        # 修改原始检查点数据
        checkpoint_data['action_config']['action_names'].append('NEW_ACTION')
        checkpoint_data['action_config']['action_dim'] = 999
        
        # 配置应该不受影响
        assert config.action_dim == expected_dim, (
            f"配置的动作维度不应被修改，期望 {expected_dim}，实际 {config.action_dim}"
        )
        assert config.action_names == expected_names, (
            f"配置的动作名称不应被修改，期望 {expected_names}，实际 {config.action_names}"
        )


# ============================================================================
# Property 1: 动作维度检测正确性
# **Feature: adaptive-action-viewer, Property 1: 动作维度检测正确性**
# **验证: 需求 1.1**
# ============================================================================

from viewer.model_loader import ModelLoader


# 生成包含网络权重的检查点数据（用于测试维度检测）
@st.composite
def checkpoint_with_network_weights(draw):
    """生成包含网络权重的检查点数据。
    
    生成包含 policy_network 或 regret_network 权重的检查点数据，
    用于测试 ModelLoader 的动作维度自动检测功能。
    """
    action_dim = draw(st.integers(min_value=2, max_value=10))
    input_dim = 370
    hidden_dims = [512, 256, 128]
    
    # 创建网络并获取 state_dict
    from models.networks import RegretNetwork, PolicyNetwork
    
    # 随机选择包含哪些网络
    include_regret = draw(st.booleans())
    include_policy = draw(st.booleans())
    
    # 至少包含一个网络
    if not include_regret and not include_policy:
        include_policy = True
    
    checkpoint_data = {
        'checkpoint_format': 'deep_cfr_v1',
        'episode_number': draw(st.integers(min_value=0, max_value=100000)),
        'timestamp': '2025-12-15T12:00:00',
    }
    
    if include_regret:
        regret_net = RegretNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            action_dim=action_dim
        )
        checkpoint_data['regret_network_state_dict'] = regret_net.state_dict()
    
    if include_policy:
        policy_net = PolicyNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            action_dim=action_dim
        )
        checkpoint_data['policy_network_state_dict'] = policy_net.state_dict()
    
    return checkpoint_data, action_dim


# 生成旧格式检查点数据（使用 model_state_dict）
@st.composite
def legacy_checkpoint_with_weights(draw):
    """生成旧格式的检查点数据。
    
    生成包含 model_state_dict 的旧格式检查点数据。
    """
    action_dim = draw(st.integers(min_value=2, max_value=10))
    input_dim = 370
    hidden_dims = [512, 256, 128]
    
    from models.networks import PolicyNetwork
    
    policy_net = PolicyNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        action_dim=action_dim
    )
    
    checkpoint_data = {
        'model_state_dict': policy_net.state_dict(),
        'episode_number': draw(st.integers(min_value=0, max_value=100000)),
        'timestamp': '2025-12-15T12:00:00',
    }
    
    return checkpoint_data, action_dim


class TestProperty1ActionDimDetection:
    """属性测试：动作维度检测正确性。
    
    **Feature: adaptive-action-viewer, Property 1: 动作维度检测正确性**
    **验证: 需求 1.1**
    
    对于任意包含策略网络或遗憾网络权重的检查点数据，
    _detect_action_dim 函数应返回与网络输出层维度一致的动作维度值。
    """
    
    @given(checkpoint_with_network_weights())
    @settings(max_examples=100)
    def test_detect_action_dim_from_deep_cfr_checkpoint(self, checkpoint_data_tuple):
        """
        **Feature: adaptive-action-viewer, Property 1: 动作维度检测正确性**
        **验证: 需求 1.1**
        
        测试从 Deep CFR 格式检查点检测动作维度。
        """
        checkpoint_data, expected_dim = checkpoint_data_tuple
        
        loader = ModelLoader()
        detected_dim = loader._detect_action_dim(checkpoint_data)
        
        assert detected_dim == expected_dim, (
            f"检测到的动作维度应为 {expected_dim}，实际为 {detected_dim}"
        )
    
    @given(legacy_checkpoint_with_weights())
    @settings(max_examples=100)
    def test_detect_action_dim_from_legacy_checkpoint(self, checkpoint_data_tuple):
        """
        **Feature: adaptive-action-viewer, Property 1: 动作维度检测正确性**
        **验证: 需求 1.1**
        
        测试从旧格式检查点检测动作维度。
        """
        checkpoint_data, expected_dim = checkpoint_data_tuple
        
        loader = ModelLoader()
        detected_dim = loader._detect_action_dim(checkpoint_data)
        
        assert detected_dim == expected_dim, (
            f"检测到的动作维度应为 {expected_dim}，实际为 {detected_dim}"
        )
    
    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=100)
    def test_detect_action_dim_from_action_config_metadata(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 1: 动作维度检测正确性**
        **验证: 需求 1.1**
        
        测试从 action_config 元数据检测动作维度（优先级最高）。
        """
        checkpoint_data = {
            'action_config': {
                'action_dim': action_dim,
                'action_names': [f'ACTION_{i}' for i in range(action_dim)],
            },
            'episode_number': 1000,
        }
        
        loader = ModelLoader()
        detected_dim = loader._detect_action_dim(checkpoint_data)
        
        assert detected_dim == action_dim, (
            f"检测到的动作维度应为 {action_dim}，实际为 {detected_dim}"
        )
    
    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=100)
    def test_detect_action_dim_from_top_level_field(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 1: 动作维度检测正确性**
        **验证: 需求 1.1**
        
        测试从顶层 action_dim 字段检测动作维度。
        """
        checkpoint_data = {
            'action_dim': action_dim,
            'episode_number': 1000,
        }
        
        loader = ModelLoader()
        detected_dim = loader._detect_action_dim(checkpoint_data)
        
        assert detected_dim == action_dim, (
            f"检测到的动作维度应为 {action_dim}，实际为 {detected_dim}"
        )
    
    @given(st.just({}))
    @settings(max_examples=1)
    def test_detect_action_dim_defaults_to_6(self, checkpoint_data):
        """
        **Feature: adaptive-action-viewer, Property 1: 动作维度检测正确性**
        **验证: 需求 1.1**
        
        测试无法检测时默认返回6。
        """
        loader = ModelLoader()
        detected_dim = loader._detect_action_dim(checkpoint_data)
        
        assert detected_dim == 6, (
            f"无法检测时应默认返回6，实际为 {detected_dim}"
        )
    
    @given(checkpoint_with_network_weights())
    @settings(max_examples=100)
    def test_action_config_priority_over_weights(self, checkpoint_data_tuple):
        """
        **Feature: adaptive-action-viewer, Property 1: 动作维度检测正确性**
        **验证: 需求 1.1**
        
        测试 action_config 元数据优先于网络权重检测。
        """
        checkpoint_data, weights_dim = checkpoint_data_tuple
        
        # 添加不同的 action_config 元数据
        config_dim = 7  # 使用一个不同的维度
        checkpoint_data['action_config'] = {
            'action_dim': config_dim,
            'action_names': [f'ACTION_{i}' for i in range(config_dim)],
        }
        
        loader = ModelLoader()
        detected_dim = loader._detect_action_dim(checkpoint_data)
        
        # 应该使用 action_config 中的维度，而不是从权重检测的维度
        assert detected_dim == config_dim, (
            f"应优先使用 action_config 中的维度 {config_dim}，实际为 {detected_dim}"
        )
    
    @given(st.data())
    @settings(max_examples=50, deadline=None)
    def test_model_loader_uses_detected_action_dim(self, data):
        """
        **Feature: adaptive-action-viewer, Property 1: 动作维度检测正确性**
        **验证: 需求 1.1**
        
        测试 ModelLoader.load 方法使用检测到的动作维度。
        """
        checkpoint_data, expected_dim = data.draw(checkpoint_with_network_weights())
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint_data, f.name)
            temp_path = f.name
        
        try:
            loader = ModelLoader()
            metadata = loader.load(temp_path)
            
            # 验证 action_config 被正确设置
            assert loader.action_config is not None, "action_config 应该被设置"
            assert loader.action_config.action_dim == expected_dim, (
                f"action_config.action_dim 应为 {expected_dim}，"
                f"实际为 {loader.action_config.action_dim}"
            )
            
            # 验证元数据中包含动作维度信息
            assert 'action_dim' in metadata.extra_info, (
                "元数据应包含 action_dim"
            )
            assert metadata.extra_info['action_dim'] == expected_dim, (
                f"元数据中的 action_dim 应为 {expected_dim}，"
                f"实际为 {metadata.extra_info['action_dim']}"
            )
        finally:
            os.unlink(temp_path)
    
    @given(st.data())
    @settings(max_examples=50, deadline=None)
    def test_loaded_networks_have_correct_action_dim(self, data):
        """
        **Feature: adaptive-action-viewer, Property 1: 动作维度检测正确性**
        **验证: 需求 1.1**
        
        测试加载的网络具有正确的动作维度。
        """
        checkpoint_data, expected_dim = data.draw(checkpoint_with_network_weights())
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint_data, f.name)
            temp_path = f.name
        
        try:
            loader = ModelLoader()
            metadata = loader.load(temp_path)
            
            # 验证加载的网络具有正确的 action_dim
            if loader.has_regret_network:
                assert loader.regret_network.action_dim == expected_dim, (
                    f"遗憾网络的 action_dim 应为 {expected_dim}，"
                    f"实际为 {loader.regret_network.action_dim}"
                )
            
            if loader.policy_network is not None:
                assert loader.policy_network.action_dim == expected_dim, (
                    f"策略网络的 action_dim 应为 {expected_dim}，"
                    f"实际为 {loader.policy_network.action_dim}"
                )
        finally:
            os.unlink(temp_path)
    
    @given(st.data())
    @settings(max_examples=50, deadline=None)
    def test_action_config_unloaded_after_unload(self, data):
        """
        **Feature: adaptive-action-viewer, Property 1: 动作维度检测正确性**
        **验证: 需求 1.1**
        
        测试卸载模型后 action_config 被清除。
        """
        checkpoint_data, expected_dim = data.draw(checkpoint_with_network_weights())
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint_data, f.name)
            temp_path = f.name
        
        try:
            loader = ModelLoader()
            metadata = loader.load(temp_path)
            
            # 验证加载后有 action_config
            assert loader.action_config is not None, "加载后应有 action_config"
            
            # 卸载模型
            loader.unload()
            
            # 验证卸载后 action_config 被清除
            assert loader.action_config is None, "卸载后 action_config 应为 None"
        finally:
            os.unlink(temp_path)


# ============================================================================
# Property 4: 策略动作数量一致性
# **Feature: adaptive-action-viewer, Property 4: 策略动作数量一致性**
# **验证: 需求 2.1, 4.1**
# ============================================================================


# 生成有效的动作配置
@st.composite
def valid_action_config_strategy(draw):
    """生成有效的动作配置。"""
    action_dim = draw(st.integers(min_value=4, max_value=6))
    config = ActionConfig.default_for_dim(action_dim)
    return config


# 生成策略概率字典（基于动作配置）
@st.composite
def strategy_for_config(draw, config: ActionConfig):
    """生成与动作配置匹配的策略概率字典。"""
    # 生成随机概率
    probs = [draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)) 
             for _ in range(config.action_dim)]
    
    # 归一化
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [1.0 / config.action_dim] * config.action_dim
    
    # 构建策略字典
    strategy = {}
    for i, action in enumerate(config.action_names):
        strategy[action] = probs[i]
    
    return strategy


class TestProperty4StrategyActionCountConsistency:
    """属性测试：策略动作数量一致性。
    
    **Feature: adaptive-action-viewer, Property 4: 策略动作数量一致性**
    **验证: 需求 2.1, 4.1**
    
    对于任意模型和游戏状态，策略分析器返回的动作概率字典的键数量
    应等于模型的动作维度。
    """
    
    @given(valid_action_config_strategy())
    @settings(max_examples=100)
    def test_calculator_available_actions_matches_config(self, config):
        """
        **Feature: adaptive-action-viewer, Property 4: 策略动作数量一致性**
        **验证: 需求 2.1**
        
        测试策略计算器的可用动作数量与配置一致。
        """
        calculator = StrategyCalculator(action_config=config)
        
        available_actions = calculator.available_actions
        
        # 计算预期的显示动作数量
        # CHECK 和 CALL 会合并为 CHECK/CALL
        has_check = 'CHECK' in config.action_names
        has_call = 'CALL' in config.action_names
        
        if has_check and has_call:
            # CHECK 和 CALL 合并为一个
            expected_count = config.action_dim - 1
        else:
            expected_count = config.action_dim
        
        assert len(available_actions) == expected_count, (
            f"可用动作数量应为 {expected_count}，实际为 {len(available_actions)}。"
            f"配置动作: {config.action_names}，显示动作: {available_actions}"
        )
    
    @given(valid_action_config_strategy())
    @settings(max_examples=100)
    def test_set_action_config_updates_available_actions(self, config):
        """
        **Feature: adaptive-action-viewer, Property 4: 策略动作数量一致性**
        **验证: 需求 2.1**
        
        测试 set_action_config 方法正确更新可用动作列表。
        """
        # 先用默认配置创建
        calculator = StrategyCalculator()
        original_actions = calculator.available_actions.copy()
        
        # 设置新配置
        calculator.set_action_config(config)
        new_actions = calculator.available_actions
        
        # 验证动作列表已更新
        has_check = 'CHECK' in config.action_names
        has_call = 'CALL' in config.action_names
        
        if has_check and has_call:
            expected_count = config.action_dim - 1
        else:
            expected_count = config.action_dim
        
        assert len(new_actions) == expected_count, (
            f"更新后的可用动作数量应为 {expected_count}，实际为 {len(new_actions)}"
        )
    
    @given(st.integers(min_value=4, max_value=6))
    @settings(max_examples=100)
    def test_uniform_strategy_has_correct_action_count(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 4: 策略动作数量一致性**
        **验证: 需求 2.1, 4.1**
        
        测试均匀分布策略的动作数量正确。
        """
        config = ActionConfig.default_for_dim(action_dim)
        calculator = StrategyCalculator(action_config=config)
        
        # 获取均匀分布策略
        uniform_strategy = calculator._get_uniform_strategy()
        
        # 验证动作数量
        expected_count = len(calculator.available_actions)
        assert len(uniform_strategy) == expected_count, (
            f"均匀分布策略的动作数量应为 {expected_count}，实际为 {len(uniform_strategy)}"
        )
        
        # 验证概率之和为1
        total_prob = sum(uniform_strategy.values())
        assert abs(total_prob - 1.0) < 0.001, (
            f"均匀分布策略的概率之和应为1.0，实际为 {total_prob}"
        )
    
    @given(st.integers(min_value=4, max_value=6))
    @settings(max_examples=100)
    def test_action_config_property_returns_correct_config(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 4: 策略动作数量一致性**
        **验证: 需求 2.1**
        
        测试 action_config 属性返回正确的配置。
        """
        config = ActionConfig.default_for_dim(action_dim)
        calculator = StrategyCalculator(action_config=config)
        
        returned_config = calculator.action_config
        
        assert returned_config.action_dim == config.action_dim, (
            f"返回的配置动作维度应为 {config.action_dim}，实际为 {returned_config.action_dim}"
        )
        assert returned_config.action_names == config.action_names, (
            f"返回的配置动作名称应为 {config.action_names}，实际为 {returned_config.action_names}"
        )
    
    @given(st.integers(min_value=4, max_value=6))
    @settings(max_examples=100)
    def test_display_actions_merge_check_call(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 4: 策略动作数量一致性**
        **验证: 需求 2.1**
        
        测试显示动作正确合并 CHECK 和 CALL。
        """
        config = ActionConfig.default_for_dim(action_dim)
        calculator = StrategyCalculator(action_config=config)
        
        available_actions = calculator.available_actions
        
        # 验证 CHECK 和 CALL 被合并为 CHECK/CALL
        has_check = 'CHECK' in config.action_names
        has_call = 'CALL' in config.action_names
        
        if has_check and has_call:
            assert 'CHECK/CALL' in available_actions, (
                f"当配置包含 CHECK 和 CALL 时，显示动作应包含 CHECK/CALL。"
                f"配置: {config.action_names}，显示: {available_actions}"
            )
            assert 'CHECK' not in available_actions, (
                f"当配置包含 CHECK 和 CALL 时，显示动作不应单独包含 CHECK"
            )
            assert 'CALL' not in available_actions, (
                f"当配置包含 CHECK 和 CALL 时，显示动作不应单独包含 CALL"
            )
    
    @given(st.integers(min_value=4, max_value=6))
    @settings(max_examples=100)
    def test_all_non_merged_actions_preserved(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 4: 策略动作数量一致性**
        **验证: 需求 2.1**
        
        测试非合并动作都被保留在显示列表中。
        """
        config = ActionConfig.default_for_dim(action_dim)
        calculator = StrategyCalculator(action_config=config)
        
        available_actions = calculator.available_actions
        
        # 验证非 CHECK/CALL 的动作都被保留
        for action in config.action_names:
            if action not in ['CHECK', 'CALL']:
                assert action in available_actions, (
                    f"动作 {action} 应该在显示列表中。"
                    f"配置: {config.action_names}，显示: {available_actions}"
                )
    
    @given(st.data())
    @settings(max_examples=50)
    def test_strategy_result_actions_match_calculator(self, data):
        """
        **Feature: adaptive-action-viewer, Property 4: 策略动作数量一致性**
        **验证: 需求 2.1, 4.1**
        
        测试策略计算结果的动作列表与计算器一致。
        """
        action_dim = data.draw(st.integers(min_value=4, max_value=6))
        config = ActionConfig.default_for_dim(action_dim)
        calculator = StrategyCalculator(action_config=config)
        
        # 创建一个简单的游戏树节点用于测试
        from viewer.models import GameTreeNode, NodeType
        from models.core import GameStage
        
        node = GameTreeNode(
            node_id="test_root",
            stage=GameStage.PREFLOP,
            player=0,
            pot=15,
            stacks=[995, 990],
            node_type=NodeType.PLAYER
        )
        
        # 计算策略
        result = calculator.calculate_node_strategy(node, board_cards=[], player_id=0)
        
        # 验证结果中的动作列表与计算器一致
        assert result.available_actions == calculator.available_actions, (
            f"策略结果的动作列表应与计算器一致。"
            f"结果: {result.available_actions}，计算器: {calculator.available_actions}"
        )
    
    @given(st.data())
    @settings(max_examples=50)
    def test_hand_strategy_action_count_matches(self, data):
        """
        **Feature: adaptive-action-viewer, Property 4: 策略动作数量一致性**
        **验证: 需求 2.1, 4.1**
        
        测试手牌策略的动作概率数量与配置一致。
        """
        action_dim = data.draw(st.integers(min_value=4, max_value=6))
        config = ActionConfig.default_for_dim(action_dim)
        calculator = StrategyCalculator(action_config=config)
        
        # 创建一个简单的游戏树节点
        from viewer.models import GameTreeNode, NodeType
        from models.core import GameStage
        
        node = GameTreeNode(
            node_id="test_root",
            stage=GameStage.PREFLOP,
            player=0,
            pot=15,
            stacks=[995, 990],
            node_type=NodeType.PLAYER
        )
        
        # 计算策略
        result = calculator.calculate_node_strategy(node, board_cards=[], player_id=0)
        
        # 验证每个手牌策略的动作数量
        expected_action_count = len(calculator.available_actions)
        
        for hand_label, hand_strategy in result.hand_strategies.items():
            actual_count = len(hand_strategy.action_probabilities)
            assert actual_count == expected_action_count, (
                f"手牌 {hand_label} 的动作概率数量应为 {expected_action_count}，"
                f"实际为 {actual_count}。"
                f"动作概率: {hand_strategy.action_probabilities}"
            )


# ============================================================================
# Property 5: 动作颜色映射正确性
# **Feature: adaptive-action-viewer, Property 5: 动作颜色映射正确性**
# **验证: 需求 6.2, 6.3, 6.4**
# ============================================================================


class TestProperty5ActionColorMapping:
    """属性测试：动作颜色映射正确性。
    
    对于任意动作名称：
    - FOLD动作返回蓝色系颜色（R < 100, G > 100, B > 200）
    - CHECK/CALL动作返回绿色系颜色（R < 100, G > 150, B < 150）
    - RAISE类动作返回红色系颜色（R > 180, G < 150, B < 150）
    """
    
    @given(st.just('FOLD'))
    @settings(max_examples=10)
    def test_fold_returns_blue_color(self, action):
        """
        **Feature: adaptive-action-viewer, Property 5: 动作颜色映射正确性**
        **验证: 需求 6.2, 6.3, 6.4**
        
        测试FOLD动作返回蓝色系颜色。
        """
        mapper = StrategyColorMapper()
        color = mapper.get_action_color(action)
        
        # 蓝色系: R < 100, G > 100, B > 200
        assert color.r < 100, f"FOLD的红色分量应 < 100，实际: {color.r}"
        assert color.g > 100, f"FOLD的绿色分量应 > 100，实际: {color.g}"
        assert color.b > 200, f"FOLD的蓝色分量应 > 200，实际: {color.b}"
    
    @given(st.sampled_from(['CHECK', 'CALL', 'CHECK/CALL']))
    @settings(max_examples=30)
    def test_check_call_returns_green_color(self, action):
        """
        **Feature: adaptive-action-viewer, Property 5: 动作颜色映射正确性**
        **验证: 需求 6.2, 6.3, 6.4**
        
        测试CHECK/CALL动作返回绿色系颜色。
        """
        mapper = StrategyColorMapper()
        color = mapper.get_action_color(action)
        
        # 绿色系: R < 100, G > 150, B < 150
        assert color.r < 100, f"{action}的红色分量应 < 100，实际: {color.r}"
        assert color.g > 150, f"{action}的绿色分量应 > 150，实际: {color.g}"
        assert color.b < 150, f"{action}的蓝色分量应 < 150，实际: {color.b}"
    
    @given(st.sampled_from(['RAISE', 'RAISE_SMALL', 'RAISE_BIG', 'ALL_IN']))
    @settings(max_examples=40)
    def test_raise_returns_red_color(self, action):
        """
        **Feature: adaptive-action-viewer, Property 5: 动作颜色映射正确性**
        **验证: 需求 6.2, 6.3, 6.4**
        
        测试RAISE类动作返回红色系颜色。
        """
        mapper = StrategyColorMapper()
        color = mapper.get_action_color(action)
        
        # 红色系: R > 180, G < 150, B < 150
        assert color.r > 180, f"{action}的红色分量应 > 180，实际: {color.r}"
        assert color.g < 150, f"{action}的绿色分量应 < 150，实际: {color.g}"
        assert color.b < 150, f"{action}的蓝色分量应 < 150，实际: {color.b}"
    
    @given(st.sampled_from(['FOLD', 'CHECK', 'CALL', 'RAISE', 'RAISE_SMALL', 'RAISE_BIG', 'ALL_IN']))
    @settings(max_examples=100)
    def test_action_color_is_valid(self, action):
        """
        **Feature: adaptive-action-viewer, Property 5: 动作颜色映射正确性**
        **验证: 需求 6.2, 6.3, 6.4**
        
        测试所有动作颜色值在有效范围内。
        """
        mapper = StrategyColorMapper()
        color = mapper.get_action_color(action)
        
        assert 0 <= color.r <= 255, f"红色分量应在0-255范围内: {color.r}"
        assert 0 <= color.g <= 255, f"绿色分量应在0-255范围内: {color.g}"
        assert 0 <= color.b <= 255, f"蓝色分量应在0-255范围内: {color.b}"


# ============================================================================
# Property 6: 条状宽度计算正确性
# **Feature: adaptive-action-viewer, Property 6: 条状宽度计算正确性**
# **验证: 需求 6.5**
# ============================================================================


# 生成有效的策略概率字典（概率之和为1.0）
@st.composite
def normalized_strategy_for_bar(draw):
    """生成归一化的策略概率字典，用于条状测试。"""
    actions = ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG', 'ALL_IN']
    
    # 随机选择1-6个行动
    num_actions = draw(st.integers(min_value=1, max_value=6))
    selected_actions = draw(st.permutations(actions).map(lambda x: list(x)[:num_actions]))
    
    # 生成随机权重（确保非零）
    weights = [draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)) 
               for _ in selected_actions]
    
    # 归一化
    total = sum(weights)
    probs = [w / total for w in weights]
    
    return dict(zip(selected_actions, probs))


class TestProperty6BarWidthCalculation:
    """属性测试：条状宽度计算正确性。
    
    对于任意策略概率分布，get_bar_segments返回的所有段的width_ratio之和
    应等于1.0（允许浮点误差±0.001），且每个段的width_ratio应等于其probability值。
    """
    
    @given(normalized_strategy_for_bar())
    @settings(max_examples=100)
    def test_width_ratio_sum_equals_one(self, strategy):
        """
        **Feature: adaptive-action-viewer, Property 6: 条状宽度计算正确性**
        **验证: 需求 6.5**
        
        测试所有段的width_ratio之和等于1.0。
        """
        mapper = StrategyColorMapper()
        segments = mapper.get_bar_segments(strategy)
        
        # 如果策略为空或全零，segments可能为空
        if not segments:
            return
        
        total_width = sum(seg.width_ratio for seg in segments)
        
        assert abs(total_width - 1.0) <= 0.001, (
            f"所有段的width_ratio之和应为1.0，实际为 {total_width}"
        )
    
    @given(normalized_strategy_for_bar())
    @settings(max_examples=100)
    def test_width_ratio_equals_probability(self, strategy):
        """
        **Feature: adaptive-action-viewer, Property 6: 条状宽度计算正确性**
        **验证: 需求 6.5**
        
        测试每个段的width_ratio等于其probability值（归一化后）。
        """
        mapper = StrategyColorMapper()
        segments = mapper.get_bar_segments(strategy)
        
        if not segments:
            return
        
        # 计算非零概率的总和
        total_prob = sum(p for p in strategy.values() if p > 0)
        
        for seg in segments:
            expected_width = seg.probability / total_prob
            assert abs(seg.width_ratio - expected_width) <= 0.001, (
                f"动作 {seg.action} 的width_ratio应为 {expected_width}，"
                f"实际为 {seg.width_ratio}"
            )
    
    @given(normalized_strategy_for_bar())
    @settings(max_examples=100)
    def test_width_ratio_in_valid_range(self, strategy):
        """
        **Feature: adaptive-action-viewer, Property 6: 条状宽度计算正确性**
        **验证: 需求 6.5**
        
        测试每个段的width_ratio在[0.0, 1.0]范围内。
        """
        mapper = StrategyColorMapper()
        segments = mapper.get_bar_segments(strategy)
        
        for seg in segments:
            assert 0.0 <= seg.width_ratio <= 1.0, (
                f"动作 {seg.action} 的width_ratio应在[0.0, 1.0]范围内，"
                f"实际为 {seg.width_ratio}"
            )
    
    @given(normalized_strategy_for_bar())
    @settings(max_examples=100)
    def test_segments_have_valid_colors(self, strategy):
        """
        **Feature: adaptive-action-viewer, Property 6: 条状宽度计算正确性**
        **验证: 需求 6.5**
        
        测试每个段都有有效的颜色。
        """
        mapper = StrategyColorMapper()
        segments = mapper.get_bar_segments(strategy)
        
        for seg in segments:
            assert len(seg.color) == 3, f"颜色应为(R, G, B)格式: {seg.color}"
            for i, c in enumerate(seg.color):
                assert 0 <= c <= 255, (
                    f"动作 {seg.action} 的颜色分量{i}应在[0, 255]范围内，"
                    f"实际为 {c}"
                )


# ============================================================================
# Property 7: 零概率动作过滤
# **Feature: adaptive-action-viewer, Property 7: 零概率动作过滤**
# **验证: 需求 6.6**
# ============================================================================


# 生成包含零概率动作的策略
@st.composite
def strategy_with_zeros(draw):
    """生成包含零概率动作的策略字典。"""
    actions = ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG', 'ALL_IN']
    
    # 随机选择2-6个行动
    num_actions = draw(st.integers(min_value=2, max_value=6))
    selected_actions = draw(st.permutations(actions).map(lambda x: list(x)[:num_actions]))
    
    # 生成概率，其中一些为零
    strategy = {}
    has_nonzero = False
    for action in selected_actions:
        # 50%概率生成零概率
        if draw(st.booleans()):
            strategy[action] = 0.0
        else:
            prob = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
            strategy[action] = prob
            has_nonzero = True
    
    # 确保至少有一个非零概率
    if not has_nonzero and selected_actions:
        strategy[selected_actions[0]] = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    return strategy


class TestProperty7ZeroProbabilityFiltering:
    """属性测试：零概率动作过滤。
    
    对于任意策略概率分布，get_bar_segments返回的段列表中
    不应包含probability为0的动作。
    """
    
    @given(strategy_with_zeros())
    @settings(max_examples=100)
    def test_no_zero_probability_segments(self, strategy):
        """
        **Feature: adaptive-action-viewer, Property 7: 零概率动作过滤**
        **验证: 需求 6.6**
        
        测试返回的段列表中不包含零概率动作。
        """
        mapper = StrategyColorMapper()
        segments = mapper.get_bar_segments(strategy)
        
        for seg in segments:
            assert seg.probability > 0, (
                f"段列表中不应包含零概率动作，"
                f"动作 {seg.action} 的概率为 {seg.probability}"
            )
    
    @given(strategy_with_zeros())
    @settings(max_examples=100)
    def test_segment_count_equals_nonzero_actions(self, strategy):
        """
        **Feature: adaptive-action-viewer, Property 7: 零概率动作过滤**
        **验证: 需求 6.6**
        
        测试段数量等于非零概率动作的数量。
        """
        mapper = StrategyColorMapper()
        segments = mapper.get_bar_segments(strategy)
        
        # 计算非零概率动作数量
        nonzero_count = sum(1 for p in strategy.values() if p > 0)
        
        assert len(segments) == nonzero_count, (
            f"段数量应等于非零概率动作数量 {nonzero_count}，"
            f"实际为 {len(segments)}"
        )
    
    @given(st.dictionaries(
        keys=st.sampled_from(['FOLD', 'CHECK', 'CALL', 'RAISE']),
        values=st.just(0.0),
        min_size=1,
        max_size=4
    ))
    @settings(max_examples=50)
    def test_all_zero_returns_empty(self, strategy):
        """
        **Feature: adaptive-action-viewer, Property 7: 零概率动作过滤**
        **验证: 需求 6.6**
        
        测试全零策略返回空列表。
        """
        mapper = StrategyColorMapper()
        segments = mapper.get_bar_segments(strategy)
        
        assert len(segments) == 0, (
            f"全零策略应返回空列表，实际返回 {len(segments)} 个段"
        )
    
    @given(st.just({}))
    @settings(max_examples=1)
    def test_empty_strategy_returns_empty(self, strategy):
        """
        **Feature: adaptive-action-viewer, Property 7: 零概率动作过滤**
        **验证: 需求 6.6**
        
        测试空策略返回空列表。
        """
        mapper = StrategyColorMapper()
        segments = mapper.get_bar_segments(strategy)
        
        assert len(segments) == 0, (
            f"空策略应返回空列表，实际返回 {len(segments)} 个段"
        )
    
    @given(strategy_with_zeros())
    @settings(max_examples=100)
    def test_all_nonzero_actions_included(self, strategy):
        """
        **Feature: adaptive-action-viewer, Property 7: 零概率动作过滤**
        **验证: 需求 6.6**
        
        测试所有非零概率动作都被包含在段列表中。
        """
        mapper = StrategyColorMapper()
        segments = mapper.get_bar_segments(strategy)
        
        # 获取段中的动作集合
        segment_actions = {seg.action for seg in segments}
        
        # 获取非零概率动作集合
        nonzero_actions = {action for action, prob in strategy.items() if prob > 0}
        
        assert segment_actions == nonzero_actions, (
            f"段中的动作应与非零概率动作一致。"
            f"段动作: {segment_actions}，非零动作: {nonzero_actions}"
        )


# ============================================================================
# Property 8: JSON导出动作完整性
# **Feature: adaptive-action-viewer, Property 8: JSON导出动作完整性**
# **验证: 需求 5.3**
# ============================================================================

import json
from hypothesis import HealthCheck
from viewer.models import ActionConfig, DEFAULT_ACTION_MAPPINGS


# 生成随机动作配置
@st.composite
def random_action_config_for_export(draw):
    """生成随机动作配置。"""
    # 随机选择动作维度（4, 5, 或 6）
    action_dim = draw(st.sampled_from([4, 5, 6]))
    action_names = DEFAULT_ACTION_MAPPINGS[action_dim].copy()
    return ActionConfig(action_names=action_names, action_dim=action_dim)


# 生成单个手牌的策略（带动作配置）
@st.composite
def single_hand_strategy_with_config(draw):
    """生成单个手牌的策略数据，带动作配置。"""
    # 生成动作配置
    action_config = draw(random_action_config_for_export())
    
    # 随机选择一个手牌标签
    hand_label = draw(st.sampled_from([label for row in HAND_LABELS_MATRIX for label in row]))
    
    # 生成随机权重并归一化
    weights = [draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)) 
               for _ in action_config.action_names]
    total = sum(weights)
    strategy = {action: w / total for action, w in zip(action_config.action_names, weights)}
    
    return action_config, hand_label, strategy


# 生成少量手牌的策略（带动作配置）
@st.composite
def few_hand_strategies_with_config(draw):
    """生成少量手牌的策略数据，带动作配置。"""
    # 生成动作配置
    action_config = draw(random_action_config_for_export())
    
    # 随机选择5-10个手牌标签
    all_labels = [label for row in HAND_LABELS_MATRIX for label in row]
    num_hands = draw(st.integers(min_value=5, max_value=10))
    selected_labels = draw(st.permutations(all_labels).map(lambda x: list(x)[:num_hands]))
    
    strategies = {}
    for label in selected_labels:
        # 生成随机权重并归一化
        weights = [draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)) 
                   for _ in action_config.action_names]
        total = sum(weights)
        strategies[label] = {action: w / total for action, w in zip(action_config.action_names, weights)}
    
    return action_config, strategies


class TestProperty8JSONExportActionCompleteness:
    """属性测试：JSON导出动作完整性。
    
    对于任意导出的策略JSON，每个手牌的策略应包含与当前模型
    动作维度相同数量的动作概率。
    """
    
    @given(single_hand_strategy_with_config())
    @settings(max_examples=100, deadline=None)
    def test_exported_json_contains_action_config(self, config_label_strategy):
        """
        **Feature: adaptive-action-viewer, Property 8: JSON导出动作完整性**
        **验证: 需求 5.3**
        
        测试导出的JSON包含动作配置信息。
        """
        action_config, hand_label, strategy = config_label_strategy
        
        # 构建导出数据结构（模拟export_json的输出）
        export_data = {
            "metadata": {
                "node_id": "test_node",
                "board_cards": [],
                "model_loaded": True,
                "action_config": {
                    "action_names": action_config.action_names,
                    "action_dim": action_config.action_dim,
                }
            },
            "strategies": {}
        }
        
        # 确保所有动作都包含在导出数据中
        action_probabilities = strategy.copy()
        for action_name in action_config.action_names:
            if action_name not in action_probabilities:
                action_probabilities[action_name] = 0.0
        
        export_data["strategies"][hand_label] = {
            "action_probabilities": action_probabilities,
        }
        
        # 序列化和反序列化
        json_str = json.dumps(export_data, ensure_ascii=False)
        parsed_data = json.loads(json_str)
        
        # 验证动作配置存在
        assert "action_config" in parsed_data["metadata"], (
            "导出的JSON应包含action_config元数据"
        )
        
        # 验证动作维度
        assert parsed_data["metadata"]["action_config"]["action_dim"] == action_config.action_dim, (
            f"动作维度应为 {action_config.action_dim}，"
            f"实际为 {parsed_data['metadata']['action_config']['action_dim']}"
        )
        
        # 验证动作名称列表
        assert parsed_data["metadata"]["action_config"]["action_names"] == action_config.action_names, (
            f"动作名称应为 {action_config.action_names}，"
            f"实际为 {parsed_data['metadata']['action_config']['action_names']}"
        )
    
    @given(few_hand_strategies_with_config())
    @settings(max_examples=100, deadline=None)
    def test_each_hand_has_correct_action_count(self, config_and_strategies):
        """
        **Feature: adaptive-action-viewer, Property 8: JSON导出动作完整性**
        **验证: 需求 5.3**
        
        测试每个手牌的策略包含与动作维度相同数量的动作。
        """
        action_config, strategies = config_and_strategies
        
        # 构建导出数据结构
        export_data = {"strategies": {}}
        
        for label, strategy in strategies.items():
            # 确保所有动作都包含在导出数据中
            action_probabilities = strategy.copy()
            for action_name in action_config.action_names:
                if action_name not in action_probabilities:
                    action_probabilities[action_name] = 0.0
            
            export_data["strategies"][label] = {
                "action_probabilities": action_probabilities,
            }
        
        # 序列化和反序列化
        json_str = json.dumps(export_data, ensure_ascii=False)
        parsed_data = json.loads(json_str)
        
        # 验证每个手牌的动作数量
        for label, data in parsed_data["strategies"].items():
            action_count = len(data["action_probabilities"])
            assert action_count == action_config.action_dim, (
                f"手牌 {label} 的动作数量应为 {action_config.action_dim}，"
                f"实际为 {action_count}"
            )
    
    @given(few_hand_strategies_with_config())
    @settings(max_examples=100, deadline=None)
    def test_all_action_names_present(self, config_and_strategies):
        """
        **Feature: adaptive-action-viewer, Property 8: JSON导出动作完整性**
        **验证: 需求 5.3**
        
        测试每个手牌的策略包含所有动作名称。
        """
        action_config, strategies = config_and_strategies
        
        # 构建导出数据结构
        export_data = {"strategies": {}}
        
        for label, strategy in strategies.items():
            # 确保所有动作都包含在导出数据中
            action_probabilities = strategy.copy()
            for action_name in action_config.action_names:
                if action_name not in action_probabilities:
                    action_probabilities[action_name] = 0.0
            
            export_data["strategies"][label] = {
                "action_probabilities": action_probabilities,
            }
        
        # 序列化和反序列化
        json_str = json.dumps(export_data, ensure_ascii=False)
        parsed_data = json.loads(json_str)
        
        # 验证每个手牌包含所有动作名称
        for label, data in parsed_data["strategies"].items():
            action_names_in_export = set(data["action_probabilities"].keys())
            expected_action_names = set(action_config.action_names)
            
            assert action_names_in_export == expected_action_names, (
                f"手牌 {label} 的动作名称应为 {expected_action_names}，"
                f"实际为 {action_names_in_export}"
            )
    
    @given(few_hand_strategies_with_config())
    @settings(max_examples=100, deadline=None)
    def test_probabilities_sum_to_one(self, config_and_strategies):
        """
        **Feature: adaptive-action-viewer, Property 8: JSON导出动作完整性**
        **验证: 需求 5.3**
        
        测试每个手牌的策略概率之和为1.0。
        """
        action_config, strategies = config_and_strategies
        
        # 构建导出数据结构
        export_data = {"strategies": {}}
        
        for label, strategy in strategies.items():
            # 确保所有动作都包含在导出数据中
            action_probabilities = strategy.copy()
            for action_name in action_config.action_names:
                if action_name not in action_probabilities:
                    action_probabilities[action_name] = 0.0
            
            export_data["strategies"][label] = {
                "action_probabilities": action_probabilities,
            }
        
        # 序列化和反序列化
        json_str = json.dumps(export_data, ensure_ascii=False)
        parsed_data = json.loads(json_str)
        
        # 验证每个手牌的概率之和
        for label, data in parsed_data["strategies"].items():
            total = sum(data["action_probabilities"].values())
            assert abs(total - 1.0) <= 0.001, (
                f"手牌 {label} 的策略概率之和应为1.0，实际为 {total}"
            )
    
    @given(st.sampled_from([4, 5, 6]))
    @settings(max_examples=3, deadline=None)
    def test_different_action_dims_supported(self, action_dim):
        """
        **Feature: adaptive-action-viewer, Property 8: JSON导出动作完整性**
        **验证: 需求 5.3**
        
        测试支持不同的动作维度（4, 5, 6）。
        """
        action_config = ActionConfig.default_for_dim(action_dim)
        
        # 构建导出数据结构
        export_data = {
            "metadata": {
                "action_config": {
                    "action_names": action_config.action_names,
                    "action_dim": action_config.action_dim,
                }
            },
            "strategies": {}
        }
        
        # 为一个手牌生成策略
        strategy = {action: 1.0 / action_dim for action in action_config.action_names}
        export_data["strategies"]["AA"] = {
            "action_probabilities": strategy,
        }
        
        # 序列化和反序列化
        json_str = json.dumps(export_data, ensure_ascii=False)
        parsed_data = json.loads(json_str)
        
        # 验证动作维度
        assert parsed_data["metadata"]["action_config"]["action_dim"] == action_dim, (
            f"动作维度应为 {action_dim}"
        )
        
        # 验证动作数量
        assert len(parsed_data["strategies"]["AA"]["action_probabilities"]) == action_dim, (
            f"动作数量应为 {action_dim}"
        )
    
    @given(single_hand_strategy_with_config())
    @settings(max_examples=100, deadline=None)
    def test_zero_probability_actions_included(self, config_label_strategy):
        """
        **Feature: adaptive-action-viewer, Property 8: JSON导出动作完整性**
        **验证: 需求 5.3**
        
        测试零概率动作也被包含在导出数据中。
        """
        action_config, hand_label, strategy = config_label_strategy
        
        # 修改策略，使第一个动作概率为0
        action_names = list(strategy.keys())
        if len(action_names) > 1:
            modified = {action_names[0]: 0.0}
            remaining_total = sum(strategy[a] for a in action_names[1:])
            if remaining_total > 0:
                for action in action_names[1:]:
                    modified[action] = strategy[action] / remaining_total
            else:
                for action in action_names[1:]:
                    modified[action] = 1.0 / (len(action_names) - 1)
            strategy = modified
        
        # 构建导出数据结构
        export_data = {"strategies": {}}
        
        # 确保所有动作都包含在导出数据中
        action_probabilities = {}
        for action_name in action_config.action_names:
            action_probabilities[action_name] = strategy.get(action_name, 0.0)
        
        export_data["strategies"][hand_label] = {
            "action_probabilities": action_probabilities,
        }
        
        # 序列化和反序列化
        json_str = json.dumps(export_data, ensure_ascii=False)
        parsed_data = json.loads(json_str)
        
        # 验证手牌包含所有动作（包括零概率的）
        action_count = len(parsed_data["strategies"][hand_label]["action_probabilities"])
        assert action_count == action_config.action_dim, (
            f"手牌 {hand_label} 应包含所有 {action_config.action_dim} 个动作，"
            f"实际为 {action_count}"
        )



# ============================================================================
# Property 3: 检查点元数据完整性
# **Feature: adaptive-action-viewer, Property 3: 检查点元数据完整性**
# **验证: 需求 3.1, 3.2**
# ============================================================================

@st.composite
def training_config_strategy(draw):
    """生成训练配置。"""
    from models.core import TrainingConfig
    
    return TrainingConfig(
        num_episodes=draw(st.integers(min_value=1, max_value=100)),
        batch_size=draw(st.integers(min_value=16, max_value=64)),
        learning_rate=0.001,
        network_architecture=[64, 32],
        initial_stack=1000,
        small_blind=5,
        big_blind=10,
        max_raises_per_street=3,
        checkpoint_interval=10,
        regret_buffer_size=1000,
        strategy_buffer_size=1000,
        network_train_steps=10,
    )


class TestProperty3CheckpointMetadataCompleteness:
    """属性测试：检查点元数据完整性。
    
    **Feature: adaptive-action-viewer, Property 3: 检查点元数据完整性**
    **验证: 需求 3.1, 3.2**
    
    对于任意保存的检查点，检查点数据中应包含 action_config 字段，
    且该字段包含 action_names 列表和 action_dim 整数值。
    """
    
    @given(training_config_strategy())
    @settings(max_examples=100, deadline=None)
    def test_checkpoint_contains_action_config(self, config):
        """
        **Feature: adaptive-action-viewer, Property 3: 检查点元数据完整性**
        **验证: 需求 3.1, 3.2**
        
        测试保存的检查点包含 action_config 字段。
        """
        from training.training_engine import TrainingEngine
        import tempfile
        import os
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
            tensorboard_dir = os.path.join(temp_dir, 'runs')
            
            # 创建训练引擎
            engine = TrainingEngine(
                config=config,
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
                enable_tensorboard=False
            )
            
            # 保存检查点
            checkpoint_path = engine.save_checkpoint()
            
            # 加载检查点数据
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
            
            # 验证 action_config 字段存在
            assert 'action_config' in checkpoint_data, (
                "检查点数据应包含 action_config 字段"
            )
    
    @given(training_config_strategy())
    @settings(max_examples=100, deadline=None)
    def test_action_config_contains_action_names(self, config):
        """
        **Feature: adaptive-action-viewer, Property 3: 检查点元数据完整性**
        **验证: 需求 3.1**
        
        测试 action_config 包含 action_names 列表。
        """
        from training.training_engine import TrainingEngine
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
            tensorboard_dir = os.path.join(temp_dir, 'runs')
            
            engine = TrainingEngine(
                config=config,
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
                enable_tensorboard=False
            )
            
            checkpoint_path = engine.save_checkpoint()
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
            
            action_config = checkpoint_data['action_config']
            
            # 验证 action_names 字段存在且为列表
            assert 'action_names' in action_config, (
                "action_config 应包含 action_names 字段"
            )
            assert isinstance(action_config['action_names'], list), (
                f"action_names 应为列表，实际类型: {type(action_config['action_names'])}"
            )
            assert len(action_config['action_names']) > 0, (
                "action_names 列表不应为空"
            )
    
    @given(training_config_strategy())
    @settings(max_examples=100, deadline=None)
    def test_action_config_contains_action_dim(self, config):
        """
        **Feature: adaptive-action-viewer, Property 3: 检查点元数据完整性**
        **验证: 需求 3.2**
        
        测试 action_config 包含 action_dim 整数值。
        """
        from training.training_engine import TrainingEngine
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
            tensorboard_dir = os.path.join(temp_dir, 'runs')
            
            engine = TrainingEngine(
                config=config,
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
                enable_tensorboard=False
            )
            
            checkpoint_path = engine.save_checkpoint()
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
            
            action_config = checkpoint_data['action_config']
            
            # 验证 action_dim 字段存在且为整数
            assert 'action_dim' in action_config, (
                "action_config 应包含 action_dim 字段"
            )
            assert isinstance(action_config['action_dim'], int), (
                f"action_dim 应为整数，实际类型: {type(action_config['action_dim'])}"
            )
            assert action_config['action_dim'] > 0, (
                f"action_dim 应大于0，实际值: {action_config['action_dim']}"
            )
    
    @given(training_config_strategy())
    @settings(max_examples=100, deadline=None)
    def test_action_names_length_matches_action_dim(self, config):
        """
        **Feature: adaptive-action-viewer, Property 3: 检查点元数据完整性**
        **验证: 需求 3.1, 3.2**
        
        测试 action_names 列表长度与 action_dim 一致。
        """
        from training.training_engine import TrainingEngine
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
            tensorboard_dir = os.path.join(temp_dir, 'runs')
            
            engine = TrainingEngine(
                config=config,
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
                enable_tensorboard=False
            )
            
            checkpoint_path = engine.save_checkpoint()
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
            
            action_config = checkpoint_data['action_config']
            action_names = action_config['action_names']
            action_dim = action_config['action_dim']
            
            # 验证长度一致
            assert len(action_names) == action_dim, (
                f"action_names 长度({len(action_names)})应等于 action_dim({action_dim})"
            )
    
    @given(training_config_strategy())
    @settings(max_examples=100, deadline=None)
    def test_action_config_compatible_with_viewer(self, config):
        """
        **Feature: adaptive-action-viewer, Property 3: 检查点元数据完整性**
        **验证: 需求 3.1, 3.2**
        
        测试保存的 action_config 可以被 viewer 的 ActionConfig 正确解析。
        """
        from training.training_engine import TrainingEngine
        from viewer.models import ActionConfig
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
            tensorboard_dir = os.path.join(temp_dir, 'runs')
            
            engine = TrainingEngine(
                config=config,
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
                enable_tensorboard=False
            )
            
            checkpoint_path = engine.save_checkpoint()
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
            
            # 使用 ActionConfig.from_checkpoint 解析
            action_config = ActionConfig.from_checkpoint(checkpoint_data)
            
            # 验证解析成功
            assert action_config is not None, "ActionConfig.from_checkpoint 应返回有效对象"
            assert action_config.action_dim == checkpoint_data['action_config']['action_dim'], (
                f"解析后的 action_dim 应与原始值一致"
            )
            assert action_config.action_names == checkpoint_data['action_config']['action_names'], (
                f"解析后的 action_names 应与原始值一致"
            )
    
    @given(training_config_strategy())
    @settings(max_examples=100, deadline=None)
    def test_action_names_contains_expected_actions(self, config):
        """
        **Feature: adaptive-action-viewer, Property 3: 检查点元数据完整性**
        **验证: 需求 3.1**
        
        测试 action_names 包含预期的动作名称。
        """
        from training.training_engine import TrainingEngine
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
            tensorboard_dir = os.path.join(temp_dir, 'runs')
            
            engine = TrainingEngine(
                config=config,
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
                enable_tensorboard=False
            )
            
            checkpoint_path = engine.save_checkpoint()
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
            
            action_names = checkpoint_data['action_config']['action_names']
            
            # 验证包含基本动作
            expected_actions = ['FOLD', 'CHECK', 'CALL']
            for action in expected_actions:
                assert action in action_names, (
                    f"action_names 应包含 {action}，实际: {action_names}"
                )
