"""CardAbstraction和AbstractionCache的属性测试。

本模块包含CardAbstraction和AbstractionCache的属性测试，验证：
- 属性45：抽象状态等价性
- 属性52：抽象结果持久化往返一致性
- 属性53：桶ID查询O(1)时间复杂度
- 属性55：抽象元数据完整性
"""

import os
import time
import tempfile
import shutil
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from models.core import Card
from abstraction import (
    CardAbstraction,
    AbstractionCache,
    AbstractionConfig,
    AbstractionResult,
    get_canonical_hand_index,
)


# ============================================================================
# 测试策略（生成器）
# ============================================================================

@st.composite
def small_abstraction_config_strategy(draw):
    """生成小规模的抽象配置（用于快速测试）。"""
    flop_buckets = draw(st.integers(min_value=3, max_value=20))
    turn_buckets = draw(st.integers(min_value=3, max_value=20))
    river_buckets = draw(st.integers(min_value=3, max_value=20))
    
    return AbstractionConfig(
        flop_buckets=flop_buckets,
        turn_buckets=turn_buckets,
        river_buckets=river_buckets,
        kmeans_restarts=2,
        kmeans_max_iters=10,
        random_seed=draw(st.integers(min_value=0, max_value=10000))
    )


@st.composite
def card_strategy(draw):
    """生成单张扑克牌。"""
    rank = draw(st.integers(min_value=2, max_value=14))
    suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
    return Card(rank=rank, suit=suit)


@st.composite
def unique_cards_strategy(draw, count: int):
    """生成指定数量的不重复扑克牌。"""
    cards = []
    used = set()
    
    while len(cards) < count:
        rank = draw(st.integers(min_value=2, max_value=14))
        suit = draw(st.sampled_from(['h', 'd', 'c', 's']))
        key = (rank, suit)
        if key not in used:
            used.add(key)
            cards.append(Card(rank=rank, suit=suit))
    
    return cards


@st.composite
def hole_cards_strategy(draw):
    """生成手牌（两张不重复的牌）。"""
    cards = draw(unique_cards_strategy(2))
    return (cards[0], cards[1])


@st.composite
def hole_and_community_strategy(draw, num_community: int):
    """生成手牌和公共牌组合。"""
    total_cards = 2 + num_community
    cards = draw(unique_cards_strategy(total_cards))
    hole_cards = (cards[0], cards[1])
    community_cards = cards[2:]
    return hole_cards, community_cards


# ============================================================================
# 属性测试
# ============================================================================

class TestAbstractionStateEquivalence:
    """属性45：抽象状态等价性测试。
    
    Feature: texas-holdem-ai-training, Property 45: 抽象状态等价性
    验证需求：11.6
    """
    
    @given(config=small_abstraction_config_strategy())
    @settings(max_examples=20, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_same_bucket_returns_same_id(self, config):
        """同一桶内的手牌应该返回相同的桶ID。
        
        Feature: texas-holdem-ai-training, Property 45: 抽象状态等价性
        验证需求：11.6
        """
        # 创建抽象管理器并生成抽象
        abstraction = CardAbstraction(config)
        result = abstraction.generate_abstraction()
        
        # 验证翻牌前：相同规范化索引的手牌应该返回相同桶ID
        # 测试几个等价的手牌对（花色同构性）
        test_cases = [
            # AhKh 和 AsKs 是等价的（同花AK）
            ((Card(14, 'h'), Card(13, 'h')), (Card(14, 's'), Card(13, 's'))),
            # AhKs 和 AsKh 是等价的（异花AK）
            ((Card(14, 'h'), Card(13, 's')), (Card(14, 's'), Card(13, 'h'))),
            # 对子：AhAs 和 AcAd 是等价的
            ((Card(14, 'h'), Card(14, 's')), (Card(14, 'c'), Card(14, 'd'))),
        ]
        
        for hand1, hand2 in test_cases:
            bucket1 = abstraction.get_bucket_id(hand1, [])
            bucket2 = abstraction.get_bucket_id(hand2, [])
            
            # 规范化索引应该相同
            idx1 = get_canonical_hand_index(hand1)
            idx2 = get_canonical_hand_index(hand2)
            assert idx1 == idx2, f"规范化索引不同：{idx1} vs {idx2}"
            
            # 桶ID应该相同
            assert bucket1 == bucket2, \
                f"等价手牌的桶ID不同：{bucket1} vs {bucket2}"
    
    @given(hole_cards=hole_cards_strategy())
    @settings(max_examples=50, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_preflop_bucket_consistency(self, hole_cards):
        """翻牌前桶ID查询应该一致。
        
        Feature: texas-holdem-ai-training, Property 45: 抽象状态等价性
        验证需求：11.6
        """
        config = AbstractionConfig(
            flop_buckets=10,
            turn_buckets=10,
            river_buckets=10,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstraction = CardAbstraction(config)
        abstraction.generate_abstraction()
        
        # 多次查询同一手牌，应该返回相同的桶ID
        bucket1 = abstraction.get_bucket_id(hole_cards, [])
        bucket2 = abstraction.get_bucket_id(hole_cards, [])
        bucket3 = abstraction.get_bucket_id(hole_cards, [])
        
        assert bucket1 == bucket2 == bucket3, \
            f"同一手牌的桶ID不一致：{bucket1}, {bucket2}, {bucket3}"


class TestAbstractionPersistenceRoundTrip:
    """属性52：抽象结果持久化往返一致性测试。
    
    Feature: texas-holdem-ai-training, Property 52: 抽象结果持久化往返一致性
    验证需求：14.1, 14.2
    """
    
    @given(config=small_abstraction_config_strategy())
    @settings(max_examples=15, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_save_load_roundtrip(self, config):
        """保存后加载的抽象应该与原始抽象等价。
        
        Feature: texas-holdem-ai-training, Property 52: 抽象结果持久化往返一致性
        验证需求：14.1, 14.2
        """
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 生成抽象
            abstraction1 = CardAbstraction(config)
            result1 = abstraction1.generate_abstraction()
            
            # 保存
            save_path = os.path.join(temp_dir, 'test_abstraction')
            abstraction1.save(save_path)
            
            # 加载
            abstraction2 = CardAbstraction()
            result2 = abstraction2.load(save_path)
            
            # 验证配置一致
            assert result1.config.flop_buckets == result2.config.flop_buckets
            assert result1.config.turn_buckets == result2.config.turn_buckets
            assert result1.config.river_buckets == result2.config.river_buckets
            assert result1.config.random_seed == result2.config.random_seed
            
            # 验证映射一致
            assert np.array_equal(result1.preflop_mapping, result2.preflop_mapping)
            assert np.array_equal(result1.flop_mapping, result2.flop_mapping)
            assert np.array_equal(result1.turn_mapping, result2.turn_mapping)
            assert np.array_equal(result1.river_mapping, result2.river_mapping)
            
            # 验证WCSS一致
            assert result1.wcss == result2.wcss
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir)
    
    @given(
        config=small_abstraction_config_strategy(),
        hole_cards=hole_cards_strategy()
    )
    @settings(max_examples=15, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_bucket_id_consistent_after_roundtrip(self, config, hole_cards):
        """往返后桶ID查询应该一致。
        
        Feature: texas-holdem-ai-training, Property 52: 抽象结果持久化往返一致性
        验证需求：14.1, 14.2
        """
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 生成抽象
            abstraction1 = CardAbstraction(config)
            abstraction1.generate_abstraction()
            
            # 查询桶ID
            bucket_before = abstraction1.get_bucket_id(hole_cards, [])
            
            # 保存和加载
            save_path = os.path.join(temp_dir, 'test_abstraction')
            abstraction1.save(save_path)
            
            abstraction2 = CardAbstraction()
            abstraction2.load(save_path)
            
            # 再次查询桶ID
            bucket_after = abstraction2.get_bucket_id(hole_cards, [])
            
            # 验证一致性
            assert bucket_before == bucket_after, \
                f"往返后桶ID不一致：{bucket_before} vs {bucket_after}"
            
        finally:
            shutil.rmtree(temp_dir)


class TestBucketQueryTimeComplexity:
    """属性53：桶ID查询O(1)时间复杂度测试。
    
    Feature: texas-holdem-ai-training, Property 53: 桶ID查询O(1)时间复杂度
    验证需求：14.3
    """
    
    @given(hole_cards=hole_cards_strategy())
    @settings(max_examples=30, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_query_time_is_constant(self, hole_cards):
        """桶ID查询时间应该是常数级别。
        
        Feature: texas-holdem-ai-training, Property 53: 桶ID查询O(1)时间复杂度
        验证需求：14.3
        """
        config = AbstractionConfig(
            flop_buckets=100,
            turn_buckets=100,
            river_buckets=100,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstraction = CardAbstraction(config)
        abstraction.generate_abstraction()
        
        # 预热
        for _ in range(10):
            abstraction.get_bucket_id(hole_cards, [])
        
        # 测量单次查询时间
        num_queries = 100
        start_time = time.perf_counter()
        for _ in range(num_queries):
            abstraction.get_bucket_id(hole_cards, [])
        elapsed = time.perf_counter() - start_time
        
        avg_time = elapsed / num_queries
        
        # 验证平均查询时间小于1毫秒（O(1)操作应该非常快）
        assert avg_time < 0.001, \
            f"平均查询时间过长：{avg_time * 1000:.3f}ms，应该小于1ms"
    
    @given(
        hole_and_flop=hole_and_community_strategy(3),
        hole_and_turn=hole_and_community_strategy(4),
        hole_and_river=hole_and_community_strategy(5)
    )
    @settings(max_examples=20, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_cache_query_time_is_constant(self, hole_and_flop, hole_and_turn, 
                                          hole_and_river):
        """AbstractionCache的查询时间应该是常数级别。
        
        Feature: texas-holdem-ai-training, Property 53: 桶ID查询O(1)时间复杂度
        验证需求：14.3
        """
        config = AbstractionConfig(
            flop_buckets=50,
            turn_buckets=50,
            river_buckets=50,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstraction = CardAbstraction(config)
        result = abstraction.generate_abstraction()
        
        # 创建缓存
        cache = AbstractionCache(result)
        
        # 测试不同阶段的查询时间
        test_cases = [
            (hole_and_flop[0], []),  # 翻牌前
            (hole_and_flop[0], hole_and_flop[1]),  # 翻牌
            (hole_and_turn[0], hole_and_turn[1]),  # 转牌
            (hole_and_river[0], hole_and_river[1]),  # 河牌
        ]
        
        for hole_cards, community_cards in test_cases:
            # 预热
            for _ in range(5):
                cache.get_bucket(hole_cards, community_cards)
            
            # 测量
            num_queries = 50
            start_time = time.perf_counter()
            for _ in range(num_queries):
                cache.get_bucket(hole_cards, community_cards)
            elapsed = time.perf_counter() - start_time
            
            avg_time = elapsed / num_queries
            
            # 验证平均查询时间小于1毫秒
            assert avg_time < 0.001, \
                f"缓存查询时间过长：{avg_time * 1000:.3f}ms"


class TestAbstractionMetadataCompleteness:
    """属性55：抽象元数据完整性测试。
    
    Feature: texas-holdem-ai-training, Property 55: 抽象元数据完整性
    验证需求：14.5
    """
    
    @given(config=small_abstraction_config_strategy())
    @settings(max_examples=15, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_saved_file_contains_complete_metadata(self, config):
        """保存的文件应该包含完整的配置元数据。
        
        Feature: texas-holdem-ai-training, Property 55: 抽象元数据完整性
        验证需求：14.5
        """
        import json
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 生成并保存抽象
            abstraction = CardAbstraction(config)
            abstraction.generate_abstraction()
            
            save_path = os.path.join(temp_dir, 'test_abstraction')
            abstraction.save(save_path)
            
            # 读取元数据文件
            metadata_path = os.path.join(save_path, 'abstraction.json')
            assert os.path.exists(metadata_path), "元数据文件应该存在"
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # 验证配置字段完整性
            assert 'config' in metadata, "元数据应该包含config字段"
            config_data = metadata['config']
            
            required_config_fields = [
                'preflop_buckets',
                'flop_buckets',
                'turn_buckets',
                'river_buckets',
                'equity_bins',
                'kmeans_restarts',
                'kmeans_max_iters',
                'use_potential_aware',
                'random_seed',
            ]
            
            for field in required_config_fields:
                assert field in config_data, f"配置应该包含{field}字段"
            
            # 验证其他元数据字段
            assert 'wcss' in metadata, "元数据应该包含wcss字段"
            assert 'generation_time' in metadata, "元数据应该包含generation_time字段"
            assert 'mappings' in metadata, "元数据应该包含mappings字段"
            assert 'centers' in metadata, "元数据应该包含centers字段"
            
            # 验证映射文件引用
            mappings = metadata['mappings']
            assert 'preflop' in mappings
            assert 'flop' in mappings
            assert 'turn' in mappings
            assert 'river' in mappings
            
        finally:
            shutil.rmtree(temp_dir)
    
    @given(config=small_abstraction_config_strategy())
    @settings(max_examples=15, deadline=None,
              suppress_health_check=[HealthCheck.large_base_example])
    def test_stats_contain_all_stages(self, config):
        """统计信息应该包含所有游戏阶段。
        
        Feature: texas-holdem-ai-training, Property 55: 抽象元数据完整性
        验证需求：14.5
        """
        abstraction = CardAbstraction(config)
        abstraction.generate_abstraction()
        
        stats = abstraction.get_abstraction_stats()
        
        # 验证统计信息结构
        assert 'config' in stats
        assert 'generation_time' in stats
        assert 'wcss' in stats
        assert 'stages' in stats
        
        # 验证所有阶段都有统计
        stages = stats['stages']
        for stage in ['preflop', 'flop', 'turn', 'river']:
            assert stage in stages, f"统计应该包含{stage}阶段"
            stage_stats = stages[stage]
            assert 'count' in stage_stats
            assert 'avg_size' in stage_stats
            assert 'max_size' in stage_stats
            assert 'min_size' in stage_stats


# ============================================================================
# 单元测试
# ============================================================================

class TestCardAbstractionBasic:
    """CardAbstraction基本功能测试。"""
    
    def test_initialization_with_default_config(self):
        """使用默认配置初始化应该成功。"""
        abstraction = CardAbstraction()
        assert abstraction.config is not None
        assert abstraction.result is None
    
    def test_initialization_with_custom_config(self):
        """使用自定义配置初始化应该成功。"""
        config = AbstractionConfig(
            flop_buckets=100,
            turn_buckets=100,
            river_buckets=100
        )
        abstraction = CardAbstraction(config)
        assert abstraction.config == config
    
    def test_generate_abstraction(self):
        """生成抽象应该成功。"""
        config = AbstractionConfig(
            flop_buckets=5,
            turn_buckets=5,
            river_buckets=5,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstraction = CardAbstraction(config)
        result = abstraction.generate_abstraction()
        
        assert result is not None
        assert result.is_complete()
        assert abstraction.is_loaded()
    
    def test_get_bucket_id_preflop(self):
        """翻牌前桶ID查询应该成功。"""
        config = AbstractionConfig(
            flop_buckets=5,
            turn_buckets=5,
            river_buckets=5,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstraction = CardAbstraction(config)
        abstraction.generate_abstraction()
        
        hole_cards = (Card(14, 'h'), Card(13, 'h'))  # AhKh
        bucket_id = abstraction.get_bucket_id(hole_cards, [])
        
        assert isinstance(bucket_id, int)
        assert bucket_id >= 0
    
    def test_get_bucket_id_flop(self):
        """翻牌阶段桶ID查询应该成功。"""
        config = AbstractionConfig(
            flop_buckets=5,
            turn_buckets=5,
            river_buckets=5,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstraction = CardAbstraction(config)
        abstraction.generate_abstraction()
        
        hole_cards = (Card(14, 'h'), Card(13, 'h'))
        community_cards = [Card(12, 's'), Card(11, 'd'), Card(10, 'c')]
        bucket_id = abstraction.get_bucket_id(hole_cards, community_cards)
        
        assert isinstance(bucket_id, int)
        assert bucket_id >= 0
    
    def test_get_bucket_id_without_abstraction_raises(self):
        """未生成抽象时查询应该抛出异常。"""
        abstraction = CardAbstraction()
        hole_cards = (Card(14, 'h'), Card(13, 'h'))
        
        with pytest.raises(ValueError):
            abstraction.get_bucket_id(hole_cards, [])
    
    def test_save_without_abstraction_raises(self):
        """未生成抽象时保存应该抛出异常。"""
        abstraction = CardAbstraction()
        
        with pytest.raises(ValueError):
            abstraction.save('/tmp/test')
    
    def test_load_nonexistent_raises(self):
        """加载不存在的文件应该抛出异常。"""
        abstraction = CardAbstraction()
        
        with pytest.raises(FileNotFoundError):
            abstraction.load('/nonexistent/path')
    
    def test_config_matches(self):
        """配置匹配检查应该正确。"""
        config1 = AbstractionConfig(
            flop_buckets=10,
            turn_buckets=10,
            river_buckets=10,
            random_seed=42
        )
        config2 = AbstractionConfig(
            flop_buckets=10,
            turn_buckets=10,
            river_buckets=10,
            random_seed=42
        )
        config3 = AbstractionConfig(
            flop_buckets=20,  # 不同
            turn_buckets=10,
            river_buckets=10,
            random_seed=42
        )
        
        abstraction = CardAbstraction(config1)
        abstraction.generate_abstraction()
        
        assert abstraction.config_matches(config2)
        assert not abstraction.config_matches(config3)


class TestAbstractionCacheBasic:
    """AbstractionCache基本功能测试。"""
    
    def test_initialization_empty(self):
        """空初始化应该成功。"""
        cache = AbstractionCache()
        assert not cache.is_loaded()
    
    def test_initialization_with_result(self):
        """使用结果初始化应该成功。"""
        config = AbstractionConfig(
            flop_buckets=5,
            turn_buckets=5,
            river_buckets=5,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstraction = CardAbstraction(config)
        result = abstraction.generate_abstraction()
        
        cache = AbstractionCache(result)
        assert cache.is_loaded()
    
    def test_get_bucket_preflop(self):
        """翻牌前桶查询应该成功。"""
        config = AbstractionConfig(
            flop_buckets=5,
            turn_buckets=5,
            river_buckets=5,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstraction = CardAbstraction(config)
        result = abstraction.generate_abstraction()
        
        cache = AbstractionCache(result)
        
        hole_cards = (Card(14, 'h'), Card(13, 'h'))
        bucket_id = cache.get_bucket(hole_cards, [])
        
        assert isinstance(bucket_id, int)
        assert bucket_id >= 0
    
    def test_get_canonical_hand(self):
        """规范化手牌应该正确。"""
        # 同花AK
        hand1 = (Card(14, 'h'), Card(13, 'h'))
        hand2 = (Card(14, 's'), Card(13, 's'))
        
        idx1 = AbstractionCache.get_canonical_hand(hand1)
        idx2 = AbstractionCache.get_canonical_hand(hand2)
        
        assert idx1 == idx2, "同花AK应该有相同的规范化索引"
        
        # 异花AK
        hand3 = (Card(14, 'h'), Card(13, 's'))
        hand4 = (Card(14, 's'), Card(13, 'h'))
        
        idx3 = AbstractionCache.get_canonical_hand(hand3)
        idx4 = AbstractionCache.get_canonical_hand(hand4)
        
        assert idx3 == idx4, "异花AK应该有相同的规范化索引"
        
        # 同花和异花应该不同
        assert idx1 != idx3, "同花AK和异花AK应该有不同的规范化索引"
    
    def test_cache_stats(self):
        """缓存统计应该正确。"""
        config = AbstractionConfig(
            flop_buckets=5,
            turn_buckets=5,
            river_buckets=5,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstraction = CardAbstraction(config)
        result = abstraction.generate_abstraction()
        
        cache = AbstractionCache(result)
        stats = cache.get_cache_stats()
        
        assert 'preflop_size' in stats
        assert stats['preflop_size'] > 0
    
    def test_clear_cache(self):
        """清除缓存应该成功。"""
        config = AbstractionConfig(
            flop_buckets=5,
            turn_buckets=5,
            river_buckets=5,
            kmeans_restarts=1,
            kmeans_max_iters=5
        )
        abstraction = CardAbstraction(config)
        result = abstraction.generate_abstraction()
        
        cache = AbstractionCache(result)
        assert cache.is_loaded()
        
        cache.clear()
        assert not cache.is_loaded()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
