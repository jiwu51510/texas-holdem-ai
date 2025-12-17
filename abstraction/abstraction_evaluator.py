"""抽象质量评估器模块。

本模块实现了AbstractionEvaluator类，用于评估卡牌抽象的质量。
提供多种质量指标计算和比较功能。

主要功能：
- calculate_wcss: 计算Within-Cluster Sum of Squares
- get_bucket_size_distribution: 获取桶大小分布统计
- compare_abstractions: 比较不同抽象配置
- generate_report: 生成抽象质量报告

需求：15.1, 15.2, 15.3, 15.4
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from abstraction.data_classes import AbstractionConfig, AbstractionResult
from abstraction.emd_calculator import EMDCalculator


@dataclass
class BucketSizeStats:
    """桶大小统计数据类。
    
    Attributes:
        count: 桶数量
        avg_size: 平均桶大小
        max_size: 最大桶大小
        min_size: 最小桶大小
        std_size: 桶大小标准差
        size_distribution: 桶大小分布（桶大小 -> 数量）
    """
    count: int
    avg_size: float
    max_size: int
    min_size: int
    std_size: float
    size_distribution: Dict[int, int]


@dataclass
class AbstractionQualityReport:
    """抽象质量报告数据类。
    
    Attributes:
        config: 抽象配置
        wcss: 每个阶段的WCSS
        bucket_stats: 每个阶段的桶大小统计
        generation_time: 生成耗时
        total_buckets: 总桶数
        compression_ratio: 压缩比（原始状态数 / 桶数）
    """
    config: AbstractionConfig
    wcss: Dict[str, float]
    bucket_stats: Dict[str, BucketSizeStats]
    generation_time: float
    total_buckets: int
    compression_ratio: float


class AbstractionEvaluator:
    """抽象质量评估器。
    
    提供卡牌抽象质量评估功能，包括WCSS计算、桶大小分布统计、
    多配置比较和质量报告生成。
    
    Attributes:
        emd_calculator: EMD计算器实例
    """
    
    def __init__(self):
        """初始化抽象评估器。"""
        self.emd_calculator = EMDCalculator()
    
    def calculate_wcss(self, 
                       data_points: np.ndarray,
                       labels: np.ndarray,
                       centers: np.ndarray,
                       ground_distances: Optional[np.ndarray] = None) -> float:
        """计算Within-Cluster Sum of Squares (WCSS)。
        
        WCSS是所有数据点到其聚类中心距离平方和，用于衡量聚类质量。
        WCSS越小表示聚类越紧凑。
        
        计算公式：WCSS = Σ_i d(x_i, c_{l_i})^2
        其中x_i是数据点，c_{l_i}是x_i所属聚类的中心，d是距离函数。
        
        Args:
            data_points: 数据点数组，形状为 (n_samples, n_features)
            labels: 聚类标签数组，形状为 (n_samples,)
            centers: 聚类中心数组，形状为 (n_clusters, n_features)
            ground_distances: 可选的地面距离矩阵（用于EMD计算）
            
        Returns:
            WCSS值（非负浮点数）
            
        Raises:
            ValueError: 如果输入维度不匹配
        """
        if len(data_points) == 0:
            return 0.0
        
        if len(data_points) != len(labels):
            raise ValueError(
                f"数据点数量({len(data_points)})与标签数量({len(labels)})不匹配"
            )
        
        wcss = 0.0
        
        for i in range(len(data_points)):
            cluster_idx = labels[i]
            
            if cluster_idx < 0 or cluster_idx >= len(centers):
                raise ValueError(
                    f"无效的聚类标签 {cluster_idx}，有效范围为 [0, {len(centers)-1}]"
                )
            
            # 计算数据点到聚类中心的距离
            if ground_distances is not None:
                # 使用带地面距离的EMD
                dist = self.emd_calculator.calculate_emd_with_ground_distance(
                    data_points[i], centers[cluster_idx], ground_distances
                )
            else:
                # 使用一维EMD
                dist = self.emd_calculator.calculate_emd_1d(
                    data_points[i], centers[cluster_idx]
                )
            
            wcss += dist ** 2
        
        return wcss
    
    def calculate_wcss_from_result(self, 
                                   result: AbstractionResult,
                                   stage: str,
                                   data_points: Optional[np.ndarray] = None
                                   ) -> float:
        """从抽象结果计算指定阶段的WCSS。
        
        如果抽象结果中已有WCSS值，直接返回；否则重新计算。
        
        Args:
            result: 抽象结果
            stage: 游戏阶段（'flop', 'turn', 'river'）
            data_points: 可选的数据点（如果需要重新计算）
            
        Returns:
            WCSS值
        """
        # 如果结果中已有WCSS，直接返回
        if stage in result.wcss and result.wcss[stage] > 0:
            return result.wcss[stage]
        
        # 否则返回0（需要数据点才能重新计算）
        return 0.0
    
    def get_bucket_size_distribution(self, 
                                     mapping: np.ndarray) -> BucketSizeStats:
        """获取桶大小分布统计。
        
        分析映射数组，计算桶数量、平均大小、最大/最小大小等统计信息。
        
        Args:
            mapping: 桶映射数组（元素值为桶ID）
            
        Returns:
            BucketSizeStats实例，包含完整的桶大小统计
        """
        if mapping is None or len(mapping) == 0:
            return BucketSizeStats(
                count=0,
                avg_size=0.0,
                max_size=0,
                min_size=0,
                std_size=0.0,
                size_distribution={}
            )
        
        # 统计每个桶的大小
        unique_buckets, bucket_counts = np.unique(mapping, return_counts=True)
        
        # 计算统计量
        count = len(unique_buckets)
        avg_size = float(np.mean(bucket_counts))
        max_size = int(np.max(bucket_counts))
        min_size = int(np.min(bucket_counts))
        std_size = float(np.std(bucket_counts))
        
        # 构建大小分布字典
        size_distribution = {}
        for bucket_id, bucket_count in zip(unique_buckets, bucket_counts):
            size_distribution[int(bucket_id)] = int(bucket_count)
        
        return BucketSizeStats(
            count=count,
            avg_size=avg_size,
            max_size=max_size,
            min_size=min_size,
            std_size=std_size,
            size_distribution=size_distribution
        )
    
    def get_all_bucket_stats(self, 
                             result: AbstractionResult) -> Dict[str, BucketSizeStats]:
        """获取所有阶段的桶大小统计。
        
        Args:
            result: 抽象结果
            
        Returns:
            字典，键为阶段名称，值为BucketSizeStats
        """
        stats = {}
        
        # 翻牌前
        if result.preflop_mapping is not None:
            stats['preflop'] = self.get_bucket_size_distribution(result.preflop_mapping)
        
        # 翻牌
        if result.flop_mapping is not None:
            stats['flop'] = self.get_bucket_size_distribution(result.flop_mapping)
        
        # 转牌
        if result.turn_mapping is not None:
            stats['turn'] = self.get_bucket_size_distribution(result.turn_mapping)
        
        # 河牌
        if result.river_mapping is not None:
            stats['river'] = self.get_bucket_size_distribution(result.river_mapping)
        
        return stats
    
    def compare_abstractions(self, 
                             results: List[AbstractionResult],
                             names: Optional[List[str]] = None
                             ) -> Dict[str, Any]:
        """比较不同抽象配置的质量。
        
        对多个抽象结果进行比较，生成比较报告。
        
        Args:
            results: 抽象结果列表
            names: 可选的抽象名称列表（用于标识）
            
        Returns:
            比较报告字典，包含每个抽象的质量指标和排名
        """
        if not results:
            return {'error': '没有提供抽象结果'}
        
        if names is None:
            names = [f'abstraction_{i}' for i in range(len(results))]
        
        if len(names) != len(results):
            raise ValueError(
                f"名称数量({len(names)})与结果数量({len(results)})不匹配"
            )
        
        comparison = {
            'abstractions': [],
            'rankings': {},
            'summary': {}
        }
        
        # 收集每个抽象的指标
        for name, result in zip(names, results):
            bucket_stats = self.get_all_bucket_stats(result)
            
            abstraction_info = {
                'name': name,
                'config': result.config.to_dict(),
                'wcss': result.wcss,
                'generation_time': result.generation_time,
                'bucket_stats': {
                    stage: {
                        'count': stats.count,
                        'avg_size': stats.avg_size,
                        'max_size': stats.max_size,
                        'min_size': stats.min_size,
                        'std_size': stats.std_size,
                    }
                    for stage, stats in bucket_stats.items()
                }
            }
            
            comparison['abstractions'].append(abstraction_info)
        
        # 计算排名（按总WCSS）
        total_wcss_list = []
        for i, result in enumerate(results):
            total_wcss = sum(result.wcss.values()) if result.wcss else float('inf')
            total_wcss_list.append((names[i], total_wcss))
        
        # 按WCSS排序（越小越好）
        sorted_by_wcss = sorted(total_wcss_list, key=lambda x: x[1])
        comparison['rankings']['by_wcss'] = [
            {'name': name, 'total_wcss': wcss, 'rank': i + 1}
            for i, (name, wcss) in enumerate(sorted_by_wcss)
        ]
        
        # 计算排名（按生成时间）
        time_list = [(names[i], results[i].generation_time) for i in range(len(results))]
        sorted_by_time = sorted(time_list, key=lambda x: x[1])
        comparison['rankings']['by_time'] = [
            {'name': name, 'time': time, 'rank': i + 1}
            for i, (name, time) in enumerate(sorted_by_time)
        ]
        
        # 汇总统计
        comparison['summary'] = {
            'num_abstractions': len(results),
            'best_by_wcss': sorted_by_wcss[0][0] if sorted_by_wcss else None,
            'fastest': sorted_by_time[0][0] if sorted_by_time else None,
        }
        
        return comparison
    
    def generate_report(self, result: AbstractionResult) -> AbstractionQualityReport:
        """生成抽象质量报告。
        
        生成完整的抽象质量报告，包含配置、WCSS、桶统计等信息。
        
        Args:
            result: 抽象结果
            
        Returns:
            AbstractionQualityReport实例
        """
        # 获取所有阶段的桶统计
        bucket_stats = self.get_all_bucket_stats(result)
        
        # 计算总桶数
        total_buckets = sum(stats.count for stats in bucket_stats.values())
        
        # 计算压缩比（估算原始状态数）
        # 翻牌前：169种起手牌
        # 翻牌：约130万种
        # 转牌：约5500万种
        # 河牌：约250万种
        estimated_original_states = 169 + 1300000 + 55000000 + 2500000
        compression_ratio = estimated_original_states / max(total_buckets, 1)
        
        return AbstractionQualityReport(
            config=result.config,
            wcss=result.wcss,
            bucket_stats=bucket_stats,
            generation_time=result.generation_time,
            total_buckets=total_buckets,
            compression_ratio=compression_ratio
        )
    
    def generate_report_dict(self, result: AbstractionResult) -> Dict[str, Any]:
        """生成抽象质量报告（字典格式）。
        
        生成可序列化的报告字典。
        
        Args:
            result: 抽象结果
            
        Returns:
            报告字典
        """
        report = self.generate_report(result)
        
        return {
            'config': report.config.to_dict(),
            'wcss': report.wcss,
            'bucket_stats': {
                stage: {
                    'count': stats.count,
                    'avg_size': stats.avg_size,
                    'max_size': stats.max_size,
                    'min_size': stats.min_size,
                    'std_size': stats.std_size,
                }
                for stage, stats in report.bucket_stats.items()
            },
            'generation_time': report.generation_time,
            'total_buckets': report.total_buckets,
            'compression_ratio': report.compression_ratio,
        }
    
    def verify_reproducibility(self,
                               config: AbstractionConfig,
                               generator_func,
                               num_runs: int = 2) -> Tuple[bool, str]:
        """验证抽象的可重复性。
        
        使用相同配置（包括随机种子）多次生成抽象，验证结果是否一致。
        
        Args:
            config: 抽象配置
            generator_func: 抽象生成函数，接受config参数，返回AbstractionResult
            num_runs: 运行次数
            
        Returns:
            (is_reproducible, message) 元组
        """
        if num_runs < 2:
            return True, "至少需要2次运行才能验证可重复性"
        
        results = []
        for i in range(num_runs):
            result = generator_func(config)
            results.append(result)
        
        # 比较所有结果
        reference = results[0]
        
        for i in range(1, len(results)):
            current = results[i]
            
            # 比较映射
            if not self._arrays_equal(reference.preflop_mapping, current.preflop_mapping):
                return False, f"第{i+1}次运行的翻牌前映射与第1次不同"
            
            if not self._arrays_equal(reference.flop_mapping, current.flop_mapping):
                return False, f"第{i+1}次运行的翻牌映射与第1次不同"
            
            if not self._arrays_equal(reference.turn_mapping, current.turn_mapping):
                return False, f"第{i+1}次运行的转牌映射与第1次不同"
            
            if not self._arrays_equal(reference.river_mapping, current.river_mapping):
                return False, f"第{i+1}次运行的河牌映射与第1次不同"
        
        return True, f"所有{num_runs}次运行结果一致"
    
    def _arrays_equal(self, arr1: Optional[np.ndarray], 
                      arr2: Optional[np.ndarray]) -> bool:
        """比较两个数组是否相等。"""
        if arr1 is None and arr2 is None:
            return True
        if arr1 is None or arr2 is None:
            return False
        return np.array_equal(arr1, arr2)
    
    def calculate_silhouette_score(self,
                                   data_points: np.ndarray,
                                   labels: np.ndarray,
                                   ground_distances: Optional[np.ndarray] = None
                                   ) -> float:
        """计算轮廓系数（Silhouette Score）。
        
        轮廓系数衡量聚类的紧凑性和分离性，范围为[-1, 1]。
        值越大表示聚类效果越好。
        
        Args:
            data_points: 数据点数组
            labels: 聚类标签数组
            ground_distances: 可选的地面距离矩阵
            
        Returns:
            轮廓系数
        """
        n_samples = len(data_points)
        if n_samples < 2:
            return 0.0
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return 0.0
        
        silhouette_scores = []
        
        for i in range(n_samples):
            # 计算a(i)：点i到同簇其他点的平均距离
            same_cluster_mask = labels == labels[i]
            same_cluster_indices = np.where(same_cluster_mask)[0]
            
            if len(same_cluster_indices) <= 1:
                a_i = 0.0
            else:
                distances_same = []
                for j in same_cluster_indices:
                    if i != j:
                        if ground_distances is not None:
                            dist = self.emd_calculator.calculate_emd_with_ground_distance(
                                data_points[i], data_points[j], ground_distances
                            )
                        else:
                            dist = self.emd_calculator.calculate_emd_1d(
                                data_points[i], data_points[j]
                            )
                        distances_same.append(dist)
                a_i = np.mean(distances_same) if distances_same else 0.0
            
            # 计算b(i)：点i到最近其他簇的平均距离
            b_i = float('inf')
            for cluster in unique_labels:
                if cluster == labels[i]:
                    continue
                
                other_cluster_mask = labels == cluster
                other_cluster_indices = np.where(other_cluster_mask)[0]
                
                if len(other_cluster_indices) == 0:
                    continue
                
                distances_other = []
                for j in other_cluster_indices:
                    if ground_distances is not None:
                        dist = self.emd_calculator.calculate_emd_with_ground_distance(
                            data_points[i], data_points[j], ground_distances
                        )
                    else:
                        dist = self.emd_calculator.calculate_emd_1d(
                            data_points[i], data_points[j]
                        )
                    distances_other.append(dist)
                
                avg_dist = np.mean(distances_other)
                b_i = min(b_i, avg_dist)
            
            # 计算轮廓系数
            if b_i == float('inf'):
                s_i = 0.0
            else:
                s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
            
            silhouette_scores.append(s_i)
        
        return float(np.mean(silhouette_scores))
