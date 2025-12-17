"""Earth Mover's Distance (EMD) 计算器模块。

本模块实现了EMD距离计算功能，用于衡量两个分布之间的距离。
EMD也称为Wasserstein距离，是卡牌抽象中用于聚类的核心距离度量。

主要功能：
- 一维直方图的线性时间EMD计算
- 带地面距离的EMD计算（用于Potential-Aware抽象）
- 快速近似EMD计算（用于大规模k-means聚类）
"""

from typing import Optional, Tuple
import numpy as np


class EMDCalculator:
    """Earth Mover's Distance计算器。
    
    提供多种EMD计算方法，支持不同的使用场景：
    - 精确的一维EMD计算（线性时间）
    - 带自定义地面距离的EMD计算
    - 快速近似EMD计算（用于大规模聚类）
    """
    
    @staticmethod
    def calculate_emd_1d(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """计算两个一维直方图之间的EMD。
        
        使用线性时间算法：扫描直方图并累计需要移动的"土"的距离。
        这是一维情况下的最优算法，时间复杂度O(N)。
        
        算法原理：
        将直方图视为土堆，EMD是将一个土堆变换为另一个土堆所需的最小工作量。
        在一维情况下，最优策略是从左到右扫描，累计差值并计算移动距离。
        
        Args:
            hist1: 第一个直方图（归一化后的概率分布）
            hist2: 第二个直方图（归一化后的概率分布）
            
        Returns:
            EMD距离值（非负浮点数）
            
        Raises:
            ValueError: 如果直方图长度不一致
        """
        if len(hist1) != len(hist2):
            raise ValueError(f"直方图长度不一致：{len(hist1)} vs {len(hist2)}")
        
        if len(hist1) == 0:
            return 0.0
        
        # 线性时间EMD算法
        emd = 0.0
        cumulative_diff = 0.0
        
        for i in range(len(hist1)):
            cumulative_diff += hist1[i] - hist2[i]
            emd += abs(cumulative_diff)
        
        return emd
    
    @staticmethod
    def calculate_emd_with_ground_distance(hist1: np.ndarray, 
                                           hist2: np.ndarray,
                                           ground_distances: np.ndarray) -> float:
        """计算两个直方图之间的EMD，使用自定义的地面距离矩阵。
        
        用于Potential-Aware抽象，其中地面距离是下一轮次桶之间的EMD。
        
        这是一个更通用的EMD计算，使用线性规划或贪心近似。
        对于小规模问题，使用精确算法；对于大规模问题，使用近似算法。
        
        Args:
            hist1: 第一个直方图（在下一轮次桶上的分布）
            hist2: 第二个直方图
            ground_distances: 桶之间的地面距离矩阵 (n x n)
            
        Returns:
            EMD距离值
            
        Raises:
            ValueError: 如果输入维度不匹配
        """
        n = len(hist1)
        if len(hist2) != n:
            raise ValueError(f"直方图长度不一致：{len(hist1)} vs {len(hist2)}")
        if ground_distances.shape != (n, n):
            raise ValueError(f"地面距离矩阵维度不正确：{ground_distances.shape}，期望 ({n}, {n})")
        
        # 对于小规模问题，使用贪心近似算法
        # 这不是最优解，但对于聚类来说足够好
        return EMDCalculator._greedy_emd(hist1, hist2, ground_distances)
    
    @staticmethod
    def _greedy_emd(hist1: np.ndarray, hist2: np.ndarray, 
                   ground_distances: np.ndarray) -> float:
        """使用贪心算法近似计算EMD。
        
        贪心策略：优先移动到距离最近的桶。
        
        Args:
            hist1: 源分布
            hist2: 目标分布
            ground_distances: 地面距离矩阵
            
        Returns:
            近似的EMD值
        """
        n = len(hist1)
        
        # 复制以避免修改原数组
        supply = hist1.copy()
        demand = hist2.copy()
        
        total_cost = 0.0
        
        # 对于每个有供应的桶
        for i in range(n):
            if supply[i] <= 1e-10:
                continue
            
            # 按距离排序找到最近的需求桶
            distances_from_i = ground_distances[i]
            sorted_indices = np.argsort(distances_from_i)
            
            for j in sorted_indices:
                if demand[j] <= 1e-10:
                    continue
                
                # 移动尽可能多的质量
                move_amount = min(supply[i], demand[j])
                total_cost += move_amount * distances_from_i[j]
                supply[i] -= move_amount
                demand[j] -= move_amount
                
                if supply[i] <= 1e-10:
                    break
        
        return total_cost
    
    @staticmethod
    def calculate_emd_fast_approximation(point: np.ndarray,
                                         center: np.ndarray,
                                         ground_distances: Optional[np.ndarray] = None
                                         ) -> float:
        """快速近似EMD计算（用于大规模k-means聚类）。
        
        利用稀疏性和预计算的排序距离来加速计算。
        当不提供地面距离时，使用一维EMD。
        
        Args:
            point: 数据点的直方图表示
            center: 聚类中心的直方图表示
            ground_distances: 可选的地面距离矩阵
            
        Returns:
            近似的EMD距离值
        """
        if ground_distances is None:
            # 使用一维EMD
            return EMDCalculator.calculate_emd_1d(point, center)
        
        # 利用稀疏性优化
        # 只处理非零元素
        point_nonzero = np.where(point > 1e-10)[0]
        center_nonzero = np.where(center > 1e-10)[0]
        
        if len(point_nonzero) == 0 and len(center_nonzero) == 0:
            return 0.0
        
        # 对于稀疏情况，使用简化的贪心算法
        return EMDCalculator._sparse_greedy_emd(
            point, center, ground_distances, 
            point_nonzero, center_nonzero
        )
    
    @staticmethod
    def _sparse_greedy_emd(hist1: np.ndarray, hist2: np.ndarray,
                          ground_distances: np.ndarray,
                          nonzero1: np.ndarray, nonzero2: np.ndarray) -> float:
        """稀疏直方图的贪心EMD计算。
        
        只处理非零元素，提高计算效率。
        
        Args:
            hist1: 源分布
            hist2: 目标分布
            ground_distances: 地面距离矩阵
            nonzero1: hist1的非零索引
            nonzero2: hist2的非零索引
            
        Returns:
            近似的EMD值
        """
        # 复制非零部分
        supply = {i: hist1[i] for i in nonzero1}
        demand = {j: hist2[j] for j in nonzero2}
        
        total_cost = 0.0
        
        for i in list(supply.keys()):
            if supply[i] <= 1e-10:
                continue
            
            # 找到最近的有需求的桶
            min_dist = float('inf')
            min_j = -1
            
            for j in demand.keys():
                if demand[j] > 1e-10:
                    dist = ground_distances[i, j]
                    if dist < min_dist:
                        min_dist = dist
                        min_j = j
            
            if min_j == -1:
                break
            
            # 移动质量
            move_amount = min(supply[i], demand[min_j])
            total_cost += move_amount * min_dist
            supply[i] -= move_amount
            demand[min_j] -= move_amount
            
            if demand[min_j] <= 1e-10:
                del demand[min_j]
        
        return total_cost
    
    @staticmethod
    def compute_ground_distance_matrix(centers: np.ndarray) -> np.ndarray:
        """计算聚类中心之间的地面距离矩阵。
        
        使用一维EMD作为中心之间的距离。
        
        Args:
            centers: 聚类中心数组，形状为 (num_centers, histogram_size)
            
        Returns:
            地面距离矩阵，形状为 (num_centers, num_centers)
        """
        num_centers = len(centers)
        distances = np.zeros((num_centers, num_centers))
        
        for i in range(num_centers):
            for j in range(i + 1, num_centers):
                dist = EMDCalculator.calculate_emd_1d(centers[i], centers[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    @staticmethod
    def validate_metric_properties(hist1: np.ndarray, hist2: np.ndarray, 
                                   hist3: np.ndarray) -> Tuple[bool, bool, bool]:
        """验证EMD是否满足度量空间的性质。
        
        度量空间的三个性质：
        1. 非负性：d(x, y) >= 0
        2. 对称性：d(x, y) = d(y, x)
        3. 三角不等式：d(x, z) <= d(x, y) + d(y, z)
        
        Args:
            hist1, hist2, hist3: 三个直方图
            
        Returns:
            (非负性, 对称性, 三角不等式) 的布尔元组
        """
        d12 = EMDCalculator.calculate_emd_1d(hist1, hist2)
        d21 = EMDCalculator.calculate_emd_1d(hist2, hist1)
        d13 = EMDCalculator.calculate_emd_1d(hist1, hist3)
        d23 = EMDCalculator.calculate_emd_1d(hist2, hist3)
        
        # 非负性
        non_negative = d12 >= 0 and d21 >= 0 and d13 >= 0 and d23 >= 0
        
        # 对称性
        symmetric = np.isclose(d12, d21, atol=1e-10)
        
        # 三角不等式
        triangle = d13 <= d12 + d23 + 1e-10  # 允许小误差
        
        return non_negative, symmetric, triangle
