"""Potential分析器模块。

本模块实现了Potential直方图与Solver策略之间的相关性分析功能。
包括：
- 直方图特征与策略的相关性分析
- 基于EMD距离的手牌聚类
- 聚类与策略的一致性比较
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score

from abstraction.emd_calculator import EMDCalculator
from experiments.turn_potential_validation.data_models import (
    CorrelationResult,
    ClusteringComparisonResult,
)
from experiments.turn_potential_validation.potential_histogram import (
    PotentialHistogramCalculator,
)


class PotentialAnalyzer:
    """Potential分析器类。
    
    分析Potential直方图与Solver策略之间的关系，包括：
    - 直方图特征与策略的相关性
    - 基于EMD距离的手牌聚类
    - 聚类与策略的一致性比较
    """
    
    def __init__(self, num_bins: int = 50):
        """初始化分析器。
        
        Args:
            num_bins: Potential直方图的区间数量
        """
        self.num_bins = num_bins
        self.histogram_calculator = PotentialHistogramCalculator(num_bins=num_bins)
        self.emd_calculator = EMDCalculator()
    
    def analyze_histogram_strategy_correlation(
        self,
        potential_histograms: Dict[str, np.ndarray],
        solver_strategies: Dict[str, Dict[str, float]]
    ) -> CorrelationResult:
        """分析Potential直方图与Solver策略的相关性。
        
        计算直方图特征（均值、方差、熵等）与策略动作概率之间的相关性。
        
        Args:
            potential_histograms: 每个手牌的Potential直方图，格式为 {hand_str: histogram}
            solver_strategies: Solver计算的策略，格式为 {hand_str: {action: prob}}
            
        Returns:
            CorrelationResult实例，包含各种相关性指标
        """
        if not potential_histograms or not solver_strategies:
            return CorrelationResult()
        
        # 找到共同的手牌
        common_hands = set(potential_histograms.keys()) & set(solver_strategies.keys())
        
        if len(common_hands) < 3:
            # 样本太少，无法计算有意义的相关性
            return CorrelationResult()
        
        # 提取特征
        mean_equities = []
        variances = []
        entropies = []
        
        # 提取策略动作概率（假设主要动作是bet/raise）
        aggressive_probs = []
        
        for hand in common_hands:
            histogram = potential_histograms[hand]
            strategy = solver_strategies[hand]
            
            # 计算直方图特征
            features = self.histogram_calculator.get_histogram_features(histogram)
            mean_equities.append(features['mean_equity'])
            variances.append(features['variance'])
            entropies.append(features['entropy'])
            
            # 计算激进动作概率（bet或raise）
            aggressive_prob = 0.0
            for action, prob in strategy.items():
                if 'bet' in action.lower() or 'raise' in action.lower():
                    aggressive_prob += prob
            aggressive_probs.append(aggressive_prob)
        
        # 计算相关性
        mean_equity_corr = 0.0
        variance_corr = 0.0
        entropy_corr = 0.0
        
        if len(mean_equities) >= 3:
            # 使用Spearman相关系数（更稳健）
            if np.std(mean_equities) > 1e-10 and np.std(aggressive_probs) > 1e-10:
                mean_equity_corr, _ = stats.spearmanr(mean_equities, aggressive_probs)
                if np.isnan(mean_equity_corr):
                    mean_equity_corr = 0.0
            
            if np.std(variances) > 1e-10 and np.std(aggressive_probs) > 1e-10:
                variance_corr, _ = stats.spearmanr(variances, aggressive_probs)
                if np.isnan(variance_corr):
                    variance_corr = 0.0
            
            if np.std(entropies) > 1e-10 and np.std(aggressive_probs) > 1e-10:
                entropy_corr, _ = stats.spearmanr(entropies, aggressive_probs)
                if np.isnan(entropy_corr):
                    entropy_corr = 0.0
        
        # 计算同一动作内和不同动作间的EMD距离
        intra_action_emd, inter_action_emd = self._compute_action_emd_distances(
            potential_histograms, solver_strategies, common_hands
        )
        
        return CorrelationResult(
            mean_equity_correlation=float(mean_equity_corr),
            variance_correlation=float(variance_corr),
            intra_action_emd=intra_action_emd,
            inter_action_emd=inter_action_emd,
            clustering_purity=0.0,  # 将在聚类分析中计算
            histogram_entropy_correlation=float(entropy_corr),
        )
    
    def _compute_action_emd_distances(
        self,
        potential_histograms: Dict[str, np.ndarray],
        solver_strategies: Dict[str, Dict[str, float]],
        common_hands: set
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """计算同一动作内和不同动作间的平均EMD距离。
        
        Args:
            potential_histograms: Potential直方图
            solver_strategies: Solver策略
            common_hands: 共同的手牌集合
            
        Returns:
            (同一动作内EMD, 不同动作间EMD) 元组
        """
        # 按主要动作分组手牌
        action_groups: Dict[str, List[str]] = defaultdict(list)
        
        for hand in common_hands:
            strategy = solver_strategies[hand]
            # 找到概率最高的动作
            main_action = max(strategy.items(), key=lambda x: x[1])[0]
            action_groups[main_action].append(hand)
        
        # 计算同一动作内的平均EMD
        intra_action_emd = {}
        for action, hands in action_groups.items():
            if len(hands) < 2:
                intra_action_emd[action] = 0.0
                continue
            
            distances = []
            for i in range(len(hands)):
                for j in range(i + 1, len(hands)):
                    hist1 = potential_histograms[hands[i]]
                    hist2 = potential_histograms[hands[j]]
                    dist = self.emd_calculator.calculate_emd_1d(hist1, hist2)
                    distances.append(dist)
            
            intra_action_emd[action] = float(np.mean(distances)) if distances else 0.0
        
        # 计算不同动作间的平均EMD
        inter_action_emd = {}
        actions = list(action_groups.keys())
        
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                action1, action2 = actions[i], actions[j]
                hands1 = action_groups[action1]
                hands2 = action_groups[action2]
                
                if not hands1 or not hands2:
                    continue
                
                distances = []
                for h1 in hands1:
                    for h2 in hands2:
                        hist1 = potential_histograms[h1]
                        hist2 = potential_histograms[h2]
                        dist = self.emd_calculator.calculate_emd_1d(hist1, hist2)
                        distances.append(dist)
                
                key = f"{action1}_vs_{action2}"
                inter_action_emd[key] = float(np.mean(distances)) if distances else 0.0
        
        return intra_action_emd, inter_action_emd

    
    def cluster_by_potential(
        self,
        potential_histograms: Dict[str, np.ndarray],
        num_clusters: int
    ) -> Dict[str, int]:
        """基于Potential直方图对手牌进行聚类。
        
        使用k-means聚类算法，以EMD距离作为度量。
        由于sklearn的k-means使用欧氏距离，这里使用直方图向量直接聚类，
        这是EMD的一个近似（对于归一化直方图，L1距离是EMD的上界）。
        
        Args:
            potential_histograms: 每个手牌的Potential直方图
            num_clusters: 聚类数量
            
        Returns:
            每个手牌的聚类标签映射
        """
        if not potential_histograms:
            return {}
        
        hands = list(potential_histograms.keys())
        
        if len(hands) < num_clusters:
            # 手牌数量少于聚类数，每个手牌一个聚类
            return {hand: i for i, hand in enumerate(hands)}
        
        # 构建特征矩阵
        X = np.array([potential_histograms[hand] for hand in hands])
        
        # 使用k-means聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        return {hand: int(label) for hand, label in zip(hands, labels)}
    
    def cluster_by_potential_with_emd(
        self,
        potential_histograms: Dict[str, np.ndarray],
        num_clusters: int,
        max_iterations: int = 100
    ) -> Dict[str, int]:
        """基于Potential直方图使用EMD距离进行聚类。
        
        实现基于EMD距离的k-medoids聚类算法。
        
        Args:
            potential_histograms: 每个手牌的Potential直方图
            num_clusters: 聚类数量
            max_iterations: 最大迭代次数
            
        Returns:
            每个手牌的聚类标签映射
        """
        if not potential_histograms:
            return {}
        
        hands = list(potential_histograms.keys())
        n = len(hands)
        
        if n < num_clusters:
            return {hand: i for i, hand in enumerate(hands)}
        
        # 计算EMD距离矩阵
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.emd_calculator.calculate_emd_1d(
                    potential_histograms[hands[i]],
                    potential_histograms[hands[j]]
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # k-medoids聚类
        labels = self._k_medoids(distance_matrix, num_clusters, max_iterations)
        
        return {hand: int(label) for hand, label in zip(hands, labels)}
    
    def _k_medoids(
        self,
        distance_matrix: np.ndarray,
        k: int,
        max_iterations: int
    ) -> np.ndarray:
        """k-medoids聚类算法。
        
        Args:
            distance_matrix: 距离矩阵
            k: 聚类数量
            max_iterations: 最大迭代次数
            
        Returns:
            聚类标签数组
        """
        n = len(distance_matrix)
        
        # 随机初始化medoids
        np.random.seed(42)
        medoids = np.random.choice(n, k, replace=False)
        
        for _ in range(max_iterations):
            # 分配点到最近的medoid
            labels = np.argmin(distance_matrix[:, medoids], axis=1)
            
            # 更新medoids
            new_medoids = []
            for cluster_id in range(k):
                cluster_points = np.where(labels == cluster_id)[0]
                if len(cluster_points) == 0:
                    new_medoids.append(medoids[cluster_id])
                    continue
                
                # 找到使簇内距离和最小的点
                min_cost = float('inf')
                best_medoid = medoids[cluster_id]
                
                for point in cluster_points:
                    cost = np.sum(distance_matrix[point, cluster_points])
                    if cost < min_cost:
                        min_cost = cost
                        best_medoid = point
                
                new_medoids.append(best_medoid)
            
            new_medoids = np.array(new_medoids)
            
            # 检查收敛
            if np.array_equal(medoids, new_medoids):
                break
            
            medoids = new_medoids
        
        # 最终分配
        labels = np.argmin(distance_matrix[:, medoids], axis=1)
        
        return labels
    
    def compare_clustering_with_strategy(
        self,
        cluster_labels: Dict[str, int],
        solver_strategies: Dict[str, Dict[str, float]]
    ) -> ClusteringComparisonResult:
        """比较基于Potential的聚类与Solver策略的一致性。
        
        计算聚类纯度、归一化互信息等指标。
        
        Args:
            cluster_labels: 聚类标签
            solver_strategies: Solver策略
            
        Returns:
            ClusteringComparisonResult实例
        """
        if not cluster_labels or not solver_strategies:
            return ClusteringComparisonResult()
        
        # 找到共同的手牌
        common_hands = set(cluster_labels.keys()) & set(solver_strategies.keys())
        
        if len(common_hands) < 2:
            return ClusteringComparisonResult()
        
        hands = list(common_hands)
        
        # 获取聚类标签
        cluster_ids = [cluster_labels[h] for h in hands]
        
        # 获取主要动作标签
        action_labels = []
        for hand in hands:
            strategy = solver_strategies[hand]
            main_action = max(strategy.items(), key=lambda x: x[1])[0]
            action_labels.append(main_action)
        
        # 将动作转换为数字标签
        unique_actions = list(set(action_labels))
        action_to_id = {a: i for i, a in enumerate(unique_actions)}
        action_ids = [action_to_id[a] for a in action_labels]
        
        # 计算聚类纯度
        purity = self._compute_purity(cluster_ids, action_ids)
        
        # 计算归一化互信息
        nmi = normalized_mutual_info_score(cluster_ids, action_ids)
        
        # 计算每个聚类的动作分布
        action_distribution = self._compute_action_distribution_per_cluster(
            hands, cluster_labels, solver_strategies
        )
        
        # 计算聚类大小
        cluster_sizes = defaultdict(int)
        for label in cluster_ids:
            cluster_sizes[label] += 1
        
        # 计算轮廓系数（如果有足够的数据）
        silhouette = 0.0
        num_clusters = len(set(cluster_ids))
        if num_clusters > 1 and len(hands) > num_clusters:
            # 需要特征矩阵来计算轮廓系数
            # 这里使用简化方法：基于动作概率向量
            X = []
            for hand in hands:
                strategy = solver_strategies[hand]
                # 创建动作概率向量
                probs = [strategy.get(a, 0.0) for a in unique_actions]
                X.append(probs)
            X = np.array(X)
            
            try:
                silhouette = silhouette_score(X, cluster_ids)
            except Exception:
                silhouette = 0.0
        
        return ClusteringComparisonResult(
            num_clusters=num_clusters,
            purity=float(purity),
            normalized_mutual_info=float(nmi),
            action_distribution_per_cluster=action_distribution,
            cluster_sizes=dict(cluster_sizes),
            silhouette_score=float(silhouette),
        )
    
    def _compute_purity(
        self,
        cluster_labels: List[int],
        true_labels: List[int]
    ) -> float:
        """计算聚类纯度。
        
        纯度 = 每个聚类中最常见类别的样本数之和 / 总样本数
        
        Args:
            cluster_labels: 聚类标签
            true_labels: 真实标签
            
        Returns:
            纯度值 [0, 1]
        """
        n = len(cluster_labels)
        if n == 0:
            return 0.0
        
        # 统计每个聚类中各类别的数量
        cluster_to_class_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        for cluster, true_class in zip(cluster_labels, true_labels):
            cluster_to_class_counts[cluster][true_class] += 1
        
        # 计算纯度
        correct = 0
        for cluster, class_counts in cluster_to_class_counts.items():
            correct += max(class_counts.values())
        
        return correct / n
    
    def _compute_action_distribution_per_cluster(
        self,
        hands: List[str],
        cluster_labels: Dict[str, int],
        solver_strategies: Dict[str, Dict[str, float]]
    ) -> Dict[int, Dict[str, float]]:
        """计算每个聚类的动作分布。
        
        Args:
            hands: 手牌列表
            cluster_labels: 聚类标签
            solver_strategies: Solver策略
            
        Returns:
            每个聚类的动作分布
        """
        # 收集每个聚类的动作概率
        cluster_action_probs: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        for hand in hands:
            cluster = cluster_labels[hand]
            strategy = solver_strategies[hand]
            
            for action, prob in strategy.items():
                cluster_action_probs[cluster][action].append(prob)
        
        # 计算平均动作概率
        result = {}
        for cluster, action_probs in cluster_action_probs.items():
            result[cluster] = {
                action: float(np.mean(probs))
                for action, probs in action_probs.items()
            }
        
        return result
    
    def get_histogram_similarity_matrix(
        self,
        potential_histograms: Dict[str, np.ndarray]
    ) -> Tuple[List[str], np.ndarray]:
        """计算所有手牌之间的EMD相似度矩阵。
        
        Args:
            potential_histograms: Potential直方图
            
        Returns:
            (手牌列表, 距离矩阵) 元组
        """
        hands = list(potential_histograms.keys())
        n = len(hands)
        
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.emd_calculator.calculate_emd_1d(
                    potential_histograms[hands[i]],
                    potential_histograms[hands[j]]
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return hands, distance_matrix
    
    def find_similar_hands(
        self,
        potential_histograms: Dict[str, np.ndarray],
        target_hand: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """找到与目标手牌最相似的手牌。
        
        Args:
            potential_histograms: Potential直方图
            target_hand: 目标手牌
            top_k: 返回的相似手牌数量
            
        Returns:
            (手牌, EMD距离) 元组列表，按距离升序排列
        """
        if target_hand not in potential_histograms:
            return []
        
        target_hist = potential_histograms[target_hand]
        
        distances = []
        for hand, hist in potential_histograms.items():
            if hand == target_hand:
                continue
            dist = self.emd_calculator.calculate_emd_1d(target_hist, hist)
            distances.append((hand, dist))
        
        # 按距离排序
        distances.sort(key=lambda x: x[1])
        
        return distances[:top_k]
