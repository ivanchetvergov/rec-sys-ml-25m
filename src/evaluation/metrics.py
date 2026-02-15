"""Evaluation metrics for recommender systems."""
import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def precision_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """
    Calculate Precision@K.

    Precision@K = (# of recommended items @K that are relevant) / K

    Args:
        recommended: List of recommended item IDs (ordered by score)
        relevant: Set of relevant/liked item IDs (ground truth)
        k: Number of top recommendations to consider

    Returns:
        Precision@K score [0, 1]
    """
    if k <= 0:
        return 0.0

    recommended_at_k = recommended[:k]
    hits = len(set(recommended_at_k) & relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """
    Calculate Recall@K.

    Recall@K = (# of recommended items @K that are relevant) / (# of relevant items)

    Args:
        recommended: List of recommended item IDs (ordered by score)
        relevant: Set of relevant/liked item IDs (ground truth)
        k: Number of top recommendations to consider

    Returns:
        Recall@K score [0, 1]
    """
    if len(relevant) == 0:
        return 0.0

    recommended_at_k = recommended[:k]
    hits = len(set(recommended_at_k) & relevant)
    return hits / len(relevant)


def average_precision(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """
    Calculate Average Precision@K.

    AP@K = (sum of P@i for each relevant item i in top K) / min(K, # relevant)

    Args:
        recommended: List of recommended item IDs (ordered by score)
        relevant: Set of relevant/liked item IDs (ground truth)
        k: Number of top recommendations to consider

    Returns:
        Average Precision score [0, 1]
    """
    if len(relevant) == 0:
        return 0.0

    recommended_at_k = recommended[:k]
    score = 0.0
    num_hits = 0.0

    for i, item_id in enumerate(recommended_at_k, start=1):
        if item_id in relevant:
            num_hits += 1
            score += num_hits / i

    return score / min(len(relevant), k)


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@K.

    DCG@K = sum(rel_i / log2(i + 1)) for i in [1, K]
    NDCG@K = DCG@K / IDCG@K

    Assumes binary relevance (1 if relevant, 0 otherwise).

    Args:
        recommended: List of recommended item IDs (ordered by score)
        relevant: Set of relevant/liked item IDs (ground truth)
        k: Number of top recommendations to consider

    Returns:
        NDCG@K score [0, 1]
    """
    if len(relevant) == 0:
        return 0.0

    recommended_at_k = recommended[:k]

    # Calculate DCG
    dcg = 0.0
    for i, item_id in enumerate(recommended_at_k, start=1):
        if item_id in relevant:
            dcg += 1.0 / np.log2(i + 1)

    # Calculate IDCG (ideal DCG - all relevant items at top)
    idcg = 0.0
    for i in range(1, min(len(relevant), k) + 1):
        idcg += 1.0 / np.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mean_average_precision_at_k(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, Set[int]],
    k: int = 10
) -> float:
    """
    Calculate Mean Average Precision@K across all users.

    MAP@K = mean(AP@K) across all users

    Args:
        recommendations: Dict[user_id -> list of recommended item IDs]
        ground_truth: Dict[user_id -> set of relevant item IDs]
        k: Number of top recommendations to consider

    Returns:
        MAP@K score [0, 1]
    """
    if not recommendations:
        return 0.0

    aps = []
    for user_id, recommended in recommendations.items():
        if user_id in ground_truth:
            relevant = ground_truth[user_id]
            ap = average_precision(recommended, relevant, k)
            aps.append(ap)

    return np.mean(aps) if aps else 0.0


def coverage(
    recommendations: Dict[int, List[int]],
    catalog: Set[int],
    k: int = 10
) -> float:
    """
    Calculate catalog coverage - fraction of items that are ever recommended.

    Coverage = (# of unique items recommended) / (# of items in catalog)

    Args:
        recommendations: Dict[user_id -> list of recommended item IDs]
        catalog: Set of all available item IDs
        k: Number of top recommendations to consider per user

    Returns:
        Coverage score [0, 1]
    """
    if len(catalog) == 0:
        return 0.0

    recommended_items = set()
    for recommended in recommendations.values():
        recommended_items.update(recommended[:k])

    return len(recommended_items) / len(catalog)


def evaluate_recommendations(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, Set[int]],
    catalog: Set[int],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluate recommendations with multiple metrics at different K values.

    Args:
        recommendations: Dict[user_id -> list of recommended item IDs]
        ground_truth: Dict[user_id -> set of relevant item IDs]
        catalog: Set of all available item IDs
        k_values: List of K values to evaluate at

    Returns:
        Dictionary with all metrics
    """
    results = {}

    for k in k_values:
        precisions = []
        recalls = []
        ndcgs = []

        for user_id, recommended in recommendations.items():
            if user_id not in ground_truth:
                continue

            relevant = ground_truth[user_id]
            if len(relevant) == 0:
                continue

            precisions.append(precision_at_k(recommended, relevant, k))
            recalls.append(recall_at_k(recommended, relevant, k))
            ndcgs.append(ndcg_at_k(recommended, relevant, k))

        # Average metrics
        results[f"precision_at_{k}"] = np.mean(precisions) if precisions else 0.0
        results[f"recall_at_{k}"] = np.mean(recalls) if recalls else 0.0
        results[f"ndcg_at_{k}"] = np.mean(ndcgs) if ndcgs else 0.0
        results[f"map_at_{k}"] = mean_average_precision_at_k(recommendations, ground_truth, k)
        results[f"coverage_at_{k}"] = coverage(recommendations, catalog, k)

    # Log summary
    logger.info(f"Evaluated {len(recommendations)} users with {len(k_values)} K values")
    for k in k_values:
        logger.info(
            f"  K={k}: P={results[f'precision_at_{k}']:.4f}, "
            f"R={results[f'recall_at_{k}']:.4f}, "
            f"NDCG={results[f'ndcg_at_{k}']:.4f}, "
            f"Coverage={results[f'coverage_at_{k}']:.4f}"
        )

    return results


def create_ground_truth(
    test_df: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
    relevance_threshold: float = 4.0
) -> Dict[int, Set[int]]:
    """
    Create ground truth dict from test dataframe.

    Items with rating >= threshold are considered relevant.

    Args:
        test_df: Test dataframe with user-item interactions
        user_col: Column name for user ID
        item_col: Column name for item ID
        rating_col: Column name for rating
        relevance_threshold: Minimum rating to consider item relevant

    Returns:
        Dict[user_id -> set of relevant item IDs]
    """
    relevant_df = test_df[test_df[rating_col] >= relevance_threshold]
    ground_truth = {}

    for user_id, group in relevant_df.groupby(user_col):
        ground_truth[user_id] = set(group[item_col].values)

    logger.info(
        f"Created ground truth: {len(ground_truth)} users, "
        f"avg {np.mean([len(v) for v in ground_truth.values()]):.1f} relevant items/user"
    )

    return ground_truth
