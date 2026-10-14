"""Ranking / rating metrics for recommender evaluation."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple


def rmse(predictions: Sequence[float], actuals: Sequence[float]) -> float:
    if len(predictions) != len(actuals) or not predictions:
        raise ValueError("predictions and actuals must be non-empty and equal length")
    return math.sqrt(
        sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)
    )


def mae(predictions: Sequence[float], actuals: Sequence[float]) -> float:
    if len(predictions) != len(actuals) or not predictions:
        raise ValueError("predictions and actuals must be non-empty and equal length")
    return sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(predictions)


def precision_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k recommendations that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = list(recommended)[:k]
    if not top:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for item in top if item in relevant_set) / len(top)


def recall_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of all relevant items surfaced in the top-k."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = list(recommended)[:k]
    return sum(1 for item in top if item in relevant_set) / len(relevant_set)


def ndcg_at_k(
    recommended: Sequence[str],
    relevance: Dict[str, float],
    k: int,
) -> float:
    """Normalised discounted cumulative gain at k.

    `relevance` maps item -> graded relevance (>= 0). Recommended items not
    in the map count as relevance 0. Returns 0 when there is no relevance.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top = list(recommended)[:k]
    dcg = sum(
        (2 ** relevance.get(item, 0.0) - 1) / math.log2(rank + 2)
        for rank, item in enumerate(top)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(
        (2 ** rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(ideal)
    )
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    recommendations_by_user: Dict[str, List[str]],
    relevant_by_user: Dict[str, Dict[str, float]],
    k: int = 10,
) -> Dict[str, float]:
    """Mean precision@k / recall@k / nDCG@k over users that have relevance."""
    precisions, recalls, ndcgs = [], [], []
    for user, relevance in relevant_by_user.items():
        recs = recommendations_by_user.get(user, [])
        if not relevance:
            continue
        precisions.append(precision_at_k(recs, relevance.keys(), k))
        recalls.append(recall_at_k(recs, relevance.keys(), k))
        ndcgs.append(ndcg_at_k(recs, relevance, k))
    if not precisions:
        return {f"precision@{k}": 0.0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0}
    n = len(precisions)
    return {
        f"precision@{k}": sum(precisions) / n,
        f"recall@{k}": sum(recalls) / n,
        f"ndcg@{k}": sum(ndcgs) / n,
    }
