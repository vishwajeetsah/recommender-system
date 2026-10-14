"""Evaluation helpers: score a fitted model on held-out ratings and produce
top-N recommendation lists that exclude items the user already rated."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from .data import Ratings
from .metrics import mae, rmse


def evaluate_ratings(
    predict: Callable[[str, str], float],
    test: Ratings,
) -> Dict[str, float]:
    """RMSE / MAE of `predict(user_id, item_id)` on a held-out Ratings set."""
    preds, actuals = [], []
    for u, i, r in test.triples:
        preds.append(predict(test.users[u], test.items[i]))
        actuals.append(r)
    return {"rmse": rmse(preds, actuals), "mae": mae(preds, actuals), "n": len(actuals)}


def recommend_top_n(
    predict: Callable[[str, str], float],
    train: Ratings,
    user: str,
    n: int = 10,
    candidates: Sequence[str] | None = None,
) -> List[str]:
    """Top-N items for `user`, excluding anything they rated in `train`.

    Unknown users get the globally highest-predicted items (their predictions
    fall back to item means inside the model), which is a sane cold-start
    default: popular / highly rated items first.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    rated = set()
    if user in train.user_index:
        u = train.user_index[user]
        rated = {i for uu, i, _ in train.triples if uu == u}
    pool = list(candidates) if candidates is not None else list(train.items)
    scored = [
        (predict(user, train.items[i]), train.items[i])
        for i in (train.item_index[it] for it in pool if it in train.item_index)
        if i not in rated
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [item for _, item in scored[:n]]
