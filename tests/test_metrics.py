import math

import pytest

from recommender.metrics import (
    evaluate_ranking,
    mae,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


def test_rmse_and_mae_known_values():
    assert rmse([3.0, 4.0], [3.0, 5.0]) == pytest.approx(0.7071, abs=1e-4)
    assert mae([3.0, 4.0], [3.0, 5.0]) == pytest.approx(0.5)
    assert rmse([2.0], [2.0]) == 0.0


def test_rmse_rejects_mismatched_or_empty():
    with pytest.raises(ValueError):
        rmse([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        mae([], [])


def test_precision_recall_known_values():
    recs = ["a", "b", "c"]
    assert precision_at_k(recs, {"a", "d"}, k=2) == pytest.approx(0.5)
    assert recall_at_k(recs, {"a", "d"}, k=2) == pytest.approx(0.5)
    assert precision_at_k([], {"a"}, k=5) == 0.0
    assert recall_at_k(recs, set(), k=5) == 0.0


def test_ndcg_perfect_and_worst():
    relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(["a", "b", "c"], relevance, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], relevance, k=2) == pytest.approx(0.0)
    assert ndcg_at_k(["a"], {}, k=1) == 0.0


def test_ndcg_partial_ordering():
    relevance = {"a": 3.0, "b": 1.0}
    best = ndcg_at_k(["a", "b"], relevance, k=2)
    worst = ndcg_at_k(["b", "a"], relevance, k=2)
    assert best > worst > 0


def test_evaluate_ranking_averages_users():
    recs = {"u1": ["a", "b"], "u2": ["c", "d"]}
    rel = {"u1": {"a": 2.0}, "u2": {"z": 1.0}}
    out = evaluate_ranking(recs, rel, k=2)
    assert out["precision@2"] == pytest.approx(0.25)
    assert out["recall@2"] == pytest.approx(0.5)


def test_k_validation():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, k=0)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": 1.0}, k=-1)
