"""Collaborative-filtering recommender: neighbourhood CF + matrix factorisation.

Data format: iterable of (user_id, item_id, rating) triples, or a CSV with
columns user_id, item_id, rating. IDs may be strings or ints; they are mapped
to contiguous indices internally.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass
class Ratings:
    """A ratings table with integer index mappings for users and items."""

    users: List[str] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    triples: List[Tuple[int, int, float]] = field(default_factory=list)
    user_index: dict = field(default_factory=dict)
    item_index: dict = field(default_factory=dict)

    @classmethod
    def from_triples(cls, rows: Iterable[Tuple[object, object, float]]) -> "Ratings":
        ratings = cls()
        for user, item, rating in rows:
            u = str(user)
            i = str(item)
            if u not in ratings.user_index:
                ratings.user_index[u] = len(ratings.users)
                ratings.users.append(u)
            if i not in ratings.item_index:
                ratings.item_index[i] = len(ratings.items)
                ratings.items.append(i)
            ratings.triples.append(
                (ratings.user_index[u], ratings.item_index[i], float(rating))
            )
        return ratings

    @classmethod
    def from_csv(cls, path: str | Path) -> "Ratings":
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or not {
                "user_id",
                "item_id",
                "rating",
            }.issubset(reader.fieldnames):
                raise ValueError(
                    "CSV must have header columns: user_id, item_id, rating"
                )
            rows = [
                (row["user_id"], row["item_id"], float(row["rating"]))
                for row in reader
            ]
        return cls.from_triples(rows)

    def __len__(self) -> int:
        return len(self.triples)


def split_random(
    ratings: Ratings, test_fraction: float = 0.2, seed: int = 42
) -> Tuple[Ratings, Ratings]:
    """Random train/test split. Every user keeps at least one rating in train
    when they have more than one, so test users are never fully cold-start."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    rng = random.Random(seed)
    triples = list(ratings.triples)
    rng.shuffle(triples)
    counts: dict[int, int] = {}
    for u, _, _ in triples:
        counts[u] = counts.get(u, 0) + 1
    train_triples: List[Tuple[int, int, float]] = []
    test_triples: List[Tuple[int, int, float]] = []
    train_counts: dict[int, int] = {}
    for t in triples:
        u = t[0]
        remaining_for_user = counts[u] - train_counts.get(u, 0)
        # Keep at least one rating per user in train.
        if (
            len(test_triples) < round(len(triples) * test_fraction)
            and remaining_for_user > 1
            and rng.random() < test_fraction
        ):
            test_triples.append(t)
        else:
            train_triples.append(t)
        train_counts[u] = train_counts.get(u, 0) + 1

    def _rebuild(triple_list: Sequence[Tuple[int, int, float]]) -> Ratings:
        out = Ratings(users=list(ratings.users), items=list(ratings.items),
                      user_index=dict(ratings.user_index),
                      item_index=dict(ratings.item_index))
        out.triples = list(triple_list)
        return out

    return _rebuild(train_triples), _rebuild(test_triples)


def synthetic_ratings(
    n_users: int = 60,
    n_items: int = 40,
    n_factors: int = 3,
    density: float = 0.25,
    seed: int = 7,
) -> Ratings:
    """Generate plausible ratings from a low-rank latent-factor model.

    Users and items get random latent vectors plus biases; observed ratings
    are the dot product plus biases, noise, clipped to [1, 5] and rounded to
    the nearest 0.5. Because the data really is (approximately) low-rank,
    matrix factorisation should beat a global-mean baseline on it — a
    built-in sanity check for the whole pipeline.
    """
    import numpy as np

    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    rng = np.random.default_rng(seed)
    user_factors = rng.normal(0, 1, (n_users, n_factors))
    item_factors = rng.normal(0, 1, (n_items, n_factors))
    user_bias = rng.normal(0, 0.4, n_users)
    item_bias = rng.normal(0, 0.4, n_items)
    global_mean = 3.5
    rows = []
    for u in range(n_users):
        for i in range(n_items):
            if rng.random() > density:
                continue
            score = (
                global_mean
                + user_bias[u]
                + item_bias[i]
                + float(user_factors[u] @ item_factors[i])
                + rng.normal(0, 0.3)
            )
            score = round(min(5.0, max(1.0, score)) * 2) / 2
            rows.append((f"user_{u:03d}", f"item_{i:03d}", score))
    return Ratings.from_triples(rows)
