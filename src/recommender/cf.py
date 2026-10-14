"""Neighbourhood collaborative filtering (user-based and item-based).

Similarity is cosine over mean-centred ratings, with shrinkage so that pairs
sharing only one or two co-rated items do not get a spuriously perfect
similarity. Predictions are similarity-weighted deviations from the
neighbour's mean, falling back to the user mean, then the global mean.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .data import Ratings


class _NeighbourhoodModel:
    kind: str  # "user" or "item"

    def __init__(
        self,
        k: int = 20,
        shrinkage: float = 10.0,
        min_similarity: float = 0.0,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
    ):
        if k < 1:
            raise ValueError("k must be >= 1")
        if shrinkage < 0:
            raise ValueError("shrinkage must be >= 0")
        self.k = k
        self.shrinkage = shrinkage
        self.min_similarity = min_similarity
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.global_mean = 0.0
        self._ratings: dict = {}
        self._means: dict = {}
        self._sim: dict = {}
        self._n_users = 0
        self._n_items = 0

    # -- fitting ---------------------------------------------------------
    def fit(self, ratings: Ratings) -> "_NeighbourhoodModel":
        self._n_users = len(ratings.users)
        self._n_items = len(ratings.items)
        if not ratings.triples:
            raise ValueError("cannot fit on empty ratings")
        self.global_mean = float(
            np.mean([r for _, _, r in ratings.triples])
        )
        # Group ratings by the "axis" entity (user for user-based, item for
        # item-based) and by the "other" entity.
        axis_ratings: dict[int, dict[int, float]] = defaultdict(dict)
        other_sums: dict[int, list[float]] = defaultdict(list)
        for u, i, r in ratings.triples:
            axis = u if self.kind == "user" else i
            other = i if self.kind == "user" else u
            axis_ratings[axis][other] = r
            other_sums[other].append(r)
        self._means = {
            a: float(np.mean(list(rs.values()))) for a, rs in axis_ratings.items()
        }
        self._ratings = dict(axis_ratings)
        # Pairwise cosine similarity over mean-centred co-rated entries.
        axes = sorted(axis_ratings)
        centred = {
            a: {o: r - self._means[a] for o, r in rs.items()}
            for a, rs in axis_ratings.items()
        }
        norms = {
            a: math.sqrt(sum(v * v for v in rs.values())) for a, rs in centred.items()
        }
        sim: dict[tuple[int, int], float] = {}
        for x_pos, a in enumerate(axes):
            for b in axes[x_pos + 1:]:
                common = set(centred[a]) & set(centred[b])
                if not common:
                    continue
                num = sum(centred[a][o] * centred[b][o] for o in common)
                den = norms[a] * norms[b]
                if den == 0:
                    continue
                raw = num / den
                # Shrinkage: discount pairs with few co-rated items.
                value = raw * len(common) / (len(common) + self.shrinkage)
                if abs(value) >= self.min_similarity:
                    sim[(a, b)] = value
                    sim[(b, a)] = value
        self._sim = sim
        return self

    def _neighbours(self, axis: int) -> list[tuple[float, int]]:
        scored = [
            (sim, other)
            for (a, other), sim in self._sim.items()
            if a == axis
        ]
        scored.sort(key=lambda t: -t[0])
        return scored[: self.k]

    # -- prediction ------------------------------------------------------
    def predict(self, user: str, item: str, ratings: Ratings) -> float:
        """Predict a rating. `ratings` supplies the id mappings; the model
        itself holds the fitted rating table."""
        u = ratings.user_index.get(user)
        i = ratings.item_index.get(item)
        if u is None or i is None:
            return self.global_mean
        if self.kind == "user":
            axis, other = u, i
        else:
            axis, other = i, u
        neighbours = [
            (sim, nb)
            for sim, nb in self._neighbours(axis)
            if other in self._ratings.get(nb, {})
        ]
        if not neighbours:
            # Fallbacks: axis mean, then global mean.
            value = self._means.get(axis, self.global_mean)
        else:
            num = sum(
                sim * (self._ratings[nb][other] - self._means[nb])
                for sim, nb in neighbours
            )
            den = sum(abs(sim) for sim, _ in neighbours)
            if den == 0:
                value = self._means.get(axis, self.global_mean)
            else:
                value = self._means.get(axis, self.global_mean) + num / den
        # Neighbourhood deviations can overshoot the rating scale; clip.
        return float(min(self.max_rating, max(self.min_rating, value)))


class UserBasedCF(_NeighbourhoodModel):
    kind = "user"


class ItemBasedCF(_NeighbourhoodModel):
    kind = "item"
