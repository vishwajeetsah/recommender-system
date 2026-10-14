"""Biased matrix factorisation trained with SGD.

Model: r_hat(u, i) = global_mean + b_u + b_i + p_u . q_i

Trained with L2-regularised SGD over observed ratings, with early stopping
on a held-out validation slice. Unknown users/items at predict time fall
back to the item mean, user mean, then the global mean. Predictions are
clipped to the observed rating range (default 1-5) so the model can never
emit an absurd rating.
"""

from __future__ import annotations

import numpy as np

from .data import Ratings


class MatrixFactorization:
    def __init__(
        self,
        n_factors: int = 16,
        lr: float = 0.01,
        reg: float = 0.02,
        n_epochs: int = 30,
        seed: int = 42,
        val_fraction: float = 0.1,
        patience: int = 5,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
    ):
        if n_factors < 1:
            raise ValueError("n_factors must be >= 1")
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed
        self.val_fraction = val_fraction
        self.patience = patience
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.global_mean = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.train_rmse_history: list[float] = []
        self.val_rmse_history: list[float] = []
        self.n_epochs_run = 0
        self._item_means: dict[int, float] = {}
        self._user_means: dict[int, float] = {}
        self._user_index: dict = {}
        self._item_index: dict = {}

    def fit(self, ratings: Ratings) -> "MatrixFactorization":
        if not ratings.triples:
            raise ValueError("cannot fit on empty ratings")
        rng = np.random.default_rng(self.seed)
        self._user_index = dict(ratings.user_index)
        self._item_index = dict(ratings.item_index)
        n_users, n_items = len(ratings.users), len(ratings.items)
        triples = np.array(
            [(u, i, r) for u, i, r in ratings.triples], dtype=float
        )
        self.global_mean = float(triples[:, 2].mean())
        # Validation split for early stopping (deterministic shuffle).
        order = rng.permutation(len(triples))
        n_val = int(len(triples) * self.val_fraction) if self.val_fraction else 0
        val_idx, train_idx = order[:n_val], order[n_val:]
        train = triples[train_idx]
        val = triples[val_idx] if n_val else triples[:0]

        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.user_factors = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (n_items, self.n_factors))

        # Fallback means from the training slice.
        for arr, key, store in (
            (train[:, 0], 0, self._user_means),
            (train[:, 1], 1, self._item_means),
        ):
            for idx in np.unique(arr.astype(int)):
                mask = train[:, key] == idx
                store[int(idx)] = float(train[mask, 2].mean())

        best_val = float("inf")
        best_state = None
        epochs_without_improve = 0
        for epoch in range(self.n_epochs):
            rng.shuffle(train)
            for u, i, r in train:
                u, i = int(u), int(i)
                pred = self._raw_predict(u, i)
                err = r - pred
                bu = self.user_bias[u]
                bi = self.item_bias[i]
                pu = self.user_factors[u].copy()
                qi = self.item_factors[i].copy()
                self.user_bias[u] += self.lr * (err - self.reg * bu)
                self.item_bias[i] += self.lr * (err - self.reg * bi)
                self.user_factors[u] += self.lr * (err * qi - self.reg * pu)
                self.item_factors[i] += self.lr * (err * pu - self.reg * qi)
            self.n_epochs_run = epoch + 1
            train_rmse = self._rmse_on(train)
            self.train_rmse_history.append(train_rmse)
            if n_val:
                val_rmse = self._rmse_on(val)
                self.val_rmse_history.append(val_rmse)
                if val_rmse < best_val - 1e-4:
                    best_val = val_rmse
                    best_state = (
                        self.user_bias.copy(),
                        self.item_bias.copy(),
                        self.user_factors.copy(),
                        self.item_factors.copy(),
                    )
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                    if epochs_without_improve >= self.patience:
                        break
        if best_state is not None:
            (self.user_bias, self.item_bias,
             self.user_factors, self.item_factors) = best_state
        return self

    def _raw_predict(self, u: int, i: int) -> float:
        return float(
            self.global_mean
            + self.user_bias[u]
            + self.item_bias[i]
            + self.user_factors[u] @ self.item_factors[i]
        )

    def _rmse_on(self, triples: np.ndarray) -> float:
        if len(triples) == 0:
            return 0.0
        preds = np.array([self._raw_predict(int(u), int(i)) for u, i, _ in triples])
        return float(np.sqrt(np.mean((preds - triples[:, 2]) ** 2)))

    def predict(self, user: str, item: str) -> float:
        u = self._user_index.get(user)
        i = self._item_index.get(item)
        if u is None and i is None:
            value = self.global_mean
        elif u is None:
            value = self._item_means.get(i, self.global_mean)
        elif i is None:
            value = self._user_means.get(u, self.global_mean)
        else:
            value = self._raw_predict(u, i)
        return float(min(self.max_rating, max(self.min_rating, value)))
