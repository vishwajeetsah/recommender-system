"""End-to-end demo: synthetic data -> train MF -> evaluate -> recommend."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommender import (  # noqa: E402
    MatrixFactorization,
    evaluate_ratings,
    recommend_top_n,
    split_random,
    synthetic_ratings,
)


def main() -> None:
    ratings = synthetic_ratings(n_users=60, n_items=40, seed=7)
    train, test = split_random(ratings, test_fraction=0.2, seed=42)
    model = MatrixFactorization(n_factors=8, n_epochs=30, seed=42).fit(train)
    result = evaluate_ratings(lambda u, i: model.predict(u, i), test)
    print(f"held-out: RMSE={result['rmse']:.4f}  MAE={result['mae']:.4f}  (n={result['n']})")
    user = train.users[0]
    print(f"\nTop 5 recommendations for {user}:")
    for rank, item in enumerate(
        recommend_top_n(lambda u, i: model.predict(u, i), train, user, n=5), 1
    ):
        print(f"  {rank}. {item} (predicted {model.predict(user, item):.2f})")


if __name__ == "__main__":
    main()
