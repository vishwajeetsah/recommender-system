"""Command-line interface for the recommender.

Examples:
    python -m recommender.cli generate --out data/sample_ratings.csv
    python -m recommender.cli evaluate --csv data/sample_ratings.csv
    python -m recommender.cli recommend --csv data/sample_ratings.csv \\
        --user user_000 --n 5 --model mf
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .cf import ItemBasedCF, UserBasedCF
from .data import Ratings, split_random, synthetic_ratings
from .evaluate import evaluate_ratings, recommend_top_n
from .mf import MatrixFactorization


def _build_model(name: str, **kwargs):
    if name == "mf":
        return MatrixFactorization(**kwargs)
    if name == "user_cf":
        return UserBasedCF()
    if name == "item_cf":
        return ItemBasedCF()
    raise ValueError(f"unknown model: {name} (choose mf, user_cf, item_cf)")


def _predict_fn(model, name: str, train: Ratings):
    if name == "mf":
        return lambda user, item: model.predict(user, item)
    return lambda user, item: model.predict(user, item, train)


def cmd_generate(args) -> int:
    ratings = synthetic_ratings(
        n_users=args.users, n_items=args.items, seed=args.seed
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["user_id", "item_id", "rating"])
        for u, i, r in ratings.triples:
            writer.writerow([ratings.users[u], ratings.items[i], r])
    print(f"wrote {len(ratings)} ratings to {out}")
    return 0


def cmd_evaluate(args) -> int:
    ratings = Ratings.from_csv(args.csv)
    train, test = split_random(ratings, test_fraction=args.test_fraction, seed=42)
    results = {}
    for name in args.models:
        model = _build_model(name)
        model.fit(train)
        results[name] = evaluate_ratings(_predict_fn(model, name, train), test)
    # Global-mean baseline for context.
    mean = sum(r for _, _, r in train.triples) / len(train.triples)
    results["global_mean"] = evaluate_ratings(lambda u, i: mean, test)
    print(json.dumps(results, indent=2))
    return 0


def cmd_recommend(args) -> int:
    ratings = Ratings.from_csv(args.csv)
    model = _build_model(args.model)
    model.fit(ratings)
    recs = recommend_top_n(
        _predict_fn(model, args.model, ratings), ratings, args.user, n=args.n
    )
    for rank, item in enumerate(recs, start=1):
        print(f"{rank}. {item}  (predicted {model_predict(model, args.model, ratings, args.user, item):.2f})")
    return 0


def model_predict(model, name: str, train: Ratings, user: str, item: str) -> float:
    if name == "mf":
        return model.predict(user, item)
    return model.predict(user, item, train)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="recommender")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="generate synthetic ratings CSV")
    p_gen.add_argument("--out", default="data/sample_ratings.csv")
    p_gen.add_argument("--users", type=int, default=60)
    p_gen.add_argument("--items", type=int, default=40)
    p_gen.add_argument("--seed", type=int, default=7)
    p_gen.set_defaults(func=cmd_generate)

    p_eval = sub.add_parser("evaluate", help="train models and report RMSE/MAE")
    p_eval.add_argument("--csv", required=True)
    p_eval.add_argument("--models", nargs="+", default=["mf", "user_cf", "item_cf"])
    p_eval.add_argument("--test-fraction", type=float, default=0.2)
    p_eval.set_defaults(func=cmd_evaluate)

    p_rec = sub.add_parser("recommend", help="top-N recommendations for a user")
    p_rec.add_argument("--csv", required=True)
    p_rec.add_argument("--user", required=True)
    p_rec.add_argument("--n", type=int, default=10)
    p_rec.add_argument("--model", default="mf", choices=["mf", "user_cf", "item_cf"])
    p_rec.set_defaults(func=cmd_recommend)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
