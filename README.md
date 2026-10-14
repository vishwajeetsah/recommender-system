# recommender-system

Collaborative-filtering recommender built from scratch in NumPy — no
Surprise, no LightFM, no black boxes. Two classic approaches, side by side:

- **Neighbourhood CF** (user-based and item-based): mean-centred cosine
  similarity with shrinkage, similarity-weighted deviation predictions.
- **Biased matrix factorisation**: `r̂ = μ + b_u + b_i + p_u · q_i`, trained
  with regularised SGD and early stopping on a held-out validation slice.

Plus the evaluation toolkit that makes a recommender honest: RMSE / MAE for
rating prediction, and Precision@K / Recall@K / nDCG@K for ranking quality.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic ratings (low-rank latent-factor model, seed 7)
python -m recommender.cli generate --out data/sample_ratings.csv

# Train all three models and compare on a held-out split
python -m recommender.cli evaluate --csv data/sample_ratings.csv

# Top-5 recommendations for one user (matrix factorisation)
python -m recommender.cli recommend --csv data/sample_ratings.csv \
    --user user_000 --n 5 --model mf
```

Or run the end-to-end demo:

```bash
PYTHONPATH=src python examples/demo.py
```

Use your own data with any CSV that has `user_id,item_id,rating` columns.

## How it works

**Data** (`data.py`) — `Ratings` maps arbitrary user/item IDs to contiguous
indices. `split_random` holds out 20% of ratings but guarantees every user
keeps at least one rating in train, so test users are never fully cold-start.
`synthetic_ratings` generates plausible data from a true low-rank model
(latent factors + user/item biases + noise, clipped to [1, 5]), so there is
always a dataset to experiment with and the models have real signal to find.

**Neighbourhood CF** (`cf.py`) — similarity is cosine over mean-centred
ratings, discounted by `|common| / (|common| + shrinkage)` so pairs sharing
one or two co-rated items can't claim a perfect similarity. Predictions are
similarity-weighted deviations from neighbours' means, falling back to the
user/item mean, then the global mean. Predictions are clipped to [1, 5].

**Matrix factorisation** (`mf.py`) — biases plus latent factors, SGD with L2
regularisation. Early stopping watches a 10% validation slice and restores
the best weights. Unknown users/items fall back to item mean / user mean /
global mean, so cold-start queries return something sane instead of crashing.

**Evaluation** (`evaluate.py`, `metrics.py`) — `evaluate_ratings` scores a
model on held-out ratings; `recommend_top_n` returns the highest-predicted
items the user hasn't rated yet. Ranking metrics cover graded relevance
(nDCG) as well as binary hit/miss (Precision@K, Recall@K).

## Measured results

On the bundled synthetic data (60 users × 40 items, 566 ratings, 20% held
out, n = 89 test ratings), from `python -m recommender.cli evaluate`:

| model | RMSE | MAE |
|---|---|---|
| item-based CF | **1.088** | **0.822** |
| user-based CF | 1.140 | 0.872 |
| matrix factorisation (8 factors) | 1.187 | 0.944 |
| global-mean baseline | 1.196 | 0.976 |

All three models beat the baseline, and item-based CF wins on this small
synthetic set — a genuine, expected result: neighbourhood methods are strong
when the item catalogue is small and dense, while factorisation usually pulls
ahead on larger, sparser data. Try `--users 300 --items 200` and compare.

Test suite: **25 tests pass** (`pytest`, covering data loading/splitting,
metrics against hand-computed values, both CF variants, MF learning and
cold-start behaviour, top-N exclusion of already-rated items, and the CLI).

## Project structure

```
src/recommender/
  data.py       Ratings container, CSV loading, train/test split, synthetic data
  cf.py         UserBasedCF / ItemBasedCF (cosine + shrinkage)
  mf.py         MatrixFactorization (biased SGD + early stopping)
  metrics.py    RMSE, MAE, Precision@K, Recall@K, nDCG@K
  evaluate.py   held-out rating evaluation + top-N recommendation
  cli.py        generate / evaluate / recommend commands
tests/          pytest suite (25 tests)
examples/demo.py  end-to-end demo
data/sample_ratings.csv  generated synthetic ratings
```

## Limitations (honest)

- **Explicit feedback only** — it models 1–5 star ratings, not clicks, views,
  or other implicit signals (those need a different loss, e.g. BPR).
- **No content features** — new items with no ratings get only the global
  mean; there is no genre/title/description model to fall back on.
- **CF is O(n²) in users/items** — the pairwise similarity table is fine for
  thousands of users, not millions. Production systems use approximate
  nearest neighbours or factorisation alone.
- **Synthetic data is well-behaved** — real ratings are messier (selection
  bias, spam, drift). The numbers above describe the demo set, not a claim
  about MovieLens or production data.
- **Popularity bias is uncorrected** — top-N lists lean towards widely liked
  items; no diversity or novelty re-ranking is implemented.

## What I Learned

- **Why the baseline matters.** The global-mean RMSE (1.196) is the number
  every model has to beat before it means anything. One of my first test
  configs — user-CF with k=10, shrinkage=10 — scored 1.239, *worse* than
  guessing the mean. Without the baseline in the same table, that failure
  would have looked like a plausible result.
- **Shrinkage is the quiet hero of neighbourhood methods.** Raw cosine
  similarity treats two users who co-rated one item as confidently as two
  who co-rated fifty. Multiplying by `|common| / (|common| + shrinkage)`
  fixed most of the overconfident-neighbour damage.
- **Clip your predictions.** Neighbourhood deviation predictions can overshoot
  the 1–5 scale; an unclipped model can emit a "6.2 star" prediction and no
  metric will complain. The fix is one line, but the test that caught it
  (asserting every prediction is in range) is the real lesson.
- **Small data flatters simple models.** Item-CF beating matrix factorisation
  here is not an upset — factorisation needs enough data to earn its
  parameters. Comparing models on one dataset size and declaring a winner is
  how bad conclusions get shipped.
- **Determinism is a debugging superpower.** Seeded data generation, splits,
  and SGD meant every failure reproduced exactly, so the two real bugs found
  during testing (unclipped CF output, a test counting its own log line)
  could be fixed and re-verified in seconds.
