import pytest

from recommender.cf import ItemBasedCF, UserBasedCF
from recommender.data import Ratings, split_random, synthetic_ratings
from recommender.evaluate import evaluate_ratings, recommend_top_n
from recommender.mf import MatrixFactorization


@pytest.fixture(scope="module")
def data():
    ratings = synthetic_ratings(n_users=60, n_items=40, seed=7)
    train, test = split_random(ratings, test_fraction=0.2, seed=42)
    return ratings, train, test


def test_user_cf_prediction_in_range_and_better_than_mean(data):
    _, train, test = data
    model = UserBasedCF().fit(train)
    preds = [model.predict(test.users[u], test.items[i], train) for u, i, _ in test.triples]
    assert all(1.0 <= p <= 5.0 for p in preds)
    global_mean = sum(r for _, _, r in train.triples) / len(train.triples)
    from recommender.metrics import rmse

    cf_rmse = rmse(preds, [r for _, _, r in test.triples])
    mean_rmse = rmse([global_mean] * len(test.triples), [r for _, _, r in test.triples])
    assert cf_rmse < mean_rmse


def test_item_cf_prediction_in_range(data):
    _, train, test = data
    model = ItemBasedCF(k=10).fit(train)
    for u, i, _ in test.triples[:20]:
        p = model.predict(test.users[u], test.items[i], train)
        assert 1.0 <= p <= 5.0


def test_cf_cold_start_falls_back_to_global_mean(data):
    _, train, _ = data
    model = UserBasedCF().fit(train)
    assert model.predict("nobody", "nothing", train) == pytest.approx(model.global_mean)


def test_cf_rejects_empty_fit():
    with pytest.raises(ValueError):
        UserBasedCF().fit(Ratings())


def test_mf_beats_global_mean_and_learns(data):
    _, train, test = data
    model = MatrixFactorization(n_factors=8, n_epochs=30, seed=42).fit(train)
    result = evaluate_ratings(lambda u, i: model.predict(u, i), test)
    global_mean = sum(r for _, _, r in train.triples) / len(train.triples)
    from recommender.metrics import rmse

    mean_rmse = rmse([global_mean] * len(test.triples), [r for _, _, r in test.triples])
    assert result["rmse"] < mean_rmse
    # Training error should improve over epochs.
    assert model.train_rmse_history[0] > model.train_rmse_history[-1]
    assert model.n_epochs_run >= 1


def test_mf_predictions_clipped_and_cold_start(data):
    _, train, _ = data
    model = MatrixFactorization(n_epochs=5).fit(train)
    assert 1.0 <= model.predict("nobody", "nothing") <= 5.0
    assert 1.0 <= model.predict(train.users[0], "nothing") <= 5.0


def test_mf_rejects_empty_fit():
    with pytest.raises(ValueError):
        MatrixFactorization().fit(Ratings())


def test_recommend_top_n_excludes_rated(data):
    ratings, train, _ = data
    model = MatrixFactorization(n_epochs=5).fit(train)
    user = train.users[0]
    rated = {i for u, i, _ in train.triples if u == 0}
    recs = recommend_top_n(lambda u, i: model.predict(u, i), train, user, n=5)
    assert len(recs) == 5
    assert not set(recs) & {train.items[i] for i in rated}
    # Sorted by descending predicted score.
    scores = [model.predict(user, item) for item in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommend_cold_start_user(data):
    ratings, train, _ = data
    model = MatrixFactorization(n_epochs=5).fit(train)
    recs = recommend_top_n(lambda u, i: model.predict(u, i), train, "brand_new_user", n=3)
    assert len(recs) == 3
