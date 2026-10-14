import csv

import pytest

from recommender.data import Ratings, split_random, synthetic_ratings


def _small():
    return Ratings.from_triples(
        [
            ("u1", "i1", 5.0),
            ("u1", "i2", 3.0),
            ("u1", "i3", 4.0),
            ("u2", "i1", 4.0),
            ("u2", "i2", 2.0),
            ("u2", "i4", 5.0),
            ("u3", "i2", 4.0),
            ("u3", "i3", 5.0),
            ("u3", "i4", 3.0),
        ]
    )


def test_from_triples_builds_mappings():
    r = _small()
    assert r.users == ["u1", "u2", "u3"]
    assert r.items == ["i1", "i2", "i3", "i4"]
    assert len(r) == 9


def test_from_csv_roundtrip(tmp_path):
    path = tmp_path / "r.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "item_id", "rating"])
        w.writerow(["alice", "movie_a", 4.5])
        w.writerow(["bob", "movie_b", 2.0])
    r = Ratings.from_csv(path)
    assert r.users == ["alice", "bob"]
    assert len(r) == 2


def test_from_csv_bad_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError):
        Ratings.from_csv(path)


def test_split_keeps_every_user_in_train():
    r = synthetic_ratings(n_users=20, n_items=10, seed=1)
    train, test = split_random(r, test_fraction=0.2, seed=3)
    train_users = {u for u, _, _ in train.triples}
    assert train_users == {u for u, _, _ in r.triples}
    assert len(train) + len(test) == len(r)
    assert 0 < len(test) < len(r)


def test_split_deterministic():
    r = synthetic_ratings(n_users=20, n_items=10, seed=1)
    a = split_random(r, seed=9)
    b = split_random(r, seed=9)
    assert a[0].triples == b[0].triples and a[1].triples == b[1].triples


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        split_random(_small(), test_fraction=1.5)


def test_synthetic_ratings_in_range_and_deterministic():
    a = synthetic_ratings(n_users=10, n_items=8, seed=5)
    b = synthetic_ratings(n_users=10, n_items=8, seed=5)
    assert a.triples == b.triples
    assert all(1.0 <= r <= 5.0 for _, _, r in a.triples)
    assert len(a) > 0
