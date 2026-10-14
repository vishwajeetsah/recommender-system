import json

from recommender.cli import main
from recommender.data import Ratings


def test_cli_generate_and_evaluate(tmp_path, capsys):
    out = tmp_path / "ratings.csv"
    assert main(["generate", "--out", str(out), "--users", "30", "--items", "20"]) == 0
    assert out.exists()
    assert len(Ratings.from_csv(out)) > 0

    assert main(["evaluate", "--csv", str(out), "--models", "mf"]) == 0
    captured = capsys.readouterr().out
    results = json.loads(captured[captured.index("{"):])
    assert "mf" in results and "global_mean" in results
    assert results["mf"]["rmse"] < results["global_mean"]["rmse"]


def test_cli_recommend(tmp_path, capsys):
    out = tmp_path / "ratings.csv"
    main(["generate", "--out", str(out), "--users", "30", "--items", "20"])
    capsys.readouterr()  # discard the generate line
    assert main(["recommend", "--csv", str(out), "--user", "user_000", "--n", "3"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("1.")
