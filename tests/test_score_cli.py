import pytest

from bench.__main__ import main
from bench.collect.corpus import CORPUS_SCHEMA, write_corpus
from bench.points.points_file import POINTS_SCHEMA, write_points
from bench.score.cli import ScoreFileError, read_ledger, read_summary
from tests.github_fakes import FakeClient
from tests.score_fixtures import finding, inline_point, mnb_case, pr_record, round_case, write_run


def test_score_writes_summary_and_ledger(tmp_path, monkeypatch):
    write_corpus({"schema": CORPUS_SCHEMA, "repo": "o/r", "pin": "p", "prs": [pr_record(1)]}, tmp_path / "corpus")
    write_points({"schema": POINTS_SCHEMA, "repo": "o/r", "points": [inline_point("a", 1, 1, 10, True)]},
                 tmp_path / "points")
    run = tmp_path / "runs" / "2026-10-08"
    write_run(run, "every-hunk", 1, "h1", [finding(10)], [round_case(1, 1)])
    write_run(run, "every-hunk", 1, "h2", [], [round_case(1, 2), mnb_case(1)])
    (run / "_scratch").mkdir()
    monkeypatch.setattr("bench.score.cli.GitHubClient", lambda cache: FakeClient({}, {}))
    args = ["score", "--run", str(run), "--corpus", str(tmp_path / "corpus"), "--points", str(tmp_path / "points"),
            "--out", str(tmp_path / "results")]
    assert main(args) == 0
    published = read_summary(tmp_path / "results" / "o__r.json")
    assert published["reportable"] is False and "tools" not in published   # no numbers below the minimums
    assert "1 acted-on points (minimum 20)" in published["insufficient_data"]
    full = read_summary(run / "scores" / "o__r.json")                      # full metrics stay with the run
    assert list(full["tools"]) == ["every-hunk"] and full["tools"]["every-hunk"]["catches"]["0"]["all"]["caught"] == 1
    ledger = read_ledger(run / "ledger.jsonl")
    assert ledger == [{"tool": "every-hunk", "repo": "o/r", "point": "a", "pr": 1, "round": 1, "acted_on": True, "distance": 0,
                       "blocking_distance": None, "head_sha": "h1", "finding": 0, "mapped_line": 10}]


def test_score_fails_cleanly_without_points(tmp_path, monkeypatch):
    write_corpus({"schema": CORPUS_SCHEMA, "repo": "o/r", "pin": "p", "prs": []}, tmp_path / "corpus")
    (tmp_path / "run").mkdir()
    monkeypatch.setattr("bench.score.cli.GitHubClient", lambda cache: FakeClient({}, {}))
    assert main(["score", "--run", str(tmp_path / "run"), "--corpus", str(tmp_path / "corpus"),
                 "--points", str(tmp_path / "none")]) == 1


def test_readers_reject_bad_files(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ScoreFileError, match="not a score summary"):
        read_summary(bad)
    bad.write_text("{oops\n", encoding="utf-8")
    with pytest.raises(ScoreFileError, match="cannot read ledger"):
        read_ledger(bad)
