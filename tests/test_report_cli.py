from bench.__main__ import main
from bench.collect.corpus import CORPUS_SCHEMA, write_corpus
from bench.points.points_file import POINTS_SCHEMA, write_points
from tests.github_fakes import FakeClient
from tests.score_fixtures import finding, inline_point, mnb_case, pr_record, round_case, write_run


def test_score_calibrate_report_end_to_end(tmp_path, monkeypatch, capsys):
    write_corpus({"schema": CORPUS_SCHEMA, "repo": "o/r", "pin": "p" * 40, "prs": [pr_record(1)]}, tmp_path / "corpus")
    points = [{**inline_point(f"p{n}", 1, 1, n, True), "acted_basis": "direct"} for n in range(1, 6)]
    write_points({"schema": POINTS_SCHEMA, "repo": "o/r", "points": points}, tmp_path / "points")
    run = tmp_path / "2026-10-08"
    run.mkdir()
    (run / "run.json").write_text('{"run_started_at": "2026-10-08T00:00:00+00:00"}', encoding="utf-8")
    for tool in ("every-hunk", "alpha"):
        write_run(run, tool, 1, "h1", [finding(2)], [round_case(1, 1)])
        write_run(run, tool, 1, "h2", [], [round_case(1, 2), mnb_case(1)])
    for module in ("bench.score.cli", "bench.report.cli"):
        monkeypatch.setattr(f"{module}.GitHubClient", lambda cache: FakeClient({}, {}))
    common = ["--run", str(run), "--points", str(tmp_path / "points")]
    results = ["--results", str(tmp_path / "results")]
    assert main(["score", *common[:2], "--corpus", str(tmp_path / "corpus"), "--points", str(tmp_path / "points"),
                 "--out", str(tmp_path / "results")]) == 0
    assert main(["calibrate", "draw", *common, *results, "--labels", str(tmp_path / "labels")]) == 0
    assert "0 reportable repo(s); short: 0 of 50" in capsys.readouterr().out   # below-minimum repo left out
    assert main(["calibrate", "draw", *common, *results]) == 1          # no silent redraw
    assert main(["report", *common, *results, "--labels", str(tmp_path / "labels")]) == 0
    page = (tmp_path / "results" / "summary.md").read_text()
    assert "# DriftBench results: 2026-10-08" in page and "**Insufficient data:**" in page
    assert "**unvalidated**" in page
