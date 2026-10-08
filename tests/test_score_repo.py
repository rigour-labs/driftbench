from bench.score.metrics import false_blocks
from bench.score.repo import score_repo
from tests.score_fixtures import (SameFileVersions, finding, inline_point, mnb_case, pr_record, round_case,
                                  write_run)


def corpus(prs: int = 1) -> dict:
    return {"repo": "o/r", "pin": "p", "prs": [pr_record(n) for n in range(1, prs + 1)]}


def points_file(points: list) -> dict:
    return {"points": points}


def test_catches_by_window_and_denominators(tmp_path):
    points = [inline_point("a", 1, 1, 10, True), inline_point("b", 1, 2, 50, False), inline_point("c", 1, 2, 80, None)]
    write_run(tmp_path, "tool", 1, "h1", [finding(12)], [round_case(1, 1)])
    write_run(tmp_path, "tool", 1, "h2", [finding(60), finding(81, blocking=True)], [round_case(1, 2), mnb_case(1)])
    summary, ledger = score_repo(corpus(), points_file(points), tmp_path, ["tool"], SameFileVersions())
    catches = summary["tools"]["tool"]["catches"]
    assert catches["0"]["all"]["caught"] == 0
    assert catches["3"]["all"] == {"caught": 2, "points": 3, "rate": 0.6667}      # a (2 away), c (1 away)
    assert catches["3"]["acted_on"] == {"caught": 1, "points": 1, "rate": 1.0}
    assert catches["10"]["all"]["caught"] == 3
    assert {row["point"]: row["distance"] for row in ledger} == {"a": 2, "b": 10, "c": 1}
    assert {row["point"]: row["blocking_distance"] for row in ledger} == {"a": None, "b": 31, "c": 1}
    assert summary["tools"]["tool"]["catches_blocking"]["3"]["all"]["caught"] == 1
    blocks = summary["tools"]["tool"]["false_blocks"]
    assert (blocks["approved_heads"], blocks["approved_heads_blocked"], blocks["blocks_per_approved_head"]) == (1, 1, 1.0)
    assert summary["tools"]["tool"]["findings_per_100_changed_lines"] == 1.5
    assert summary["corpus"]["points_acted_on"] == 1 and summary["corpus"]["acted_on_unknown"] == 1


def test_noise_label_against_the_every_hunk_ceiling(tmp_path):
    points = [inline_point("a", 1, 1, 10, True)]
    for tool, found in (("every-hunk", [finding(10)] * 20), ("sprayer", [finding(10)] * 15), ("quiet", [finding(10)])):
        write_run(tmp_path, tool, 1, "h1", found, [round_case(1, 1)])
    summary, _ = score_repo(corpus(), points_file(points), tmp_path, ["every-hunk", "sprayer", "quiet"],
                            SameFileVersions())
    labels = {name: metrics["label"] for name, metrics in summary["tools"].items()}
    assert labels == {"every-hunk": "ceiling", "sprayer": "noise", "quiet": None}


def test_minimums_decide_reportable(tmp_path):
    few, _ = score_repo(corpus(1), points_file([inline_point("a", 1, 1, 10, True)]), tmp_path, [], SameFileVersions())
    assert few["reportable"] is False
    many_points = [inline_point(f"p{n}", 1, 1, n, True) for n in range(1, 21)]
    many, _ = score_repo(corpus(10), points_file(many_points), tmp_path, [], SameFileVersions())
    assert many["reportable"] is True and many["corpus"]["approved_heads"] == 10


def test_false_blocks_groups():
    def record(source, overridden=False, blocks=0, verdict="pass"):
        return [{"verdict": verdict, "cases": [mnb_case(1, source, overridden)],
                 "findings": [finding(1, blocking=True)] * blocks}]
    result = false_blocks({1: record("approved", blocks=2), 2: record("approved"), 3: record("merged", blocks=1),
                           4: record("approved", overridden=True, blocks=1), 5: record("approved", verdict="error"),
                           6: [{"verdict": "pass", "cases": [round_case(6, 1)], "findings": []}]})
    assert result == {"approved_heads": 2, "approved_heads_blocked": 1, "false_block_rate": 0.5,
                      "blocks_per_approved_head": 1.0, "merged_fallback": {"heads": 1, "blocked": 1},
                      "overridden_excluded": 1, "not_scored": 1}
