from bench.report.classes import per_class
from bench.report.markdown import pct, render
from bench.score.repo import score_repo
from tests.score_fixtures import SameFileVersions, finding, inline_point, mnb_case, pr_record, round_case, write_run


def summary(tmp_path, prs: int, points: int):
    corpus = {"repo": "o/r", "pin": "p" * 40, "prs": [pr_record(n) for n in range(1, prs + 1)]}
    pts = [inline_point(f"p{n}", 1, 1, n, True) for n in range(1, points + 1)]
    write_run(tmp_path, "every-hunk", 1, "h1", [{**finding(1), "end_line": 40}], [round_case(1, 1)])
    write_run(tmp_path, "every-hunk", 1, "h2", [], [round_case(1, 2), mnb_case(1)])
    return score_repo(corpus, {"points": pts}, tmp_path, ["every-hunk"], SameFileVersions())


def test_pct_always_shows_n_and_interval():
    assert pct(10, 20) == "50% (10/20; 95% CI 30 to 70%)" and pct(0, 0) == "n/a (n=0)"


def test_reportable_repo_renders_tables_with_intervals_and_unvalidated_status(tmp_path):
    result, ledger = summary(tmp_path, prs=10, points=25)
    classes = per_class(ledger, {"p1": "mechanical", "p2": "judgment"})
    page = render("2026-10-08", [result], {"o/r": classes}, {"validated": False, "location": {}, "acted_on": {}})
    assert "## o/r" in page and "| every-hunk | 1 | 100% (25/25; 95% CI 87 to 100%)" in page
    assert "Sensitivity" in page and "By class" in page and "unclassified" in page
    assert "**unvalidated**" in page and "—" not in page


def test_small_repo_says_insufficient_data(tmp_path):
    result, _ = summary(tmp_path, prs=1, points=3)
    page = render("2026-10-08", [result], {}, {"validated": True, "short": ["short: 3 of 50 location matches"],
                                               "location": {"alpha": {"yes": 3}}, "acted_on": {}})
    assert "**Insufficient data:** 3 acted-on points (minimum 20)" in page and "| Entrant |" not in page
    assert "Hand-checked sample: complete (short: 3 of 50 location matches)." in page
    assert "alpha, same issue at a location match: yes 3" in page and "left out of the calibration sample" in page


def test_per_class_uses_only_given_labels():
    ledger = [{"tool": "t", "point": "a", "distance": 0}, {"tool": "t", "point": "b", "distance": 9},
              {"tool": "t", "point": "c", "distance": None}]
    assert per_class(ledger, {"a": "mechanical", "b": "mechanical"}) == {
        "t": {"mechanical": {"caught": 1, "points": 2}, "unclassified": {"caught": 0, "points": 1}}}


def test_withheld_per_class_note_replaces_the_table(tmp_path):
    result, ledger = summary(tmp_path, prs=10, points=25)
    classes = per_class(ledger, {"p1": "mechanical"})
    page = render("r", [result], {"o/r": classes}, {"validated": False, "location": {}, "acted_on": {}},
                  {"o/r": "o__r.yaml was committed after the run started"})
    assert "Results by class withheld: o__r.yaml was committed after the run started." in page
    assert "By class" not in page


def test_range_note_when_acted_on_rests_mostly_on_range(tmp_path):
    from bench.report.markdown import range_note
    result, _ = summary(tmp_path, prs=10, points=25)
    result["corpus"]["acted_on_by_basis"] = {"range": 20, "direct": 5}
    assert range_note(result, {}) == ("Acted-on points here rest mostly on the `range` basis (20 of 25); "
                                      "its calibration agreement: not yet hand-checked.")
    checked = {"acted_on": {"range": {"agree": 8, "disagree": 2}}}
    assert "agree 8, disagree 2" in range_note(result, checked)
    page = render("r", [result], {}, {"validated": False, "short": [], "location": {}, "acted_on": {}})
    assert "rest mostly on the `range` basis" in page
    result["corpus"]["acted_on_by_basis"] = {"range": 10, "ancestor": 15}
    assert range_note(result, checked) == ""


def test_a_tool_that_never_blocks_shows_n_a_not_zero(tmp_path):
    from bench.report.markdown import block_row, tool_row
    result, _ = summary(tmp_path, prs=10, points=25)
    metrics = {**result["tools"]["every-hunk"], "catches_blocking": None, "false_blocks": None,
               "blocking_semantics": False}
    row = tool_row("claude-code-review", metrics)
    assert row.count("n/a (never blocks)") == 3 and "0%" not in row.split("|")[5]
    assert block_row("claude-code-review", metrics).count("n/a (never blocks)") == 4
