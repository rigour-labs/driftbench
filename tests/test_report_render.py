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
    assert "these intervals are wide" in page                              # caveat on the class table itself
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
                                      "range calibration agreement (all repos): not yet hand-checked.")
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


def test_model_assisted_labels_report_blind_agreement_and_anchoring(tmp_path):
    result, ledger = summary(tmp_path, prs=10, points=25)
    classes = per_class(ledger, {"p1": "mechanical"})
    agreement = {"model": "example-org/example-model", "blind": {"agree": 8, "points": 10, "pre_suggestion": 0, "rate": 0.8},
                 "anchored": {"accepted": 30, "overridden": 6}, "unseen": 2}
    page = render("r", [result], {"o/r": classes}, {"validated": False, "location": {}, "acted_on": {}},
                  {}, {"o/r": agreement})
    assert "assisted by model suggestions (example-org/example-model)" in page
    assert "agree with the model on 8 of 10: 80%" in page and "accepted 30 and overrode 6" in page
    few = {**agreement, "blind": {"agree": 4, "points": 6, "rate": None}}
    page = render("r", [result], {"o/r": classes}, {"validated": False, "location": {}, "acted_on": {}},
                  {}, {"o/r": few})
    assert "too few for a rate (n<10)" in page
    pre = {**agreement, "blind": {"agree": 40, "points": 48, "pre_suggestion": 40, "rate": 0.8333}}
    page = render("r", [result], {"o/r": classes}, {"validated": False, "location": {}, "acted_on": {}},
                  {}, {"o/r": pre})
    assert "40 of 48 (40 of them blind: pre-suggestion" in page and ": 83%" in page


def test_ai_consensus_labels_are_named_with_their_disagreements_next_to_the_class_table(tmp_path):
    result, ledger = summary(tmp_path, prs=10, points=25)
    classes = per_class(ledger, {"p1": "mechanical"})
    consensus = {"sources": {"first": "claude-opus-5-5", "second": "example-org/example-model"},
                 "first_role": "a Claude model working as DriftBench's builder, which knows the benchmark and its guide",
                 "both_labelled": 48, "agreed": 42, "kappa": 0.81,
                 "dropped_by_class": {"first": {"judgment": 4, "mechanical": 2}, "second": {"mechanical": 6}}}
    page = render("r", [result], {"o/r": classes}, {"validated": False, "location": {}, "acted_on": {}},
                  {}, {"o/r": {"consensus": consensus}})
    assert "By class (N=3), from AI-consensus labels on the random sample only" in page
    assert "working as DriftBench's builder, which knows the benchmark and its guide" in page
    assert "agree on 42 of 48 points (kappa 0.81)" in page
    table_end = page.index("Dropped as disagreements")
    assert page.index("| Entrant | mechanical") < table_end                    # right after the class table
    assert "claude-opus-5-5: judgment 4, mechanical 2; example-org/example-model: mechanical 6" in page
    assert "human-labelled" not in page and "agree with the model" not in page


def test_the_spot_check_sits_under_the_headline_table_and_the_basis_note_follows_the_data(tmp_path):
    from bench.report.calibration import summarise_calibration
    from bench.report.markdown import basis_note
    result, _ = summary(tmp_path, prs=10, points=25)
    entries = [{"kind": "location", "tool": "every-hunk", "repo": "o/r", "point": f"p{n}", "verdict": v,
                "verdict_by": "model: example-org/example-model"} for n, v in enumerate(("no", "no", "partly"))]
    entries += [{"kind": "acted_on", "repo": "o/r", "point": f"a{n}", "basis": b, "acted_on": True, "verdict": v,
                 "verdict_by": "consensus: x + y"}
                for n, (b, v) in enumerate([("ancestor", "yes")] * 4 + [("ancestor", "no")] + [("direct", "yes")] * 8
                                           + [("direct", "no")] * 3)]
    calibration = summarise_calibration({"seed": 1, "short": [], "entries": entries})
    page = render("r", [result], {}, calibration)
    line = "- every-hunk: same issue on spot-check: 0 of 3 checked, partly 1 (AI verdict, non-Claude model)."
    assert line in page
    assert page.index("| every-hunk | 1 |") < page.index(line) < page.index("False blocks in detail")
    assert "not clearly at these sample sizes; acted-on is pooled" in page
    assert basis_note({"ancestor": {"agree": 30}, "range": {"agree": 2, "disagree": 28}}).startswith(
        "Per-basis agreement differs clearly")
    assert basis_note({"direct": {"agree": 3}}) == ""
