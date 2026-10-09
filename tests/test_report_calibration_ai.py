import pytest

from bench.harness.budget import Budget
from bench.labels.openrouter import OpenRouterError
from bench.report.calibration import summarise_calibration
from bench.report.calibration_ai import ENTRANT, VerdictError, entry_key, judge, merge_verdicts, parse_verdict
from bench.report.calibration_evidence import around, window_diff
from bench.report.markdown import calibration_status

ACTED = [{"kind": "acted_on", "repo": "o/r", "point": f"a{i}", "pr": 1, "basis": "direct", "acted_on": i < 2,
          "verdict": None} for i in range(4)]
LOCATED = {"kind": "location", "tool": "rigour", "repo": "o/r", "point": "l0", "pr": 1, "head_sha": "h",
           "finding": 0, "verdict": None}


def calibration(*extra):
    return {"seed": 1, "short": [], "entries": [*ACTED, LOCATED, *extra]}


def model(verdicts):
    return {"model": "example-org/example-model", "verdicts": {k: {"verdict": v} for k, v in verdicts.items()}}


def keys():
    return [entry_key(e) for e in ACTED], entry_key(LOCATED)


def test_acted_on_needs_both_judges_and_location_takes_the_model_alone():
    acted, located = keys()
    claude = {"labeller": "claude-opus-5-5", "verdicts": {acted[0]: "yes", acted[1]: "no", acted[2]: "no"}}
    merged = merge_verdicts(calibration(), claude, model({acted[0]: "yes", acted[1]: "yes", acted[2]: "no",
                                                          acted[3]: "no", located: "partly"}))
    by_point = {e["point"]: e for e in merged["entries"]}
    assert by_point["a0"]["verdict"] == "yes" and by_point["a0"]["verdict_by"].startswith("consensus: claude-opus-5-5")
    assert by_point["a1"]["verdict"] is None and by_point["a1"]["disputed"] is True
    assert by_point["a3"]["verdict"] is None and "disputed" not in by_point["a3"]          # only the model judged it
    assert by_point["l0"] == {**LOCATED, "verdict": "partly", "verdict_by": "model: example-org/example-model"}


def test_a_claude_labeller_may_not_judge_an_entrant_and_human_verdicts_win():
    acted, located = keys()
    with pytest.raises(VerdictError, match="Rigour's reviewer runs on Claude"):
        merge_verdicts(calibration(), {"verdicts": {located: "yes"}}, model({}))
    human = {**ACTED[0], "point": "h0", "verdict": "no"}
    merged = merge_verdicts(calibration(human), {"verdicts": {}}, model({entry_key(human): "yes"}))
    kept = next(e for e in merged["entries"] if e["point"] == "h0")
    assert kept["verdict"] == "no" and kept["verdict_by"] == "human"


def test_agreement_matches_the_scorer_and_ai_verdicts_are_named_in_the_status():
    acted, located = keys()
    claude = {"labeller": "claude-opus-5-5", "verdicts": {acted[0]: "yes", acted[1]: "no", acted[2]: "yes", acted[3]: "no"}}
    merged = merge_verdicts(calibration(), claude, model({acted[0]: "yes", acted[1]: "yes", acted[2]: "yes",
                                                          acted[3]: "no", located: "no"}))
    summary = summarise_calibration(merged)
    assert summary["validated"] is True and summary["disputed"] == 1
    assert summary["acted_on"]["direct"] == {"agree": 2, "disagree": 1}   # a0 yes/yes, a3 no/no agree; a2 yes on a no
    status = calibration_status(summary)
    assert "with AI verdicts, not a human check" in status and "1 acted-on entries disputed" in status
    assert "model: example-org/example-model: 1" in status


def test_a_model_verdict_is_charged_and_validated():
    calls = []

    def reply(messages):
        calls.append(messages)
        return {"content": '{"verdict": "partly", "reason": "related"}', "cost_usd": 0.001, "served_model": "m"}
    budget = Budget(1.0, {ENTRANT: 0.05})
    result, why = judge(LOCATED, "evidence text", reply, budget)
    assert result["verdict"] == "partly" and why == "" and budget.spent == 0.001
    assert "same issue" in calls[0][0]["content"] and calls[0][1]["content"] == "evidence text"
    with pytest.raises(VerdictError):
        parse_verdict('{"verdict": "partly"}', "acted_on")                 # partly is for location only

    def fail(messages):
        raise OpenRouterError("OpenRouter answered HTTP 500")
    result, why = judge(ACTED[0], "e", fail, budget)
    assert result is None and "500" in why and budget.spent == 0.051


def test_the_evidence_shows_only_the_change_near_the_anchor():
    old = "\n".join(f"line {n}" for n in range(1, 101))
    new = old.replace("line 50", "line 50 changed").replace("line 95", "line 95 changed")
    diff = window_diff(old, new, 48)
    assert "+line 50 changed" in diff and "95" not in diff
    assert window_diff(old, old, 48) == "(no change within 12 lines of line 48)"
    assert window_diff(old, None, 48).startswith("(the file was removed")
    assert around(old, 3, width=1) == "    2 line 2\n    3 line 3\n    4 line 4"
