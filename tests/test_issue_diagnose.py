import json

import pytest

from bench.harness.budget import Budget
from bench.issues.diagnose import ENTRANT, DiagnosisError, bucket, candidates, diagnose_point, messages_for, parse
from bench.issues.diagnose_cli import summarise, target_ids

POINT = {"id": "7-inline-1-0", "pr": 7, "round": 1, "anchor": {"path": "a.go", "line": 12}}
RECORD = {"paid_output": {
    "findings": [{"path": "a.go", "line": 40, "message": "unrelated naming nit"}],
    "held_back": {"dropped": [{"file": "a.go", "line": 12, "issue": "off by one in the bound", "why": "low confidence"}],
                  "unverified": [], "disputed": [{"file": "b.go", "line": 3, "issue": "leaks the handle", "why": "split"}],
                  "dismissed": [], "shown": {"folded": 2}, "counts": {}}}}


def reply(content, cost=0.003):
    return lambda messages: {"content": json.dumps(content), "cost_usd": cost}


def test_served_and_held_back_findings_are_mixed_and_the_judge_is_never_told_which_list():
    found = candidates([RECORD])
    assert [f["list"] for f in found] == ["served", "dropped", "disputed"]
    text = " ".join(m["content"] for m in messages_for(POINT, "Off by one here", found)).lower()
    assert "1. a.go:40: unrelated naming nit" in text and "2. a.go:12: off by one in the bound low confidence" in text
    assert not any(word in text for word in ("served", "dropped", "disputed", "unverified", "dismissed", "held"))


def test_each_point_lands_in_one_bucket():
    found = candidates([RECORD])
    assert bucket(found, {2}, set()) == "held_back:dropped"
    assert bucket(found, {1}, {2}) == "served"
    assert bucket(found, set(), {3}) == "held_back:disputed"
    assert bucket(found, set(), set()) == "absent"


def test_answers_are_validated_and_every_call_is_charged():
    assert parse('{"same": [2], "partly": [], "reason": "same bound"}', 3) == ({2}, set(), "same bound")
    with pytest.raises(DiagnosisError, match="finding numbers"):
        parse('{"same": [9], "partly": []}', 3)
    budget = Budget(1.0, {ENTRANT: 0.05})
    out = diagnose_point(POINT, "Off by one", candidates([RECORD]), reply({"same": [2], "partly": [], "reason": "r"}),
                         budget)
    assert out["bucket"] == "held_back:dropped" and out["lists"] == ["dropped"] and budget.spent == 0.003
    assert diagnose_point(POINT, "x", [], reply({}), budget) == {"bucket": "absent", "findings": 0}
    assert budget.spent == 0.003                                         # nothing to show: no call


def test_targets_are_raised_by_one_and_missed_by_the_other():
    def j(cc, rig, **extra):
        return {"repo": "o/r", "verdicts": {"claude-code-review": {"verdict": cc}, "rigour-reviewer": {"verdict": rig}},
                **extra}
    judgments = {"a": j("yes", "no"), "b": j("partly", "no"), "c": j("yes", "yes"), "d": j("no", "no"),
                 "e": j("yes", "no", error="budget")}
    assert target_ids(judgments, "claude-code-review", "rigour-reviewer") == {"a": "o/r", "b": "o/r"}
    assert summarise({"a": {"bucket": "absent"}, "b": {"bucket": "served"}, "c": {"error": "x"}}) == {
        "absent": 1, "error": 1, "served": 1}
