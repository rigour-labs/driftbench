import json

import pytest

from bench.harness.budget import Budget
from bench.issues.inputs import REVIEW_CHARS, entrant_review, head_review
from bench.issues.judge import ENTRANT, JudgeError, judge_point, messages_for, order, parse
from bench.issues.summary import consistency, per_entrant
from bench.report.markdown import issue_judge_section, issue_lines

POINT = {"id": "7-inline-1-0", "pr": 7, "round": 2, "anchor": {"path": "a.go", "line": 12}}
ROUNDS = [{"index": 1, "head_sha": "h1"}, {"index": 2, "head_sha": "h2"}, {"index": 3, "head_sha": "h3"}]
ENTRANTS = ["claude-code-review", "rigour-reviewer"]


def reply(content):
    return lambda messages: {"content": json.dumps(content), "cost_usd": 0.004, "served_model": "m"}


def test_order_is_seeded_per_point_and_both_orders_occur():
    assert order("p1", ENTRANTS, 1) == order("p1", ENTRANTS, 1)
    firsts = {order(f"p{i}", ENTRANTS, 1)[0] for i in range(40)}
    assert firsts == set(ENTRANTS)


def test_the_prompt_never_names_an_entrant():
    messages = messages_for(POINT, "Off by one here", {"A": "the loop bound is wrong", "B": "- a.go:12: off by one"})
    text = " ".join(m["content"] for m in messages).lower()
    assert "a.go line 12" in text and "review a:" in text and "review b:" in text
    assert not any(name in text for name in ("claude", "rigour", "code-review", "reviewer:"))


def test_reviews_come_from_eligible_heads_only_and_are_bounded():
    reviews = {"h1": "first", "h3": "after the human spoke"}
    assert entrant_review(POINT, ROUNDS, reviews) == "first"
    assert entrant_review(POINT, ROUNDS, {"h3": "too late"}) is None
    rigour = {"paid_output": {"review_text": "", "findings": [{"path": "a.go", "line": 3, "message": "full text"}]}}
    assert head_review(rigour) == "- a.go:3: full text"
    long = head_review({"paid_output": {"review_text": "x" * (REVIEW_CHARS + 50)}})
    assert long.endswith("[review cut at 12,000 characters]") and len(long) < REVIEW_CHARS + 60


def test_a_point_is_judged_blind_with_a_missing_review_recorded_and_the_call_charged():
    budget = Budget(1.0, {ENTRANT: 0.05})
    out = judge_point(POINT, "Off by one", {"claude-code-review": None, "rigour-reviewer": "- a.go:12: off by one"},
                      1, reply({"A": {"verdict": "yes", "reason": "same bound"}}), budget)
    assert out["order"] == ["rigour-reviewer"] and out["verdicts"]["claude-code-review"] == {"verdict": "missing"}
    assert out["verdicts"]["rigour-reviewer"] == {"verdict": "yes", "reason": "same bound"} and budget.spent == 0.004
    bad = judge_point(POINT, "x", {"claude-code-review": "r", "rigour-reviewer": "s"}, 1,
                      reply({"A": {"verdict": "maybe"}}), budget)
    assert bad["error"].startswith("invalid output") and bad["cost_usd"] == 0.004
    with pytest.raises(JudgeError):
        parse('{"A": {"verdict": "yes"}}', ["A", "B"])


def test_results_per_entrant_and_self_consistency_reach_the_page():
    def item(repo, a, b):
        return {"repo": repo, "verdicts": {"claude-code-review": {"verdict": a}, "rigour-reviewer": {"verdict": b}}}
    judgments = {"p1": item("o/r", "yes", "no"), "p2": item("o/r", "partly", "yes"), "p3": item("o/r", "no", "yes"),
                 "p4": {**item("o/r", "yes", "yes"), "error": "budget"}}
    table = per_entrant(judgments, "o/r")
    assert table["claude-code-review"]["yes"] == 1 and table["claude-code-review"]["judged"] == 3
    assert table["rigour-reviewer"]["rate"] == round(2 / 3, 4)
    same = consistency(judgments, {"p1": item("o/r", "yes", "partly"), "p4": item("o/r", "yes", "yes")})
    assert same == {"same": 1, "asked": 2, "rate": 0.5}
    lines = "\n".join(issue_lines(table))
    assert "AI judge, non-Claude, blind to entrant" in lines and "| claude-code-review | 33% (1/3" in lines
    section = "\n".join(issue_judge_section({"model": "example-org/example-model", "consistency": same, "repos": {}}))
    assert "outside the Claude family" in section and "repeated its verdict on 1 of 2" in section


def test_words_that_identify_an_entrant_are_redacted():
    from bench.issues.inputs import redact
    text = ("- **CLAUDE.md:** none exists.\nI did not look for claude.md files.\nRigour flags this too.\n"
            "🤖 Generated with Claude Code\nThe loop bound is wrong.")
    out = redact(text)
    assert "claude" not in out.lower() and "rigour" not in out.lower() and "generated" not in out.lower()
    assert "[the repository's agent instructions file]" in out and "The loop bound is wrong." in out
