import urllib.error
from pathlib import Path

import pytest

from bench.harness.budget import Budget
from bench.labels import openrouter
from bench.labels.model_agreement import model_agreement
from bench.labels.model_file import blind_ids, shown_suggestion
from bench.labels.openrouter import OpenRouterError, chat
from bench.labels.prelabel import ENTRANT, ReplyError, anchored_hunk, guide_excerpt, parse_reply, run_prelabels
from bench.labels.rules import CLASSES
from tests.fixture_files import load_json

FIXTURE = load_json("openrouter-chat.json")
GUIDE = Path(__file__).resolve().parents[1] / "docs" / "LABELLING.md"


def reply(cost=0.02, content='{"class": "mechanical", "reason": "a rename"}'):
    return lambda messages: {"content": content, "cost_usd": cost, "served_model": "m"}


def targets(n, text="rename x"):
    return [{"id": f"p{i}", "text": text, "messages": []} for i in range(n)]


def fresh(model="example-org/example-model"):
    return {"model": model, "spent_usd": 0.0, "points": {}, "unsuggested": {}}


def run(data, items, call, budget):
    saved = []
    return run_prelabels(data, items, call, budget, saved.append), saved


def test_chat_reads_the_recorded_reply_and_asks_for_usage():
    sent = []
    result = chat("example-org/example-model", [{"role": "user", "content": "x"}], 200,
                  transport=lambda body: sent.append(body) or FIXTURE)
    assert result == {"content": FIXTURE["choices"][0]["message"]["content"], "cost_usd": 0.000412,
                      "served_model": "example-org/example-model-2026-09"}
    assert sent[0]["usage"] == {"include": True} and sent[0]["temperature"] == 0 and sent[0]["max_tokens"] == 200
    no_cost = chat("m", [], 10, transport=lambda body: {**FIXTURE, "usage": {}})
    assert no_cost["cost_usd"] is None


def test_the_key_is_required_and_never_appears_in_an_error(monkeypatch):
    monkeypatch.delenv(openrouter.KEY_NAME, raising=False)
    with pytest.raises(OpenRouterError, match=openrouter.KEY_NAME):
        openrouter.post({})
    monkeypatch.setenv(openrouter.KEY_NAME, "sk-test-not-a-real-key")

    def refuse(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)
    monkeypatch.setattr(openrouter.urllib.request, "urlopen", refuse)
    with pytest.raises(OpenRouterError) as caught:
        openrouter.post({})
    assert "401" in str(caught.value) and "sk-test" not in str(caught.value)


def test_a_known_class_and_a_one_line_reason_are_read():
    assert parse_reply('{"class": "performance", "reason": "extra\\n query"}') == ("performance", "extra query")
    fenced = '```json\n{"class": "security/privacy", "reason": "logs the email"}\n```'
    assert parse_reply(fenced) == ("security/privacy", "logs the email")


@pytest.mark.parametrize("content", ['{"class": "style", "reason": "x"}', "mechanical", '{"class": ', None])
def test_anything_else_is_no_suggestion(content):
    with pytest.raises(ReplyError, match="invalid output"):
        parse_reply(content)


def test_the_cap_stops_before_a_call_that_could_exceed_it():
    budget = Budget(0.04, {ENTRANT: 0.02})
    calls = []
    data, saved = run(fresh(), targets(4), lambda m: calls.append(1) or reply()(m), budget)
    assert len(calls) == 2 and data["spent_usd"] == 0.04         # 0.02 + 0.02 lands exactly on the cap
    assert set(data["unsuggested"]) == {"p2", "p3"} and all(v.startswith("budget:") for v in data["unsuggested"].values())
    assert len(saved) == 4 and budget.exhausted


def test_failures_and_unreported_cost_are_charged_at_the_bound_and_listed():
    def fail(messages):
        raise OpenRouterError("OpenRouter answered HTTP 502")
    budget = Budget(1.0, {ENTRANT: 0.05})
    data, _ = run(fresh(), targets(1), fail, budget)
    assert data["unsuggested"] == {"p0": "OpenRouter answered HTTP 502"} and budget.spent == 0.05
    data, _ = run(fresh(), targets(1), reply(cost=None), budget)
    assert "no cost" in data["unsuggested"]["p0"] and budget.spent == 0.1 and data["points"] == {}


def test_an_invalid_answer_is_paid_for_but_is_no_suggestion():
    data, _ = run(fresh(), targets(1), reply(content="I think mechanical"), Budget(1.0, {ENTRANT: 0.05}))
    assert data["points"]["p0"]["suggested"] is None and data["unsuggested"]["p0"].startswith("invalid output")
    assert data["spent_usd"] == 0.02


def test_paid_points_are_not_asked_again_and_text_changes_are_listed():
    budget = Budget(1.0, {ENTRANT: 0.05})
    data, _ = run(fresh(), targets(2), reply(), budget)
    calls = []
    again, _ = run(data, [*targets(2), {"id": "p9", "text": None, "messages": []}],
                   lambda m: calls.append(1) or reply()(m), budget)
    assert calls == [] and again["points"]["p0"]["suggested_by"] == "example-org/example-model"
    assert "TEXT-1" in again["unsuggested"]["p9"]


def test_the_blind_subset_is_seeded_and_never_shows_a_suggestion():
    sample = {"repo": "o/r", "seed": 1, "point_ids": [f"p{i:02}" for i in range(50)]}
    blind = blind_ids(sample)
    assert len(blind) == 10 and blind == blind_ids(sample) and set(blind) <= set(sample["point_ids"])
    assert blind != blind_ids({**sample, "seed": 2})
    data = {"blind_ids": blind, "points": {pid: {"suggested": "mechanical", "reason": "r"} for pid in sample["point_ids"]}}
    assert shown_suggestion(data, blind[0]) is None
    other = next(pid for pid in sample["point_ids"] if pid not in blind)
    assert shown_suggestion(data, other)["suggested"] == "mechanical" and shown_suggestion(None, other) is None


def test_agreement_is_rated_on_blind_labels_only_and_from_ten_up():
    blind = [f"b{i}" for i in range(10)]
    data = {"model": "m", "blind_ids": blind,
            "points": {**{pid: {"suggested": "mechanical"} for pid in blind}, "a1": {"suggested": "judgment"},
                       "u1": {"suggested": "judgment"}}}
    entries = {**{pid: {"label": "mechanical" if i < 7 else "judgment", "blind": True} for i, pid in enumerate(blind)},
               "a1": {"label": "judgment", "blind": False, "suggestion": "accepted"},
               "u1": {"label": "judgment", "blind": True}}
    usable = {pid: e["label"] for pid, e in entries.items()}
    sample = [*blind, "a1", "u1"]
    result = model_agreement(entries, usable, data, sample)
    assert result["blind"] == {"agree": 7, "points": 10, "pre_suggestion": 0, "rate": 0.7}
    assert result["anchored"] == {"accepted": 1, "overridden": 0} and result["unseen"] == 1
    fewer = model_agreement(entries, {k: v for k, v in usable.items() if k != "b9"}, data, sample)
    assert fewer["blind"]["rate"] is None and fewer["blind"]["points"] == 9


def test_a_sample_answered_in_full_before_suggestions_counts_as_blind():
    """`label next` goes in point-ID order, so only a complete pass is a random set; a partial one is not."""
    sample = [f"p{i:02}" for i in range(12)]
    data = {"model": "m", "blind_ids": ["p00", "p01"], "points": {pid: {"suggested": "mechanical"} for pid in sample}}
    entries = {pid: {"label": "mechanical", "blind": True} for pid in sample[:11]}
    entries["p11"] = {"skipped": True}                                     # skipped counts as answered
    usable = {pid: e["label"] for pid, e in entries.items() if e.get("label")}
    complete = model_agreement(entries, usable, data, sample)
    assert complete["blind"] == {"agree": 11, "points": 11, "pre_suggestion": 9, "rate": 1.0}
    assert complete["unseen"] == 0
    partial = {k: v for k, v in entries.items() if k not in ("p10", "p11")}
    result = model_agreement(partial, {k: v for k, v in usable.items() if k in partial}, data, sample)
    assert result["blind"]["pre_suggestion"] == 0 and result["blind"]["points"] == 2 and result["unseen"] == 8
    shown = {**entries, "p05": {"label": "judgment", "blind": False, "suggestion": "overridden"}}
    after = model_agreement(shown, {**usable, "p05": "judgment"}, data, sample)
    assert after["blind"]["pre_suggestion"] == 0 and after["anchored"]["overridden"] == 1


def test_the_prompt_carries_the_guide_classes_and_the_anchored_hunk():
    excerpt = guide_excerpt(GUIDE)
    assert all(f"**{label}**" in excerpt for label in CLASSES) and "Decision order" in excerpt
    comments = [{"id": 5, "diff_hunk": "\n".join(f"line {i}" for i in range(100))}, {"id": 6, "diff_hunk": "other"}]
    hunk = anchored_hunk(comments, 5)
    assert hunk.splitlines()[-1] == "line 99" and len(hunk.splitlines()) == 40 and anchored_hunk(comments, 7) == ""


def test_the_calibration_sample_never_reads_model_suggestions():
    source = (Path(__file__).resolve().parents[1] / "bench" / "report" / "calibration.py").read_text()
    assert "model_file" not in source and "prelabel" not in source and "openrouter" not in source
