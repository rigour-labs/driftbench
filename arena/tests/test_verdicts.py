import pytest

from arena.pipeline import score_tool
from arena.verdicts import Verdicts, merge, pack, unjudged

LABELS = {"prs": {
    "1": [{"fix_sha": "real", "fix_subject": "fix: a", "path": "src/a.ts", "lines": [10]},
          {"fix_sha": "noise", "fix_subject": "fix: b", "path": "src/b.ts", "lines": [5]}],
    "2": [{"fix_sha": "real", "fix_subject": "fix: a", "path": "src/a.ts", "lines": [40]}],
}}
RESULTS = {"prs": {
    "1": {"findings": [{"path": "src/a.ts", "line": 11, "id": "c1"}, {"path": "src/b.ts", "line": 5, "id": "c2"}]},
    "2": {"findings": [{"path": "src/a.ts", "line": 41, "id": "c3"}]},
}, "tool": "bot"}


def judged(*answers) -> Verdicts:
    v = Verdicts()
    merge(v, [{"fix": "real", "real": "yes", "needs": "local", "reason": "r"},
              {"fix": "noise", "real": "no", "needs": None, "reason": "moved code"}, *answers])
    return v


def pair(pr, finding, describes):
    return {"tool": "bot", "pr": pr, "finding": finding, "fix": "real", "describes": describes, "reason": "r"}


def test_proximity_counts_every_nearby_finding_but_judged_counts_only_real_bugs_it_describes():
    assert score_tool(LABELS, RESULTS).recall.value == 1.0
    s = score_tool(LABELS, RESULTS, verdicts=judged(pair("1", "c1", True), pair("2", "c3", False)))
    assert (s.bugs, s.recall.value) == (2, 0.5)


def test_an_unjudged_pair_is_not_a_hit_and_is_reported():
    v = judged(pair("1", "c1", True))
    assert score_tool(LABELS, RESULTS, verdicts=v).recall.value == 0.5
    assert unjudged(LABELS, {"bot": RESULTS}, v) == (0, 1)


def test_pack_skips_fixes_judged_not_real_and_pairs_already_judged():
    todo = pack(LABELS, {"bot": RESULTS}, judged(pair("1", "c1", True)))
    assert [(f["fix"], [c["finding"] for c in f["candidates"]]) for f in todo] == [("real", ["c3"])]
    fresh = pack(LABELS, {"bot": RESULTS}, Verdicts())
    assert {f["fix"] for f in fresh} == {"real", "noise"}


@pytest.mark.parametrize("answer", [
    {"fix": "x", "real": "maybe", "needs": None, "reason": "r"},
    {"fix": "x", "real": "yes", "needs": "vibes", "reason": "r"},
    {"tool": "bot", "pr": "1", "finding": "c1", "fix": "x", "describes": "yes", "reason": "r"},
    {"tool": "bot", "pr": "1", "fix": "x", "describes": True, "reason": "r"},
])
def test_malformed_answers_are_rejected(answer):
    with pytest.raises(ValueError):
        merge(Verdicts(), [answer])


def test_config_flags_expand_environment_variables_and_refuse_unset_ones(monkeypatch):
    from arena.__main__ import _expand
    monkeypatch.setenv("RIGOUR_MODEL_PATH", "/models/candidate.gguf")
    assert _expand(["--max", "--model-path", "${RIGOUR_MODEL_PATH}"]) == ("--max", "--model-path", "/models/candidate.gguf")
    monkeypatch.delenv("RIGOUR_MODEL_PATH")
    with pytest.raises(SystemExit, match="unset environment variable"):
        _expand(["--model-path", "${RIGOUR_MODEL_PATH}"])
