from arena.score import Finding, PrResult, hits, score

BUGS = {"pr1": [("a.ts", [10])], "pr2": [("b.ts", [5, 6])], "pr3": []}
CHANGED = {"pr1": ("a.ts", range(1, 60)), "pr2": ("b.ts", range(1, 40)), "pr3": ("c.ts", range(1, 30))}


def _results(findings: dict[str, list[Finding]]) -> list[PrResult]:
    return [PrResult(pr, BUGS[pr], findings.get(pr, [])) for pr in BUGS]


def test_hits_within_tolerance_in_the_same_file_only():
    assert hits(Finding("a.ts", 12), ("a.ts", [10]))
    assert not hits(Finding("a.ts", 14), ("a.ts", [10]))
    assert not hits(Finding("b.ts", 10), ("a.ts", [10]))


def test_a_targeted_tool_scores_recall_and_hit_rate():
    s = score(_results({"pr1": [Finding("a.ts", 11)], "pr2": [Finding("b.ts", 30)]}))
    assert (s.bugs, s.comments) == (2, 2)
    assert s.recall.value == 0.5
    assert s.hit_rate.value == 0.5
    assert s.recall.low is not None and s.recall.low <= 0.5 <= s.recall.high


def test_flagging_every_changed_line_cannot_win():
    everything = {pr: [Finding(path, line) for line in lines] for pr, (path, lines) in CHANGED.items()}
    noisy = score(_results(everything))
    targeted = score(_results({"pr1": [Finding("a.ts", 10)], "pr2": [Finding("b.ts", 5)]}))
    assert noisy.recall.value == targeted.recall.value == 1.0
    assert noisy.hit_rate.value < 0.1 < targeted.hit_rate.value
    assert noisy.comments_per_pr.value > 40 * targeted.comments_per_pr.value


def test_silence_scores_zero_recall_and_no_precision_claim():
    s = score(_results({}))
    assert s.recall.value == 0.0
    assert s.hit_rate.value is None and s.precision.value is None


def test_precision_comes_only_from_hand_labels():
    s = score(_results({"pr1": [Finding("a.ts", 40, correct=True), Finding("a.ts", 50, correct=False), Finding("a.ts", 55)]}))
    assert s.precision.value == 0.5
    assert s.hit_rate.value == 0.0


def test_code_scope_excludes_docs_config_and_tests():
    from arena.score import in_scope
    assert in_scope("packages/router/src/link.tsx", "code")
    assert not in_scope("docs/SKILL.md", "code")
    assert not in_scope("packages/router/tests/link.test.tsx", "code")
    assert not in_scope("e2e/app/src/main.tsx", "code")
    assert in_scope("docs/SKILL.md", "all")
