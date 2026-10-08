from bench.labels.session import links, run_session
from bench.labels.store import empty_labels, text_sha256

POINTS = {
    "p1": {"id": "p1", "pr": 7, "kind": "inline", "source_id": 1,
           "anchor": {"path": "a.go", "line": 12, "start_line": 10, "commit_sha": "abc"}},
    "p2": {"id": "p2", "pr": 7, "kind": "body", "source_id": 55, "anchor": None},
    "p3": {"id": "p3", "pr": 7, "kind": "conversation", "source_id": 66, "anchor": None},
}
TEXTS = {"p1": "this leaks the handle", "p2": "rename x", "p3": None}


def session(replies, labels=None, include_skipped=False):
    saved, shown = [], []
    answers = iter(replies)
    context = {"repo": "o/r", "points": POINTS, "sample_ids": ["p1", "p2", "p3"],
               "text": lambda p: TEXTS[p["id"]], "labeller": "ash", "save": saved.append,
               "include_skipped": include_skipped}
    result = run_session(context, labels or empty_labels("o/r"), lambda prompt: next(answers), shown.append)
    return result, saved, shown


def test_label_skip_and_unreadable_text():
    labels, saved, shown = session(["x", "3", "s"])     # "x" is rejected and asked again
    p1, p2 = labels["points"]["p1"], labels["points"]["p2"]
    assert (p1["label"], p1["blind"], p1["labeller"]) == ("claim/contract", True, "ash")
    assert p1["text_sha256"] == text_sha256("this leaks the handle") and p1["suggested"] == "claim/contract"
    assert p2["label"] is None and p2["skipped"] is True
    assert len(saved) == 2 and any("TEXT-1" in line for line in shown)
    assert "suggested" not in shown[0]                  # blind: no suggestion shown


def test_quit_keeps_progress_and_resume_skips_done_and_skipped():
    labels, saved, _ = session(["1", "q"])
    assert labels["points"]["p1"]["label"] == "mechanical" and "p2" not in labels["points"] and len(saved) == 1
    labels, _, _ = session(["s"], labels)
    again, saved, shown = session([], labels)
    assert saved == [] and not any("p1" in line or "p2" in line for line in shown)
    relabel, _, _ = session(["5"], labels, include_skipped=True)
    assert relabel["points"]["p2"]["label"] == "judgment" and "skipped" not in relabel["points"]["p2"]


def test_links_point_at_the_anchor_commit_and_the_pr():
    assert links("o/r", POINTS["p1"]) == ["code: https://github.com/o/r/blob/abc/a.go#L10-L12",
                                          "PR: https://github.com/o/r/pull/7"]
    assert links("o/r", POINTS["p2"]) == ["PR: https://github.com/o/r/pull/7#pullrequestreview-55"]
    assert links("o/r", POINTS["p3"]) == ["PR: https://github.com/o/r/pull/7#issuecomment-66"]
