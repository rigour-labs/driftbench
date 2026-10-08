import hashlib

import pytest

from bench.points.points_file import PointsError, build_points, read_points, write_points
from tests.github_fakes import FakeClient

TEXTS = {
    "inline": {100: "This leaks the handle.", 101: "fixed", 102: "agreed, and also the other one", 103: "LGTM"},
    "review": {10: "Two things:\n\n- add a test\n- rename `f`\n\n> quoted earlier text\n\nThanks!", 11: "LGTM"},
    "conversation": {200: "Does this work on Windows?", 201: "yes", 202: "/retest", 203: "edited later"},
}


def digest(text: str) -> dict:
    return {"body_sha256": hashlib.sha256(text.encode()).hexdigest(), "body_chars": len(text)}


def inline(cid: int, **extra) -> dict:
    entry = {"id": cid, "round": 1, "by_author": False, "in_reply_to": None, "path": "a.go", "line": 5,
             "start_line": None, "side": "RIGHT", "commit_sha": "A", **digest(TEXTS["inline"][cid])}
    entry.update(extra)
    return entry


def conversation(cid: int, **extra) -> dict:
    entry = {"id": cid, "round": 1, "by_author": False, "head_sha": "A", "head_source": "push",
             **digest(TEXTS["conversation"][cid])}
    entry.update(extra)
    return entry


def corpus() -> dict:
    pr = {
        "number": 7, "head_sha": "H", "commits": [{"sha": "H"}],
        "comments": [inline(100), inline(101, by_author=True), inline(102, in_reply_to=100), inline(103)],
        "reviews": [{"id": 10, "round": 1, "commit_sha": "A", **digest(TEXTS["review"][10])},
                    {"id": 11, "round": None, "commit_sha": "A", **digest(TEXTS["review"][11])}],
        "conversation": [conversation(200, round=None, head_sha="Z"), conversation(201, by_author=True),
                         conversation(202), conversation(203, body_sha256="0" * 64)],
    }
    return {"repo": "o/r", "pin": "p" * 40, "collected_at": "2026-10-08T00:00:00Z", "prs": [pr]}


def client() -> FakeClient:
    lists = {
        "repos/o/r/pulls/7/comments": [{"id": k, "body": v} for k, v in TEXTS["inline"].items()],
        "repos/o/r/pulls/7/reviews": [{"id": k, "body": v} for k, v in TEXTS["review"].items()],
        "repos/o/r/issues/7/comments": [{"id": k, "body": v} for k, v in TEXTS["conversation"].items()],
    }
    compare = {"status": "ahead", "files": [{"filename": "a.go", "patch": "@@ -5,1 +5,1 @@\n-x\n+y\n"}]}
    return FakeClient({}, lists, {"repos/o/r/compare/A...H": compare})


def by_source(points: list[dict]) -> dict:
    return {(p["kind"], p["source_id"], p["span"] and tuple(p["span"])): p for p in points}


def test_drop_rules_and_scorability():
    result = build_points(corpus(), client())
    dropped = sorted((p["kind"], p["source_id"], p["dropped"]) for p in result["points"] if p["dropped"])
    assert dropped == [
        ("body", 10, "ACK-1"), ("body", 10, "QUOTE-1"), ("body", 11, "ACK-1"),
        ("conversation", 201, "AUTHOR-1"), ("conversation", 202, "CMD-1"), ("conversation", 203, "TEXT-1"),
        ("inline", 101, "AUTHOR-1"), ("inline", 102, "THREAD-1"), ("inline", 103, "ACK-1"),
    ]
    kept = [p for p in result["points"] if not p["dropped"]]
    assert sorted((p["kind"], p["source_id"]) for p in kept) == [
        ("body", 10), ("body", 10), ("body", 10), ("conversation", 200), ("inline", 100)]
    assert [p["scorable"] for p in kept if p["kind"] == "conversation"] == [False]  # no reviewed head


def test_spans_index_the_text_and_inline_is_acted_on():
    result = build_points(corpus(), client())
    text = TEXTS["review"][10]
    body = [text[slice(*p["span"])] for p in result["points"] if p["kind"] == "body" and p["source_id"] == 10]
    assert body == ["Two things:", "- add a test", "- rename `f`", "> quoted earlier text", "Thanks!"]
    first = next(p for p in result["points"] if p["source_id"] == 100)
    assert (first["acted_on"], first["acted_basis"]) == (True, "ancestor")
    assert first["anchor"]["line"] == 5 and first["id"] == "7-inline-100-0"


def test_summary_and_no_text_in_file(tmp_path):
    result = build_points(corpus(), client())
    assert result["summary"]["scorable"] == 4 and result["summary"]["unscored_no_round"] == 1
    assert result["summary"]["acted_on"] == {"true/ancestor": 1}
    path = write_points(result, tmp_path)
    raw = path.read_text(encoding="utf-8")
    assert all(text not in raw for group in TEXTS.values() for text in group.values() if len(text) > 6)
    assert read_points(path)["repo"] == "o/r"


def test_read_points_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text('{"schema": 0}', encoding="utf-8")
    with pytest.raises(PointsError):
        read_points(bad)
