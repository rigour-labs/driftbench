import hashlib
from datetime import datetime, timezone

import pytest

from bench.collect.corpus import CorpusError, collect_repo, read_corpus, write_corpus
from bench.repos import PinnedRepo
from tests.github_fakes import (AUTHOR, BOT, REVIEWER, FakeClient, commit_objects, make_comment, make_commit,
                                make_conversation, make_pr, make_push, make_review)

REPO = PinnedRepo("o/r", "MIT", "main", "a" * 40, datetime(2026, 6, 1, tzinfo=timezone.utc))
BASE = "repos/o/r/pulls"
ISSUES = "repos/o/r/issues"


def reviewed_pr_fixture() -> dict[str, list[dict]]:
    """PR 1: two rounds, then a bare approval. PR 2: approvals only. PR 3: reviewed. PR 4: conversation only."""
    return {
        f"{BASE}/1/reviews": [
            make_review(10, "CHANGES_REQUESTED", "h1", "2026-02-01T00:00:00Z", body="see inline"),
            make_review(11, "COMMENTED", "h1", "2026-02-01T01:00:00Z", body="thanks", user=AUTHOR),
            make_review(12, "COMMENTED", "h2", "2026-02-03T00:00:00Z", body="one more thing"),
            make_review(13, "APPROVED", "h3", "2026-02-05T00:00:00Z"),
        ],
        f"{BASE}/1/comments": [
            make_comment(100, 10, "2026-02-01T00:00:00Z"),
            make_comment(101, 11, "2026-02-01T01:00:00Z", user=AUTHOR, in_reply_to_id=100),
            make_comment(102, 10, "2026-02-01T00:00:00Z", user=BOT),
            make_comment(103, 12, "2026-03-05T00:00:00Z"),  # after merge
            make_comment(104, 12, "2026-02-03T00:00:00Z", original_line=None, original_commit_id="h2"),
        ],
        f"{BASE}/1/commits": [make_commit("h3", "2026-02-04T00:00:00Z")],
        f"{ISSUES}/1/timeline": [
            make_push("h1", "2026-01-30T00:00:00Z"),
            make_push("h2", "2026-02-02T00:00:00Z"),
            make_push("h3", "2026-02-04T00:00:00Z"),
        ],
        f"{ISSUES}/1/comments": [
            make_conversation(200, "2026-02-02T12:00:00Z", "why not reuse the parser?"),  # head h2, round 2
            make_conversation(201, "2026-02-04T12:00:00Z", "and the docs?"),              # head h3, unreviewed
            make_conversation(202, "2026-02-04T13:00:00Z", "/retest"),
            make_conversation(203, "2026-02-04T14:00:00Z", "done", user=AUTHOR),
        ],
        f"{BASE}/2/reviews": [make_review(20, "APPROVED", "h9", "2026-02-01T00:00:00Z")],
        f"{ISSUES}/2/comments": [make_conversation(210, "2026-02-01T00:00:00Z", "@dependabot rebase")],
        f"{BASE}/3/reviews": [make_review(30, "COMMENTED", "h7", "2026-02-01T00:00:00Z", body="why?")],
        f"{ISSUES}/4/comments": [make_conversation(400, "2026-02-01T00:00:00Z", "this breaks the CLI flag")],
        f"{BASE}/4/commits": [make_commit("c4", "2026-01-20T00:00:00Z")],
    }


def run_collection(max_prs: int) -> tuple[dict, FakeClient]:
    prs = [make_pr(n, "2026-03-01T00:00:00Z", f"2026-05-0{n}T00:00:00Z", head="h3") for n in (4, 3, 2, 1)]
    dates = {"h1": "2026-01-30T00:00:00Z", "h2": "2026-02-02T00:00:00Z", "h7": "2026-01-15T00:00:00Z"}
    client = FakeClient({BASE: [prs]}, reviewed_pr_fixture(), commit_objects("o/r", dates))
    return collect_repo(client, REPO, max_prs=max_prs, max_listed=100), client


def pr_number(corpus: dict, number: int) -> dict:
    return next(p for p in corpus["prs"] if p["number"] == number)


def test_collect_keeps_reviewed_prs_in_order_and_skips_bare_approvals_and_commands():
    corpus, client = run_collection(max_prs=5)
    assert [pr["number"] for pr in corpus["prs"]] == [4, 3, 1]
    assert f"{BASE}/2/commits" not in client.calls
    assert corpus["schema"] == 2
    assert corpus["selection"] == {"max_prs": 5, "max_listed": 100, "merged_candidates": 4}


def test_collect_stops_at_max_prs():
    corpus, client = run_collection(max_prs=1)
    assert [pr["number"] for pr in corpus["prs"]] == [4]
    assert f"{BASE}/3/reviews" not in client.calls


def test_every_human_review_is_kept_and_the_approved_head_survives():
    """CHANGES_REQUESTED on h1, a comment on h2, then a bare APPROVED on h3."""
    pr = pr_number(run_collection(max_prs=5)[0], 1)
    assert [(r["id"], r["substantive"], r["round"]) for r in pr["reviews"]] == [(10, True, 1), (12, True, 2), (13, False, None)]
    assert [(r["index"], r["head_sha"]) for r in pr["rounds"]] == [(1, "h1"), (2, "h2")]
    assert pr["approved_head_sha"] == "h3"
    assert {r["commit_check"] for r in pr["reviews"]} == {"ok"}


def test_record_inline_comments():
    pr = pr_number(run_collection(max_prs=5)[0], 1)
    comments = {c["id"]: c for c in pr["comments"]}
    assert set(comments) == {100, 101, 104}  # bot and post-merge comments dropped
    assert comments[100]["kind"] == "inline" and comments[100]["round"] == 1 and comments[100]["line"] == 10
    assert comments[101]["by_author"] and comments[101]["round"] is None and comments[101]["in_reply_to"] == 100
    assert comments[104]["line"] is None and comments[104]["commit_sha"] == "h2"


def test_conversation_comments_anchor_to_the_head_they_were_written_against():
    pr = pr_number(run_collection(max_prs=5)[0], 1)
    conversation = {c["id"]: c for c in pr["conversation"]}
    assert (conversation[200]["head_sha"], conversation[200]["round"]) == ("h2", 2)
    assert (conversation[201]["head_sha"], conversation[201]["round"]) == ("h3", None)  # h3 never reviewed
    assert conversation[201]["head_source"] == "push" and conversation[201]["kind"] == "conversation"
    assert conversation[203]["by_author"]
    assert [h["sha"] for h in pr["head_history"] if h["source"] == "push"] == ["h1", "h2", "h3"]
    assert [h["sha"] for h in pr["head_history"] if h["source"] == "review"] == ["h1", "h2", "h3"]


def test_conversation_only_pr_falls_back_to_commit_dates():
    pr = pr_number(run_collection(max_prs=5)[0], 4)
    assert pr["rounds"] == [] and pr["approved_head_sha"] is None
    assert pr["conversation"][0]["head_sha"] == "c4" and pr["conversation"][0]["head_source"] == "commit_date"


def test_record_stores_text_digests_never_text(tmp_path):
    corpus, _ = run_collection(max_prs=5)
    path = write_corpus(corpus, tmp_path)
    raw = path.read_text(encoding="utf-8")
    assert path.name == "o__r.json"
    for text in ("please handle the empty case", "see inline", "why not reuse the parser?", REVIEWER["login"]):
        assert text not in raw
    assert '"body"' not in raw
    review = pr_number(read_corpus(path), 1)["reviews"][0]
    assert review["body_sha256"] == hashlib.sha256(b"see inline").hexdigest()
    assert review["body_chars"] == len("see inline")


def test_read_corpus_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(CorpusError, match="cannot read"):
        read_corpus(bad)
    bad.write_text('{"schema": 1}', encoding="utf-8")
    with pytest.raises(CorpusError, match="schema"):
        read_corpus(bad)
