import hashlib
from datetime import datetime, timezone

import pytest

from bench.collect.corpus import CorpusError, collect_repo, read_corpus, write_corpus
from bench.repos import PinnedRepo
from tests.github_fakes import (AUTHOR, BOT, FakeClient, make_comment, make_commit, make_pr,
                                make_review)

REPO = PinnedRepo("o/r", "MIT", "main", "a" * 40, datetime(2026, 6, 1, tzinfo=timezone.utc))
BASE = "repos/o/r/pulls"


def reviewed_pr_fixture() -> dict[str, list[dict]]:
    """PR 1: two rounds of human review. PR 2: approvals only. PR 3: reviewed."""
    return {
        f"{BASE}/1/reviews": [
            make_review(10, "CHANGES_REQUESTED", "h1", "2026-02-01T00:00:00Z", body="see inline"),
            make_review(11, "COMMENTED", "h1", "2026-02-01T01:00:00Z", body="thanks", user=AUTHOR),
            make_review(12, "APPROVED", "h2", "2026-02-03T00:00:00Z", body="looks good now"),
        ],
        f"{BASE}/1/comments": [
            make_comment(100, 10, "2026-02-01T00:00:00Z"),
            make_comment(101, 11, "2026-02-01T01:00:00Z", user=AUTHOR, in_reply_to_id=100),
            make_comment(102, 10, "2026-02-01T00:00:00Z", user=BOT),
            make_comment(103, 12, "2026-03-05T00:00:00Z"),  # after merge
            make_comment(104, 12, "2026-02-03T00:00:00Z", original_line=None, original_commit_id="h2"),
        ],
        f"{BASE}/1/commits": [make_commit("h1", "2026-01-30T00:00:00Z"), make_commit("h2", "2026-02-02T00:00:00Z")],
        f"{BASE}/2/reviews": [make_review(20, "APPROVED", "h9", "2026-02-01T00:00:00Z")],
        f"{BASE}/3/reviews": [make_review(30, "COMMENTED", "h7", "2026-02-01T00:00:00Z", body="why?")],
    }


def run_collection(max_prs: int) -> tuple[dict, FakeClient]:
    prs = [make_pr(n, "2026-03-01T00:00:00Z", f"2026-05-0{n}T00:00:00Z") for n in (3, 2, 1)]
    client = FakeClient({BASE: [prs]}, reviewed_pr_fixture())
    return collect_repo(client, REPO, max_prs=max_prs, max_listed=100), client


def test_collect_keeps_reviewed_prs_in_order_and_skips_bare_approvals():
    corpus, client = run_collection(max_prs=5)
    assert [pr["number"] for pr in corpus["prs"]] == [3, 1]
    assert f"{BASE}/2/commits" not in client.calls
    assert corpus["selection"] == {"max_prs": 5, "max_listed": 100, "merged_candidates": 3}


def test_collect_stops_at_max_prs():
    corpus, client = run_collection(max_prs=1)
    assert [pr["number"] for pr in corpus["prs"]] == [3]
    assert f"{BASE}/1/reviews" not in client.calls


def test_record_rounds_reviews_and_comments():
    pr = next(p for p in run_collection(max_prs=5)[0]["prs"] if p["number"] == 1)
    assert [(r["index"], r["head_sha"]) for r in pr["rounds"]] == [(1, "h1"), (2, "h2")]
    assert [(r["id"], r["round"]) for r in pr["reviews"]] == [(10, 1), (12, 2)]
    comments = {c["id"]: c for c in pr["comments"]}
    assert set(comments) == {100, 101, 104}  # bot and post-merge comments dropped
    assert comments[100]["round"] == 1 and comments[100]["line"] == 10
    assert comments[101]["by_author"] and comments[101]["round"] is None and comments[101]["in_reply_to"] == 100
    assert comments[104]["line"] is None and comments[104]["commit_sha"] == "h2"
    assert pr["head_sha"] == "h2" and [c["sha"] for c in pr["commits"]] == ["h1", "h2"]


def test_record_stores_text_digests_never_text(tmp_path):
    corpus, _ = run_collection(max_prs=5)
    path = write_corpus(corpus, tmp_path)
    raw = path.read_text(encoding="utf-8")
    assert path.name == "o__r.json"
    assert "please handle the empty case" not in raw and "see inline" not in raw
    assert '"body"' not in raw
    review = read_corpus(path)["prs"][1]["reviews"][0]
    assert review["body_sha256"] == hashlib.sha256(b"see inline").hexdigest()
    assert review["body_chars"] == len("see inline")


def test_read_corpus_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(CorpusError, match="cannot read"):
        read_corpus(bad)
    bad.write_text('{"schema": 99}', encoding="utf-8")
    with pytest.raises(CorpusError, match="schema"):
        read_corpus(bad)
