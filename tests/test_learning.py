"""The learning run's free steps: which pull requests a store learns from, and the leak assertions."""
from __future__ import annotations

import json

from bench.learning import crawl as crawl_mod
from bench.learning.leaks import check_store, source_id
from bench.learning.precheck import precheck

CUTOFF = "2026-01-20T00:00:00Z"


def pr(number: int, created: str, merged: str | None) -> dict:
    return {"number": number, "created_at": created, "merged_at": merged, "merge_commit_sha": f"m{number}" if merged else None,
            "user": {"login": f"author{number}", "type": "User"}, "title": "dropped by trimming"}


class Listing:
    """A GitHub client answering the closed-pull-request listing, newest opened first, 100 a page."""

    def __init__(self, prs: list[dict]):
        self.prs = sorted(prs, key=lambda p: p["created_at"], reverse=True)
        self.pages = 0

    def get(self, path, params):
        self.pages += 1
        page = int(params["page"])
        return self.prs[(page - 1) * 100: page * 100]

    def get_all(self, path):
        return [{"id": 1, "body": "text", "user": {"login": "r", "type": "User", "site_admin": False},
                 "created_at": "2026-01-01T00:00:00Z", "extra": "dropped"}]


def test_a_store_learns_from_the_limit_most_recently_merged_before_its_cutoff():
    merged = [pr(n, f"2026-01-{n:02d}T00:00:00Z", f"2026-01-{n + 1:02d}T00:00:00Z") for n in range(1, 25)]
    old = crawl_mod.LIMIT
    crawl_mod.LIMIT = 3
    try:
        assert crawl_mod.window(merged, "2026-01-10T00:00:00Z") == [8, 7, 6]       # merged on the 9th, 8th, 7th
        assert crawl_mod.needed(merged, ["2026-01-10T00:00:00Z", "2026-01-12T00:00:00Z"]) == [6, 7, 8, 9, 10]
    finally:
        crawl_mod.LIMIT = old


def test_the_listing_stops_once_the_earliest_cutoff_has_its_window_and_the_margin():
    from datetime import datetime, timedelta
    day = lambda n: (datetime(2025, 1, 1) + timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")
    prs = [pr(n, day(n), None if n % 5 == 0 else day(n).replace("T00", "T06")) for n in range(300)]
    client = Listing(prs)
    merged = crawl_mod.list_merged(client, "o/r", day(243))
    assert client.pages == 2                       # page 1 (days 299-200) has 34 merged before the cutoff
    assert all(p["merged_at"] for p in merged) and "title" not in merged[0] and len(merged) == 160


def test_the_crawl_keeps_only_what_the_learner_reads():
    out = crawl_mod.crawl(Listing([pr(1, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")]), "o/r", [CUTOFF])
    comment = out["reviews"]["1"]["comments"][0]
    assert out["prs"][0]["number"] == 1 and "extra" not in comment and comment["user"] == {"login": "r", "type": "User"}


CRAWL = {"prs": [{"number": 1, "merged_at": "2026-01-05T00:00:00Z"}, {"number": 4, "merged_at": "2026-01-22T00:00:00Z"}],
         "reviews": {"1": {"comments": [{"id": 11, "created_at": "2026-01-03T00:00:00Z", "updated_at": "2026-01-21T00:00:00Z"},
                                        {"id": 12, "created_at": "2026-01-25T00:00:00Z", "updated_at": ""}],
                           "reviews": [{"id": 77, "submitted_at": "2026-01-04T00:00:00Z"}]},
                     "4": {"comments": [{"id": 41, "created_at": "2026-01-19T00:00:00Z"}], "reviews": []}}}


def lesson(*evidence: dict) -> dict:
    return {"id": "L", "state": "verified", "evidence": list(evidence)}


def test_evidence_from_before_the_cutoff_is_clean_and_a_late_edit_is_counted():
    out = check_store([lesson({"kind": "point", "pr": 1, "comment": "11"}, {"kind": "point", "pr": 1, "comment": "review-77-0"},
                              {"kind": "counter", "pr": 1, "comment": "counter-1-5", "at": CUTOFF})], 3, CUTOFF, CRAWL)
    assert out == {"leaks": [], "edited_after": ["L"], "rejected": 0}
    assert source_id("review-77-0") == "review-77" and source_id("11") == "11"


def test_every_kind_of_leak_is_caught():
    cases = {
        "the pull request under review": {"kind": "point", "pr": 3, "comment": "31"},
        "not merged before the cutoff": {"kind": "point", "pr": 4, "comment": "41"},
        "after the cutoff": {"kind": "point", "pr": 1, "comment": "12"},
        "not in the crawl": {"kind": "point", "pr": 1, "comment": "99"},
        "dated 2026-01-21": {"kind": "outcome", "pr": 1, "comment": "outcome-abc", "at": "2026-01-21T00:00:00Z"},
    }
    for expected, evidence in cases.items():
        leaks = check_store([lesson(evidence)], 3, CUTOFF, CRAWL)["leaks"]
        assert len(leaks) == 1 and expected in leaks[0], (expected, leaks)


def test_a_leaking_store_is_never_served_and_the_totals_count_served_heads(tmp_path):
    stores = tmp_path / "stores"
    stores.mkdir()
    (stores / "3.json").write_text(json.dumps({"lessons": [lesson({"kind": "point", "pr": 1, "comment": "11"})]}))
    (stores / "5.json").write_text(json.dumps({"lessons": [lesson({"kind": "point", "pr": 1, "comment": "12"})]}))
    served = {"repo": "o/r", "core": "6.12.4", "limits": {}, "prs": [
        {"pr": 3, "cutoff": CUTOFF, "store": "stores/3.json",
         "heads": {"a": {"verified": ["L"], "all": ["L"]}, "b": {"verified": [], "all": ["L"]}, "c": {"error": "gone"}}},
        {"pr": 5, "cutoff": CUTOFF, "store": "stores/5.json", "heads": {"d": {"verified": ["L"], "all": ["L"]}}}]}
    result = precheck(served, tmp_path, {**CRAWL, "limit": 100})
    assert result["prs"][1]["heads"]["d"] == {"error": "the store leaks; never served"}
    assert result["totals"] == {"prs": 2, "heads": 4, "errors": 2, "leaking_prs": 1, "edited_after_lessons": 1,
                                "heads_served_verified": 1, "heads_served_all": 2,
                                "lessons_served_verified": 1, "lessons_served_all": 2}
