"""The review comments a time-correct lesson store learns from (docs/LEARNING.md).

Each pull request in the sample gets a cutoff: the moment it was opened. Its
store is built from the `LIMIT` most recently merged pull requests before the
cutoff, Rigour's own default for `rigour learn-reviews`. One listing per repo
finds them all; then each one's review comments and reviews are fetched once.
The crawl holds review text: it is never an artifact or a release asset, only
the Actions cache (like the corpus API cache).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from bench.collect.github import GitHubClient

LIMIT = 100          # `rigour learn-reviews --limit` default
MARGIN_DAYS = 30     # keep listing this far past the earliest cutoff, for pull requests merged long after they opened
MAX_PAGES = 200

USER_KEYS = ("login", "type")
PR_KEYS = ("number", "created_at", "merged_at", "merge_commit_sha")
COMMENT_KEYS = ("id", "path", "line", "original_line", "start_line", "original_start_line", "commit_id",
                "original_commit_id", "in_reply_to_id", "body", "created_at", "updated_at")
REVIEW_KEYS = ("id", "commit_id", "body", "state", "submitted_at")


def trimmed(item: dict, keys: tuple[str, ...]) -> dict:
    user = item.get("user") or {}
    return {**{k: item.get(k) for k in keys}, "user": {k: user.get(k) for k in USER_KEYS}}


def parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def list_merged(client: GitHubClient, repo: str, earliest_cutoff: str) -> list[dict]:
    """Merged pull requests, newest opened first, until LIMIT merged before the earliest cutoff and the
    listing is MARGIN_DAYS past it."""
    stop = parse(earliest_cutoff) - timedelta(days=MARGIN_DAYS)
    merged: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        batch = client.get(f"repos/{repo}/pulls", {"state": "closed", "sort": "created", "direction": "desc",
                                                   "per_page": "100", "page": str(page)})
        if not batch:
            break
        merged += [trimmed(pr, PR_KEYS) for pr in batch if pr.get("merged_at") and pr.get("merge_commit_sha")]
        before = sum(1 for pr in merged if pr["merged_at"] < earliest_cutoff)
        if before >= LIMIT and parse(batch[-1]["created_at"]) < stop:
            break
    return merged


def window(merged: list[dict], cutoff: str) -> list[int]:
    """The LIMIT pull requests merged last before the cutoff."""
    before = sorted((pr for pr in merged if pr["merged_at"] < cutoff), key=lambda pr: pr["merged_at"], reverse=True)
    return [pr["number"] for pr in before[:LIMIT]]


def needed(merged: list[dict], cutoffs: list[str]) -> list[int]:
    return sorted({n for cutoff in cutoffs for n in window(merged, cutoff)})


def fetch_reviews(client: GitHubClient, repo: str, number: int) -> dict:
    comments = client.get_all(f"repos/{repo}/pulls/{number}/comments")
    reviews = client.get_all(f"repos/{repo}/pulls/{number}/reviews")
    return {"comments": [trimmed(c, COMMENT_KEYS) for c in comments],
            "reviews": [trimmed(r, REVIEW_KEYS) for r in reviews]}


def crawl(client: GitHubClient, repo: str, cutoffs: list[str]) -> dict:
    """{repo, prs: listing, reviews: {number: {comments, reviews}}} for every store's window."""
    merged = list_merged(client, repo, min(cutoffs))
    numbers = needed(merged, cutoffs)
    listed = {pr["number"]: pr for pr in merged}
    return {"repo": repo, "limit": LIMIT, "prs": [listed[n] for n in numbers],
            "reviews": {str(n): fetch_reviews(client, repo, n) for n in numbers}}
