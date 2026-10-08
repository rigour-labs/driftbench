"""Which pull requests and reviews enter the corpus (docs/SPEC.md, "Pull requests")."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from bench.collect.github import GitHubClient
from bench.collect.text_rules import is_review_point

PAGE_SIZE = 100


def parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def is_bot(user: dict | None) -> bool:
    if not user:
        return True
    return user.get("type") == "Bot" or str(user.get("login", "")).endswith("[bot]")


def is_human_reviewer(user: dict | None, author_login: str) -> bool:
    return not is_bot(user) and (user or {}).get("login") != author_login


def candidate_prs(client: GitHubClient, repo: str, pinned_at: datetime, limit: int) -> list[dict]:
    """Merged PRs, merged at or before the pin, sorted by updated then merged, newest first.

    Lists at most `limit` closed PRs, most recently updated first.
    """
    merged = [
        pr for pr in list_closed(client, repo, limit)
        if (merged_at := parse_time(pr.get("merged_at"))) and merged_at <= pinned_at
    ]
    return sorted(merged, key=lambda pr: (pr["updated_at"], pr["merged_at"]), reverse=True)


def list_closed(client: GitHubClient, repo: str, limit: int) -> Iterator[dict]:
    seen = 0
    for page in range(1, limit // PAGE_SIZE + 2):
        params = {"state": "closed", "sort": "updated", "direction": "desc",
                  "per_page": str(PAGE_SIZE), "page": str(page)}
        batch = client.get(f"repos/{repo}/pulls", params) or []
        for pr in batch[: limit - seen]:
            yield pr
        seen += len(batch)
        if len(batch) < PAGE_SIZE or seen >= limit:
            return


def human_reviews(reviews: list[dict], author: str, merged_at: datetime) -> list[dict]:
    """Reviews by a human other than the author, on a known commit, before the merge."""
    kept = []
    for review in reviews:
        submitted = parse_time(review.get("submitted_at"))
        if not submitted or submitted > merged_at or not review.get("commit_id"):
            continue
        if is_human_reviewer(review.get("user"), author):
            kept.append(review)
    return kept


def substantive_ids(reviews: list[dict], comments: list[dict]) -> set[int]:
    """IDs of the reviews that say something.

    A review counts if it requested changes, or has a body or inline comment
    that is a review point (not an acknowledgement, ACK-1, or a command, CMD-1).
    A bare approval doesn't count.
    """
    with_inline = {c.get("pull_request_review_id") for c in comments if is_review_point(c.get("body"))}
    return {
        review["id"] for review in reviews
        if review.get("state") == "CHANGES_REQUESTED"
        or is_review_point(review.get("body"))
        or review["id"] in with_inline
    }


def substantive_conversation(conversation: list[dict], author: str, merged_at: datetime) -> set[int]:
    """IDs of conversation comments by a human other than the author, before merge, that are review points."""
    return {
        c["id"] for c in conversation
        if is_human_reviewer(c.get("user"), author)
        and parse_time(c.get("created_at")) <= merged_at
        and is_review_point(c.get("body"))
    }


def approved_head(reviews: list[dict]) -> str | None:
    """The commit of the last approval before the merge: the must-not-block head."""
    approvals = [r for r in reviews if r.get("state") == "APPROVED" and r.get("commit_id")]
    if not approvals:
        return None
    return max(approvals, key=lambda r: parse_time(r["submitted_at"]))["commit_id"]
