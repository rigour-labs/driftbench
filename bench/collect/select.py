"""Which pull requests and reviews enter the corpus (docs/SPEC.md, "Pull requests")."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from bench.collect.github import GitHubClient
from bench.collect.text_rules import is_acknowledgement

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


def substantive_reviews(reviews: list[dict], comments: list[dict], author: str, merged_at: datetime) -> list[dict]:
    """Reviews by a human other than the author, before merge, that say something.

    A review counts if it requested changes, has a body that isn't just an
    acknowledgement (rule ACK-1), or has inline comments that aren't. A bare
    approval doesn't count.
    """
    with_inline = {
        c.get("pull_request_review_id") for c in comments if not is_acknowledgement(c.get("body"))
    }
    kept = []
    for review in reviews:
        submitted = parse_time(review.get("submitted_at"))
        if not submitted or submitted > merged_at or not review.get("commit_id"):
            continue
        if not is_human_reviewer(review.get("user"), author):
            continue
        says_something = (
            review.get("state") == "CHANGES_REQUESTED"
            or not is_acknowledgement(review.get("body"))
            or review.get("id") in with_inline
        )
        if says_something:
            kept.append(review)
    return kept
