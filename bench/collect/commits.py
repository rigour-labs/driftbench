"""Which commit a review was really made on (docs/SPEC.md, rule COMMIT-1).

GitHub can report a review's `commit_id` as a commit created after the review
was submitted: an approval appears to move forward onto a later force-pushed
head. A commit can't be reviewed before it exists, so:

- the reported commit is trusted when its committer date is at or before the
  review (check "ok");
- otherwise, the commit the review's own inline comments were written on
  (`original_commit_id`), if that predates the review (check "from_comment");
- otherwise the review has no trusted commit (check "untrusted"): it is kept in
  the record but defines no round and is not an approved head.
"""
from __future__ import annotations

from bench.collect.github import GitHubClient
from bench.collect.select import parse_time


class CommitDates:
    """Committer dates by SHA: known ones first, the rest fetched once and cached."""

    def __init__(self, client: GitHubClient, repo: str, known: dict[str, str]):
        self.client = client
        self.repo = repo
        self.dates = dict(known)

    def date(self, sha: str | None) -> str | None:
        if not sha:
            return None
        if sha not in self.dates:
            commit = self.client.get_optional(f"repos/{self.repo}/commits/{sha}") or {}
            self.dates[sha] = ((commit.get("commit") or {}).get("committer") or {}).get("date")
        return self.dates[sha]

    def existed_by(self, sha: str | None, when: str) -> bool:
        created = self.date(sha)
        return bool(created) and parse_time(created) <= parse_time(when)


def known_dates(commits: list[dict], timeline: list[dict]) -> dict[str, str]:
    dates = {c["sha"]: c["commit"]["committer"]["date"] for c in commits}
    for event in timeline:
        if event.get("event") == "committed" and (event.get("committer") or {}).get("date"):
            dates.setdefault(event["sha"], event["committer"]["date"])
    return dates


def resolve_review(review: dict, comments: list[dict], dates: CommitDates) -> dict:
    """A copy of the review with `commit_id` set to the trusted commit (or None)."""
    submitted = review["submitted_at"]
    resolved = {**review, "reported_commit_id": review["commit_id"]}
    if dates.existed_by(review["commit_id"], submitted):
        return {**resolved, "commit_check": "ok"}
    written_on = sorted(
        (c for c in comments if c.get("pull_request_review_id") == review["id"]
         and dates.existed_by(c.get("original_commit_id"), submitted)),
        key=lambda c: parse_time(c["created_at"]),
    )
    if written_on:
        return {**resolved, "commit_id": written_on[-1]["original_commit_id"], "commit_check": "from_comment"}
    return {**resolved, "commit_id": None, "commit_check": "untrusted"}
