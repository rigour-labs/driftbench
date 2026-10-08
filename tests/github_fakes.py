"""Synthetic GitHub API objects for tests. No real repository data."""
from __future__ import annotations

AUTHOR = {"login": "author", "type": "User"}
REVIEWER = {"login": "reviewer", "type": "User"}
BOT = {"login": "ci[bot]", "type": "Bot"}


def make_pr(number: int, merged_at: str | None, updated_at: str, head: str = "h2") -> dict:
    return {
        "number": number,
        "user": AUTHOR,
        "merged_at": merged_at,
        "updated_at": updated_at,
        "created_at": "2026-01-01T00:00:00Z",
        "merge_commit_sha": f"m{number}",
        "base": {"ref": "main", "sha": "b0"},
        "head": {"sha": head},
    }


def make_review(review_id: int, state: str, commit: str, at: str, body: str = "", user: dict = REVIEWER) -> dict:
    return {"id": review_id, "state": state, "commit_id": commit, "submitted_at": at, "body": body, "user": user}


def make_comment(comment_id: int, review_id: int, at: str, user: dict = REVIEWER, **extra) -> dict:
    comment = {
        "id": comment_id,
        "pull_request_review_id": review_id,
        "created_at": at,
        "user": user,
        "path": "src/app.py",
        "original_line": 10,
        "original_start_line": None,
        "side": "RIGHT",
        "original_commit_id": "h1",
        "in_reply_to_id": None,
        "body": "please handle the empty case",
    }
    comment.update(extra)
    return comment


def make_commit(sha: str, at: str) -> dict:
    return {"sha": sha, "commit": {"committer": {"date": at}}}


class FakeClient:
    """Answers `get` and `get_all` from a dict keyed by API path, and records calls."""

    def __init__(self, pages: dict[str, list[list[dict]]], lists: dict[str, list[dict]]):
        self.pages = pages
        self.lists = lists
        self.calls: list[str] = []

    def get(self, path: str, params: dict[str, str] | None = None):
        self.calls.append(path)
        page = int((params or {}).get("page", "1"))
        pages = self.pages.get(path, [])
        return pages[page - 1] if page <= len(pages) else []

    def get_all(self, path: str, params: dict[str, str] | None = None) -> list:
        self.calls.append(path)
        return self.lists.get(path, [])
