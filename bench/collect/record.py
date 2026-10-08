"""The frozen record of one pull request: IDs, SHAs, positions, timestamps. No text.

Comment and review text is not stored. Its SHA-256 and length are, so a later
fetch by ID can prove the text is unchanged (docs/SPEC.md, "Storage").
"""
from __future__ import annotations

import dataclasses
import hashlib

from bench.collect.rounds import ReviewRound, round_of_review
from bench.collect.select import is_bot, parse_time


def text_digest(text: str | None) -> dict:
    body = text or ""
    return {"body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), "body_chars": len(body)}


def review_entry(review: dict, rounds: dict[int, int]) -> dict:
    return {
        "id": review["id"],
        "round": rounds[review["id"]],
        "state": review.get("state"),
        "commit_sha": review["commit_id"],
        "submitted_at": review["submitted_at"],
        **text_digest(review.get("body")),
    }


def comment_entry(comment: dict, rounds: dict[int, int], author: str) -> dict:
    """An inline comment, anchored where it was written (the `original_*` fields)."""
    return {
        "id": comment["id"],
        "review_id": comment.get("pull_request_review_id"),
        "round": rounds.get(comment.get("pull_request_review_id")),
        "in_reply_to": comment.get("in_reply_to_id"),
        "by_author": (comment.get("user") or {}).get("login") == author,
        "path": comment.get("path"),
        "line": comment.get("original_line"),
        "start_line": comment.get("original_start_line"),
        "side": comment.get("side"),
        "commit_sha": comment.get("original_commit_id"),
        "created_at": comment.get("created_at"),
        **text_digest(comment.get("body")),
    }


def pr_record(pr: dict, reviews: list[dict], comments: list[dict], commits: list[dict], rounds: list[ReviewRound]) -> dict:
    """Assemble the record; `reviews` are the substantive reviews only."""
    author = (pr.get("user") or {}).get("login", "")
    merged_at = parse_time(pr["merged_at"])
    review_rounds = round_of_review(rounds)
    kept_comments = [
        c for c in comments
        if not is_bot(c.get("user")) and parse_time(c.get("created_at")) <= merged_at
    ]
    return {
        "number": pr["number"],
        "base_ref": pr["base"]["ref"],
        "base_sha": pr["base"]["sha"],
        "head_sha": pr["head"]["sha"],
        "merge_commit_sha": pr.get("merge_commit_sha"),
        "created_at": pr["created_at"],
        "merged_at": pr["merged_at"],
        "updated_at": pr["updated_at"],
        "commits": [{"sha": c["sha"], "committed_at": c["commit"]["committer"]["date"]} for c in commits],
        "rounds": [dataclasses.asdict(r) for r in rounds],
        "reviews": [review_entry(r, review_rounds) for r in reviews],
        "comments": [comment_entry(c, review_rounds, author) for c in kept_comments],
    }
