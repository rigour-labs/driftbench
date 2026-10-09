"""The frozen record of one pull request: IDs, SHAs, positions, timestamps. No text.

Comment and review text is not stored. Its SHA-256 and length are, so a later
fetch by ID can prove the text is unchanged (docs/SPEC.md, "Storage").
"""
from __future__ import annotations

import dataclasses
import hashlib

from bench.collect.rounds import ReviewRound, round_for_head, round_of_review
from bench.collect.select import approval_overridden, approved_head, is_bot, parse_time
from bench.collect.timeline import head_at, head_history


@dataclasses.dataclass(frozen=True)
class PrFetch:
    """What the API returned for one PR.

    `reviews` are the human, pre-merge ones, with `commit_id` resolved by rule COMMIT-1.
    """
    pr: dict
    reviews: list[dict]
    comments: list[dict]
    conversation: list[dict]
    commits: list[dict]
    timeline: list[dict]

    @property
    def author(self) -> str:
        return (self.pr.get("user") or {}).get("login", "")


def text_digest(text: str | None) -> dict:
    body = text or ""
    return {"body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), "body_chars": len(body)}


def review_entry(review: dict, rounds: dict[int, int], substantive: set[int]) -> dict:
    return {
        "id": review["id"],
        "substantive": review["id"] in substantive,
        "round": rounds.get(review["id"]),
        "state": review.get("state"),
        "commit_sha": review["commit_id"],
        "reported_commit_sha": review.get("reported_commit_id", review["commit_id"]),
        "commit_check": review.get("commit_check", "ok"),
        "submitted_at": review["submitted_at"],
        **text_digest(review.get("body")),
    }


def comment_entry(comment: dict, rounds: dict[int, int], author: str) -> dict:
    """An inline comment, anchored where it was written (the `original_*` fields)."""
    return {
        "id": comment["id"],
        "kind": "inline",
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


def conversation_entry(comment: dict, rounds: list[ReviewRound], history: list[dict], author: str) -> dict:
    """A conversation-tab comment: no anchor; tied to the head it was written against.

    `round` is the round reviewed on that head, or None: such a point is
    counted but not scored, as there is no reviewed head to run a tool on.
    """
    head = head_at(history, comment["created_at"]) or {}
    return {
        "id": comment["id"],
        "kind": "conversation",
        "by_author": (comment.get("user") or {}).get("login") == author,
        "created_at": comment["created_at"],
        "head_sha": head.get("sha"),
        "head_source": head.get("source"),
        "round": round_for_head(rounds, head.get("sha")),
        **text_digest(comment.get("body")),
    }


def human_before_merge(items: list[dict], merged_at: str) -> list[dict]:
    cutoff = parse_time(merged_at)
    return [c for c in items if not is_bot(c.get("user")) and parse_time(c.get("created_at")) <= cutoff]


def pr_record(fetch: PrFetch, substantive: set[int], rounds: list[ReviewRound]) -> dict:
    pr = fetch.pr
    review_rounds = round_of_review(rounds)
    history = head_history(fetch.timeline, fetch.commits, fetch.reviews)
    return {
        "number": pr["number"],
        "base_ref": pr["base"]["ref"],
        "base_sha": pr["base"]["sha"],
        "head_sha": pr["head"]["sha"],
        "approved_head_sha": approved_head(fetch.reviews),
        "approval_overridden": approval_overridden(fetch.reviews),
        "merge_commit_sha": pr.get("merge_commit_sha"),
        "created_at": pr["created_at"],
        "merged_at": pr["merged_at"],
        "updated_at": pr["updated_at"],
        "commits": [{"sha": c["sha"], "committed_at": c["commit"]["committer"]["date"]} for c in fetch.commits],
        "head_history": history,
        "rounds": [dataclasses.asdict(r) for r in rounds],
        "reviews": [review_entry(r, review_rounds, substantive) for r in fetch.reviews],
        "comments": [
            comment_entry(c, review_rounds, fetch.author)
            for c in human_before_merge(fetch.comments, pr["merged_at"])
        ],
        "conversation": [
            conversation_entry(c, rounds, history, fetch.author)
            for c in human_before_merge(fetch.conversation, pr["merged_at"])
        ],
    }
