"""Review rounds: each distinct head a human reviewed, in the order it was first reviewed.

Round k is what a tool is allowed to see when it is compared with round k's
reviews: the code at that head, and only reviews from earlier rounds.
"""
from __future__ import annotations

import dataclasses

from bench.collect.select import parse_time


@dataclasses.dataclass(frozen=True)
class ReviewRound:
    index: int
    head_sha: str
    first_review_at: str
    review_ids: tuple[int, ...]


def build_rounds(reviews: list[dict]) -> list[ReviewRound]:
    """Group reviews by the commit they were made on, ordered by first review time."""
    by_head: dict[str, list[dict]] = {}
    for review in sorted(reviews, key=lambda r: parse_time(r["submitted_at"])):
        by_head.setdefault(review["commit_id"], []).append(review)
    return [
        ReviewRound(
            index=index,
            head_sha=head,
            first_review_at=group[0]["submitted_at"],
            review_ids=tuple(r["id"] for r in group),
        )
        for index, (head, group) in enumerate(by_head.items(), start=1)
    ]


def round_of_review(rounds: list[ReviewRound]) -> dict[int, int]:
    return {review_id: rnd.index for rnd in rounds for review_id in rnd.review_ids}
