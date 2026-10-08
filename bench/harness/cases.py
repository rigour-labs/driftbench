"""What a tool reviews for one pull request (docs/SPEC.md, "Rounds" and "Must-not-block cases").

- one case per round: the round's trusted head;
- one must-not-block case: the approved head, or the merged head when there
  is no trusted approval (`source: merged`).
A tool runs once per distinct head; every case on that head shares the result.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ReviewCase:
    case_id: str
    pr: int
    head_sha: str
    kind: str          # "round" or "must_not_block"
    round: int | None
    source: str = ""   # must_not_block only: "approved" or "merged"
    approval_overridden: bool = False


def cases_for_pr(pr: dict) -> list[ReviewCase]:
    cases = [
        ReviewCase(f"{pr['number']}-round-{r['index']}", pr["number"], r["head_sha"], "round", r["index"])
        for r in pr["rounds"]
    ]
    approved = pr.get("approved_head_sha")
    cases.append(ReviewCase(
        case_id=f"{pr['number']}-must-not-block",
        pr=pr["number"],
        head_sha=approved or pr["head_sha"],
        kind="must_not_block",
        round=None,
        source="approved" if approved else "merged",
        approval_overridden=bool(pr.get("approval_overridden")),
    ))
    return cases


def heads_to_run(cases: list[ReviewCase]) -> dict[str, list[ReviewCase]]:
    """Cases grouped by head, in first-seen order."""
    grouped: dict[str, list[ReviewCase]] = {}
    for case in cases:
        grouped.setdefault(case.head_sha, []).append(case)
    return grouped
