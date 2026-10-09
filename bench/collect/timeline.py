"""Which commit was the PR's head at a given moment (docs/SPEC.md, "Rounds").

Built from the issue timeline, with no text:
- `head_ref_force_pushed` events carry the new head and the push time
  (source "push", exact);
- `committed` events carry the commit and its committer date, which can be
  earlier than the push (source "commit_date", approximate).
If the timeline has neither, the PR's commit list gives committer dates
(source "commit_date"). The timeline only lists commits still in the PR, so
heads that were rebased away are missing. Every review with a trusted commit
(rule COMMIT-1) adds one more entry: that commit existed and was under review
at that moment (source "review").
"""
from __future__ import annotations

from bench.collect.select import parse_time


def head_history(timeline: list[dict], commits: list[dict], reviews: list[dict]) -> list[dict]:
    """`[{sha, at, source}]`, oldest first."""
    history = []
    for event in timeline:
        if event.get("event") == "head_ref_force_pushed" and event.get("commit_id"):
            history.append({"sha": event["commit_id"], "at": event["created_at"], "source": "push"})
        elif event.get("event") == "committed" and (event.get("committer") or {}).get("date"):
            history.append({"sha": event["sha"], "at": event["committer"]["date"], "source": "commit_date"})
    if not history:
        history = [
            {"sha": c["sha"], "at": c["commit"]["committer"]["date"], "source": "commit_date"}
            for c in commits
        ]
    history += [
        {"sha": r["commit_id"], "at": r["submitted_at"], "source": "review"}
        for r in reviews if r.get("commit_id")
    ]
    return sorted(history, key=lambda entry: parse_time(entry["at"]))


def head_at(history: list[dict], when: str) -> dict | None:
    """The latest history entry at or before `when`, or None if there is none."""
    moment = parse_time(when)
    earlier = [entry for entry in history if parse_time(entry["at"]) <= moment]
    return earlier[-1] if earlier else None
