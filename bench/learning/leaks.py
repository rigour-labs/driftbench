"""Every lesson in a store must come from before its pull request's cutoff (docs/LEARNING.md, "Leakage").

Checked on the store the learner wrote, against the crawl, for each piece of
evidence on each lesson:
- never the pull request under review (no round of it, ever);
- a review point: its pull request merged before the cutoff, and the comment
  or review it came from written before the cutoff;
- an outcome on main (a later fix, a revert): dated before the cutoff.
A pull request with any leak has every head marked as an error, never served.
Comments edited after the cutoff are counted: only today's text exists.
"""
from __future__ import annotations

POINT = "point"
DATED = ("outcome", "lines")


def written(crawl: dict) -> dict[tuple[int, str], dict]:
    """(pr, evidence comment id) -> {at, edited} for every comment and review in the crawl."""
    found = {}
    for number, entry in crawl["reviews"].items():
        for c in entry["comments"]:
            found[(int(number), str(c["id"]))] = {"at": c.get("created_at") or "", "edited": c.get("updated_at") or ""}
        for r in entry["reviews"]:
            found[(int(number), f"review-{r['id']}")] = {"at": r.get("submitted_at") or "", "edited": ""}
    return found


def source_id(comment: str) -> str:
    """A review body's points are `review-<id>-<n>`; inline comments are their own id."""
    parts = comment.split("-")
    return "-".join(parts[:2]) if comment.startswith("review-") and len(parts) == 3 else comment


def evidence_leak(e: dict, pr: int, cutoff: str, merged: dict[int, str], comments: dict) -> str | None:
    if e.get("pr") == pr:
        return f"evidence from the pull request under review ({e.get('comment')})"
    kind = e.get("kind") or POINT
    if kind == POINT:
        if not merged.get(e.get("pr"), "9999") < cutoff:
            return f"pull request #{e.get('pr')} not merged before the cutoff"
        seen = comments.get((e.get("pr"), source_id(str(e.get("comment")))))
        if seen is None:
            return f"comment {e.get('comment')} is not in the crawl"
        if not seen["at"] < cutoff:
            return f"comment {e.get('comment')} written at {seen['at']}, after the cutoff"
    elif kind in DATED and not str(e.get("at") or "9999") < cutoff:
        return f"{kind} {e.get('comment')} dated {e.get('at')}, after the cutoff"
    return None


def check_store(lessons: list[dict], pr: int, cutoff: str, crawl: dict) -> dict:
    """{leaks: [...], edited_after: n, rejected: n} for one pull request's store."""
    merged = {p["number"]: p["merged_at"] for p in crawl["prs"]}
    comments = written(crawl)
    leaks, edited = [], set()
    for lesson in lessons:
        for e in lesson.get("evidence", []):
            leak = evidence_leak(e, pr, cutoff, merged, comments)
            if leak:
                leaks.append(f"{lesson.get('id')}: {leak}")
            seen = comments.get((e.get("pr"), source_id(str(e.get("comment")))))
            if seen and seen["edited"] >= cutoff:
                edited.add(lesson.get("id"))
    return {"leaks": leaks, "edited_after": sorted(edited),
            "rejected": sum(1 for lesson in lessons if lesson.get("state") == "rejected")}
