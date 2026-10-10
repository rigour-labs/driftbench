"""What the issue judge reads: each human point, and each entrant's review of the heads it may use.

The population is the headline's: location-scorable points of the reviewed
pull requests. An entrant's review of a point is its output on the point's
eligible heads (rounds up to the point's own, time-correct), taken from the
run's `paid_output`: Claude Code's whole answer, Rigour's findings with
their file, line and full message. Nothing names the entrant: words that
would (a tool's name, a CLAUDE.md check, a "generated with" footer) are
redacted before the judge reads them.
"""
from __future__ import annotations

import re
from pathlib import Path

from bench.harness.runner import read_record
from bench.repos import slug_of
from bench.score.match import eligible_heads, location_scorable

REVIEW_CHARS = 12_000  # per head; longer answers are cut and say so
IDENTIFYING = (
    (re.compile(r"\bclaude\.md\b", re.IGNORECASE), "[the repository's agent instructions file]"),
    (re.compile(r"^.*generated with.*$", re.IGNORECASE | re.MULTILINE), ""),
    (re.compile(r"\b(?:claude(?:[ -]code)?|anthropic|rigour)\b", re.IGNORECASE), "[tool]"),
)


def redact(text: str) -> str:
    """The review without words that would tell the judge which entrant wrote it."""
    for pattern, replacement in IDENTIFYING:
        text = pattern.sub(replacement, text)
    return text


def head_review(record: dict) -> str:
    """One head's review as text, from what the entrant wrote; empty if it wrote nothing."""
    output = record.get("paid_output") or {}
    text = (output.get("review_text") or "").strip()
    if not text:
        text = "\n".join(f"- {f.get('path')}:{f.get('line')}: {f.get('message', '')}"
                         for f in output.get("findings") or [])
    text = redact(text)
    if len(text) > REVIEW_CHARS:
        text = text[:REVIEW_CHARS] + "\n[review cut at 12,000 characters]"
    return text


def reviews_by_head(run_dir: Path, tool: str, repo: str) -> dict[str, str]:
    reviews = {}
    for path in sorted((run_dir / tool / slug_of(repo)).glob("*/*.json")):
        record = read_record(path)
        reviews[record["head_sha"]] = head_review(record)
    return reviews


def entrant_review(point: dict, rounds: list[dict], reviews: dict[str, str]) -> str | None:
    """The entrant's reviews of the point's eligible heads, oldest first; None if it reviewed none of them."""
    found = [reviews[head] for head in eligible_heads(point, rounds) if head in reviews]
    if not found:
        return None
    return "\n\n---\n\n".join(text or "(no comments)" for text in found)


def judged_points(corpus: dict, points_file: dict) -> list[dict]:
    """Location-scorable points of the pull requests the (restricted) corpus holds."""
    numbers = {pr["number"] for pr in corpus["prs"]}
    return [p for p in points_file["points"] if p["pr"] in numbers and location_scorable(p)]
