"""Did a tool flag the same spot as a human point, before the human did (docs/SPEC.md, "Matching")?

A point is location-scorable when it is scorable (kept, has a round) and is
an inline comment with a line on the new side. A tool may use any head from
round 1 up to the point's own round: the code existed then, and the review
that raised the point did not. Each finding on one of those heads, in the
point's file, is carried to the anchor commit and its distance to the
anchored lines is measured. The closest one decides.
"""
from __future__ import annotations

import dataclasses

from bench.score.linemap import FileVersions

WINDOWS = (0, 3, 10)
HEADLINE_WINDOW = 3
SCORED_VERDICTS = ("pass", "fail")


@dataclasses.dataclass(frozen=True)
class Match:
    head_sha: str
    finding_index: int
    mapped_line: int
    distance: int


def location_scorable(point: dict) -> bool:
    anchor = point.get("anchor") or {}
    return bool(point["scorable"] and point["kind"] == "inline" and anchor.get("line") and anchor.get("side") != "LEFT")


def distance(line: int, anchor: dict, end_line: int | None = None) -> int:
    """Gap between a finding's lines [line, end_line] and the anchored lines; 0 when they overlap."""
    first, last = sorted((line, end_line or line))
    start = anchor.get("start_line") or anchor["line"]
    if first <= anchor["line"] and start <= last:
        return 0
    return start - last if last < start else first - anchor["line"]


def carried_span(finding: dict, head: str, anchor: dict, versions: FileVersions) -> tuple[int, int | None] | None:
    """The finding's line (and end line) at the anchor commit, or None if they can't be carried."""
    line = versions.carry(anchor["path"], finding["line"], head, anchor["commit_sha"])
    if line is None:
        return None
    if not finding.get("end_line"):
        return line, None
    end = versions.carry(anchor["path"], finding["end_line"], head, anchor["commit_sha"])
    return line, end


def eligible_heads(point: dict, rounds: list[dict]) -> list[str]:
    """Heads of rounds 1..point's round, deduplicated, oldest first."""
    heads: list[str] = []
    for rnd in rounds:
        if rnd["index"] <= point["round"] and rnd["head_sha"] not in heads:
            heads.append(rnd["head_sha"])
    return heads


def closest_match(point: dict, heads: list[str], records: dict[str, dict], versions: FileVersions,
                  blocking_only: bool = False) -> Match | None:
    """The closest finding (only blocking ones if `blocking_only`) on the eligible heads."""
    anchor = point["anchor"]
    best: Match | None = None
    for head in heads:
        record = records.get(head)
        if not record or record["verdict"] not in SCORED_VERDICTS:
            continue
        for index, finding in enumerate(record["findings"]):
            if finding.get("path") != anchor["path"] or not finding.get("line"):
                continue
            if blocking_only and not finding.get("blocking"):
                continue
            span = carried_span(finding, head, anchor, versions)
            if span is None:
                continue
            gap = distance(span[0], anchor, span[1])
            if best is None or gap < best.distance:
                best = Match(head, index, span[0], gap)
    return best
