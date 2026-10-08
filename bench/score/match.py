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


def distance(line: int, anchor: dict) -> int:
    start = anchor.get("start_line") or anchor["line"]
    if start <= line <= anchor["line"]:
        return 0
    return min(abs(line - start), abs(line - anchor["line"]))


def eligible_heads(point: dict, rounds: list[dict]) -> list[str]:
    """Heads of rounds 1..point's round, deduplicated, oldest first."""
    heads: list[str] = []
    for rnd in rounds:
        if rnd["index"] <= point["round"] and rnd["head_sha"] not in heads:
            heads.append(rnd["head_sha"])
    return heads


def closest_match(point: dict, heads: list[str], records: dict[str, dict], versions: FileVersions) -> Match | None:
    anchor = point["anchor"]
    best: Match | None = None
    for head in heads:
        record = records.get(head)
        if not record or record["verdict"] not in SCORED_VERDICTS:
            continue
        for index, finding in enumerate(record["findings"]):
            if finding.get("path") != anchor["path"] or not finding.get("line"):
                continue
            mapped = versions.carry(anchor["path"], finding["line"], head, anchor["commit_sha"])
            if mapped is None:
                continue
            gap = distance(mapped, anchor)
            if best is None or gap < best.distance:
                best = Match(head, index, mapped, gap)
    return best
