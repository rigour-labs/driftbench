"""Unified-diff hunks: parse them, map a line across them, and test overlap.

Every location in the arena (a reviewer's comment, a tool's finding, a
later bug fix) is moved onto one version of the file, the PR's merge
commit, before anything is compared. These functions do that from the
output of `git diff -U0`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int

    @property
    def old_end(self) -> int:
        """Last old line the hunk replaces (old_start - 1 for a pure insertion)."""
        return self.old_start + self.old_count - 1


def parse_hunks(diff_text: str) -> list[Hunk]:
    """Hunks of a single-file `git diff -U0`, in order."""
    hunks = []
    for line in diff_text.splitlines():
        match = _HUNK.match(line)
        if match:
            old_start, old_count, new_start, new_count = match.groups()
            hunks.append(Hunk(
                int(old_start), 1 if old_count is None else int(old_count),
                int(new_start), 1 if new_count is None else int(new_count),
            ))
    return hunks


def touches(hunks: list[Hunk], first: int, last: int) -> bool:
    """Whether any hunk removes or replaces an old line in [first, last]."""
    return any(h.old_count > 0 and h.old_start <= last and h.old_end >= first for h in hunks)


def map_line(hunks: list[Hunk], line: int) -> tuple[int, bool]:
    """(line in the new version, whether the line itself was changed).

    An unchanged line moves by the net size of the hunks before it. A changed
    line maps to the start of the hunk that replaced it (clamped to the last
    new line when the hunk only deletes).
    """
    shift = 0
    for hunk in hunks:
        if hunk.old_count > 0 and hunk.old_start <= line <= hunk.old_end:
            return max(1, hunk.new_start + min(line - hunk.old_start, max(hunk.new_count - 1, 0))), True
        if _before(hunk, line):
            shift += hunk.new_count - hunk.old_count
    return line + shift, False


def _before(hunk: Hunk, line: int) -> bool:
    """A hunk that sits entirely before `line` in the old version."""
    if hunk.old_count == 0:
        return hunk.old_start < line  # insertion after old_start
    return hunk.old_end < line
