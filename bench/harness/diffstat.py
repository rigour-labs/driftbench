"""What a unified diff changed, on the head side: hunks and a changed-line count."""
from __future__ import annotations

import dataclasses
import re

FILE_RE = re.compile(r"^(?:\+\+\+|---) (?:[ab]/)?(.*)$")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclasses.dataclass(frozen=True)
class Hunk:
    path: str
    first_line: int  # first added line, or the hunk's head-side start when it only deletes
    added: int
    removed: int


def parse_hunks(diff: str) -> list[Hunk]:
    """Hunks per file; a deleted file is reported under its old path.

    `---`/`+++` lines are file headers only between `diff --git` and the
    first `@@`; inside a hunk they are content (e.g. a removed "-- comment").
    """
    hunks: list[Hunk] = []
    path: str | None = None
    current: dict | None = None
    in_header = True  # also accept a plain unified diff with no "diff --git" line
    for row in diff.splitlines():
        if row.startswith("diff --git"):
            flush(hunks, current)
            current, path, in_header = None, None, True
        elif in_header and row.startswith(("--- ", "+++ ")) and (match := FILE_RE.match(row)):
            path = match.group(1) if match.group(1) != "/dev/null" else path
        elif hunk := HUNK_RE.match(row):
            flush(hunks, current)
            in_header = False
            current = {"path": path, "start": int(hunk.group(1)), "line": int(hunk.group(1)),
                       "first": None, "added": 0, "removed": 0}
        elif current is not None and not in_header:
            track(current, row)
    flush(hunks, current)
    return hunks


def track(current: dict, row: str) -> None:
    if row.startswith("+"):
        current["first"] = current["first"] or current["line"]
        current["added"] += 1
        current["line"] += 1
    elif row.startswith("-"):
        current["removed"] += 1
    elif not row.startswith("\\"):
        current["line"] += 1


def flush(hunks: list[Hunk], current: dict | None) -> None:
    if current is None or current["path"] is None:
        return
    hunks.append(Hunk(current["path"], current["first"] or current["start"], current["added"], current["removed"]))


def changed_lines(hunks: list[Hunk]) -> int:
    """Added plus removed lines: the denominator for findings per 100 changed lines."""
    return sum(h.added + h.removed for h in hunks)
