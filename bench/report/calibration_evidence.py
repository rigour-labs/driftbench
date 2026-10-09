"""What a calibration judge reads for each entry: the point, and the code it is about.

- acted-on: the review comment and the file's change around the anchor, from
  the commit the comment was written on to the merged head;
- location: the review comment, the code around its anchor, and the tool's
  finding as published (its reduced message, file and line).

The same text goes to every judge (human, Claude labeller, or model), so
their verdicts rest on the same evidence.
"""
from __future__ import annotations

import difflib

from bench.collect.github import GitHubClient
from bench.points.file_at import file_at

CONTEXT_LINES = 12


def around(text: str | None, line: int, width: int = CONTEXT_LINES) -> str:
    """Numbered lines of `text` within `width` of `line`."""
    if text is None:
        return "(file not readable)"
    lines = text.splitlines()
    start, end = max(1, line - width), min(len(lines), line + width)
    return "\n".join(f"{n:>5} {lines[n - 1]}" for n in range(start, end + 1))


def window_diff(old: str | None, new: str | None, line: int, width: int = CONTEXT_LINES) -> str:
    """The unified diff of old -> new, keeping only hunks that touch `line` +- `width` on the old side."""
    if old is None:
        return "(the file at the reviewed commit is not readable)"
    if new is None:
        return "(the file was removed or is not readable at the merged head)"
    low, high = line - width, line + width
    kept: list[str] = []
    matcher = difflib.SequenceMatcher(a=old.splitlines(), b=new.splitlines(), autojunk=False)
    for group in matcher.get_grouped_opcodes(3):
        if group[0][1] + 1 > high or group[-1][2] < low:
            continue
        i1, i2, j1, j2 = group[0][1], group[-1][2], group[0][3], group[-1][4]
        kept.append(f"@@ -{i1 + 1},{i2 - i1} +{j1 + 1},{j2 - j1} @@")
        for tag, a1, a2, b1, b2 in group:
            kept += [f" {x}" for x in matcher.a[a1:a2]] if tag == "equal" else []
            kept += [f"-{x}" for x in matcher.a[a1:a2]] if tag in ("replace", "delete") else []
            kept += [f"+{x}" for x in matcher.b[b1:b2]] if tag in ("replace", "insert") else []
    return "\n".join(kept) or f"(no change within {width} lines of line {line})"


def acted_evidence(client: GitHubClient, repo: str, point: dict, text: str, merged_head: str) -> str:
    anchor = point["anchor"]
    old = file_at(client, repo, anchor["path"], anchor["commit_sha"]).text
    new = file_at(client, repo, anchor["path"], merged_head).text
    return (f"Review comment on {anchor['path']} line {anchor['line']}:\n{text}\n\n"
            f"Change to that file around the line, from the reviewed commit to the merged head:\n"
            f"{window_diff(old, new, anchor['line'])}")


def location_evidence(client: GitHubClient, repo: str, point: dict, text: str, tool: str, finding: dict) -> str:
    anchor = point["anchor"]
    code = around(file_at(client, repo, anchor["path"], anchor["commit_sha"]).text, anchor["line"])
    return (f"Review comment on {anchor['path']} line {anchor['line']}:\n{text}\n\nCode at the review:\n{code}\n\n"
            f"Finding by {tool} at {finding['path']} line {finding['line']}: {finding['message']}")
