"""Findings from a review written as prose: the file:line places it cites in changed files.

Claude Code's /code-review answers in text. A finding is a cited place in a
file the change touched, in any of the shapes reviews use:
- `path:12`, `path:12-15`, `path:L12-L15`, with or without backticks;
- a markdown link whose text or target cites the place, e.g. [path:12](...);
- a link to the line, `.../blob/<sha>/path#L12-L15` or `path#L12`.
A cited path may carry a prefix (a URL, `./`, a checkout directory); it
counts when it ends in a changed path. Each place counts once.

Alongside, `diagnostics` gives numbers only, never text: how long the
answer was and how many citation-like and link-like strings it had, so a
review that cites nothing can be told apart from one the parser misread.
"""
from __future__ import annotations

import re

from bench.harness.types import Finding

PATH = r"[\w./-]+\.\w+"
COLON_RE = re.compile(rf"({PATH}):L?(\d+)(?:-L?(\d+))?")
ANCHOR_RE = re.compile(rf"({PATH})#L(\d+)(?:-L(\d+))?")
LINK_RE = re.compile(r"https?://\S+")


def changed_path(cited: str, changed: set[str]) -> str | None:
    """The changed file a cited path names, allowing a leading URL, `./` or directory prefix."""
    while cited.startswith("./"):
        cited = cited[2:]
    if cited in changed:
        return cited
    return next((p for p in sorted(changed, key=len, reverse=True) if cited.endswith("/" + p)), None)


def places(text: str) -> list[tuple[str, str, int, int | None]]:
    """(line of text, cited path, first line, last line) for every citation-shaped string, in order."""
    found = []
    for row in text.splitlines():
        for pattern in (COLON_RE, ANCHOR_RE):
            for match in pattern.finditer(row):
                found.append((row, match.group(1), int(match.group(2)), int(match.group(3)) if match.group(3) else None))
    return found


def citations(text: str, changed: set[str]) -> list[Finding]:
    """Each cited place in a changed file, once, as a non-blocking finding."""
    seen: set[tuple[str, int, int | None]] = set()
    findings = []
    for row, cited, first, last in places(text):
        path = changed_path(cited, changed)
        if path and (path, first, last) not in seen:
            seen.add((path, first, last))
            findings.append(Finding(path, first, False, row.strip(), "code-review", end_line=last))
    return findings


def diagnostics(text: str, changed: set[str], num_turns: int | None) -> dict:
    """Numbers only: the answer's length, citation-like and link-like strings, and how many were usable."""
    found = places(text)
    return {"result_chars": len(text), "citation_like": len(found), "link_like": len(LINK_RE.findall(text)),
            "cited_changed": len(citations(text, changed)), "num_turns": num_turns}
