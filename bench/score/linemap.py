"""Carry a line number from one version of a file to another (docs/SPEC.md, "Matching").

A finding on an earlier round's head is compared with a human anchor on a
later commit, so its line is moved through the diff between the two
versions: an unchanged line keeps its place; a line inside a changed region
maps to where that region starts in the newer version.
"""
from __future__ import annotations

import difflib

from bench.collect.github import GitHubClient
from bench.points.file_at import file_at


def map_line(old: str, new: str, line: int) -> int | None:
    """`line` (1-based) in `old`, as a line number in `new`; None if `line` is outside `old`."""
    old_lines, new_lines = old.splitlines(), new.splitlines()
    index = line - 1
    if not 0 <= index < max(len(old_lines), 1):
        return None
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, i1, i2, j1, _ in matcher.get_opcodes():
        if i1 <= index < i2 or (i1 == i2 == index):
            return j1 + (index - i1) + 1 if tag == "equal" else j1 + 1
    return None


class FileVersions:
    """File text by (path, commit), fetched once through the cached API."""

    def __init__(self, client: GitHubClient, repo: str):
        self.client = client
        self.repo = repo
        self.texts: dict[tuple[str, str], str | None] = {}

    def text_at(self, path: str, sha: str) -> str | None:
        key = (path, sha)
        if key not in self.texts:
            self.texts[key] = file_at(self.client, self.repo, path, sha).text
        return self.texts[key]

    def carry(self, path: str, line: int, from_sha: str, to_sha: str) -> int | None:
        """`line` of `path` at `from_sha`, as a line at `to_sha`; None if either version is unreadable."""
        if from_sha == to_sha:
            return line
        old, new = self.text_at(path, from_sha), self.text_at(path, to_sha)
        if old is None or new is None:
            return None
        return map_line(old, new, line)
