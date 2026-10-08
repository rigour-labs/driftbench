"""A file's text at a commit, and the old-side lines that differ between two versions."""
from __future__ import annotations

import base64
import binascii
import dataclasses
import difflib
from urllib.parse import quote

from bench.collect.github import GitHubClient


@dataclasses.dataclass(frozen=True)
class FileAt:
    found: bool
    text: str | None  # None when found but GitHub returned no content (too large, binary)


def file_at(client: GitHubClient, repo: str, path: str, sha: str) -> FileAt:
    data = client.get_optional(f"repos/{repo}/contents/{quote(path)}?ref={sha}")
    if data is None:
        return FileAt(found=False, text=None)
    if not isinstance(data, dict) or data.get("encoding") != "base64" or not data.get("content"):
        return FileAt(found=True, text=None)
    try:
        return FileAt(found=True, text=base64.b64decode(data["content"]).decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError):
        return FileAt(found=True, text=None)


def changed_lines_between(old: str, new: str) -> set[int]:
    """Old-side line numbers replaced, deleted, or inserted after (same convention as a patch)."""
    changed: set[int] = set()
    matcher = difflib.SequenceMatcher(a=old.splitlines(), b=new.splitlines(), autojunk=False)
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            changed.update(range(i1 + 1, i2 + 1))
        elif tag == "insert":
            changed.add(max(i1, 1))
    return changed
