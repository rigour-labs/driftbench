"""ACTED-1 for a pull request rebased onto newer upstream commits: the `range` basis.

A direct diff between the anchor commit A and the merged head H mixes the
pull request's own edits with upstream changes. Instead, compare the pull
request's OWN change to the anchored file at both points:

- P_A: the patch from A's fork point to A (`compare/<base>...A`);
- P_H: the patch from H's fork point to H (`compare/<base>...H`),

where <base> is the pull request's recorded base commit, so each compare
starts at that version's merge base and upstream drift is in neither patch.

- If the anchored lines include lines the pull request added at A: acted on
  when any of those lines is no longer added, as is, at H.
- If the anchored lines are context only: acted on when P_H adds or removes
  lines, not already in P_A, near where the anchored lines sit at H.
- If the pull request no longer touches the file at H: acted on if its
  anchored lines were its own (the change was taken out), else unknown.
Unknown (None) when a patch can't be read (too big, list truncated, commit
gone) or the file's text at A or H can't be read.
"""
from __future__ import annotations

import dataclasses
import re
from collections import Counter

from bench.collect.github import GitHubClient
from bench.points.file_at import file_at
from bench.score.linemap import map_line

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
COMPARE_FILE_CAP = 300


@dataclasses.dataclass(frozen=True)
class OwnPatch:
    added: list[tuple[int, str]]    # (head-side line, text)
    removed: list[tuple[int, str]]  # (head-side line it sits before, text)


MISSING = OwnPatch([], [])  # the pull request doesn't touch the file at this point


def parse_own_patch(patch: str) -> OwnPatch:
    added, removed, line = [], [], 0
    for row in patch.splitlines():
        hunk = HUNK_RE.match(row)
        if hunk:
            line = int(hunk.group(1))
        elif row.startswith("+"):
            added.append((line, row[1:]))
            line += 1
        elif row.startswith("-"):
            removed.append((line, row[1:]))
        elif not row.startswith("\\"):
            line += 1
    return OwnPatch(added, removed)


def own_patch(client: GitHubClient, repo: str, base_sha: str, sha: str, path: str) -> OwnPatch | None:
    """The pull request's own patch to `path` at `sha`; MISSING if it doesn't touch it; None if unreadable."""
    compare = client.get_optional(f"repos/{repo}/compare/{base_sha}...{sha}")
    if not compare:
        return None
    files = compare.get("files") or []
    entry = next((f for f in files if path in (f.get("filename"), f.get("previous_filename"))), None)
    if entry is None:
        return None if len(files) >= COMPARE_FILE_CAP else MISSING
    return parse_own_patch(entry["patch"]) if "patch" in entry else None


def mapped_window(old: str, new: str, window: set[int]) -> set[int]:
    """The anchored window's place in the newer text, widened to cover both mapped ends."""
    ends = [map_line(old, new, line) for line in (min(window), max(window)) if line >= 1]
    ends = [end for end in ends if end is not None]
    return set(range(min(ends), max(ends) + 1)) if ends else set()


def context_only(client: GitHubClient, repo: str, anchor: dict, own: tuple[OwnPatch, OwnPatch],
                 heads: tuple[str, set[int]]) -> bool | None:
    """Anchored lines were context: did the pull request newly change lines near them by H?"""
    own_a, own_h = own
    merged_head, window = heads
    old = file_at(client, repo, anchor["path"], anchor["commit_sha"]).text
    new = file_at(client, repo, anchor["path"], merged_head).text
    if old is None or new is None:
        return None
    added_a = {text for _, text in own_a.added}
    removed_a = {text for _, text in own_a.removed}
    fresh = {line for line, text in own_h.added if text not in added_a}
    fresh |= {line for line, text in own_h.removed if text not in removed_a}
    return bool(fresh & mapped_window(old, new, window))


def from_range(client: GitHubClient, repo: str, anchor: dict, base_sha: str, merged_head: str,
               window: set[int]) -> bool | None:
    own_a = own_patch(client, repo, base_sha, anchor["commit_sha"], anchor["path"])
    own_h = own_patch(client, repo, base_sha, merged_head, anchor["path"])
    if own_a is None or own_a is MISSING or own_h is None:
        return None
    anchored_own = {text for line, text in own_a.added if line in window}
    if own_h is MISSING:
        return True if anchored_own else None
    if anchored_own:
        still_added = Counter(text for _, text in own_h.added)
        return any(still_added[text] == 0 for text in anchored_own)
    return context_only(client, repo, anchor, (own_a, own_h), (merged_head, window))
