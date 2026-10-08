"""Was an inline point acted on before merge (docs/SPEC.md, rule ACTED-1)?

Yes if the code changed within ACTED_WINDOW lines of the anchor between the
commit the comment was written on and the merged head, or the file was
removed. How the change is found (`basis`):
- "ancestor": the anchor commit is an ancestor of the merged head; the
  compare patch is exactly the later commits' change.
- "direct": the branch was amended or force-pushed in place (no upstream
  commits gained); the file at the two commits is diffed directly.
- "rebased": the merged head was rebased onto newer upstream commits; a
  direct diff would mix in upstream changes, so the answer is unknown.
Unknown also for left-side or lineless comments, a commit GitHub no longer
has, or a file GitHub returns no content for.
"""
from __future__ import annotations

import re

from bench.collect.github import GitHubClient
from bench.points.file_at import changed_lines_between, file_at

ACTED_WINDOW = 3
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")
ANCESTOR_STATUSES = ("ahead", "identical")
Verdict = tuple[bool | None, str | None]


def changed_old_lines(patch: str) -> set[int]:
    """Old-side line numbers a unified diff removes, or inserts after.

    An insertion lands on the old line it follows, so a replacement (a removed
    line, then an added one) marks one line, not two.
    """
    changed: set[int] = set()
    old = 0
    for row in patch.splitlines():
        hunk = HUNK_RE.match(row)
        if hunk:
            old = int(hunk.group(1))
        elif row.startswith("-"):
            changed.add(old)
            old += 1
        elif row.startswith("+"):
            changed.add(max(old - 1, 1))
        elif not row.startswith("\\"):
            old += 1
    return changed


def window(anchor: dict) -> set[int]:
    first = anchor.get("start_line") or anchor["line"]
    return set(range(first - ACTED_WINDOW, anchor["line"] + ACTED_WINDOW + 1))


def file_entry(compare: dict, path: str) -> dict | None:
    return next(
        (f for f in compare.get("files") or [] if path in (f.get("filename"), f.get("previous_filename"))),
        None,
    )


def from_patch(compare: dict, anchor: dict) -> bool | None:
    entry = file_entry(compare, anchor["path"])
    if entry is None:
        return False
    if entry.get("status") == "removed":
        return True
    return bool(changed_old_lines(entry["patch"]) & window(anchor)) if "patch" in entry else None


def from_contents(client: GitHubClient, repo: str, anchor: dict, compare: dict, merged_head: str) -> bool | None:
    entry = file_entry(compare, anchor["path"]) or {}
    old = file_at(client, repo, anchor["path"], anchor["commit_sha"])
    new = file_at(client, repo, entry.get("filename", anchor["path"]), merged_head)
    if not new.found:
        return True if old.found else None
    if old.text is None or new.text is None:
        return None
    return bool(changed_lines_between(old.text, new.text) & window(anchor))


def acted_on(client: GitHubClient, repo: str, anchor: dict | None, merged_head: str, pr_commits: int) -> Verdict:
    """(acted on, basis) for one inline anchor."""
    if not anchor or not anchor.get("line") or anchor.get("side") == "LEFT":
        return None, None
    if anchor["commit_sha"] == merged_head:
        return False, "ancestor"
    compare = client.get_optional(f"repos/{repo}/compare/{anchor['commit_sha']}...{merged_head}")
    if not compare:
        return None, None
    if compare.get("status") in ANCESTOR_STATUSES:
        return from_patch(compare, anchor), "ancestor"
    if compare.get("ahead_by", pr_commits + 1) <= pr_commits:
        return from_contents(client, repo, anchor, compare, merged_head), "direct"
    return None, "rebased"
