"""Was an inline point acted on before merge (docs/SPEC.md, rule ACTED-1)?

Yes if the code changed within ACTED_WINDOW lines of the anchor between the
commit the comment was written on and the merged head, or the file was
removed. How the change is found (`basis`):
- "ancestor": the anchor commit is an ancestor of the merged head; the
  compare patch is exactly the later commits' change. When the patch can't
  tell (no patch for a big file, or a file list truncated at 300), the file
  at the two commits is diffed instead, which is exact here too.
- "direct": the branch was amended or force-pushed in place (no upstream
  commits gained); the file at the two commits is diffed directly.
- "range": the merged head gained upstream commits (a rebase, or the base
  merged in); the pull request's own patch to the file at A and at H are
  compared, so upstream changes cancel out (bench/points/range_basis.py).
- "rebased": as for "range", but even that can't decide (a patch or file
  can't be read, or the pull request never touched the file at A).
Unknown also for left-side or lineless comments, a commit GitHub no longer
has, or a file GitHub returns no content for. A rename that a truncated file
list hides reads as a removal, so it counts as acted on.
"""
from __future__ import annotations

import re

from bench.collect.github import GitHubClient
from bench.points.file_at import changed_lines_between, file_at
from bench.points.range_basis import from_range

ACTED_WINDOW = 3
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")
ANCESTOR_STATUSES = ("ahead", "identical")
COMPARE_FILE_CAP = 300  # GitHub lists at most this many files in a compare
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
    """The answer from the compare patch, or None when the patch can't tell.

    It can't tell when the file has no patch (GitHub omits big ones), or is
    absent from a file list GitHub truncated at COMPARE_FILE_CAP.
    """
    files = compare.get("files") or []
    entry = file_entry(compare, anchor["path"])
    if entry is None:
        return None if len(files) >= COMPARE_FILE_CAP else False
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


def acted_on(client: GitHubClient, repo: str, anchor: dict | None, pr: dict) -> Verdict:
    """(acted on, basis) for one inline anchor; `pr` is the PR's record (head_sha, base_sha, commits)."""
    if not anchor or not anchor.get("line") or anchor.get("side") == "LEFT":
        return None, None
    merged_head, pr_commits = pr["head_sha"], len(pr["commits"])
    if anchor["commit_sha"] == merged_head:
        return False, "ancestor"
    compare = client.get_optional(f"repos/{repo}/compare/{anchor['commit_sha']}...{merged_head}")
    if not compare:
        return None, None
    if compare.get("status") in ANCESTOR_STATUSES:
        verdict = from_patch(compare, anchor)
        if verdict is None:
            verdict = from_contents(client, repo, anchor, compare, merged_head)
        return verdict, "ancestor"
    if compare.get("ahead_by", pr_commits + 1) <= pr_commits:
        return from_contents(client, repo, anchor, compare, merged_head), "direct"
    verdict = from_range(client, repo, anchor, pr["base_sha"], merged_head, window(anchor))
    return (verdict, "range") if verdict is not None else (None, "rebased")
