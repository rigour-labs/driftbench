"""SZZ: bugs a PR introduced that were fixed after it merged.

For each later fix commit, the lines it removes or replaces are blamed just
before the fix. Lines whose origin is the PR's merge commit (or one of the
PR's own commits) were written by the PR, so the PR introduced a bug there.
No reviewer's comment caused these fixes, which makes them fair ground
truth for every tool: each is judged on whether it pointed at those lines
when the PR was reviewed.

Known noise, reduced but not removed: fixes that also refactor, and
commits that say "fix" about something other than a defect.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from arena.diffmap import map_line
from arena.gitrepo import Commit, Git

FIX_SUBJECT = re.compile(r"\b(fix(e[sd])?|bug(fix)?|regression|hotfix|revert)\b", re.IGNORECASE)
NOT_A_DEFECT = re.compile(r"\b(typo|lint|format(ting)?|docs?|readme|changelog|test(s)?|ci|deps?|bump)\b", re.IGNORECASE)
MAX_FIX_FILES = 15
TRIVIAL_LINE = re.compile(r"^\s*(//.*|/\*.*|\*.*|\*/|[{}()\[\];,]*|import .*|export \{.*\} from .*)?\s*$")


@dataclass
class Bug:
    fix_sha: str
    fix_subject: str
    path: str
    #: Lines in the PR's merge commit that the fix changed.
    lines: list[int] = field(default_factory=list)


def is_fix(subject: str) -> bool:
    return bool(FIX_SUBJECT.search(subject)) and not NOT_A_DEFECT.search(subject)


def bugs_introduced(git: Git, merge_sha: str, pr_shas: set[str], until: str,
                    history: list[Commit] | None = None, blame_cache: dict | None = None) -> list[Bug]:
    """Bugs in files the PR changed, fixed by first-parent commits after the merge up to `until`.

    `history` (from `Git.first_parent_history`, covering the merge) and `blame_cache`
    are shared across PRs; without them each PR reads and blames on its own.
    """
    pr_files = set(git.changed_files(git.parent(merge_sha), merge_sha))
    origins = pr_shas | {merge_sha} | git.branch_commits(merge_sha)
    bugs: list[Bug] = []
    for fix in _after(history, merge_sha) if history is not None else git.first_parent_history(merge_sha, until):
        if not is_fix(fix.subject) or len(fix.files) > MAX_FIX_FILES:
            continue
        for path in sorted(pr_files.intersection(fix.files)):
            lines = _introduced_lines(git, fix, path, origins, merge_sha, blame_cache)
            if lines:
                bugs.append(Bug(fix.sha, fix.subject, path, sorted(set(lines))))
    return bugs


def _after(history: list[Commit], merge_sha: str) -> list[Commit]:
    for index, commit in enumerate(history):
        if commit.sha == merge_sha:
            return history[index + 1:]
    raise ValueError(f"{merge_sha} is not a first-parent commit in the shared history")


def _introduced_lines(git: Git, fix: Commit, path: str, origins: set[str], merge_sha: str,
                      cache: dict | None = None) -> list[int]:
    lines: list[int] = []
    for origin, line, text, origin_path in _fix_blame(git, fix, path, cache):
        # A line blamed to another file came there by a copy git followed; it is not locatable here.
        if origin in origins and origin_path == path and not TRIVIAL_LINE.match(text):
            lines.append(line if origin == merge_sha else _to_merge(git, origin, merge_sha, path, line))
    return lines


def _fix_blame(git: Git, fix: Commit, path: str, cache: dict | None) -> list[tuple[str, int, str, str]]:
    """Blame of every line the fix removed or replaced in `path`, just before the fix.

    It does not depend on the PR being labelled, so one run is shared by every
    PR through `cache`; all hunks go to a single git process.
    """
    key = (fix.sha, path)
    if cache is not None and key in cache:
        return cache[key]
    ranges = [(h.old_start, h.old_end) for h in git.hunks(fix.parent, fix.sha, path) if h.old_count > 0]
    rows = git.blame_ranges(fix.parent, path, ranges)
    if cache is not None:
        cache[key] = rows
    return rows


def _to_merge(git: Git, origin: str, merge_sha: str, path: str, line: int) -> int:
    """A line in one of the PR's own commits, located in the merge commit."""
    mapped, _ = map_line(git.hunks(origin, merge_sha, path), line)
    return mapped
