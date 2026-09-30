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
from arena.gitrepo import Git

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


def bugs_introduced(git: Git, merge_sha: str, pr_shas: set[str], until: str) -> list[Bug]:
    """Bugs in files the PR changed, fixed by first-parent commits after the merge up to `until`."""
    pr_files = set(git.changed_files(git.parent(merge_sha), merge_sha))
    origins = pr_shas | {merge_sha}
    bugs: list[Bug] = []
    for fix_sha, subject in git.commits_after(merge_sha, until):
        if not is_fix(subject):
            continue
        fix_files = git.changed_files(git.parent(fix_sha), fix_sha)
        if len(fix_files) > MAX_FIX_FILES:
            continue
        for path in pr_files.intersection(fix_files):
            lines = _introduced_lines(git, fix_sha, path, origins, merge_sha)
            if lines:
                bugs.append(Bug(fix_sha, subject, path, sorted(set(lines))))
    return bugs


def _introduced_lines(git: Git, fix_sha: str, path: str, origins: set[str], merge_sha: str) -> list[int]:
    before = git.parent(fix_sha)
    lines: list[int] = []
    for hunk in git.hunks(before, fix_sha, path):
        if hunk.old_count == 0:
            continue
        for origin, line, text in git.blame(before, path, hunk.old_start, hunk.old_end):
            if origin in origins and not TRIVIAL_LINE.match(text):
                lines.append(line if origin == merge_sha else _to_merge(git, origin, merge_sha, path, line))
    return lines


def _to_merge(git: Git, origin: str, merge_sha: str, path: str, line: int) -> int:
    """A line in one of the PR's own commits, located in the merge commit."""
    mapped, _ = map_line(git.hunks(origin, merge_sha, path), line)
    return mapped
