"""Git plumbing for the arena, run against a local full-history clone."""
from __future__ import annotations

import subprocess
from pathlib import Path

from dataclasses import dataclass

from arena.diffmap import Hunk, parse_hunks


@dataclass(frozen=True)
class Commit:
    sha: str
    parent: str
    subject: str
    files: tuple[str, ...]


class Git:
    def __init__(self, path: str | Path):
        self.path = str(path)

    def run(self, *args: str, check: bool = True) -> str:
        result = subprocess.run(
            ["git", "-C", self.path, *args], capture_output=True, text=True, check=False,
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    def hunks(self, old: str, new: str, path: str) -> list[Hunk]:
        """`git diff -U0` hunks for one file between two commits (histories may differ)."""
        return parse_hunks(self.run("diff", "-U0", "--no-color", old, new, "--", path))

    def changed_files(self, old: str, new: str) -> list[str]:
        return [f for f in self.run("diff", "--name-only", "--no-renames", old, new).splitlines() if f]

    def has_commit(self, sha: str) -> bool:
        return subprocess.run(
            ["git", "-C", self.path, "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True, check=False,
        ).returncode == 0

    def exists(self, sha: str, path: str) -> bool:
        return subprocess.run(
            ["git", "-C", self.path, "cat-file", "-e", f"{sha}:{path}"], capture_output=True, check=False,
        ).returncode == 0

    def parent(self, sha: str) -> str:
        return self.run("rev-parse", f"{sha}^").strip()

    def first_parent_history(self, since: str, until: str) -> list[Commit]:
        """First-parent commits after `since` up to `until`, oldest first, with changed files.

        One `git log` pass, so SZZ over many PRs does not spawn a process per commit.
        """
        out = self.run("log", "--first-parent", "--reverse", "--no-renames", "--name-only",
                       "--format=%x01%H%x00%P%x00%s", f"{since}..{until}")
        commits: list[Commit] = []
        for record in out.split("\x01")[1:]:
            header, _, body = record.partition("\n")
            sha, parents, subject = header.split("\x00", 2)
            files = tuple(f for f in body.splitlines() if f)
            commits.append(Commit(sha, parents.split(" ")[0] if parents else "", subject, files))
        return commits

    def branch_commits(self, merge_sha: str) -> set[str]:
        """Commits a merge brought in from its second parent: a PR's own commits. Empty for squash merges."""
        parents = self.run("rev-list", "--parents", "-n", "1", merge_sha).split()[1:]
        if len(parents) < 2:
            return set()
        return set(self.run("rev-list", f"{parents[0]}..{parents[1]}").split())

    def blame(self, sha: str, path: str, first: int, last: int) -> list[tuple[str, int, str, str]]:
        """(origin commit, line in that commit, text, path in that commit) for lines first..last of path at sha."""
        return self.blame_ranges(sha, path, [(first, last)])

    def blame_ranges(self, sha: str, path: str, ranges: list[tuple[int, int]]) -> list[tuple[str, int, str, str]]:
        """Like blame, for several line ranges in one git process.

        -M follows lines moved within the file; -C -C follows lines moved or copied from
        other files (any file, when the commit created this one), so a refactor that
        relocates code is not taken as its author.
        """
        if not ranges:
            return []
        args = [arg for first, last in ranges for arg in ("-L", f"{first},{last}")]
        out = self.run("blame", "-w", "-M", "-C", "-C", "--porcelain", *args, sha, "--", path)
        rows: list[tuple[str, int, str, str]] = []
        origin: tuple[str, int] | None = None
        origin_path = path
        for line in out.splitlines():
            if line.startswith("\t"):
                if origin:
                    rows.append((origin[0], origin[1], line[1:], origin_path))
                continue
            if line.startswith("filename "):
                origin_path = line[len("filename "):]
                continue
            parts = line.split(" ")
            if len(parts) >= 3 and len(parts[0]) == 40 and parts[1].isdigit():
                origin = (parts[0], int(parts[1]))
        return rows
