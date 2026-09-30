"""Git plumbing for the arena, run against a local full-history clone."""
from __future__ import annotations

import subprocess
from pathlib import Path

from arena.diffmap import Hunk, parse_hunks


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

    def commits_after(self, sha: str, until: str) -> list[tuple[str, str]]:
        """(sha, subject) of first-parent commits after `sha` up to `until`, oldest first."""
        out = self.run("log", "--first-parent", "--reverse", "--format=%H%x00%s", f"{sha}..{until}")
        return [tuple(line.split("\x00", 1)) for line in out.splitlines() if line]  # type: ignore[misc]

    def blame(self, sha: str, path: str, first: int, last: int) -> list[tuple[str, int, str]]:
        """(origin commit, line number in that commit, text) for lines first..last of path at sha."""
        out = self.run("blame", "-w", "-M", "--porcelain", "-L", f"{first},{last}", sha, "--", path)
        rows: list[tuple[str, int, str]] = []
        origin: tuple[str, int] | None = None
        for line in out.splitlines():
            if line.startswith("\t"):
                if origin:
                    rows.append((origin[0], origin[1], line[1:]))
                continue
            parts = line.split(" ")
            if len(parts) >= 3 and len(parts[0]) == 40 and parts[1].isdigit():
                origin = (parts[0], int(parts[1]))
        return rows
