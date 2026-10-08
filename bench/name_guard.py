"""Pre-push and pre-release guard: refuse to publish text that names a blocked party.

The blocked names live outside this repository (one per line in the file named
by `BENCH_BLOCKED_NAMES`, default `~/.config/driftbench/blocked-names.txt`) so
the list itself is never published. A missing or empty list fails closed.
Names match as whole words, case-insensitively, so a short name doesn't match
inside an unrelated longer word.
"""
from __future__ import annotations

import dataclasses
import os
import re
import subprocess
from pathlib import Path

DEFAULT_LIST = Path("~/.config/driftbench/blocked-names.txt")


class GuardConfigError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class GuardHit:
    where: str
    line: int
    text: str


def blocked_names_path() -> Path:
    return Path(os.environ.get("BENCH_BLOCKED_NAMES", str(DEFAULT_LIST))).expanduser()


def load_pattern(path: Path) -> re.Pattern[str]:
    """One alternation of every listed name, as whole words, ignoring case."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GuardConfigError(f"blocked-name list not readable at {path}: {exc}") from exc
    stripped = (line.strip() for line in lines)
    names = [name for name in stripped if name and not name.startswith("#")]
    if not names:
        raise GuardConfigError(f"blocked-name list at {path} is empty")
    alternation = "|".join(re.escape(name).replace(r"\ ", r"\s+") for name in names)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alternation})(?![A-Za-z0-9])", re.IGNORECASE)


def scan_text(pattern: re.Pattern[str], where: str, text: str) -> list[GuardHit]:
    return [
        GuardHit(where, number, line.strip()[:160])
        for number, line in enumerate(text.splitlines(), start=1)
        if pattern.search(line)
    ]


def scan_paths(pattern: re.Pattern[str], paths: list[Path]) -> list[GuardHit]:
    """Scan files, and every file under directories; binary files are skipped."""
    hits: list[GuardHit] = []
    for path in paths:
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            text = read_text(file)
            if text is not None:
                hits += scan_text(pattern, str(file), text)
    return hits


def tracked_files(repo: Path) -> list[Path]:
    out = run_git(repo, "ls-files", "-z")
    return [repo / name for name in out.split("\0") if name]


def commit_messages(repo: Path, rev_range: str) -> str:
    """Messages and author lines of the commits about to be pushed."""
    return run_git(repo, "log", "--format=%an <%ae>%n%B", rev_range)


def read_text(path: Path) -> str | None:
    """File text, or None for binary files; an unreadable file stops the guard."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise GuardConfigError(f"cannot read {path}: {exc}") from exc
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise GuardConfigError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout
