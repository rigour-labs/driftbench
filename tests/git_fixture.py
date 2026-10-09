"""A tiny local origin repository: base on main, then two pull request heads."""
from __future__ import annotations

import subprocess
from pathlib import Path

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@x.test", "-c", "init.defaultBranch=main"]


def git(repo: Path, *args: str) -> str:
    return subprocess.run([*GIT, "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def commit_file(repo: Path, name: str, text: str, message: str) -> str:
    (repo / name).write_text(text, encoding="utf-8")
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def make_origin(root: Path) -> dict[str, str]:
    origin = root / "origin"
    origin.mkdir()
    subprocess.run([*GIT, "init", "-q", str(origin)], check=True)
    base = commit_file(origin, "app.py", "def f():\n    return 1\n", "base")
    git(origin, "checkout", "-q", "-b", "feature")
    head1 = commit_file(origin, "app.py", "def f():\n    return 2\n\n\ndef g():\n    return 3\n", "round 1")
    head2 = commit_file(origin, "app.py", "def f():\n    return 2\n\n\ndef g():\n    return 4\n", "round 2")
    git(origin, "checkout", "-q", "main")
    return {"path": str(origin), "base": base, "head1": head1, "head2": head2}
