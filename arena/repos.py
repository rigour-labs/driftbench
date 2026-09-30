"""Full-history clones in a local cache (SZZ needs history; blame needs blobs)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from arena.gitrepo import Git

CACHE = Path(os.environ.get("ARENA_CACHE", Path.home() / ".cache" / "driftbench" / "arena"))


def clone(repo: str) -> Git:
    """A full clone of `owner/name`, fetched up to date."""
    target = CACHE / "repos" / repo.replace("/", "__")
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", f"https://github.com/{repo}.git", str(target)], check=True)
    else:
        subprocess.run(["git", "-C", str(target), "fetch", "-q", "origin"], check=True)
    return Git(target)


def ensure_commits(git: Git, pr_number: int, shas: list[str]) -> None:
    """Fetch the PR's head ref when any of its commits (e.g. a reviewed commit) is missing."""
    missing = [sha for sha in shas if not git.has_commit(sha)]
    if missing:
        git.run("fetch", "-q", "origin", f"pull/{pr_number}/head")

