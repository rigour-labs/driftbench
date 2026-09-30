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
    """Fetch commits a comment was made on: the PR's head ref first, then any commit that was
    force-pushed off the branch, by SHA (GitHub still serves it). Unfetchable commits stay missing;
    callers drop what cannot be located."""
    missing = {sha for sha in shas if not git.has_commit(sha)}
    if not missing:
        return
    git.run("fetch", "-q", "origin", f"pull/{pr_number}/head", check=False)
    for sha in sorted(sha for sha in missing if not git.has_commit(sha)):
        git.run("fetch", "-q", "origin", sha, check=False)

