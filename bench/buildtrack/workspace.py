"""The checkout an agent works in (docs/BUILD_TRACK.md, "Tasks" and "Leakage").

The task starts at the pull request's parent: the merge base of its first
head and its base commit. The agent gets a snapshot of that tree as a new
repository with one commit and no remote, no other ref and no history, so
nothing after the parent exists in it. The final diff is taken against that
one commit, so whatever the agent commits or leaves uncommitted is counted.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from bench.harness.gitrepo import GitError, RepoCheckout

START = "task start"
FIXED = {"GIT_AUTHOR_NAME": "driftbench", "GIT_AUTHOR_EMAIL": "bench@invalid", "GIT_COMMITTER_NAME": "driftbench",
         "GIT_COMMITTER_EMAIL": "bench@invalid", "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
         "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z"}


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False,
                            env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(repo), **(env or {})})
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args[:2])}: {result.stderr.strip()[:300]}")
    return result.stdout


def parent(checkout: RepoCheckout, task: dict) -> str:
    """The merge base of the task's first head and its base commit."""
    for sha in (task["base_sha"], task["first_head"]):
        if not checkout.ensure_commit(sha, task["pr"]):
            raise GitError(f"commit {sha[:12]} of #{task['pr']} is unavailable")
    return checkout.merge_base(task["base_sha"], task["first_head"])


def snapshot(checkout: RepoCheckout, sha: str, dest: Path) -> Path:
    """`sha`'s tree as a fresh one-commit repository at `dest`, with no remote and no history."""
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(checkout.path), "archive", "--format=tar", sha],
                             capture_output=True, check=False)
    if archive.returncode != 0:
        raise GitError(f"git archive {sha[:12]}: {archive.stderr.decode(errors='replace')[:300]}")
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    git(dest, "init", "-q", "-b", "main")
    git(dest, "add", "-A")
    git(dest, "commit", "-q", "--no-verify", "-m", START, env=FIXED)
    return dest


def start_commit(repo: Path) -> str:
    return git(repo, "rev-list", "--max-parents=0", "HEAD").strip()


def final_diff(repo: Path) -> str:
    """Everything the agent changed since the start, committed or not, new files included."""
    git(repo, "add", "-A")
    return git(repo, "diff", "--cached", "--no-color", start_commit(repo))
