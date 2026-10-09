"""One local clone per repository, checked out at each reviewed head in turn.

Every checkout is forced and followed by `git clean -ffdx`. A tool never
runs in this clone: each review gets an isolated copy with no refs (see
bench/harness/sandbox.py), deleted after the run. Clones are blobless by default: file contents are fetched
only for the commits that are checked out.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def run_or_raise(args: list[str], context: str) -> None:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise GitError(f"{context}: {result.stderr.strip()[:300]}")


class RepoCheckout:
    def __init__(self, url: str, path: Path, blobless: bool = True):
        self.url = url
        self.path = path
        self.blobless = blobless

    def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True, text=True, check=False)
        if check and result.returncode != 0:
            raise GitError(f"git {' '.join(args)}: {result.stderr.strip()[:300]}")
        return result

    def ensure_clone(self) -> None:
        if (self.path / ".git").exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        args = ["git", "clone", "--no-checkout", "--quiet", *(["--filter=blob:none"] if self.blobless else [])]
        run_or_raise([*args, self.url, str(self.path)], f"git clone {self.url}")

    def has_commit(self, sha: str) -> bool:
        return self.git("cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0

    def ensure_commit(self, sha: str, pr_number: int) -> bool:
        """Fetch `sha` if it's missing: by SHA, then through the PR's head ref."""
        for refspec in (sha, f"pull/{pr_number}/head"):
            if self.has_commit(sha):
                return True
            self.git("fetch", "--quiet", "origin", refspec, check=False)
        return self.has_commit(sha)

    def checkout(self, sha: str) -> None:
        self.git("checkout", "--quiet", "--force", "--detach", sha)
        self.git("clean", "-ffdxq")

    def merge_base(self, base_sha: str, head_sha: str) -> str:
        return self.git("merge-base", base_sha, head_sha).stdout.strip()

    def isolated_copy(self, head_sha: str, dest: Path) -> None:
        """A repo at `dest` with this clone's objects, no refs, and `head_sha` checked out (detached).

        Call after `checkout(head_sha)`, which fetches the head's file contents
        into this clone; the copy reads them through git alternates.
        """
        run_or_raise(["git", "init", "--quiet", str(dest)], f"isolated copy at {head_sha[:12]}")
        objects = (self.path / ".git" / "objects").resolve()
        (dest / ".git" / "objects" / "info" / "alternates").write_text(f"{objects}\n", encoding="utf-8")
        run_or_raise(["git", "-C", str(dest), "-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach",
                      head_sha], f"isolated copy at {head_sha[:12]}")

    def diff(self, base_sha: str, head_sha: str) -> str:
        return self.git("diff", "--no-color", "--no-ext-diff", base_sha, head_sha).stdout
