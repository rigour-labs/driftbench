"""Fix the labels at run start by content (docs/LABELLING.md, "Order").

`bench run` records the repository's HEAD commit and the git blob hash of
every file under labels/. The report uses results by class only if each
repository's label and sample files are still byte-identical to what the run
recorded, and committed. The draft release notes carry the same record, so
the public copy fixes it too. This is a content check, not a timestamp: dates
in git can be set by hand, file contents can't change without changing the
hash.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from bench.repos import slug_of


def git_out(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def label_fingerprint(labels_dir: Path) -> dict:
    """{"commit": HEAD SHA or "", "files": {file name: blob hash}} for every labels/*.yaml."""
    files = sorted(labels_dir.glob("*.yaml")) if labels_dir.is_dir() else []
    return {
        "commit": git_out(labels_dir if labels_dir.is_dir() else Path("."), "rev-parse", "HEAD"),
        "files": {path.name: git_out(path.parent, "hash-object", path.name) for path in files},
    }


def repo_label_files(repo: str) -> tuple[str, str]:
    slug = slug_of(repo)
    return f"{slug}.yaml", f"{slug}.sample.yaml"


def labels_unchanged(labels_dir: Path, recorded: dict | None, repo: str) -> tuple[bool, str]:
    """(ok, reason): this repo's label and sample files are committed and match the run's record."""
    if not recorded:
        return False, "the run recorded no label fingerprint"
    for name in repo_label_files(repo):
        path = labels_dir / name
        if not path.exists():
            return False, f"{name} is missing"
        if git_out(labels_dir, "status", "--porcelain", "--", name):
            return False, f"{name} has uncommitted changes"
        current = git_out(labels_dir, "hash-object", name)
        if recorded.get("files", {}).get(name) != current:
            return False, f"{name} differs from the version fixed when the run started"
    return True, ""
