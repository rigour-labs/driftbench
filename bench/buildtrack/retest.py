"""Re-run a finished build run's hidden tests on its recorded diffs, free (docs/BUILD_TRACK.md).

For each task record: the parent's snapshot, the agent's diff applied, the
pull request's test files written in, the test command run. The outcome now
keeps the first error lines, which the pilot's records did not. No agent and no
model: the diffs come from the records.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from bench.buildtrack.arms import ARMS
from bench.buildtrack.evaluate import files_at, run_tests, write_files
from bench.buildtrack.reference import prepare_deps
from bench.buildtrack.toolchains import Toolchain
from bench.buildtrack.workspace import snapshot
from bench.harness.gitrepo import GitError, RepoCheckout
from bench.harness.publish import PAID_OUTPUT


def apply(repo: Path, diff: str) -> str | None:
    """Apply the agent's diff to the snapshot; why it would not apply, or None."""
    if not diff.strip():
        return None
    result = subprocess.run(["git", "-C", str(repo), "apply", "--whitespace=nowarn", "-"], input=diff, text=True,
                            capture_output=True, check=False)
    return None if result.returncode == 0 else (result.stderr.strip() or "git apply failed")[:200]


def retest_arm(checkout: RepoCheckout, record: dict, task: dict, arm: str, toolchain: Toolchain, dest: Path,
               env: dict[str, str]) -> dict:
    repo = snapshot(checkout, record["parent"], dest)
    prepare_deps(repo, toolchain, env)
    diff = (record["arms"][arm]["agent"].get(PAID_OUTPUT) or {}).get("diff") or ""
    refused = apply(repo, diff)
    if refused:
        return {"outcome": "diff-not-applied", "errors": [refused]}
    write_files(repo, files_at(checkout, task["merged_head"], task["test_files"]))
    return run_tests(toolchain, task["test_files"], repo, env)


def retest(checkout: RepoCheckout, records: list[dict], tasks: dict[int, dict], toolchain: Toolchain, scratch: Path,
           env: dict[str, str]) -> dict:
    """{pr: {arm: outcome with errors}} for every task record."""
    out: dict = {}
    for record in records:
        task = tasks[record["pr"]]
        try:
            out[str(record["pr"])] = {arm: retest_arm(checkout, record, task, arm, toolchain,
                                                      scratch / f"{record['pr']}-{arm}", env) for arm in ARMS}
        except GitError as exc:
            print(f"retest: #{record['pr']}: {exc}", file=sys.stderr)
            out[str(record["pr"])] = {"error": str(exc)[:200]}
    return out
