"""A task's reference checks, before any agent spends (docs/BUILD_TRACK.md).

- Do the hidden tests discriminate? They must fail or not build at the
  parent (with the tests added) and pass on the merged head.
- False blocks: the real merged change, written uncommitted into a snapshot of
  its own base (where its final head forked) with Rigour set up by default,
  must pass Rigour's hooks. Its own base, not the task's parent: a branch
  rebased onto newer main would otherwise carry everything main gained.
Dependencies are fetched with the network before anything runs offline.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from bench.buildtrack import hooks
from bench.buildtrack.arms import ArmError, setup_rigour
from bench.buildtrack.evaluate import files_at, run_tests, write_files
from bench.buildtrack.toolchains import Toolchain
from bench.buildtrack.workspace import snapshot
from bench.harness.gitrepo import GitError, RepoCheckout

PREPARE_TIMEOUT_S = 1800


def prepare_deps(repo: Path, toolchain: Toolchain, env: dict[str, str]) -> None:
    """Fill the shared dependency cache for this tree, with the network (the agent later runs without it)."""
    result = subprocess.run(list(toolchain.prepare), cwd=repo, env=env, capture_output=True, text=True,
                            timeout=PREPARE_TIMEOUT_S, check=False)
    if result.returncode != 0:
        raise ArmError(f"{' '.join(toolchain.prepare)} failed: {(result.stderr or result.stdout).strip()[:300]}")


def tests_at(checkout: RepoCheckout, sha: str, task: dict, toolchain: Toolchain, dest: Path,
             env: dict[str, str]) -> dict:
    """The hidden tests' outcome on `sha`'s tree with the merged head's test files written in."""
    repo = snapshot(checkout, sha, dest)
    prepare_deps(repo, toolchain, env)
    write_files(repo, files_at(checkout, task["merged_head"], task["test_files"]))
    return run_tests(toolchain, task["test_files"], repo, env)


def discrimination(checkout: RepoCheckout, task: dict, parent: str, toolchain: Toolchain, scratch: Path,
                   env: dict[str, str]) -> dict:
    at_parent = tests_at(checkout, parent, task, toolchain, scratch / "parent", env)
    at_merged = tests_at(checkout, task["merged_head"], task, toolchain, scratch / "merged", env)
    return {"parent": at_parent, "merged": at_merged,
            "discriminates": at_parent["outcome"] in ("fail", "no-build") and at_merged["outcome"] == "pass"}


def change_base(checkout: RepoCheckout, base_sha: str, merged_head: str) -> str:
    """Where the merged head forked from its base: the pull request's own change is base..merged_head, without
    anything main gained when the branch was rebased or updated."""
    return checkout.merge_base(base_sha, merged_head)


def merged_files(checkout: RepoCheckout, base: str, merged_head: str) -> dict[str, str]:
    """Every file the pull request changed, as merged (deleted files left out)."""
    names = checkout.git("diff", "--name-only", "--diff-filter=d", base, merged_head).stdout.split()
    return files_at(checkout, merged_head, names)


def false_blocks(checkout: RepoCheckout, task: dict, home: Path, dest: Path, env: dict[str, str],
                 version: str) -> dict:
    """The approved change, written uncommitted into its own base's snapshot with Rigour set up by default:
    every hook block on it is a false block."""
    try:
        base = change_base(checkout, task["base_sha"], task["merged_head"])
        repo = snapshot(checkout, base, dest)
        setup = setup_rigour(repo, env, version)
        found = hooks.false_blocks(repo, home, env, merged_files(checkout, base, task["merged_head"]))
    except (ArmError, GitError) as exc:
        raise ArmError(f"false-block check: {exc}") from exc
    return {**found, "base": base, "setup": setup}
