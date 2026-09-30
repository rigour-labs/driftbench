"""Rigour run on a PR as it merged: `rigour review --json` over the merge diff.

The CLI is pinned: `RIGOUR_CLI` points at a built `cli.js` (recorded by
path and version), otherwise `npx @rigour-labs/cli@<version>`. Runs happen
in a throwaway worktree, with `HOME` pointed at an arena directory so
local Rigour state never mixes with the developer's own.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from arena.gitrepo import Git
from arena.repos import CACHE
from arena.score import Finding

TIMEOUT_S = 600


@dataclass(frozen=True)
class RigourConfig:
    #: Name shown on the scoreboard, e.g. "rigour-semantic".
    name: str
    #: rigour.yml to use, or None for Rigour's defaults.
    config: Path | None = None
    #: Extra review flags, e.g. ["--deep", "--pro"].
    flags: tuple[str, ...] = ()


@dataclass
class RigourRun:
    findings: list[Finding]
    seconds: float
    status: str
    error: str = ""


def review(git: Git, merge_sha: str, cfg: RigourConfig) -> RigourRun:
    workdir = Path(tempfile.mkdtemp(prefix="arena-rigour-"))
    tree = workdir / "tree"
    try:
        git.run("worktree", "add", "-q", "--detach", str(tree), merge_sha)
        diff = workdir / "pr.diff"
        diff.write_text(git.run("diff", "--no-color", git.parent(merge_sha), merge_sha))
        return _run(tree, diff, cfg)
    finally:
        git.run("worktree", "remove", "--force", str(tree), check=False)
        shutil.rmtree(workdir, ignore_errors=True)


def _run(tree: Path, diff: Path, cfg: RigourConfig) -> RigourRun:
    args = [*_cli(), "review", "--json", "--diff", str(diff), *cfg.flags]
    if cfg.config:
        args += ["-c", str(cfg.config.resolve())]
    env = {**os.environ, "HOME": os.environ.get("ARENA_RIGOUR_HOME", str(CACHE / "home"))}
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = subprocess.run(args, cwd=tree, env=env, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
    seconds = time.monotonic() - started
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        return RigourRun([], seconds, "error", (result.stderr or result.stdout)[-500:])
    status = report.get("status")
    if status not in ("PASS", "FAIL"):
        # A run that did not review (bad config, deep analysis that could not start)
        # is an error, never "no findings".
        return RigourRun([], seconds, "error", json.dumps(report)[:500])
    findings = [_finding(f) for f in report.get("failures", []) if f.get("file")]
    return RigourRun(findings, seconds, status)


def _finding(failure: dict) -> Finding:
    """A located failure, keeping what a judge needs to tell whether it describes a bug."""
    line = int(failure["line"])
    rule = failure.get("id") or failure.get("gate", "")
    message = f"[{failure.get('severity', '')}/{failure.get('gate', '')}] {failure.get('message', '')}"
    return Finding(failure["file"], line, id=f"{rule}@{failure['file']}:{line}", message=message)


def _cli() -> list[str]:
    cli = os.environ.get("RIGOUR_CLI")
    if cli:
        return ["node", cli]
    return ["npx", "--yes", f"@rigour-labs/cli@{os.environ.get('RIGOUR_VERSION', 'latest')}"]
