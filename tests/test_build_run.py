"""A whole build task, end to end on a synthetic Go repository: a fake agent, fake Rigour hooks, real git and go."""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from bench.buildtrack import arms, run
from bench.buildtrack.reference import discrimination
from bench.buildtrack.toolchains import Toolchain
from bench.buildtrack.workspace import FIXED, parent
from bench.harness.budget import Budget
from bench.harness.gitrepo import RepoCheckout

pytestmark = pytest.mark.skipif(shutil.which("go") is None, reason="go is not installed")
OFFLINE_GO = Toolchain(prepare=("true",), env={"GOFLAGS": "-mod=mod", "GOTOOLCHAIN": "local"},
                       commands=("Bash(go test:*)",), test=("go", "test", "-count=1", "-json"))
ADD = "package a\n\nfunc Add(x, y int) int { return x + y }\n"
TEST = 'package a\n\nimport "testing"\n\nfunc TestAdd(t *testing.T) { if Add(1, 2) != 3 { t.Fatal() } }\n'


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env={"PATH": os.environ["PATH"], "HOME": str(repo), **FIXED}).stdout


def commit(repo: Path, files: dict[str, str], message: str) -> str:
    for path, content in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(content)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """A repo whose pull request adds Add and its test; a fake `claude` that writes Add; fake Rigour hooks."""
    src = tmp_path / "src"
    subprocess.run(["git", "init", "-q", "-b", "main", str(src)], check=True)
    base = commit(src, {"go.mod": "module example.com/a\n\ngo 1.21\n", "a/doc.go": "package a\n"}, "base")
    git(src, "checkout", "-q", "-b", "pr")
    head = commit(src, {"a/a.go": ADD, "a/a_test.go": TEST}, "add Add")
    checkout = RepoCheckout(str(src), tmp_path / "clone", blobless=False)
    checkout.ensure_clone()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude = bin_dir / "claude"
    result = {"type": "result", "subtype": "success", "total_cost_usd": 0.3, "num_turns": 4,
              "modelUsage": {"m": {"inputTokens": 10, "outputTokens": 5}}}
    claude.write_text(f"#!/bin/sh\nmkdir -p a\nprintf '%s' '{ADD}' > a/a.go\necho '{json.dumps(result)}'\n")
    claude.chmod(claude.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    def fake_setup(repo: Path, env: dict, version: str) -> dict:
        home = Path(env["HOME"])
        (home / ".claude").mkdir(parents=True, exist_ok=True)
        hook = [{"type": "command", "command": "mkdir -p .rigour; echo '{\"type\":\"stop_review\",\"blocked\":false}' "
                                               ">> .rigour/events.jsonl; exit 0"}]
        (home / ".claude" / "settings.json").write_text(json.dumps({"hooks": {
            "Stop": [{"hooks": hook}], "PostToolUse": [{"matcher": "Write", "hooks": hook}]}}))
        (repo / ".git" / "info").mkdir(parents=True, exist_ok=True)
        (repo / ".git" / "info" / "exclude").write_text(".rigour/\n")
        return arms.installed(home, repo)
    monkeypatch.setattr(arms, "setup_rigour", fake_setup)
    monkeypatch.setattr("bench.buildtrack.reference.setup_rigour", fake_setup)
    task = {"pr": 1, "base_sha": base, "first_head": head, "merged_head": head, "test_files": ["a/a_test.go"],
            "points": ["1-inline-9-0"]}
    config = run.BuildConfig(repo="o/r", out=tmp_path / "out", scratch=tmp_path / "scratch", npm_cache=tmp_path / "npm",
                             model="anthropic/m-1", provider_env={}, toolchain=OFFLINE_GO, max_turns=5, task_bound=1.0,
                             timeout_s=60, seed=1, lessons=None)
    return checkout, task, config


def test_a_task_runs_both_arms_tests_their_work_and_checks_the_merged_change(world):
    checkout, task, config = world
    start = parent(checkout, task)
    env = run.base_env({"PATH": os.environ["PATH"], "HOME": str(config.scratch)}, config)
    check = discrimination(checkout, task, start, config.toolchain, config.scratch / "ref", env)
    assert check["parent"]["outcome"] == "no-build" and check["merged"]["outcome"] == "pass" and check["discriminates"]
    budget = Budget(5.0, {f"build-{arm}": 1.0 for arm in arms.ARMS})
    record = run.run_task(checkout, task, start, arms.prompt_for("Add an Add function."), config, budget)
    for arm in arms.ARMS:
        assert record["arms"][arm]["tests"]["outcome"] == "pass" and "a/a.go" in record["arms"][arm]["agent"]["paid_output"]["diff"]
    assert record["arms"]["rigour"]["setup"]["hooks"] and "setup" not in record["arms"]["alone"]
    assert record["arms"]["rigour"]["rigour"]["events"]["by_type"] == {}             # the fake agent fires no hooks
    assert record["reference"]["false_blocks"]["false_blocks"] == 0 and budget.spent == 0.6
    assert sorted(record["arm_order"]) == sorted(arms.ARMS)


def test_a_dry_run_runs_everything_but_the_agent(world):
    checkout, task, config = world
    dry = run.BuildConfig(**{**config.__dict__, "dry_run": True})
    budget = Budget(5.0, {f"build-{arm}": 1.0 for arm in arms.ARMS})
    record = run.run_task(checkout, task, parent(checkout, task), "p", dry, budget)
    assert all(record["arms"][arm]["agent"]["dry_run"] for arm in arms.ARMS) and budget.spent == 0
    assert record["arms"]["alone"]["tests"]["outcome"] == "no-build"                  # nothing written: tests can't build
