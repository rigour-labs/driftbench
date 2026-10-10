"""A build run: tasks in their seeded order, both arms on each (docs/BUILD_TRACK.md).

For each task, until `count` have run: check the statement against its hash,
check the hidden tests discriminate (else pass it over), then run both arms
in a seeded order, each in a fresh HOME and a fresh snapshot of the parent,
then the hidden tests on each arm's result and the false-block check. Before
the first task, the hook check must pass. A task starts only if the cap
leaves room for both arms at the per-task bound. Each task's record is
written to `<out>/tasks/<pr>.json` as soon as it is done.
"""
from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

from bench.buildtrack import arms, hooks
from bench.buildtrack.agent import run_agent
from bench.buildtrack.evaluate import files_at, rigour_events, run_tests, write_files
from bench.buildtrack.reference import false_blocks, prepare_deps
from bench.buildtrack.toolchains import Toolchain
from bench.buildtrack.workspace import final_diff, snapshot
from bench.harness.budget import Budget
from bench.harness.publish import PAID_OUTPUT
from bench.harness.gitrepo import RepoCheckout
from bench.harness.sandbox import sandbox


@dataclasses.dataclass(frozen=True)
class BuildConfig:
    repo: str
    out: Path
    scratch: Path
    npm_cache: Path
    model: str
    provider_env: dict[str, str]   # the provider's variables only (bench/adapters/claude_cli.py)
    toolchain: Toolchain
    max_turns: int
    task_bound: float
    timeout_s: int
    seed: int
    lessons: Path | None           # the stores of the learning run, by pull request (docs/LEARNING.md)
    rigour_version: str = arms.RIGOUR_VERSION
    dry_run: bool = False          # everything but the agent: free, to prove the pipeline before a paid run


def base_env(env: dict[str, str], config: BuildConfig) -> dict[str, str]:
    caches = config.scratch / "caches"
    return {**env, "GOMODCACHE": str(caches / "gomod"), "GOCACHE": str(caches / "gobuild")}


def run_arm(arm: str, checkout: RepoCheckout, task: dict, parent: str, prompt: str, config: BuildConfig) -> dict:
    with sandbox(config.scratch, config.npm_cache) as box:
        env = base_env(box.env, config)
        repo = snapshot(checkout, parent, box.repo)
        prepare_deps(repo, config.toolchain, env)
        record: dict = {}
        if arm == "rigour":
            record["setup"] = arms.setup_rigour(repo, env, config.rigour_version)
        store = config.lessons / "stores" / f"{task['pr']}.json" if config.lessons and arm == "rigour" else None
        if config.dry_run:
            record["agent"] = {"dry_run": True, "cost_usd": 0.0, "diff": final_diff(repo)}
        else:
            command = arms.agent_command(arm, config.model, config.toolchain, config.max_turns, config.task_bound,
                                         prompt)
            agent_env = {**arms.agent_env(env, config.toolchain, store), **config.provider_env}
            record["agent"] = run_agent(command, repo, agent_env, config.timeout_s)
        # the agent's own work, kept in full like a paid answer (release tarball only; bench/harness/publish.py)
        record["agent"][PAID_OUTPUT] = {"diff": record["agent"].pop("diff")}
        if arm == "rigour":
            record["rigour"] = {"events": rigour_events(repo)}
        write_files(repo, files_at(checkout, task["merged_head"], task["test_files"]))
        record["tests"] = run_tests(config.toolchain, task["test_files"], repo, env)
        return record


def run_hookcheck(checkout: RepoCheckout, parent: str, config: BuildConfig) -> dict:
    with sandbox(config.scratch, config.npm_cache) as box:
        env = base_env(box.env, config)
        repo = snapshot(checkout, parent, box.repo)
        arms.setup_rigour(repo, env, config.rigour_version)
        return hooks.hookcheck(repo, box.home, env)


def run_false_blocks(checkout: RepoCheckout, task: dict, config: BuildConfig) -> dict:
    with sandbox(config.scratch, config.npm_cache) as box:
        return false_blocks(checkout, task, box.home, box.repo, base_env(box.env, config), config.rigour_version)


def charge(budget: Budget, arm: str, record: dict, bound: float) -> None:
    cost = record["agent"].get("cost_usd")
    budget.add_cost(f"build-{arm}", cost if isinstance(cost, (int, float)) else bound)


def run_task(checkout: RepoCheckout, task: dict, parent: str, prompt: str, config: BuildConfig,
             budget: Budget) -> dict:
    order = list(arms.ARMS)
    random.Random(f"{config.seed}:{task['pr']}").shuffle(order)
    record = {"repo": config.repo, "pr": task["pr"], "parent": parent, "points": task["points"], "arm_order": order,
              "arms": {}}
    for arm in order:
        record["arms"][arm] = run_arm(arm, checkout, task, parent, prompt, config)
        charge(budget, arm, record["arms"][arm], config.task_bound)
    record["reference"] = {"false_blocks": run_false_blocks(checkout, task, config)}
    return record


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")


def passed_over(task: dict, reason: str, detail: dict | None = None) -> dict:
    return {"pr": task["pr"], "reason": reason, **({"detail": detail} if detail else {})}
