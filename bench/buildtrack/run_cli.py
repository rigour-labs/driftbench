"""`bench build run`: the build track's paid run (docs/BUILD_TRACK.md). Needs --max-usd.

Writes `<out>/run.json` first (versions, model, bounds, the task file's hash,
the prompt, the hook check), then `<out>/tasks/<pr>.json` per task and
`<out>/passed-over.json`. The hook check must pass before any agent runs.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bench.adapters.claude_cli import CLAUDE_CODE_VERSION, check_model_id, provider_env, require_claude_cli
from bench.buildtrack import arms
from bench.buildtrack.arms import ArmError
from bench.buildtrack.reference import discrimination
from bench.buildtrack.run import BuildConfig, base_env, passed_over, run_hookcheck, run_task, write
from bench.buildtrack.statement import statement
from bench.buildtrack.toolchains import ToolchainError, toolchain
from bench.buildtrack.workspace import parent as parent_of
from bench.collect.github import GitHubClient, GitHubError
from bench.harness.budget import Budget
from bench.harness.cli import write_usage
from bench.harness.gitrepo import GitError, RepoCheckout
from bench.harness.types import AdapterError
from bench.labels.openrouter import OpenRouterError, key_usage
from bench.repos import slug_of


def add_run_action(actions: argparse._SubParsersAction, root: Path) -> None:
    run = actions.add_parser("run", help="both arms on each task, in the task file's order (paid)")
    run.add_argument("--tasks", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--provider", default="openrouter")
    run.add_argument("--max-usd", type=float, default=0.0, help="hard cap for the whole run (paid runs)")
    run.add_argument("--task-bound", type=float, default=0.0, help="each arm's dollar bound per task (paid runs)")
    run.add_argument("--dry-run", action="store_true", help="everything but the agent: free, proves the pipeline")
    run.add_argument("--max-turns", type=int, default=60)
    run.add_argument("--timeout", type=int, default=1800, help="seconds per arm per task")
    run.add_argument("--seed", type=int, default=2026, help="each task's arm order")
    run.add_argument("--lessons", type=Path, help="the learning run's output for these pull requests")
    run.add_argument("--repos-dir", type=Path, default=root / "work" / "repos")
    run.add_argument("--scratch", type=Path, default=root / "work" / "scratch")
    run.add_argument("--cache", type=Path, default=root / "work" / "cache", help="API cache (never published)")
    run.set_defaults(handler=cmd_run)


def manifest(args: argparse.Namespace, drawn: dict, hookcheck: dict) -> dict:
    return {"run_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "repo": drawn["repo"],
            "tasks_file": str(args.tasks), "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
            "count": drawn["count"], "model": args.model, "provider": args.provider, "claude_code": CLAUDE_CODE_VERSION,
            "rigour": arms.RIGOUR_VERSION, "max_usd": args.max_usd, "task_bound": args.task_bound,
            "max_turns": args.max_turns, "timeout_s": args.timeout, "seed": args.seed, "dry_run": args.dry_run,
            "prompt": arms.prompt_for("<statement>"), "tools": {"allowed": list(arms.AGENT_TOOLS),
                                                                "denied": list(arms.AGENT_DENIED)},
            "hookcheck": {k: hookcheck[k] for k in ("edit_blocked", "stop_blocked", "ok")}}


def usage(args: argparse.Namespace) -> dict | None:
    """The OpenRouter key's cumulative usage now (a paid run through OpenRouter), for the billed figure."""
    if args.dry_run or args.provider != "openrouter":
        return None
    reading = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "usage_usd": None}
    try:
        reading["usage_usd"] = key_usage()
    except OpenRouterError as exc:  # the billed figure becomes "unavailable"; the run goes on
        print(f"warning: OpenRouter usage unavailable: {exc}", file=sys.stderr)
        reading["error"] = str(exc)
    return reading


def setup(args: argparse.Namespace) -> tuple[dict, BuildConfig, RepoCheckout]:
    if not args.dry_run and (args.max_usd <= 0 or args.task_bound <= 0):
        raise ValueError("a build run is paid: --max-usd and --task-bound must be above zero (or --dry-run)")
    check_model_id(args.model, args.provider)
    drawn = yaml.safe_load(args.tasks.read_text(encoding="utf-8"))
    chain = toolchain(drawn["repo"])
    env = {} if args.dry_run else provider_env({}, args.provider)
    if not args.dry_run:
        require_claude_cli({**env, "PATH": os.environ.get("PATH", "")})
    config = BuildConfig(repo=drawn["repo"], out=args.out, scratch=args.scratch, npm_cache=args.scratch / "npm",
                         model=args.model, provider_env=env, toolchain=chain, max_turns=args.max_turns,
                         task_bound=args.task_bound, timeout_s=args.timeout, seed=args.seed, lessons=args.lessons,
                         dry_run=args.dry_run)
    checkout = RepoCheckout(f"https://github.com/{drawn['repo']}.git", args.repos_dir / slug_of(drawn["repo"]),
                            blobless=False)
    checkout.ensure_clone()
    return drawn, config, checkout


def ready(task: dict, drawn: dict, config: BuildConfig, checkout: RepoCheckout, client: GitHubClient) -> dict:
    """{parent, check, text} for a task that can run, or {reason[, detail]} for one passed over."""
    found = statement(client, drawn["repo"], task["pr"])
    if found.get("sha256") != task["statement"]["sha256"]:
        return {"reason": "the statement no longer matches its hash"}
    parent = parent_of(checkout, task)
    check = discrimination(checkout, task, parent, config.toolchain, config.scratch / f"ref-{task['pr']}",
                           base_env({"PATH": os.environ.get("PATH", ""), "HOME": str(config.scratch)}, config))
    if not check["discriminates"]:
        return {"reason": "hidden tests do not discriminate", "detail": check}
    return {"parent": parent, "check": check, "text": found["text"]}


def run_tasks(drawn: dict, config: BuildConfig, checkout: RepoCheckout, client: GitHubClient, budget: Budget) -> list:
    """Tasks in order until `count` have run; a task that can't be prepared or fails is listed, and the run goes on."""
    done, over = 0, []
    for task in drawn["tasks"]:
        if done >= drawn["count"]:
            break
        room = sum(budget.per_head_bound(f"build-{arm}") for arm in arms.ARMS)
        if not config.dry_run and budget.spent + room > budget.max_usd:
            over.append(passed_over(task, f"the cap leaves less than both arms' bounds (${budget.spent:.2f} spent)"))
            break
        try:
            found = ready(task, drawn, config, checkout, client)
            if "reason" in found:
                over.append(passed_over(task, found["reason"], found.get("detail")))
                continue
            record = run_task(checkout, task, found["parent"], arms.prompt_for(found["text"]), config, budget)
        except (ArmError, GitError, GitHubError, OSError) as exc:
            print(f"build: #{task['pr']}: {exc}", file=sys.stderr)
            over.append(passed_over(task, f"error: {str(exc)[:200]}"))
            continue
        write(config.out / "tasks" / f"{task['pr']}.json", {**record, "discrimination": found["check"]})
        done += 1
    return over


def cmd_run(args: argparse.Namespace) -> int:
    try:
        drawn, config, checkout = setup(args)
        first = drawn["tasks"][0]
        hookcheck = run_hookcheck(checkout, parent_of(checkout, first), config)
        write(args.out / "run.json", manifest(args, drawn, hookcheck))
        if not hookcheck["ok"]:
            print("build: Rigour's hooks did not block planted, uncommitted work; the run stops (run.json has it)",
                  file=sys.stderr)
            return 1
        budget = Budget(args.max_usd, {f"build-{arm}": args.task_bound for arm in arms.ARMS})
        start = usage(args)
        over = run_tasks(drawn, config, checkout, GitHubClient(args.cache), budget)
        if start:
            write_usage(args.out / "openrouter-usage.json", start, usage(args))
    except (ValueError, OSError, ArmError, AdapterError, GitError, GitHubError, ToolchainError) as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 1
    write(args.out / "passed-over.json", {"passed_over": over, "budget": budget.as_record()})
    print(f"build: {len(list((args.out / 'tasks').glob('*.json')))} tasks run, {len(over)} passed over; "
          f"budget {budget.as_record()}")
    return 0
