"""`bench build judge` (paid; needs --max-usd) and `bench build report` (free) over a build run.

A run directory holds one record per task, `tasks/<pr>.json`, written by the
run (both arms' agent records with their diffs, hidden-test outcomes, Rigour's
events, the reference check). The judge writes `judgments.json`, saved after
every point so a stop loses nothing and a rerun resumes; the report writes
`summary.json`, numbers only.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

from bench.buildtrack.arms import ARMS
from bench.buildtrack.judge import ENTRANT, MAX_TOKENS, judge_point, prompt_sha256
from bench.buildtrack.report import summary
from bench.collect.github import GitHubClient, GitHubError
from bench.harness.budget import Budget
from bench.labels.openrouter import chat
from bench.labels.prelabel_cli import CLAUDE_MARKERS
from bench.points.points_file import read_points
from bench.points.texts import TextSource, point_text
from bench.repos import slug_of

JUDGMENTS = "judgments.json"
SUMMARY = "summary.json"


def add_judge_actions(actions: argparse._SubParsersAction, root: Path) -> None:
    judge = actions.add_parser("judge", help="did each arm's diff repeat each human point? (paid, blind)")
    judge.add_argument("--run", type=Path, required=True)
    judge.add_argument("--model", required=True, help="full OpenRouter model ID, outside the Claude family")
    judge.add_argument("--max-usd", type=float, required=True)
    judge.add_argument("--call-bound", type=float, default=0.05)
    judge.add_argument("--seed", type=int, default=2026)
    judge.add_argument("--points", type=Path, default=root / "work" / "points")
    judge.add_argument("--cache", type=Path, default=root / "work" / "cache")
    judge.set_defaults(handler=cmd_judge)
    report = actions.add_parser("report", help="the build run's numbers per arm and paired (free)")
    report.add_argument("--run", type=Path, required=True)
    report.set_defaults(handler=cmd_report)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable {path}: {exc}") from exc


def read_tasks(run: Path) -> list[dict]:
    return [read_json(path) for path in sorted((run / "tasks").glob("*.json"))]


def load(path: Path, args: argparse.Namespace) -> dict:
    fresh = {"model": args.model, "prompt_sha256": prompt_sha256(), "seed": args.seed, "spent_usd": 0.0,
             "judgments": {}}
    data = read_json(path) if path.exists() else fresh
    for key in ("model", "prompt_sha256", "seed"):
        if data.get(key) != fresh[key]:
            raise ValueError(f"{path}: its {key} differs from this run's; move it aside to start over")
    return data


def targets(tasks: list[dict], args: argparse.Namespace) -> list[tuple[dict, str, dict]]:
    """(point, comment, arm -> diff) for every acted-on point of every task both arms ran."""
    found = []
    for task in tasks:
        if not all(arm in task.get("arms", {}) for arm in ARMS):
            continue
        points = {p["id"]: p for p in read_points(args.points / f"{slug_of(task['repo'])}.json")["points"]}
        texts = TextSource(GitHubClient(args.cache), task["repo"])
        diffs = {arm: task["arms"][arm]["agent"].get("diff") or "" for arm in ARMS}
        for point_id in task["points"]:
            comment = point_text(texts, points[point_id]) if point_id in points else None
            if comment is not None:
                found.append((points[point_id], comment, diffs))
    return found


def cmd_judge(args: argparse.Namespace) -> int:
    path = args.run / JUDGMENTS
    try:
        if args.max_usd <= 0 or any(m in args.model.lower() for m in CLAUDE_MARKERS):
            raise ValueError("the build judge is paid (--max-usd above zero) and outside the Claude family")
        data = load(path, args)
        work = targets(read_tasks(args.run), args)
    except (ValueError, OSError, GitHubError) as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 1
    budget = Budget(args.max_usd, {ENTRANT: args.call_bound})
    call = functools.partial(chat, args.model, max_tokens=MAX_TOKENS)
    for point, comment, diffs in work:
        if point["id"] in data["judgments"]:
            continue
        result = judge_point(point, comment, diffs, args.seed, call, budget)
        data["judgments"][point["id"]] = {"pr": point["pr"], **result}
        data["spent_usd"] = round(data["spent_usd"] + result.get("cost_usd", 0), 6)
        path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{len(data['judgments'])} points judged; reported spend ${data['spent_usd']:.4f}; "
          f"budget {budget.as_record()} -> {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    try:
        tasks = read_tasks(args.run)
        judged = args.run / JUDGMENTS
        judgments = read_json(judged)["judgments"] if judged.exists() else {}
    except (ValueError, OSError, KeyError) as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 1
    out = summary(tasks, judgments)
    (args.run / SUMMARY).write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(out["arms"], sort_keys=True))
    return 0
