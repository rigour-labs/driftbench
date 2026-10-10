"""`bench issues judge`: the blind, non-Claude issue judge over a run's paid entrants (paid; needs --max-usd).

Writes `<results>/issue-judgments.yaml`, saved after every point, so a stop
loses nothing and a rerun resumes. A seeded few points are asked again to
measure the judge's self-consistency.
"""
from __future__ import annotations

import argparse
import functools
import json
import random
import sys
from pathlib import Path

import yaml

from bench.collect.corpus import CorpusError, read_corpus
from bench.collect.github import GitHubClient
from bench.harness.budget import Budget
from bench.harness.cli import read_manifest
from bench.issues.diagnose_cli import add_diagnose_action
from bench.issues.inputs import entrant_review, judged_points, reviews_by_head
from bench.issues.judge import ENTRANT, MAX_TOKENS, judge_point, prompt_sha256
from bench.labels.openrouter import chat
from bench.labels.prelabel_cli import CLAUDE_MARKERS
from bench.points.points_file import read_points
from bench.points.texts import TextSource, point_text
from bench.repos import slug_of
from bench.subsample import SubsampleError, recorded_selection, restrict

OUT = "issue-judgments.yaml"


def add_issues_parser(commands: argparse._SubParsersAction, root: Path) -> None:
    issues = commands.add_parser("issues", help="the blind issue-level comparison of a run's paid entrants")
    actions = issues.add_subparsers(dest="issues_command", required=True)
    judge = actions.add_parser("judge", help="ask a non-Claude judge, per human point (paid)")
    judge.add_argument("--run", type=Path, required=True)
    judge.add_argument("--results", type=Path, required=True)
    judge.add_argument("--entrants", nargs="+", default=["claude-code-review", "rigour-reviewer"])
    judge.add_argument("--corpus", type=Path, default=root / "work" / "corpus")
    judge.add_argument("--points", type=Path, default=root / "work" / "points")
    judge.add_argument("--cache", type=Path, default=root / "work" / "cache")
    judge.add_argument("--model", required=True, help="full OpenRouter model ID, outside the Claude family")
    judge.add_argument("--max-usd", type=float, required=True)
    judge.add_argument("--call-bound", type=float, default=0.05)
    judge.add_argument("--seed", type=int, default=2026)
    judge.add_argument("--repeats", type=int, default=10, help="points asked again for self-consistency")
    judge.set_defaults(handler=cmd_judge)
    add_diagnose_action(actions, root)


def check_args(args: argparse.Namespace) -> None:
    if args.max_usd <= 0:
        raise ValueError("--max-usd must be above zero: the judge is paid")
    if any(marker in args.model.lower() for marker in CLAUDE_MARKERS):
        raise ValueError(f"{args.model}: the issue judge is outside the Claude family; Rigour's reviewer runs on Claude")


def load(path: Path, args: argparse.Namespace) -> dict:
    fresh = {"model": args.model, "prompt_sha256": prompt_sha256(), "seed": args.seed, "entrants": args.entrants,
             "spent_usd": 0.0, "judgments": {}, "repeats": {}}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else fresh
    for key in ("model", "prompt_sha256", "seed", "entrants"):
        if data.get(key) != fresh[key]:
            raise ValueError(f"{path}: its {key} differs from this run's; move it aside to start over")
    return data


def targets(args: argparse.Namespace, client: GitHubClient) -> list[tuple[dict, str, dict]]:
    """(point with its repo, comment text, entrant -> review or None) for every judged point."""
    manifest = read_manifest(args.run / "run.json")
    selection = recorded_selection(manifest)
    found = []
    for corpus_path in sorted(args.corpus.glob("*.json")):
        corpus = restrict(read_corpus(corpus_path), selection)
        if not corpus["prs"]:
            continue
        repo, rounds = corpus["repo"], {pr["number"]: pr["rounds"] for pr in corpus["prs"]}
        reviews = {e: reviews_by_head(args.run, e, repo) for e in args.entrants}
        texts = TextSource(client, repo)
        for point in judged_points(corpus, read_points(args.points / f"{slug_of(repo)}.json")):
            comment = point_text(texts, point)
            if comment is None:
                continue
            by_entrant = {e: entrant_review(point, rounds[point["pr"]], reviews[e]) for e in args.entrants}
            found.append(({**point, "repo": repo}, comment, by_entrant))
    return found


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding="utf-8")


def cmd_judge(args: argparse.Namespace) -> int:
    try:
        check_args(args)
        path = args.results / OUT
        data = load(path, args)
        work = targets(args, GitHubClient(args.cache))
    except (ValueError, OSError, CorpusError, SubsampleError, json.JSONDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    budget = Budget(args.max_usd, {ENTRANT: args.call_bound})
    call = functools.partial(chat, args.model, max_tokens=MAX_TOKENS)
    repeat_ids = set(random.Random(args.seed).sample([p["id"] for p, _, _ in work], min(args.repeats, len(work))))
    for point, comment, by_entrant in work:
        for store in ("judgments", "repeats") if point["id"] in repeat_ids else ("judgments",):
            if point["id"] in data[store]:
                continue
            result = judge_point(point, comment, by_entrant, args.seed, call, budget)
            data[store][point["id"]] = {"repo": point["repo"], "pr": point["pr"], **result}
            data["spent_usd"] = round(data["spent_usd"] + result.get("cost_usd", 0), 6)
            save(data, path)
    errors = sum(1 for j in data["judgments"].values() if j.get("error"))
    print(f"{len(data['judgments'])} points judged ({errors} without a verdict), {len(data['repeats'])} asked again; "
          f"reported spend ${data['spent_usd']:.4f}; budget {budget.as_record()} -> {path}")
    return 0
