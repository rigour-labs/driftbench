"""`bench score`: score a run directory.

Full summaries and the ledger go in the run directory (release assets). The
summaries published in results/ (committed) carry tool metrics only for
repositories that meet the reporting minimums.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench.collect.corpus import CorpusError, read_corpus
from bench.collect.github import GitHubClient, GitHubError
from bench.harness.cli import read_manifest
from bench.harness.runner import RecordError
from bench.subsample import recorded_selection, restrict, restrict_points
from bench.points.points_file import PointsError, read_points
from bench.repos import slug_of
from bench.score.linemap import FileVersions
from bench.score.repo import published_summary, score_repo
from bench.score.spend import openrouter_billed, spend_by_tool, spend_notes


def add_score_parser(commands: argparse._SubParsersAction, root: Path) -> None:
    score = commands.add_parser("score", help="score a run: catches, false blocks, noise")
    score.add_argument("--run", type=Path, required=True, help="a run directory, e.g. work/runs/2026-10-08")
    score.add_argument("--corpus", type=Path, default=root / "work" / "corpus")
    score.add_argument("--points", type=Path, default=root / "work" / "points")
    score.add_argument("--cache", type=Path, default=root / "work" / "cache", help="raw API responses (never published)")
    score.add_argument("--out", type=Path, help="summary directory (default: results/<run name>)")
    score.set_defaults(handler=cmd_score)

    spend = commands.add_parser("spend", help="a run's estimated spend per entrant, for the release notes")
    spend.add_argument("--run", type=Path, required=True)
    spend.add_argument("--openrouter", action="store_true",
                       help="fill the billed amount from the OpenRouter key's usage readings in the run")
    spend.set_defaults(handler=cmd_spend)


def tools_in(run_dir: Path) -> list[str]:
    return sorted(p.name for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))


def is_smoke(run_dir: Path) -> bool:
    """Whether the run's record marks it a smoke or diagnostic run; ValueError if run.json can't be read."""
    path = run_dir / "run.json"
    return path.exists() and bool({"smoke", "diagnostic"} & set(read_manifest(path)))


def cmd_score(args: argparse.Namespace) -> int:
    try:
        smoke = is_smoke(args.run)
        manifest = read_manifest(args.run / "run.json") if (args.run / "run.json").exists() else {}
        selection = recorded_selection(manifest)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if smoke:
        print(f"{args.run} is a smoke or diagnostic run (run.json says so); those are never scored", file=sys.stderr)
        return 1
    out = args.out or Path("results") / args.run.name
    client = GitHubClient(args.cache)
    try:
        tools = tools_in(args.run)
        ledger_path = args.run / "ledger.jsonl"
        ledger_path.write_text("", encoding="utf-8")
        for corpus_path in sorted(args.corpus.glob("*.json")):
            corpus = restrict(read_corpus(corpus_path), selection)
            points_file = restrict_points(read_points(args.points / corpus_path.name), corpus)
            summary, ledger = score_repo(corpus, points_file, args.run, tools, FileVersions(client, corpus["repo"]))
            if selection is not None:
                summary["subsample"] = manifest["subsample"]["repos"].get(corpus["repo"])
            write_summary(summary, args.run / "scores" / f"{slug_of(corpus['repo'])}.json")
            write_summary(published_summary(summary), out / f"{slug_of(corpus['repo'])}.json")
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.writelines(json.dumps(row, sort_keys=True) + "\n" for row in ledger)
            print(f"{corpus['repo']}: reportable={summary['reportable']} tools={', '.join(summary['tools'])}")
    except (OSError, CorpusError, PointsError, RecordError, GitHubError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


def cmd_spend(args: argparse.Namespace) -> int:
    try:
        budgets = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.run.glob("budget-*.json"))]
        usages = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.run.glob("openrouter-usage-*.json"))]
        billed = openrouter_billed(usages) if args.openrouter else None
        print(spend_notes(spend_by_tool(args.run, budgets), billed), end="")
    except (OSError, json.JSONDecodeError, RecordError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")


class ScoreFileError(ValueError):
    pass


def read_summary(path: Path) -> dict:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreFileError(f"cannot read summary {path}: {exc}") from exc
    if not isinstance(summary, dict) or "method_version" not in summary or "reportable" not in summary:
        raise ScoreFileError(f"{path}: not a score summary")
    return summary


def read_ledger(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreFileError(f"cannot read ledger {path}: {exc}") from exc
