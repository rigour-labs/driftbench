"""`bench run`: every chosen entrant over every frozen corpus."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from bench.adapters import select_adapters
from bench.collect.corpus import CorpusError, read_corpus
from bench.harness.gitrepo import GitError, RepoCheckout
from bench.harness.runner import RunConfig, run_corpus
from bench.repos import slug_of


def add_run_parser(commands: argparse._SubParsersAction, root: Path) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run = commands.add_parser("run", help="run entrants on every reviewed head of the frozen corpus")
    run.add_argument("--entrants", nargs="+", default=["free"], help="names, or `free` for every free entrant")
    run.add_argument("--repo", action="append", help="only this repo (repeatable)")
    run.add_argument("--corpus", type=Path, default=root / "work" / "corpus")
    run.add_argument("--out", type=Path, default=root / "work" / "runs" / today)
    run.add_argument("--repos-dir", type=Path, default=root / "work" / "repos", help="local clones")
    run.add_argument("--timeout", type=int, default=900, help="seconds per review")
    run.set_defaults(handler=cmd_run)


def cmd_run(args: argparse.Namespace) -> int:
    try:
        adapters = select_adapters(args.entrants)
        corpora = [read_corpus(p) for p in sorted(args.corpus.glob("*.json"))]
    except (ValueError, CorpusError) as exc:
        print(exc, file=sys.stderr)
        return 1
    corpora = [c for c in corpora if not args.repo or c["repo"] in args.repo]
    if not corpora:
        print(f"no matching corpus in {args.corpus}; run `bench collect` first", file=sys.stderr)
        return 1
    config = RunConfig(out_dir=args.out, scratch_dir=args.out / "_scratch",
                       npm_cache=args.repos_dir.parent / "npm-cache", timeout_s=args.timeout)
    for corpus in corpora:
        checkout = RepoCheckout(f"https://github.com/{corpus['repo']}.git", args.repos_dir / slug_of(corpus["repo"]))
        for adapter in adapters:
            try:
                counts = run_corpus(adapter, checkout, corpus, config)
            except GitError as exc:
                print(f"{corpus['repo']}: {exc}", file=sys.stderr)
                return 1
            print(f"{corpus['repo']} {adapter.name}: {counts}")
    return 0
