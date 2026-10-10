"""`bench subsample`: draw a run's selection of pull requests once, into a committed file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.collect.corpus import CorpusError, read_corpus
from bench.subsample import SubsampleError, draw, sizes_from_run, write_selection


def add_subsample_parser(commands: argparse._SubParsersAction, root: Path) -> None:
    sub = commands.add_parser("subsample", help="draw a seeded selection of whole pull requests for a costly run")
    sub.add_argument("--corpus", type=Path, default=root / "work" / "corpus")
    sub.add_argument("--sizes-run", type=Path, required=True, help="an earlier run directory: diff size per head")
    sub.add_argument("--sizes-label", required=True,
                     help="where the sizes come from, as published (e.g. the run's release tag), never a local path")
    sub.add_argument("--heads-per-repo", type=int, required=True, help="heads to aim for in each sampled repo")
    sub.add_argument("--full", action="append", default=[], metavar="REPO", help="take this repo in full (repeatable)")
    sub.add_argument("--skip", action="append", default=[], metavar="REPO=REASON",
                     help="leave this repo out of the round, with the reason the page states (repeatable)")
    sub.add_argument("--seed", type=int, required=True)
    sub.add_argument("--out", type=Path, required=True)
    sub.add_argument("--replace", action="store_true", help="redraw before any run has used it")
    sub.set_defaults(handler=cmd_subsample)


def cmd_subsample(args: argparse.Namespace) -> int:
    try:
        corpora = [read_corpus(p) for p in sorted(args.corpus.glob("*.json"))]
        skips = dict(item.split("=", 1) for item in args.skip)
        targets = {c["repo"]: (None if c["repo"] in args.full else args.heads_per_repo) for c in corpora}
        selection = draw(corpora, sizes_from_run(args.sizes_run), targets, args.seed, source=args.sizes_label,
                         skips=skips)
        write_selection(selection, args.out, args.replace)
    except (CorpusError, SubsampleError) as exc:
        print(exc, file=sys.stderr)
        return 1
    for repo, chosen in selection["repos"].items():
        print(f"{repo}: {len(chosen['prs'])} PRs, {chosen['heads']} heads ({chosen['rule']})")
    return 0
