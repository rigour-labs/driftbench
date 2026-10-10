"""`bench subsample`: draw a run's selection of pull requests once, into a committed file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.collect.corpus import CorpusError, read_corpus
from bench.subsample import SubsampleError, draw, explicit, sizes_from_run, write_selection


def add_subsample_parser(commands: argparse._SubParsersAction, root: Path) -> None:
    sub = commands.add_parser("subsample", help="draw a seeded selection of whole pull requests for a costly run")
    sub.add_argument("--corpus", type=Path, default=root / "work" / "corpus")
    sub.add_argument("--heads-from", type=Path, metavar="FILE",
                     help="a diagnostic: exact heads, one 'owner/repo sha' per line (needs --purpose; no draw)")
    sub.add_argument("--purpose", help="with --heads-from: what the diagnostic is for")
    sub.add_argument("--sizes-run", type=Path, help="an earlier run directory: diff size per head")
    sub.add_argument("--sizes-label",
                     help="where the sizes come from, as published (e.g. the run's release tag), never a local path")
    sub.add_argument("--heads-per-repo", type=int, help="heads to aim for in each sampled repo")
    sub.add_argument("--full", action="append", default=[], metavar="REPO", help="take this repo in full (repeatable)")
    sub.add_argument("--skip", action="append", default=[], metavar="REPO=REASON",
                     help="leave this repo out of the round, with the reason the page states (repeatable)")
    sub.add_argument("--seed", type=int)
    sub.add_argument("--out", type=Path, required=True)
    sub.add_argument("--replace", action="store_true", help="redraw before any run has used it")
    sub.set_defaults(handler=cmd_subsample)


def heads_listed(path: Path) -> dict[str, set[str]]:
    heads: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            repo, sha = line.split()
            heads.setdefault(repo, set()).add(sha)
    return heads


def cmd_subsample(args: argparse.Namespace) -> int:
    try:
        corpora = [read_corpus(p) for p in sorted(args.corpus.glob("*.json"))]
        if args.heads_from:
            if not args.purpose:
                raise SubsampleError("--heads-from needs --purpose")
            selection = explicit(corpora, heads_listed(args.heads_from), args.purpose)
            write_selection(selection, args.out, args.replace)
            return report(selection)
        if None in (args.sizes_run, args.sizes_label, args.heads_per_repo, args.seed):
            raise SubsampleError("a draw needs --sizes-run, --sizes-label, --heads-per-repo and --seed")
        skips = dict(item.split("=", 1) for item in args.skip)
        targets = {c["repo"]: (None if c["repo"] in args.full else args.heads_per_repo) for c in corpora}
        selection = draw(corpora, sizes_from_run(args.sizes_run), targets, args.seed, source=args.sizes_label,
                         skips=skips)
        write_selection(selection, args.out, args.replace)
    except (CorpusError, SubsampleError, ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return report(selection)


def report(selection: dict) -> int:
    for repo, chosen in selection["repos"].items():
        print(f"{repo}: {len(chosen['prs'])} PRs, {chosen['heads']} heads ({chosen['rule']})")
    return 0
