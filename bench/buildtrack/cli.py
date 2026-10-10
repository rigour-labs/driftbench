"""`bench build tasks`: draw the build track's tasks from the frozen corpus (docs/BUILD_TRACK.md).

Free: GitHub API reads only, cached in --cache (never published). Writes the
task file: IDs, SHAs, paths and each statement's hash, never its text.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from bench.buildtrack.judge_cli import add_judge_actions
from bench.buildtrack.tasks import select
from bench.collect.corpus import CorpusError, read_corpus
from bench.collect.github import GitHubClient, GitHubError
from bench.points.points_file import read_points
from bench.repos import slug_of


def add_build_parser(commands: argparse._SubParsersAction, root: Path) -> None:
    build = commands.add_parser("build", help="the build track: the same agent with and without Rigour")
    actions = build.add_subparsers(dest="build_command", required=True)
    tasks = actions.add_parser("tasks", help="order every eligible task from the frozen corpus (seeded)")
    tasks.add_argument("--repo", required=True, help="owner/repo")
    tasks.add_argument("--count", type=int, required=True, help="tasks a run takes, in order, once their tests discriminate")
    tasks.add_argument("--seed", type=int, required=True)
    tasks.add_argument("--max-changed-files", type=int, help="leave out pull requests that changed more files")
    tasks.add_argument("--out", type=Path, required=True)
    tasks.add_argument("--replace", action="store_true", help="redraw an existing task file (before any run uses it)")
    tasks.add_argument("--corpus", type=Path, default=root / "work" / "corpus")
    tasks.add_argument("--points", type=Path, default=root / "work" / "points")
    tasks.add_argument("--cache", type=Path, default=root / "work" / "cache", help="API cache (never published)")
    tasks.set_defaults(handler=cmd_tasks)
    add_judge_actions(actions, root)


def cmd_tasks(args: argparse.Namespace) -> int:
    if args.out.exists() and not args.replace:
        print(f"build: {args.out} exists; a task file is never redrawn (pass --replace before any run uses it)",
              file=sys.stderr)
        return 1
    slug = slug_of(args.repo)
    try:
        corpus = read_corpus(args.corpus / f"{slug}.json")
        points = read_points(args.points / f"{slug}.json")
        drawn = select(GitHubClient(args.cache), corpus, points, args.count, args.seed, args.max_changed_files)
    except (CorpusError, GitHubError, OSError, ValueError) as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(drawn, sort_keys=False), encoding="utf-8")
    print(f"{args.repo}: {drawn['eligible']} eligible in seeded order, a run takes {drawn['count']}; "
          f"{len(drawn['excluded'])} excluded -> {args.out}")
    return 0
