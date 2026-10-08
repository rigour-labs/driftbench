"""Command line: `python -m bench <command>`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench import name_guard
from bench.repos import RepoListError, load_repos

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    repos = commands.add_parser("repos", help="validate repos.yaml and list the repos")
    repos.add_argument("--file", type=Path, default=ROOT / "repos.yaml")
    repos.add_argument("--all", action="store_true", help="include disabled repos")
    repos.set_defaults(handler=cmd_repos)

    guard = commands.add_parser("guard", help="refuse to publish blocked names")
    guard.add_argument("paths", nargs="*", type=Path, help="extra files or dirs, e.g. release assets")
    guard.add_argument("--range", dest="rev_range", help="also scan commit messages, e.g. origin/main..HEAD")
    guard.set_defaults(handler=cmd_guard)
    return parser


def cmd_repos(args: argparse.Namespace) -> int:
    try:
        repos = load_repos(args.file, enabled_only=not args.all)
    except RepoListError as exc:
        print(f"repos.yaml invalid: {exc}", file=sys.stderr)
        return 1
    for repo in repos:
        state = "enabled" if repo.enabled else "disabled"
        print(f"{repo.name:28} {repo.licence:14} {repo.pin[:12]} {state}")
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    try:
        pattern = name_guard.load_pattern(name_guard.blocked_names_path())
        hits = name_guard.scan_paths(pattern, name_guard.tracked_files(ROOT) + args.paths)
        if args.rev_range:
            messages = name_guard.commit_messages(ROOT, args.rev_range)
            hits += name_guard.scan_text(pattern, f"commits {args.rev_range}", messages)
    except name_guard.GuardConfigError as exc:
        print(f"guard not run: {exc}", file=sys.stderr)
        return 2
    for hit in hits:
        print(f"{hit.where}:{hit.line}: {hit.text}")
    print(f"guard: {len(hits)} match(es)", file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
