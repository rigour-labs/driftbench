"""Command line: `python -m bench <command>`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench import name_guard
from bench.collect.corpus import CorpusError, collect_repo, read_corpus, write_corpus
from bench.collect.github import GitHubClient, GitHubError, require_gh
from bench.harness.cli import add_run_parser
from bench.labels.cli import add_label_parser
from bench.report.cli import add_report_parsers
from bench.score.cli import add_score_parser
from bench.issues.cli import add_issues_parser
from bench.subsample_cli import add_subsample_parser
from bench.points.points_file import build_points, write_points
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

    collect = commands.add_parser("collect", help="freeze the corpus of reviewed PRs from GitHub")
    collect.add_argument("--repo", action="append", help="only this repo (repeatable); default: all enabled")
    collect.add_argument("--max-prs", type=int, default=60, help="substantively reviewed PRs to keep per repo")
    collect.add_argument("--max-listed", type=int, default=600, help="closed PRs to list per repo")
    collect.add_argument("--out", type=Path, default=ROOT / "work" / "corpus")
    collect.add_argument("--cache", type=Path, default=ROOT / "work" / "cache", help="raw API responses (never published)")
    collect.set_defaults(handler=cmd_collect)

    points = commands.add_parser("points", help="split frozen reviews into points; check which were acted on")
    points.add_argument("--corpus", type=Path, default=ROOT / "work" / "corpus")
    points.add_argument("--out", type=Path, default=ROOT / "work" / "points")
    points.add_argument("--cache", type=Path, default=ROOT / "work" / "cache", help="raw API responses (never published)")
    points.set_defaults(handler=cmd_points)

    add_label_parser(commands, ROOT)
    add_run_parser(commands, ROOT)
    add_score_parser(commands, ROOT)
    add_report_parsers(commands, ROOT)
    add_subsample_parser(commands, ROOT)
    add_issues_parser(commands, ROOT)
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


def cmd_collect(args: argparse.Namespace) -> int:
    try:
        repos = load_repos(ROOT / "repos.yaml")
    except RepoListError as exc:
        print(f"repos.yaml invalid: {exc}", file=sys.stderr)
        return 1
    chosen = [r for r in repos if not args.repo or r.name in args.repo]
    unknown = set(args.repo or []) - {r.name for r in chosen}
    if unknown:
        print(f"not an enabled repo in repos.yaml: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 1
    try:
        require_gh()
    except GitHubError as exc:
        print(exc, file=sys.stderr)
        return 1
    client = GitHubClient(args.cache)
    for repo in chosen:
        try:
            corpus = collect_repo(client, repo, args.max_prs, args.max_listed)
        except GitHubError as exc:
            print(f"{repo.name}: collection failed: {exc}", file=sys.stderr)
            return 1
        path = write_corpus(corpus, args.out)
        print(f"{repo.name}: {len(corpus['prs'])} PRs -> {path}")
    return 0


def cmd_points(args: argparse.Namespace) -> int:
    corpora = sorted(args.corpus.glob("*.json"))
    if not corpora:
        print(f"no corpus files in {args.corpus}; run `bench collect` first", file=sys.stderr)
        return 1
    client = GitHubClient(args.cache)
    for path in corpora:
        try:
            result = build_points(read_corpus(path), client)
        except (CorpusError, GitHubError) as exc:
            print(f"{path.name}: {exc}", file=sys.stderr)
            return 1
        out = write_points(result, args.out)
        print(f"{result['repo']}: {result['summary']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
