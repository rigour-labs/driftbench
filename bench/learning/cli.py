"""`bench learning prepare|check`: the free steps of the learning run (docs/LEARNING.md).

prepare: the crawl (GitHub API, cached) and each head's cutoff and diff (a full clone).
Then `node bench/learning/stores.mjs` builds the stores with Rigour's learner.
check:   the leak assertions and the pre-check counts, numbers and lesson ids only (a leaking store's
         heads are errors; the workflow fails after uploading the record).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench.collect.corpus import CorpusError, read_corpus
from bench.collect.github import GitHubClient, GitHubError
from bench.harness.gitrepo import GitError, RepoCheckout
from bench.learning.crawl import crawl
from bench.learning.heads import cutoffs, prepare
from bench.learning.precheck import precheck
from bench.repos import slug_of
from bench.subsample import SubsampleError, read_selection, restrict


def add_learning_parser(commands: argparse._SubParsersAction, root: Path) -> None:
    learning = commands.add_parser("learning", help="the time-correct learning run's free steps")
    actions = learning.add_subparsers(dest="learning_command", required=True)
    prep = actions.add_parser("prepare", help="crawl the review comments and write each head's cutoff and diff")
    prep.add_argument("--repo", required=True, help="owner/repo")
    prep.add_argument("--subsample", type=Path, required=True)
    prep.add_argument("--corpus", type=Path, default=root / "work" / "corpus")
    prep.add_argument("--cache", type=Path, default=root / "work" / "learn-cache", help="API cache (never published)")
    prep.add_argument("--repos-dir", type=Path, default=root / "work" / "repos")
    prep.add_argument("--out", type=Path, required=True)
    prep.set_defaults(handler=cmd_prepare)
    check = actions.add_parser("check", help="leak assertions and the pre-check counts")
    check.add_argument("--out", type=Path, required=True, help="the directory prepare and stores.mjs wrote")
    check.set_defaults(handler=cmd_check)


def cmd_prepare(args: argparse.Namespace) -> int:
    try:
        corpus = restrict(read_corpus(args.corpus / f"{slug_of(args.repo)}.json"), read_selection(args.subsample))
    except (CorpusError, SubsampleError) as exc:
        print(f"learning: {exc}", file=sys.stderr)
        return 1
    if not corpus["prs"]:
        print(f"learning: the selection has no pull requests in {args.repo}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        crawled = crawl(GitHubClient(args.cache), args.repo, [pr["cutoff"] for pr in cutoffs(corpus)])
        checkout = RepoCheckout(f"https://github.com/{args.repo}.git", args.repos_dir / slug_of(args.repo),
                                blobless=False)
        heads = prepare(checkout, corpus, args.out / "diffs")
    except (GitHubError, GitError) as exc:
        print(f"learning: {exc}", file=sys.stderr)
        return 1
    (args.out / "crawl.json").write_text(json.dumps(crawled), encoding="utf-8")
    (args.out / "heads.json").write_text(json.dumps(heads, indent=1), encoding="utf-8")
    print(f"{args.repo}: {len(heads)} pull requests, {sum(len(p['heads']) for p in heads)} heads; "
          f"crawled {len(crawled['prs'])} merged pull requests")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    try:
        crawled = json.loads((args.out / "crawl.json").read_text(encoding="utf-8"))
        served = json.loads((args.out / "served.json").read_text(encoding="utf-8"))
        result = precheck(served, args.out, crawled)
    except (OSError, ValueError) as exc:
        print(f"learning: {exc}", file=sys.stderr)
        return 1
    path = args.out / f"precheck-{slug_of(result['repo'])}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{result['repo']}: {result['totals']} -> {path}")
    return 0
