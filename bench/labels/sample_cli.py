"""`bench label sample | next | agreement` and the sample counts in `status`."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from bench.collect.github import GitHubClient
from bench.labels.agreement import cohens_kappa
from bench.labels.sample import draw_sample, read_sample, sample_path, write_sample
from bench.labels.session import run_session
from bench.labels.store import effective_labels, labels_path, read_labels, write_labels
from bench.labels.workspace import current_texts, points_for_repo
from bench.points.texts import TextSource, point_text


def add_sample_actions(actions: argparse._SubParsersAction) -> None:
    sample = actions.add_parser("sample", help="draw the seeded random sample of points to label")
    sample.add_argument("--repo", required=True)
    sample.add_argument("--size", type=int, default=50)
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--replace", action="store_true", help="redraw over an existing sample")
    sample.set_defaults(label_handler=cmd_sample)
    nxt = actions.add_parser("next", help="label the sample one point at a time (blind, resumable)")
    nxt.add_argument("--repo", required=True)
    nxt.add_argument("--labeller", required=True)
    nxt.add_argument("--include-skipped", action="store_true")
    nxt.set_defaults(label_handler=cmd_next, ask=input, show=print)
    agree = actions.add_parser("agreement", help="Cohen's kappa between two labellers' label directories")
    agree.add_argument("--repo", required=True)
    agree.add_argument("--a", type=Path, required=True, help="first labeller's labels directory")
    agree.add_argument("--b", type=Path, required=True, help="second labeller's labels directory")
    agree.set_defaults(label_handler=cmd_agreement)


def cmd_sample(args: argparse.Namespace) -> int:
    sample = draw_sample(points_for_repo(args, args.repo), args.size, args.seed)
    path = sample_path(args.labels, args.repo)
    write_sample(sample, path, args.replace)
    print(f"{args.repo}: sampled {sample['size']} points {sample['by_kind']} (seed {args.seed}) -> {path}")
    return 0


def cmd_next(args: argparse.Namespace) -> int:
    sample = read_sample(sample_path(args.labels, args.repo))
    if sample is None:
        print(f"no sample for {args.repo}; run `bench label sample` first")
        return 1
    points_file = points_for_repo(args, args.repo)
    path = labels_path(args.labels, args.repo)
    texts = TextSource(GitHubClient(args.cache), args.repo)
    context = {
        "repo": args.repo, "points": {p["id"]: p for p in points_file["points"]},
        "sample_ids": sample["point_ids"], "text": lambda point: point_text(texts, point),
        "labeller": args.labeller, "save": lambda labels: write_labels(labels, path),
        "include_skipped": args.include_skipped,
    }
    run_session(context, read_labels(path, args.repo), args.ask, args.show)
    return 0


def sample_status(labels: dict, usable: dict[str, str], sample_ids: list[str]) -> dict:
    entries = labels["points"]
    in_sample = set(sample_ids)
    labelled = {pid: label for pid, label in usable.items() if pid in in_sample}
    confirmed = [pid for pid in in_sample if entries.get(pid, {}).get("label")]
    return {
        "sampled": len(in_sample),
        "labelled": len(labelled),
        "skipped": sum(1 for pid in in_sample if entries.get(pid, {}).get("skipped")),
        "stale": len(confirmed) - len(labelled),
        "by_class": dict(sorted(Counter(labelled.values()).items())),
    }


def usable_labels(args: argparse.Namespace, labels_dir: Path, repo: str) -> dict[str, str]:
    points_file = points_for_repo(args, repo)
    labels = read_labels(labels_path(labels_dir, repo), repo)
    texts = TextSource(GitHubClient(args.cache), repo)
    return effective_labels(labels, current_texts(texts, points_file, labels))


def cmd_agreement(args: argparse.Namespace) -> int:
    result = cohens_kappa(usable_labels(args, args.a, args.repo), usable_labels(args, args.b, args.repo))
    print(f"{args.repo}: {result}")
    return 0
