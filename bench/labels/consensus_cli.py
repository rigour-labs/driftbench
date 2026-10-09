"""`bench label consensus`: write AI-consensus labels for a repository's sample."""
from __future__ import annotations

import argparse
from pathlib import Path

from bench.collect.github import GitHubClient
from bench.labels.consensus import build_consensus, is_consensus
from bench.labels.model_file import model_path, read_model_file
from bench.labels.sample import read_sample, sample_path
from bench.labels.store import LabelError, labels_path, read_labels, write_labels
from bench.labels.workspace import points_for_repo
from bench.points.texts import TextSource, point_text


def add_consensus_action(actions: argparse._SubParsersAction) -> None:
    con = actions.add_parser("consensus", help="labels where a Claude labeller and the non-Claude model agree")
    con.add_argument("--repo", required=True)
    con.add_argument("--from", dest="source", type=Path, required=True,
                     help="the Claude labeller's labels directory, e.g. labels-claude")
    con.set_defaults(label_handler=cmd_consensus)


def cmd_consensus(args: argparse.Namespace) -> int:
    sample = read_sample(sample_path(args.labels, args.repo))
    model_data = read_model_file(model_path(args.labels, args.repo), args.repo)
    if sample is None or model_data is None:
        raise LabelError(f"{args.repo}: consensus needs the sample and the model suggestions file")
    path = labels_path(args.labels, args.repo)
    existing = read_labels(path, args.repo)
    if existing["points"] and not is_consensus(existing):
        raise LabelError(f"{path} holds labels that aren't AI consensus; move it aside first")
    first = read_labels(labels_path(args.source, args.repo), args.repo)
    points = {p["id"]: p for p in points_for_repo(args, args.repo)["points"]}
    texts = TextSource(GitHubClient(args.cache), args.repo)
    current = {pid: point_text(texts, points[pid]) for pid in sample["point_ids"] if pid in points}
    labels = build_consensus(args.repo, first, model_data, sample["point_ids"], current)
    write_labels(labels, path)
    c = labels["consensus"]
    print(f"{args.repo}: {c['agreed']} of {c['both_labelled']} agree (kappa {c['kappa']}); "
          f"dropped by class: {c['dropped_by_class']} -> {path}")
    return 0
