"""`bench label suggest | show | set | status`: the labelling workflow (docs/LABELLING.md)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.collect.github import GitHubClient, GitHubError
from bench.labels.rules import CLASSES, suggest
from bench.labels.store import (LabelError, confirm, label_status, labels_path, merge_suggestions, read_labels,
                                text_sha256, write_labels)
from bench.points.points_file import PointsError, read_points
from bench.points.texts import TextSource, point_text


def add_label_parser(commands: argparse._SubParsersAction, root: Path) -> None:
    label = commands.add_parser("label", help="suggest, show, confirm and count point labels")
    label.add_argument("--points", type=Path, default=root / "work" / "points")
    label.add_argument("--labels", type=Path, default=root / "labels")
    label.add_argument("--cache", type=Path, default=root / "work" / "cache", help="raw API responses (never published)")
    label.set_defaults(handler=run_label)
    actions = label.add_subparsers(dest="label_command", required=True)
    actions.add_parser("suggest", help="run the rules pass; never changes a confirmed label").set_defaults(
        label_handler=cmd_suggest)
    show = actions.add_parser("show", help="print unconfirmed points with their text (local only)")
    show.add_argument("--repo", required=True)
    show.add_argument("--limit", type=int, default=20)
    show.add_argument("--with-suggestion", action="store_true",
                      help="also print the rule suggestion (labels set after this are not blind)")
    show.set_defaults(label_handler=cmd_show)
    setter = actions.add_parser("set", help="confirm a point's class")
    setter.add_argument("--repo", required=True)
    setter.add_argument("point_id")
    setter.add_argument("label", choices=CLASSES)
    setter.add_argument("--labeller", required=True, help="who confirms, e.g. maintainer")
    setter.add_argument("--saw-suggestion", action="store_true", help="the labeller saw the rule suggestion")
    setter.set_defaults(label_handler=cmd_set)
    actions.add_parser("status", help="confirmed counts per repo").set_defaults(label_handler=cmd_status)


def kept_points(points_file: dict) -> list[dict]:
    return [p for p in points_file["points"] if not p["dropped"]]


def load_points(args: argparse.Namespace) -> list[dict]:
    paths = sorted(args.points.glob("*.json"))
    if not paths:
        raise PointsError(f"no points files in {args.points}; run `bench points` first")
    return [read_points(path) for path in paths]


def points_for_repo(args: argparse.Namespace, repo: str) -> dict:
    match = [p for p in load_points(args) if p["repo"] == repo]
    if not match:
        raise PointsError(f"no points file for {repo} in {args.points}")
    return match[0]


def cmd_suggest(args: argparse.Namespace) -> int:
    client = GitHubClient(args.cache)
    for points_file in load_points(args):
        repo = points_file["repo"]
        texts = TextSource(client, repo)
        suggestions = {}
        for point in kept_points(points_file):
            text = point_text(texts, point)
            suggestions[point["id"]] = suggest(text) if text is not None else None
        path = labels_path(args.labels, repo)
        write_labels(merge_suggestions(read_labels(path, repo), suggestions), path)
        print(f"{repo}: {len(suggestions)} points suggested -> {path}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    points_file = points_for_repo(args, args.repo)
    labels = read_labels(labels_path(args.labels, args.repo), args.repo)
    texts = TextSource(GitHubClient(args.cache), args.repo)
    pending = [p for p in kept_points(points_file) if not labels["points"].get(p["id"], {}).get("label")]
    for point in pending[: args.limit]:
        anchor = point.get("anchor") or {}
        where = f"{anchor.get('path')}:{anchor.get('line')}" if anchor else point["kind"]
        hint = ""
        if args.with_suggestion:
            hint = f"; suggested: {labels['points'].get(point['id'], {}).get('suggested')}"
        print(f"## {point['id']}  ({where}{hint})\n{point_text(texts, point)}\n")
    print(f"{len(pending)} unconfirmed point(s) in {args.repo}", file=sys.stderr)
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    point = next((p for p in kept_points(points_for_repo(args, args.repo)) if p["id"] == args.point_id), None)
    if point is None:
        raise LabelError(f"{args.point_id} is not a kept point of {args.repo}")
    text = point_text(TextSource(GitHubClient(args.cache), args.repo), point)
    if text is None:
        raise LabelError(f"{args.point_id}: its text changed since the freeze (TEXT-1); it can't be labelled")
    path = labels_path(args.labels, args.repo)
    evidence = {"text_sha256": text_sha256(text), "blind": not args.saw_suggestion}
    write_labels(confirm(read_labels(path, args.repo), args.point_id, args.label, args.labeller, evidence), path)
    return 0


def current_texts(texts: TextSource, points_file: dict, labels: dict) -> dict[str, str | None]:
    """Each kept point's current text; fetched only for confirmed points, the rest are None."""
    confirmed = {pid for pid, entry in labels["points"].items() if entry.get("label")}
    return {p["id"]: point_text(texts, p) if p["id"] in confirmed else None for p in kept_points(points_file)}


def cmd_status(args: argparse.Namespace) -> int:
    client = GitHubClient(args.cache)
    for points_file in load_points(args):
        repo = points_file["repo"]
        labels = read_labels(labels_path(args.labels, repo), repo)
        print(f"{repo}: {label_status(labels, current_texts(TextSource(client, repo), points_file, labels))}")
    return 0


def run_label(args: argparse.Namespace) -> int:
    try:
        return args.label_handler(args)
    except (LabelError, PointsError, GitHubError) as exc:
        print(exc, file=sys.stderr)
        return 1
