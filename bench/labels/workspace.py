"""Shared lookups for the label commands: points files and current point text."""
from __future__ import annotations

import argparse

from bench.points.points_file import PointsError, read_points
from bench.points.texts import TextSource, point_text


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


def current_texts(texts: TextSource, points_file: dict, labels: dict) -> dict[str, str | None]:
    """Each kept point's current text; fetched only for confirmed points, the rest are None."""
    confirmed = {pid for pid, entry in labels["points"].items() if entry.get("label")}
    return {p["id"]: point_text(texts, p) if p["id"] in confirmed else None for p in kept_points(points_file)}
