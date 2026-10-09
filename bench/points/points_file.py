"""Build, write and read one repository's points file (no text; published with each run)."""
from __future__ import annotations

import json
from collections import Counter
from functools import partial
from pathlib import Path

from bench.collect.github import GitHubClient
from bench.points.acted_on import acted_on
from bench.points.build import RULES_VERSION, points_for_pr
from bench.points.texts import TextSource
from bench.repos import slug_of

POINTS_SCHEMA = 1


class PointsError(ValueError):
    pass


def build_points(corpus: dict, client: GitHubClient) -> dict:
    texts = TextSource(client, corpus["repo"])
    points = []
    for pr in corpus["prs"]:
        acted = partial(acted_on, client, corpus["repo"], pr=pr)
        points += points_for_pr(pr, texts, acted)
    return {
        "schema": POINTS_SCHEMA,
        "rules_version": RULES_VERSION,
        "repo": corpus["repo"],
        "pin": corpus["pin"],
        "corpus_collected_at": corpus["collected_at"],
        "summary": summarise(points),
        "points": points,
    }


def summarise(points: list[dict]) -> dict:
    kept = [p for p in points if not p["dropped"]]
    return {
        "points": len(points),
        "dropped_by_rule": dict(sorted(Counter(p["dropped"] for p in points if p["dropped"]).items())),
        "kept_by_kind": dict(sorted(Counter(p["kind"] for p in kept).items())),
        "scorable": sum(p["scorable"] for p in points),
        "unscored_no_round": sum(1 for p in kept if p["round"] is None),
        "acted_on": dict(sorted(Counter(
            f"{str(p['acted_on']).lower()}/{p['acted_basis']}" for p in kept if p["kind"] == "inline"
        ).items())),
    }


def write_points(points_file: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug_of(points_file['repo'])}.json"
    path.write_text(json.dumps(points_file, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_points(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PointsError(f"cannot read points {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != POINTS_SCHEMA:
        raise PointsError(f"{path}: expected points schema {POINTS_SCHEMA}")
    return data
