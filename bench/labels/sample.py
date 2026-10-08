"""The labelled sample: which points get a human class (docs/LABELLING.md, "What gets a label").

A uniform random sample of location-scorable points (the only points class
catch rates are computed over), drawn with a recorded seed from a points file
identified by its SHA-256. The seed and hash let anyone redraw the same sample
and show it wasn't hand-picked. A sample is never replaced silently, and the
report refuses a sample whose points file has since changed.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bench.repos import slug_of
from bench.score.match import location_scorable

class SampleError(ValueError):
    pass


def sample_path(labels_dir: Path, repo: str) -> Path:
    return labels_dir / f"{slug_of(repo)}.sample.yaml"


def points_sha256(points_file: dict) -> str:
    canonical = json.dumps(points_file["points"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def draw_sample(points_file: dict, size: int, seed: int) -> dict:
    eligible = sorted(p["id"] for p in points_file["points"] if location_scorable(p))
    chosen = sorted(random.Random(seed).sample(eligible, min(size, len(eligible))))
    return {
        "repo": points_file["repo"],
        "seed": seed,
        "size": len(chosen),
        "requested_size": size,
        "eligible": len(eligible),
        "points_sha256": points_sha256(points_file),
        "drawn_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "point_ids": chosen,
    }


def write_sample(sample: dict, path: Path, replace: bool) -> None:
    if path.exists() and not replace:
        raise SampleError(f"{path} exists; pass --replace to redraw (the old draw stays in git history)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(sample, sort_keys=False), encoding="utf-8")


def read_sample(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        sample = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SampleError(f"cannot read sample {path}: {exc}") from exc
    if not isinstance(sample.get("point_ids"), list):
        raise SampleError(f"{path}: not a label sample")
    return sample
