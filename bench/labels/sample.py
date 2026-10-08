"""The labelled sample: which points get a human class (docs/LABELLING.md, "The sample").

A uniform random sample of kept, scorable points, stratified by kind in
proportion (largest remainder), drawn with a recorded seed from a points file
identified by its SHA-256. The seed and hash let anyone redraw the same sample
and show it wasn't hand-picked. A sample is never replaced silently.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bench.repos import slug_of

KINDS = ("inline", "body", "conversation")


class SampleError(ValueError):
    pass


def sample_path(labels_dir: Path, repo: str) -> Path:
    return labels_dir / f"{slug_of(repo)}.sample.yaml"


def points_sha256(points_file: dict) -> str:
    canonical = json.dumps(points_file["points"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def allocate(counts: dict[str, int], size: int) -> dict[str, int]:
    """Split `size` across kinds in proportion to `counts` (largest remainder)."""
    total = sum(counts.values())
    size = min(size, total)
    if not total:
        return {kind: 0 for kind in counts}
    exact = {kind: size * n / total for kind, n in counts.items()}
    shares = {kind: int(share) for kind, share in exact.items()}
    by_remainder = sorted(counts, key=lambda kind: (exact[kind] - shares[kind], kind), reverse=True)
    for kind in by_remainder[: size - sum(shares.values())]:
        shares[kind] += 1
    return shares


def draw_sample(points_file: dict, size: int, seed: int) -> dict:
    eligible = {kind: sorted(p["id"] for p in points_file["points"] if p["scorable"] and p["kind"] == kind)
                for kind in KINDS}
    shares = allocate({kind: len(ids) for kind, ids in eligible.items()}, size)
    rng = random.Random(seed)
    chosen = [pid for kind in KINDS for pid in sorted(rng.sample(eligible[kind], shares[kind]))]
    return {
        "repo": points_file["repo"],
        "seed": seed,
        "size": len(chosen),
        "requested_size": size,
        "by_kind": shares,
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
