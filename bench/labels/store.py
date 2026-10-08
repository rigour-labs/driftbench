"""Label files: `labels/<repo>.yaml`, committed to `main`, IDs and classes only.

```yaml
repo: o/r
guideline: 1
rules_version: 1
points:
  "7-inline-100-0": {suggested: claim/contract, label: claim/contract, labeller: maintainer}
  "7-body-10-1": {suggested: null, label: null, labeller: null}
```

A point is confirmed when `label` is set; `labeller` says who set it.
Merging new suggestions never changes a confirmed label.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from bench.labels.rules import CLASSES, RULES_VERSION
from bench.repos import slug_of

GUIDELINE_VERSION = 1


class LabelError(ValueError):
    pass


def labels_path(labels_dir: Path, repo: str) -> Path:
    return labels_dir / f"{slug_of(repo)}.yaml"


def empty_labels(repo: str) -> dict:
    return {"repo": repo, "guideline": GUIDELINE_VERSION, "rules_version": RULES_VERSION, "points": {}}


def read_labels(path: Path, repo: str) -> dict:
    if not path.exists():
        return empty_labels(repo)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise LabelError(f"cannot read labels {path}: {exc}") from exc
    if data.get("repo") != repo or not isinstance(data.get("points"), dict):
        raise LabelError(f"{path}: not a label file for {repo}")
    for point_id, entry in data["points"].items():
        if entry.get("label") not in (None, *CLASSES):
            raise LabelError(f"{path}: {point_id} has unknown class {entry.get('label')!r}")
    return data


def write_labels(labels: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(labels, sort_keys=True, allow_unicode=True), encoding="utf-8")


def merge_suggestions(labels: dict, suggestions: dict[str, str | None]) -> dict:
    """Add new points; refresh suggestions on unconfirmed ones; leave confirmed labels alone."""
    points = {key: dict(value) for key, value in labels["points"].items()}
    for point_id, suggested in suggestions.items():
        entry = points.setdefault(point_id, {"suggested": None, "label": None, "labeller": None})
        entry["suggested"] = suggested
    return {**labels, "rules_version": RULES_VERSION, "points": points}


def confirm(labels: dict, point_id: str, label: str, labeller: str) -> dict:
    if label not in CLASSES:
        raise LabelError(f"unknown class {label!r}; expected one of {', '.join(CLASSES)}")
    if point_id not in labels["points"]:
        raise LabelError(f"{point_id} is not in the label file; run `bench label suggest` first")
    points = {**labels["points"], point_id: {**labels["points"][point_id], "label": label, "labeller": labeller}}
    return {**labels, "points": points}


def label_status(labels: dict, current_ids: set[str]) -> dict:
    entries = labels["points"]
    confirmed = [e for e in entries.values() if e.get("label")]
    agreed = sum(1 for e in confirmed if e["label"] == e.get("suggested"))
    return {
        "points": len(entries),
        "confirmed": len(confirmed),
        "by_class": dict(sorted(Counter(e["label"] for e in confirmed).items())),
        "suggestion_agreement": f"{agreed}/{len(confirmed)}",
        "stale": len(set(entries) - current_ids),
    }
