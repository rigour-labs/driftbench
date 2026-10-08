"""Label files: `labels/<repo>.yaml`, committed to `main`, IDs and classes only.

```yaml
repo: o/r
guideline: 1
rules_version: 1
points:
  "7-inline-100-0": {suggested: claim/contract, label: claim/contract, labeller: maintainer,
                     blind: true, text_sha256: 9f86d0...}
  "7-body-10-1": {suggested: null, label: null, labeller: null}
```

A point is confirmed when `label` is set; `labeller` says who set it,
`blind` whether they labelled without seeing the suggestion, and
`text_sha256` which exact text they read. A label is used only while the
point's text still has that hash (otherwise it is stale). Merging new
suggestions never changes a confirmed entry.
"""
from __future__ import annotations

import hashlib
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
    """Add new points and refresh suggestions, without moving any confirmed entry.

    A confirmed entry keeps the suggestion it had when it was confirmed; the
    current rules' suggestion goes in `suggested_now`.
    """
    points = {key: dict(value) for key, value in labels["points"].items()}
    for point_id, suggested in suggestions.items():
        entry = points.setdefault(point_id, {"suggested": None, "label": None, "labeller": None})
        entry["suggested_now" if entry.get("label") else "suggested"] = suggested
    return {**labels, "rules_version": RULES_VERSION, "points": points}


def confirm(labels: dict, point_id: str, label: str, labeller: str, evidence: dict) -> dict:
    """Confirm a class. `evidence` holds `text_sha256` (the point's own text) and `blind`."""
    if label not in CLASSES:
        raise LabelError(f"unknown class {label!r}; expected one of {', '.join(CLASSES)}")
    if point_id not in labels["points"]:
        raise LabelError(f"{point_id} is not in the label file; run `bench label suggest` first")
    entry = {**labels["points"][point_id], "label": label, "labeller": labeller,
             "text_sha256": evidence["text_sha256"], "blind": evidence["blind"]}
    return {**labels, "points": {**labels["points"], point_id: entry}}


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def effective_labels(labels: dict, current_text: dict[str, str | None]) -> dict[str, str]:
    """Confirmed labels whose point still has the exact text the labeller read.

    A label on a point whose text changed (a different split, an edited
    comment) or that no longer exists is stale and is not used.
    """
    return {
        point_id: entry["label"]
        for point_id, entry in labels["points"].items()
        if entry.get("label")
        and current_text.get(point_id) is not None
        and text_sha256(current_text[point_id]) == entry.get("text_sha256")
    }


def label_status(labels: dict, current_text: dict[str, str | None]) -> dict:
    entries = labels["points"]
    usable = effective_labels(labels, current_text)
    blind = [pid for pid in usable if entries[pid].get("blind")]
    agreed = sum(1 for pid in blind if usable[pid] == entries[pid].get("suggested"))
    confirmed = sum(1 for e in entries.values() if e.get("label"))
    return {
        "points": len(entries),
        "confirmed": len(usable),
        "stale": confirmed - len(usable),
        "by_class": dict(sorted(Counter(usable.values()).items())),
        "blind_suggestion_agreement": f"{agreed}/{len(blind)}",
        "dropped_ids": len(set(entries) - set(current_text)),
    }
