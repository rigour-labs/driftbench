"""Rule suggestions, kept apart from the labels: `labels/<repo>.suggest.yaml`.

```yaml
repo: o/r
rules_version: 1
points:
  "7-inline-100-0": {suggested: claim/contract}
  "7-inline-104-0": {suggested: mechanical, suggested_now: judgment}
```

A point that is already labelled keeps the suggestion it had at that time
(`suggested`); a later rules change only adds `suggested_now`, so the
agreement figure can't shift after the fact.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from bench.labels.rules import RULES_VERSION
from bench.labels.store import LabelError
from bench.repos import slug_of


def suggestions_path(labels_dir: Path, repo: str) -> Path:
    return labels_dir / f"{slug_of(repo)}.suggest.yaml"


def read_suggestions(path: Path, repo: str) -> dict:
    if not path.exists():
        return {"repo": repo, "rules_version": RULES_VERSION, "points": {}}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise LabelError(f"cannot read suggestions {path}: {exc}") from exc
    if data.get("repo") != repo or not isinstance(data.get("points"), dict):
        raise LabelError(f"{path}: not a suggestions file for {repo}")
    return data


def write_suggestions(suggestions: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(suggestions, sort_keys=True, allow_unicode=True), encoding="utf-8")


def merge_suggestions(suggestions: dict, new: dict[str, str | None], labelled: set[str]) -> dict:
    """Refresh suggestions; for already-labelled points, keep `suggested` and set `suggested_now`."""
    points = {key: dict(value) for key, value in suggestions["points"].items()}
    for point_id, suggested in new.items():
        entry = points.setdefault(point_id, {})
        entry["suggested_now" if point_id in labelled and "suggested" in entry else "suggested"] = suggested
    return {**suggestions, "rules_version": RULES_VERSION, "points": points}


def confirm_time_suggestions(suggestions: dict) -> dict[str, str | None]:
    return {point_id: entry.get("suggested") for point_id, entry in suggestions["points"].items()}
