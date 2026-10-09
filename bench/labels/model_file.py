"""Model suggestions, kept apart from labels: `labels/<repo>.model.yaml`.

```yaml
repo: o/r
rules_version: 2
prompt_version: 1
model: openai/example-model        # the full model ID requested
blind_fraction: 0.2
blind_ids: [...]                   # seeded from the sample; labelled without any suggestion
spent_usd: 0.0123                  # OpenRouter's reported cost, summed over every run
points:
  "7-inline-100-0": {suggested_by: openai/example-model, suggested: performance, reason: "...", cost_usd: 0.0001}
unsuggested:
  "7-inline-104-0": "budget: ..."  # why a point has no suggestion; never silent
```

A suggestion is never a label: the labeller confirms or overrides it, and
the label records which. Points in `blind_ids` are suggested like the rest
but the suggestion is never shown, so model-human agreement is measured on
labels the model could not have anchored.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import yaml

from bench.labels.rules import RULES_VERSION
from bench.labels.store import LabelError
from bench.repos import slug_of

PROMPT_VERSION = 1
BLIND_FRACTION = 0.2


def model_path(labels_dir: Path, repo: str) -> Path:
    return labels_dir / f"{slug_of(repo)}.model.yaml"


def blind_ids(sample: dict, fraction: float = BLIND_FRACTION) -> list[str]:
    """A seeded random share of the sample, at least one point; anyone can redraw it from the sample file."""
    ids = sorted(sample["point_ids"])
    count = min(len(ids), max(1, math.ceil(len(ids) * fraction)))
    return sorted(random.Random(f"blind:{sample['repo']}:{sample['seed']}").sample(ids, count))


def empty_model_file(sample: dict, model: str) -> dict:
    return {"repo": sample["repo"], "rules_version": RULES_VERSION, "prompt_version": PROMPT_VERSION,
            "model": model, "blind_fraction": BLIND_FRACTION, "blind_ids": blind_ids(sample),
            "spent_usd": 0.0, "points": {}, "unsuggested": {}}


def read_model_file(path: Path, repo: str) -> dict | None:
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise LabelError(f"cannot read model suggestions {path}: {exc}") from exc
    if data.get("repo") != repo or not isinstance(data.get("points"), dict):
        raise LabelError(f"{path}: not a model suggestions file for {repo}")
    return data


def write_model_file(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding="utf-8")


def shown_suggestion(data: dict | None, point_id: str) -> dict | None:
    """The suggestion a labeller may see for this point: none for blind points or failed calls."""
    if data is None or point_id in data.get("blind_ids", []):
        return None
    entry = data["points"].get(point_id)
    return entry if entry and entry.get("suggested") else None
