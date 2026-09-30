"""Balanced evaluation items: every task's golden patch (no drift) and its drift patches."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from evalgate.sanitize import sanitize_patch

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EvalItem:
    id: str
    task_id: str
    category: str
    repository: str
    intent: str
    #: Sanitized: no comment names the answer.
    patch: str
    has_drift: bool


def load_items(datasets: Path = ROOT / "datasets") -> list[EvalItem]:
    """All items, golden and drift, in a stable order. A task whose patch file is missing fails loudly."""
    items: list[EvalItem] = []
    for task_file in sorted(datasets.glob("*/*.json")):
        task = json.loads(task_file.read_text())
        base = dict(task_id=task["id"], category=task["category"], repository=task["repository"], intent=task["intent"])
        items.append(EvalItem(id=f"{task['id']}__golden", patch=_read(task["golden_patch"]), has_drift=False, **base))
        for candidate in task.get("drift_candidates", []):
            items.append(EvalItem(id=f"{task['id']}__{candidate['id']}", patch=_read(candidate["patch"]), has_drift=True, **base))
    return items


def _read(relative: str) -> str:
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f"benchmark patch missing: {relative}")
    return sanitize_patch(path.read_text())
