"""Hand-checked samples that validate the automatic decisions (docs/SPEC.md, "Calibration").

Two samples, drawn with a fixed seed so anyone can redraw the same one:
- location matches (distance <= 3) per real entrant, judged "same issue":
  yes / partly / no; this gives each entrant's location-to-issue rate;
- acted-on decisions (10 direct-true, 5 direct-false, 5 ancestor), judged
  "did the change near the anchor respond to the point": yes / no; this
  gives agreement per ACTED-1 basis.
A human fills `verdict`; until every entry has one, the headline is unvalidated.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

import yaml

BASELINES = ("no-tool", "every-hunk")
LOCATION_TARGET = 50
ACTED_ON_QUOTAS = ((True, "direct", 10), (False, "direct", 5), (None, "ancestor", 5))
LOCATION_VERDICTS = ("yes", "partly", "no")
ACTED_VERDICTS = ("yes", "no")


class CalibrationError(ValueError):
    pass


def sample_locations(ledger: list[dict], rng: random.Random) -> list[dict]:
    tools = sorted({row["tool"] for row in ledger if row["tool"] not in BASELINES})
    per_tool = math.ceil(LOCATION_TARGET / len(tools)) if tools else 0
    sample = []
    for tool in tools:
        matches = sorted((r for r in ledger if r["tool"] == tool and r["distance"] is not None and r["distance"] <= 3),
                         key=lambda r: r["point"])
        for row in rng.sample(matches, min(per_tool, len(matches))):
            sample.append({"kind": "location", "tool": tool, "repo": row["repo"], "point": row["point"],
                           "pr": row["pr"], "head_sha": row["head_sha"], "finding": row["finding"], "verdict": None})
    return sample


def sample_acted_on(points: list[dict], rng: random.Random) -> list[dict]:
    """`points` carry their `repo`; quotas per (acted_on, basis) from ACTED_ON_QUOTAS."""
    by_id = {p["id"]: p for p in points}
    sample = []
    for value, basis, quota in ACTED_ON_QUOTAS:
        pool = sorted(pid for pid, p in by_id.items() if p.get("acted_basis") == basis
                      and p["acted_on"] is not None and (value is None or p["acted_on"] is value))
        for point_id in rng.sample(pool, min(quota, len(pool))):
            point = by_id[point_id]
            sample.append({"kind": "acted_on", "repo": point["repo"], "point": point_id, "pr": point["pr"],
                           "basis": basis, "acted_on": point["acted_on"], "verdict": None})
    return sample


def draw(ledger: list[dict], points: list[dict], seed: int) -> dict:
    rng = random.Random(seed)
    return {"seed": seed, "entries": sample_locations(ledger, rng) + sample_acted_on(points, rng)}


def write_calibration(calibration: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(calibration, sort_keys=False), encoding="utf-8")


def read_calibration(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationError(f"cannot read calibration {path}: {exc}") from exc
    for entry in data.get("entries", []):
        allowed = LOCATION_VERDICTS if entry["kind"] == "location" else ACTED_VERDICTS
        if entry.get("verdict") not in (None, *allowed):
            raise CalibrationError(f"{path}: {entry['point']} has verdict {entry['verdict']!r}; use {allowed}")
    return data


def summarise_calibration(calibration: dict | None) -> dict:
    """Per-entrant location verdicts, per-basis acted-on agreement, and whether the sample is complete."""
    if not calibration or not calibration.get("entries"):
        return {"validated": False, "location": {}, "acted_on": {}}
    entries = calibration["entries"]
    location: dict[str, Counter] = {}
    acted: dict[str, Counter] = {}
    for entry in entries:
        if entry["verdict"] is None:
            continue
        if entry["kind"] == "location":
            location.setdefault(entry["tool"], Counter())[entry["verdict"]] += 1
        else:
            acted.setdefault(entry["basis"], Counter())["agree" if entry["verdict"] == "yes" else "disagree"] += 1
    return {
        "validated": all(entry["verdict"] is not None for entry in entries),
        "location": {tool: dict(sorted(c.items())) for tool, c in sorted(location.items())},
        "acted_on": {basis: dict(sorted(c.items())) for basis, c in sorted(acted.items())},
    }
