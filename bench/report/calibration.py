"""Hand-checked samples that validate the automatic decisions (docs/SPEC.md, "Calibration").

Two samples, drawn with a fixed seed so anyone can redraw the same one:
- location matches (distance <= 3) per real entrant, judged "same issue":
  yes / partly / no; this gives each entrant's location-to-issue rate;
- acted-on decisions (10 direct-true, 5 direct-false, 5 ancestor, 5 range-true,
  5 range-false; half of each range quota from the repo that relies on range
  most), judged
  "did the change near the anchor respond to the point": yes / no; this
  gives agreement per ACTED-1 basis.
A human fills `verdict`, or AI verdicts are merged in (bench/report/calibration_ai.py)
and each records `verdict_by`; an acted-on entry the two AI judges disagree on
is `disputed` and left out. Until every entry has a verdict or is disputed,
the headline is unvalidated.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

import yaml

BASELINES = ("no-tool", "every-hunk")
LOCATION_TARGET = 50
ACTED_ON_QUOTAS = ((True, "direct", 10), (False, "direct", 5), (None, "ancestor", 5),
                   (True, "range", 5), (False, "range", 5))
FOCUS_BASIS = "range"  # half of each range quota (rounded up) comes from the repo that relies on it most
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


def range_focus_repo(points: list[dict]) -> str | None:
    """The repo whose acted-on decisions rest most on FOCUS_BASIS (largest share), or None if none use it."""
    shares = {}
    for repo in {p["repo"] for p in points}:
        decided = [p for p in points if p["repo"] == repo and p["acted_on"] is not None]
        focus = sum(1 for p in decided if p.get("acted_basis") == FOCUS_BASIS)
        if focus:
            shares[repo] = focus / len(decided)
    return max(sorted(shares), key=shares.get) if shares else None


def draw_quota(pool: list[dict], quota: int, focus: str | None, rng: random.Random) -> list[dict]:
    """`quota` points from `pool`; for FOCUS_BASIS, half (rounded up) from the focus repo first."""
    if focus is None or not pool or pool[0].get("acted_basis") != FOCUS_BASIS:
        return rng.sample(pool, min(quota, len(pool)))
    from_focus = [p for p in pool if p["repo"] == focus]
    chosen = rng.sample(from_focus, min(math.ceil(quota / 2), len(from_focus)))
    rest = [p for p in pool if p not in chosen]
    return chosen + rng.sample(rest, min(quota - len(chosen), len(rest)))


def sample_acted_on(points: list[dict], rng: random.Random) -> list[dict]:
    """`points` carry their `repo`; quotas per (acted_on, basis) from ACTED_ON_QUOTAS."""
    focus = range_focus_repo(points)
    sample = []
    for value, basis, quota in ACTED_ON_QUOTAS:
        pool = sorted((p for p in points if p.get("acted_basis") == basis and p["acted_on"] is not None
                       and (value is None or p["acted_on"] is value)), key=lambda p: p["id"])
        for point in draw_quota(pool, quota, focus, rng):
            sample.append({"kind": "acted_on", "repo": point["repo"], "point": point["id"], "pr": point["pr"],
                           "basis": basis, "acted_on": point["acted_on"], "verdict": None})
    return sample


def shortfalls(entries: list[dict]) -> list[str]:
    """Where the pools were too small for the targets; shown with the results, never hidden."""
    notes = []
    located = sum(1 for e in entries if e["kind"] == "location")
    if located < LOCATION_TARGET:
        notes.append(f"short: {located} of {LOCATION_TARGET} location matches")
    for value, basis, quota in ACTED_ON_QUOTAS:
        drawn = sum(1 for e in entries if e["kind"] == "acted_on" and e["basis"] == basis
                    and (value is None or e["acted_on"] is value))
        if drawn < quota:
            label = f"{basis}" if value is None else f"{basis} {'yes' if value else 'no'}"
            notes.append(f"short: {drawn} of {quota} acted-on ({label})")
    return notes


def draw(ledger: list[dict], points: list[dict], seed: int) -> dict:
    """Sample from the ledger and points of reportable repositories only (the caller filters)."""
    rng = random.Random(seed)
    entries = sample_locations(ledger, rng) + sample_acted_on(points, rng)
    return {"seed": seed, "short": shortfalls(entries), "entries": entries}


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
        return {"validated": False, "short": [], "location": {}, "acted_on": {}, "judges": {}, "disputed": 0}
    entries = calibration["entries"]
    location: dict[str, Counter] = {}
    acted: dict[str, Counter] = {}
    for entry in entries:
        if entry["verdict"] is None:
            continue
        if entry["kind"] == "location":
            location.setdefault(entry["tool"], Counter())[entry["verdict"]] += 1
        else:
            # The judge answers whether the change responded; that agrees with the scorer when it matches the
            # scorer's own acted-on decision, whichever way that went.
            agrees = (entry["verdict"] == "yes") == bool(entry["acted_on"])
            acted.setdefault(entry["basis"], Counter())["agree" if agrees else "disagree"] += 1
    judges = Counter(entry.get("verdict_by") or "human" for entry in entries if entry["verdict"] is not None)
    return {
        "validated": all(entry["verdict"] is not None or entry.get("disputed") for entry in entries),
        "judges": dict(sorted(judges.items())),
        "disputed": sum(1 for entry in entries if entry.get("disputed")),
        "short": list(calibration.get("short", [])),
        "location": {tool: dict(sorted(c.items())) for tool, c in sorted(location.items())},
        "acted_on": {basis: dict(sorted(c.items())) for basis, c in sorted(acted.items())},
    }
