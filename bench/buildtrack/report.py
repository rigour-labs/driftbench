"""The build track's numbers, per arm and paired (docs/BUILD_TRACK.md, "Measures").

Inputs: one record per task (both arms' agent runs, hidden-test outcomes,
Rigour's events and setup in arm B, the reference check and false blocks)
and the judge's verdicts per human point. Numbers only.
"""
from __future__ import annotations

from bench.buildtrack.arms import ARMS
from bench.score.metrics import wilson

OUTCOMES = ("pass", "fail", "no-build", "no-tests")


def repeated(judgments: dict, arm: str) -> dict:
    """Points this arm's diff repeated, out of those it applied to (repeated or avoided)."""
    verdicts = [j["verdicts"][arm]["verdict"] for j in judgments.values() if arm in j.get("verdicts", {})]
    hit, applicable = verdicts.count("repeated"), verdicts.count("repeated") + verdicts.count("avoided")
    return {"repeated": hit, "applicable": applicable, "not_applicable": verdicts.count("not_applicable"),
            "rate": round(hit / applicable, 3) if applicable else None, "ci95": wilson(hit, applicable)}


def paired(judgments: dict) -> dict:
    """Points applicable in both arms: repeated by alone only, by rigour only, by both, by neither."""
    counts = {"alone_only": 0, "rigour_only": 0, "both": 0, "neither": 0}
    for j in judgments.values():
        v = {arm: j.get("verdicts", {}).get(arm, {}).get("verdict") for arm in ARMS}
        if not all(x in ("repeated", "avoided") for x in v.values()):
            continue
        a, r = v["alone"] == "repeated", v["rigour"] == "repeated"
        counts["both" if a and r else "alone_only" if a else "rigour_only" if r else "neither"] += 1
    return counts


def arm_numbers(tasks: list[dict], arm: str) -> dict:
    runs = [t["arms"][arm] for t in tasks if arm in t.get("arms", {})]
    costs = [r["agent"].get("cost_usd") or 0.0 for r in runs]
    outcomes = {o: sum(1 for r in runs if (r.get("tests") or {}).get("outcome") == o) for o in OUTCOMES}
    numbers = {"tasks": len(runs), "errors": sum(1 for r in runs if r["agent"].get("error")),
               "leaked": sum(1 for r in runs if r["agent"].get("leak_signals")),
               "timed_out": sum(1 for r in runs if r["agent"].get("timed_out")), "tests": outcomes,
               "cost_usd": round(sum(costs), 4), "cost_per_task": round(sum(costs) / len(runs), 4) if runs else None,
               "turns": sum(r["agent"].get("turns") or 0 for r in runs),
               "wall_s": round(sum(r["agent"].get("wall_s") or 0 for r in runs), 1)}
    if arm == "rigour":
        numbers["rigour_events"] = sum((r.get("rigour") or {}).get("events", {}).get("events", 0) for r in runs)
        numbers["rigour_blocks"] = sum(sum(t.get("blocked", 0) for t in
                                           ((r.get("rigour") or {}).get("events", {}).get("by_type") or {}).values())
                                       for r in runs)
        numbers["false_blocks"] = sum((t.get("reference") or {}).get("false_blocks", {}).get("false_blocks", 0)
                                      for t in tasks)
    return numbers


def summary(tasks: list[dict], judgments: dict) -> dict:
    return {"tasks": len(tasks),
            "arms": {arm: {**arm_numbers(tasks, arm), "points": repeated(judgments, arm)} for arm in ARMS},
            "paired_points": paired(judgments),
            "judge_errors": sum(1 for j in judgments.values() if j.get("error"))}
