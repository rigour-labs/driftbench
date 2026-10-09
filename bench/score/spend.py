"""What a run spent, per entrant, for the release notes (docs/ENTRANTS.md, "Hard dollar stop").

The dollars are Claude Code's list-price estimate from token counts, for
both tool families, not a bill. A head charged at its per-head bound (a
failure, or a model that ran without reporting a cost) is counted at that
bound. Through Anthropic, the billed amount comes from the console afterwards
and is written next to the estimate by the maintainer. Through OpenRouter it
comes from data: each run job reads the key's cumulative usage before and
after, and the billed total is the latest end minus the earliest start, which
is exact when the key is used only for this run.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from bench.harness.runner import read_record


def spend_by_tool(run_dir: Path, budgets: list[dict]) -> dict[str, dict]:
    """{tool: {estimated_usd, reported_usd, heads, charged_at_bound}} over a run's records."""
    bound = {name: usd for budget in budgets for name, usd in budget.get("largest_per_head", {}).items()}
    estimate = {name: usd for budget in budgets for name, usd in budget.get("estimate_per_head", {}).items()}
    tools: dict[str, dict] = {}
    for path in sorted(run_dir.glob("*/*/*/*.json")):
        record = read_record(path)
        tool = tools.setdefault(record["tool"], {"reported_usd": 0.0, "charged_at_bound": 0, "heads": Counter()})
        tool["heads"][record["verdict"]] += 1
        if record.get("charged") == "bound":
            tool["charged_at_bound"] += 1
        elif record.get("cost_usd"):
            tool["reported_usd"] += record["cost_usd"]
    for name, tool in tools.items():
        per_bound = max(bound.get(name, 0.0), estimate.get(name, 0.0))
        tool["estimated_usd"] = round(tool["reported_usd"] + tool["charged_at_bound"] * per_bound, 4)
        tool["reported_usd"] = round(tool["reported_usd"], 4)
        tool["heads"] = dict(sorted(tool["heads"].items()))
    return dict(sorted(tools.items()))


def openrouter_billed(usages: list[dict]) -> dict:
    """{billed_usd, from, to} over every run job's usage readings, or billed_usd None with the reason."""
    starts = [u["start"] for u in usages if (u.get("start") or {}).get("usage_usd") is not None]
    ends = [u["end"] for u in usages if (u.get("end") or {}).get("usage_usd") is not None]
    if not starts or not ends:
        return {"billed_usd": None, "reason": "no readable OpenRouter usage before and after the run"}
    first, last = min(starts, key=lambda s: s["usage_usd"]), max(ends, key=lambda e: e["usage_usd"])
    return {"billed_usd": round(last["usage_usd"] - first["usage_usd"], 4), "from": min(s["at"] for s in starts),
            "to": max(e["at"] for e in ends)}


def billed_line(billed: dict, spend: dict[str, dict]) -> list[str]:
    estimated = round(sum(t["estimated_usd"] for t in spend.values()), 2)
    if billed["billed_usd"] is None:
        return ["", f"- all paid entrants: estimated ${estimated:.2f}; billed by OpenRouter: unavailable "
                    f"({billed['reason']})"]
    return ["", f"- all paid entrants: estimated ${estimated:.2f}; billed by OpenRouter ${billed['billed_usd']:.2f} "
                f"(the key's usage from {billed['from']} to {billed['to']}; exact when the key is used only "
                "for this run)"]


def spend_notes(spend: dict[str, dict], billed: dict | None = None) -> str:
    source = ("billed = OpenRouter's reported usage of the run's key, below" if billed is not None
              else "billed = the Anthropic Console amount for the run window, filled in by the maintainer")
    lines = ["Spend: list-price estimate from token counts (Claude Code's figure, both tool families); "
             f"{source}.", ""]
    for name, tool in spend.items():
        if not tool["estimated_usd"] and not tool["charged_at_bound"]:
            continue
        heads = tool["heads"]
        billed_cell = "" if billed is not None else "; billed: $____"
        lines.append(f"- {name}: estimated ${tool['estimated_usd']:.2f} ({tool['charged_at_bound']} head(s) at the "
                     f"bound), not scored {heads.get('not_scored', 0)}, leaked {heads.get('leaked', 0)}{billed_cell}")
    if billed is not None:
        lines += billed_line(billed, spend)
    return "\n".join(lines) + "\n"
