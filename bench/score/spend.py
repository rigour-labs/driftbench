"""What a run spent, per entrant, for the release notes (docs/ENTRANTS.md, "Hard dollar stop").

The dollars are Claude Code's list-price estimate from token counts, for
both tool families, not a bill. A head charged at its per-head bound (a
failure, or a model that ran without reporting a cost) is counted at that
bound. The billed amount comes from the provider's console afterwards and is
written next to the estimate by the maintainer.
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


def spend_notes(spend: dict[str, dict]) -> str:
    lines = ["Spend: list-price estimate from token counts (Claude Code's figure, both tool families); "
             "billed = the Anthropic Console amount for the run window, filled in by the maintainer.", ""]
    for name, tool in spend.items():
        if not tool["estimated_usd"] and not tool["charged_at_bound"]:
            continue
        heads = tool["heads"]
        lines.append(f"- {name}: estimated ${tool['estimated_usd']:.2f} ({tool['charged_at_bound']} head(s) at the "
                     f"bound), not scored {heads.get('not_scored', 0)}, leaked {heads.get('leaked', 0)}; billed: $____")
    return "\n".join(lines) + "\n"
