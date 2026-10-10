"""Per-tool numbers for one repository (docs/SPEC.md, "Reporting rules")."""
from __future__ import annotations

import math
from collections import Counter
from statistics import median

from bench.score.match import SCORED_VERDICTS, WINDOWS, Match


Z_95 = 1.959964


def rate(hit: int, total: int) -> float | None:
    return round(hit / total, 4) if total else None


def wilson(hit: int, total: int) -> list[float] | None:
    """95% Wilson score interval for hit/total; None when total is 0."""
    if not total:
        return None
    p = hit / total
    denominator = 1 + Z_95 ** 2 / total
    centre = (p + Z_95 ** 2 / (2 * total)) / denominator
    half = Z_95 * math.sqrt(p * (1 - p) / total + Z_95 ** 2 / (4 * total ** 2)) / denominator
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def proportion(hit: int, total: int) -> dict:
    return {"caught": hit, "points": total, "rate": rate(hit, total), "ci95": wilson(hit, total)}


def catch_rates(matches: dict[str, Match | None], acted_ids: set[str]) -> dict:
    """Catches at each window, over all location-scorable points and over acted-on ones."""
    result = {}
    for window in WINDOWS:
        caught = {pid for pid, m in matches.items() if m is not None and m.distance <= window}
        result[str(window)] = {
            "all": proportion(len(caught), len(matches)),
            "acted_on": proportion(len(caught & acted_ids), len(acted_ids)),
        }
    return result


def must_not_block_record(records: list[dict]) -> tuple[dict, dict] | None:
    """(record, case) for the PR's must-not-block head, if it was run."""
    for record in records:
        for case in record["cases"]:
            if case["kind"] == "must_not_block":
                return record, case
    return None


def false_blocks(records_by_pr: dict[int, list[dict]]) -> dict:
    """Blocking findings on approved heads; overridden approvals left out; merged fallbacks separate."""
    tally = Counter()
    for records in records_by_pr.values():
        found = must_not_block_record(records)
        if found is None:
            continue
        record, case = found
        if record["verdict"] not in SCORED_VERDICTS:
            tally["not_scored"] += 1
            continue
        blocks = sum(1 for f in record["findings"] if f["blocking"])
        group = "overridden" if case["approval_overridden"] else case["source"]
        tally[f"{group}_heads"] += 1
        tally[f"{group}_blocked"] += 1 if blocks else 0
        tally[f"{group}_blocks"] += blocks
    approved = tally["approved_heads"]
    return {
        "approved_heads": approved,
        "approved_heads_blocked": tally["approved_blocked"],
        "false_block_rate": rate(tally["approved_blocked"], approved),
        "false_block_ci95": wilson(tally["approved_blocked"], approved),
        "blocks_per_approved_head": round(tally["approved_blocks"] / approved, 3) if approved else None,
        "merged_fallback": {"heads": tally["merged_heads"], "blocked": tally["merged_blocked"]},
        "overridden_excluded": tally["overridden_heads"],
        "not_scored": tally["not_scored"],
    }


def uncited(records: list[dict]) -> dict:
    """Heads whose answer cited no place of any shape (from the numbers-only diagnostics), where recorded."""
    read = [r["diagnostics"] for r in records if isinstance(r.get("diagnostics"), dict)]
    if not read:
        return {}
    return {"uncited_heads": sum(1 for d in read if not d.get("citation_like")), "diagnosed_heads": len(read)}


def volume_and_time(records: list[dict]) -> dict:
    """Noise and cost over every distinct head the tool reviewed."""
    scored = [r for r in records if r["verdict"] in SCORED_VERDICTS]
    lines = sum(r["changed_lines"] or 0 for r in scored)
    findings = sum(len(r["findings"]) for r in scored)
    costs = [r["cost_usd"] for r in scored if r.get("cost_usd") is not None]
    walls = [r["wall_s"] for r in scored if r.get("wall_s") is not None]
    return {
        "heads": dict(sorted(Counter(r["verdict"] for r in records).items())),
        "findings": findings,
        "changed_lines": lines,
        "findings_per_100_changed_lines": round(100 * findings / lines, 2) if lines else None,
        "median_wall_s": round(median(walls), 2) if walls else None,
        "cost_usd_total": round(sum(costs), 4) if costs else None,
        "tokens_total": sum((r.get("input_tokens") or 0) + (r.get("output_tokens") or 0) for r in scored) or None,
        **uncited(scored),
    }
