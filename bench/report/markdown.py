"""Render the committed results page from score summaries. Every number comes from those files."""
from __future__ import annotations

from bench.labels.model_agreement import MIN_RATE_N
from bench.score.match import WINDOWS
from bench.score.metrics import wilson

MAIN_HEADER = ("| Entrant | Version | Same spot, N=3 (all) | Same spot, N=3 (acted on) | Same spot, N=3, "
               "blocking only (all) | False blocks (approved heads) | Blocks per approved head | "
               "Findings per 100 changed lines | Heads: error / unavailable / not scored / leaked | Median s | "
               "Cost (USD, list-price estimate from token counts, Claude Code's figure) | Label |")
MAIN_COLUMNS = 12
BLOCK_HEADER = ("| Entrant | Approved heads blocked | Merged-head fallback: blocked / heads | "
                "Overridden approvals (left out) | Must-not-block heads not scored |")


def pct(part: int, whole: int) -> str:
    """A rate with its n and 95% Wilson interval; never a bare percentage."""
    if not whole:
        return "n/a (n=0)"
    low, high = wilson(part, whole)
    return f"{100 * part / whole:.0f}% ({part}/{whole}; 95% CI {100 * low:.0f} to {100 * high:.0f}%)"


def value(number: float | int | None) -> str:
    return "n/a" if number is None else str(number)


NO_BLOCKING = "n/a (never blocks)"


def blocking_cells(metrics: dict) -> tuple[str, str, str]:
    """Blocking-only catch rate, false blocks, blocks per approved head; n/a for a tool that never blocks."""
    if metrics.get("catches_blocking") is None or metrics.get("false_blocks") is None:
        return NO_BLOCKING, NO_BLOCKING, NO_BLOCKING
    blocking = metrics["catches_blocking"]["3"]["all"]
    blocks = metrics["false_blocks"]
    return (pct(blocking["caught"], blocking["points"]),
            pct(blocks["approved_heads_blocked"], blocks["approved_heads"]),
            f"{value(blocks['blocks_per_approved_head'])} (n={blocks['approved_heads']} heads)")


def tool_row(name: str, metrics: dict) -> str:
    headline = metrics["catches"]["3"]
    heads = metrics["heads"]
    blocking_rate, false_block_rate, per_head = blocking_cells(metrics)
    cells = (
        name, value(metrics["version"]),
        pct(headline["all"]["caught"], headline["all"]["points"]),
        pct(headline["acted_on"]["caught"], headline["acted_on"]["points"]),
        blocking_rate,
        false_block_rate,
        per_head,
        f"{value(metrics['findings_per_100_changed_lines'])} (n={metrics['changed_lines']} lines)",
        f"{heads.get('error', 0)} / {heads.get('unavailable', 0)} / {heads.get('not_scored', 0)} / "
        f"{heads.get('leaked', 0)}",
        value(metrics["median_wall_s"]), value(metrics["cost_usd_total"]), metrics.get("label") or "",
    )
    return "| " + " | ".join(cells) + " |"


def block_row(name: str, metrics: dict) -> str:
    blocks = metrics["false_blocks"]
    if blocks is None:
        return f"| {name} | {NO_BLOCKING} | {NO_BLOCKING} | {NO_BLOCKING} | {NO_BLOCKING} |"
    merged = blocks["merged_fallback"]
    cells = (name, pct(blocks["approved_heads_blocked"], blocks["approved_heads"]),
             f"{merged['blocked']} / {merged['heads']}", str(blocks["overridden_excluded"]), str(blocks["not_scored"]))
    return "| " + " | ".join(cells) + " |"


def corpus_line(summary: dict) -> str:
    c = summary["corpus"]
    checks = ", ".join(f"{k} {v}" for k, v in c["review_commit_checks"].items())
    return (f"{c['prs']} PRs, {c['rounds']} rounds, {c['points_location_scorable']} location-scorable points "
            f"({c['points_acted_on']} acted on, {c['acted_on_unknown']} unknown), {c['kept_not_location_scorable']} "
            f"kept points not location-scorable, {c['approved_heads']} approved heads. Review commits: {checks}.")


def sensitivity(summary: dict) -> list[str]:
    lines = ["| Entrant | " + " | ".join(f"N={w}" for w in WINDOWS) + " |", "|---" * (len(WINDOWS) + 1) + "|"]
    for name, metrics in summary["tools"].items():
        cells = [pct(metrics["catches"][str(w)]["all"]["caught"], metrics["catches"][str(w)]["all"]["points"])
                 for w in WINDOWS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def class_table(classes: dict[str, dict]) -> list[str]:
    labels = sorted({label for per_tool in classes.values() for label in per_tool})
    lines = ["| Entrant | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for tool, per_tool in classes.items():
        cells = [pct(per_tool[l]["caught"], per_tool[l]["points"]) if l in per_tool else "n/a" for l in labels]
        lines.append(f"| {tool} | " + " | ".join(cells) + " |")
    return lines


def range_note(summary: dict, calibration: dict) -> str:
    """When most acted-on points come from the `range` basis, say so next to them, with its agreement."""
    by_basis = summary["corpus"].get("acted_on_by_basis", {})
    acted = summary["corpus"]["points_acted_on"]
    on_range = by_basis.get("range", 0)
    if not acted or on_range * 2 <= acted:
        return ""
    counts = calibration.get("acted_on", {}).get("range")
    agreement = (f"agree {counts.get('agree', 0)}, disagree {counts.get('disagree', 0)}" if counts
                 else "not yet hand-checked")
    return (f"Acted-on points here rest mostly on the `range` basis ({on_range} of {acted}); "
            f"range calibration agreement (all repos): {agreement}.")


def model_line(agreement: dict | None) -> list[str]:
    """How far the labels agree with the model suggestions; a rate only on the blind subset."""
    if not agreement:
        return []
    blind = agreement["blind"]
    rate = (f"{blind['rate']:.0%}" if blind["rate"] is not None else f"too few for a rate (n<{MIN_RATE_N})")
    pre = (f" ({blind['pre_suggestion']} of them blind: pre-suggestion, from a sample answered in full before "
           "any suggestion existed)" if blind.get("pre_suggestion") else "")
    anchored = agreement["anchored"]
    return [f"Labels were assisted by model suggestions ({agreement['model']}). On blind labels, made without "
            f"seeing them, labels agree with the model on {blind['agree']} of {blind['points']}{pre}: {rate}. "
            f"Where the suggestion was shown (anchored), the labeller accepted {anchored['accepted']} and "
            f"overrode {anchored['overridden']}; {agreement['unseen']} other labels never saw one.", ""]


def repo_section(summary: dict, classes: dict[str, dict], class_note: str = "", calibration: dict | None = None,
                 agreement: dict | None = None) -> list[str]:
    lines = [f"## {summary['repo']}", "", f"Pinned at `{summary['pin'][:12]}`. {corpus_line(summary)}", ""]
    if not summary["reportable"]:
        mins, c = summary["minimums"], summary["corpus"]
        return lines + [f"**Insufficient data:** {c['points_acted_on']} acted-on points (minimum "
                        f"{mins['acted_on_points']}) and {c['approved_heads']} approved heads (minimum "
                        f"{mins['approved_heads']}). No scores are reported for this repository, and it "
                        "is left out of the calibration sample.", ""]
    lines += [MAIN_HEADER, "|---" * MAIN_COLUMNS + "|", *(tool_row(n, m) for n, m in summary["tools"].items()), ""]
    note = range_note(summary, calibration or {})
    if note:
        lines += [note, ""]
    lines += ["False blocks in detail:", "", BLOCK_HEADER, "|---" * 5 + "|",
              *(block_row(n, m) for n, m in summary["tools"].items()), ""]
    lines += ["Sensitivity (all location-scorable points):", "", *sensitivity(summary), ""]
    if class_note:
        lines += [f"Results by class withheld: {class_note}.", ""]
    elif classes:
        lines += ["By class (N=3), from the labelled random sample only; points outside it are unclassified. "
                  "Each class has few points, so these intervals are wide: read them as rough, not as rankings.",
                  "", *class_table(classes), "", *model_line(agreement)]
    return lines


def calibration_status(calibration: dict) -> str:
    status = "complete" if calibration["validated"] else "incomplete, so the headline is **unvalidated**"
    short = f" ({'; '.join(calibration['short'])})" if calibration.get("short") else ""
    return f"Hand-checked sample: {status}{short}."


def calibration_section(calibration: dict) -> list[str]:
    lines = ["## Calibration", "", calibration_status(calibration), ""]
    for tool, counts in calibration["location"].items():
        lines.append(f"- {tool}, same issue at a location match: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for basis, counts in calibration["acted_on"].items():
        lines.append(f"- acted-on ({basis}): agree {counts.get('agree', 0)}, disagree {counts.get('disagree', 0)}")
    return lines + [""]


def render(run_name: str, summaries: list[dict], classes: dict[str, dict], calibration: dict,
           class_notes: dict[str, str] | None = None, agreements: dict[str, dict | None] | None = None) -> str:
    method = summaries[0]["method_version"] if summaries else "n/a"
    lines = [f"# DriftBench results: {run_name}", "", f"Method version {method} (docs/SPEC.md). Generated by "
             "`python -m bench report` from the score summaries in this directory.", "",
             "Every entrant runs cold: no learned lessons, team memory or past reviews. Rigour's reviewer "
             "learns from a team's history; this run doesn't measure that.", "",
             f"**Status:** {calibration_status(calibration)}", ""]
    for summary in summaries:
        lines += repo_section(summary, classes.get(summary["repo"], {}), (class_notes or {}).get(summary["repo"], ""),
                              calibration, (agreements or {}).get(summary["repo"]))
    lines += calibration_section(calibration)
    return "\n".join(lines).rstrip() + "\n"
