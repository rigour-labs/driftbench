"""Score every tool on one repository; produce the summary and the match ledger."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from bench.harness.runner import read_record
from bench.repos import slug_of
from bench.score.linemap import FileVersions
from bench.score.match import HEADLINE_WINDOW, Match, closest_match, eligible_heads, location_scorable
from bench.score.metrics import catch_rates, false_blocks, volume_and_time

METHOD_VERSION = 1
MIN_ACTED_ON_POINTS = 20
MIN_APPROVED_HEADS = 10
NOISE_CATCH_SHARE = 0.75   # NOISE-1: catch rate at least this share of every-hunk's...
NOISE_VOLUME_SHARE = 0.5   # ...and finding volume at least this share of every-hunk's
CEILING = "every-hunk"


def load_records(run_dir: Path, tool: str, repo: str) -> dict[int, list[dict]]:
    by_pr: dict[int, list[dict]] = {}
    for path in sorted((run_dir / tool / slug_of(repo)).glob("*/*.json")):
        record = read_record(path)
        by_pr.setdefault(record["pr"], []).append(record)
    return by_pr


def ledger_row(tool: str, point: dict, match: Match | None, blocking: Match | None) -> dict:
    """One match decision: the closest finding (if any), and the closest blocking one, for one point and tool."""
    row = {"tool": tool, "repo": point["repo"], "point": point["id"], "pr": point["pr"], "round": point["round"],
           "acted_on": point["acted_on"], "distance": None,
           "blocking_distance": blocking.distance if blocking else None}
    if match:
        row.update(head_sha=match.head_sha, finding=match.finding_index, mapped_line=match.mapped_line,
                   distance=match.distance)
    return row


def score_tool(tool: str, corpus: dict, points: list[dict], run_dir: Path, versions: FileVersions) -> tuple[dict, list]:
    by_pr = load_records(run_dir, tool, corpus["repo"])
    rounds = {pr["number"]: pr["rounds"] for pr in corpus["prs"]}
    matches, blocking, ledger = {}, {}, []
    for point in points:
        records = {r["head_sha"]: r for r in by_pr.get(point["pr"], [])}
        heads = eligible_heads(point, rounds[point["pr"]])
        matches[point["id"]] = closest_match(point, heads, records, versions)
        blocking[point["id"]] = closest_match(point, heads, records, versions, blocking_only=True)
        ledger.append(ledger_row(tool, point, matches[point["id"]], blocking[point["id"]]))
    acted = {p["id"] for p in points if p["acted_on"] is True}
    all_records = [r for records in by_pr.values() for r in records]
    tool_version = all_records[0]["tool_version"] if all_records else None
    metrics = {"version": tool_version, "catches": catch_rates(matches, acted),
               "catches_blocking": catch_rates(blocking, acted),
               "false_blocks": false_blocks(by_pr), **volume_and_time(all_records)}
    return metrics, ledger


def noise_labels(tools: dict[str, dict]) -> None:
    """NOISE-1: mark tools whose catches and volume come close to the every-hunk ceiling."""
    ceiling = tools.get(CEILING)
    for name, metrics in tools.items():
        if name == CEILING or ceiling is None:
            metrics["label"] = "ceiling" if name == CEILING else None
            continue
        top = ceiling["catches"][str(HEADLINE_WINDOW)]["all"]["rate"] or 0
        own = metrics["catches"][str(HEADLINE_WINDOW)]["all"]["rate"] or 0
        top_volume = ceiling["findings_per_100_changed_lines"] or 0
        own_volume = metrics["findings_per_100_changed_lines"] or 0
        close = top and own >= NOISE_CATCH_SHARE * top and own_volume >= NOISE_VOLUME_SHARE * top_volume
        metrics["label"] = "noise" if close else None


def corpus_counts(corpus: dict, points_file: dict, scorable: list[dict]) -> dict:
    kept = [p for p in points_file["points"] if not p["dropped"]]
    reviews = [r for pr in corpus["prs"] for r in pr["reviews"]]
    return {
        "prs": len(corpus["prs"]),
        "rounds": sum(len(pr["rounds"]) for pr in corpus["prs"]),
        "review_commit_checks": dict(sorted(Counter(r["commit_check"] for r in reviews).items())),
        "points_kept": len(kept),
        "points_location_scorable": len(scorable),
        "points_acted_on": sum(1 for p in scorable if p["acted_on"] is True),
        "acted_on_unknown": sum(1 for p in scorable if p["acted_on"] is None),
        "kept_not_location_scorable": len(kept) - len(scorable),
        "kept_no_round": sum(1 for p in kept if p["round"] is None),
        "approved_heads": sum(1 for pr in corpus["prs"] if pr["approved_head_sha"] and not pr["approval_overridden"]),
    }


def score_repo(corpus: dict, points_file: dict, run_dir: Path, tools: list[str], versions: FileVersions) -> tuple[dict, list]:
    scorable = [{**p, "repo": corpus["repo"]} for p in points_file["points"] if location_scorable(p)]
    results, ledger = {}, []
    for tool in tools:
        results[tool], rows = score_tool(tool, corpus, scorable, run_dir, versions)
        ledger += rows
    noise_labels(results)
    counts = corpus_counts(corpus, points_file, scorable)
    reportable = counts["points_acted_on"] >= MIN_ACTED_ON_POINTS and counts["approved_heads"] >= MIN_APPROVED_HEADS
    summary = {"repo": corpus["repo"], "pin": corpus["pin"], "method_version": METHOD_VERSION,
               "minimums": {"acted_on_points": MIN_ACTED_ON_POINTS, "approved_heads": MIN_APPROVED_HEADS},
               "reportable": reportable, "corpus": counts, "tools": results}
    return summary, ledger


def published_summary(summary: dict) -> dict:
    """What goes in results/: everything when reportable; otherwise counts and the reason, no tool metrics."""
    if summary["reportable"]:
        return summary
    corpus, mins = summary["corpus"], summary["minimums"]
    reason = (f"{corpus['points_acted_on']} acted-on points (minimum {mins['acted_on_points']}) and "
              f"{corpus['approved_heads']} approved heads (minimum {mins['approved_heads']})")
    kept = ("repo", "pin", "method_version", "minimums", "reportable", "corpus")
    return {**{key: summary[key] for key in kept}, "insufficient_data": reason}
