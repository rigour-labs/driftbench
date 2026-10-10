"""Issue-level results per repository and entrant, and the judge's self-consistency."""
from __future__ import annotations

from collections import Counter

from bench.score.metrics import wilson


def per_entrant(judgments: dict[str, dict], repo: str) -> dict[str, dict]:
    """{entrant: {yes, partly, no, missing, judged, rate, ci95}}: rate = yes over judged points."""
    tally: dict[str, Counter] = {}
    for item in judgments.values():
        if item["repo"] != repo or item.get("error"):
            continue
        for entrant, verdict in item["verdicts"].items():
            tally.setdefault(entrant, Counter())[verdict["verdict"]] += 1
    out = {}
    for entrant, counts in sorted(tally.items()):
        judged = counts["yes"] + counts["partly"] + counts["no"]
        out[entrant] = {**{k: counts[k] for k in ("yes", "partly", "no", "missing")}, "judged": judged,
                        "rate": round(counts["yes"] / judged, 4) if judged else None,
                        "ci95": wilson(counts["yes"], judged)}
    return out


def consistency(judgments: dict[str, dict], repeats: dict[str, dict]) -> dict:
    """How often the judge gave the same verdict when asked the same point again."""
    same = total = 0
    for point_id, again in repeats.items():
        first = judgments.get(point_id) or {}
        if first.get("error") or again.get("error"):
            continue
        for entrant, verdict in again["verdicts"].items():
            if verdict["verdict"] == "missing":
                continue
            total += 1
            same += verdict["verdict"] == first["verdicts"][entrant]["verdict"]
    return {"same": same, "asked": total, "rate": round(same / total, 4) if total else None}
