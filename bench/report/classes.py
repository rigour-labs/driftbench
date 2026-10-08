"""Catch rates per class (docs/SPEC.md, "Labels"), from confirmed, non-stale labels only."""
from __future__ import annotations

from collections import Counter, defaultdict

from bench.labels.rules import CLASSES
from bench.score.match import HEADLINE_WINDOW

UNCLASSIFIED = "unclassified"


def class_of(point_id: str, labels: dict[str, str]) -> str:
    return labels.get(point_id, UNCLASSIFIED)


def per_class(ledger: list[dict], labels: dict[str, str]) -> dict[str, dict[str, dict]]:
    """{tool: {class: {caught, points}}} at the headline window, over every location-scorable point."""
    totals: dict[str, Counter] = defaultdict(Counter)
    caught: dict[str, Counter] = defaultdict(Counter)
    for row in ledger:
        label = class_of(row["point"], labels)
        totals[row["tool"]][label] += 1
        if row["distance"] is not None and row["distance"] <= HEADLINE_WINDOW:
            caught[row["tool"]][label] += 1
    order = (*CLASSES, UNCLASSIFIED)
    return {
        tool: {label: {"caught": caught[tool][label], "points": totals[tool][label]}
               for label in order if totals[tool][label]}
        for tool in sorted(totals)
    }
