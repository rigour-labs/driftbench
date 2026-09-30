"""Scoring review tools against the same PRs.

Every location is on the PR's merge commit. A finding "hits" a bug when it
is in the same file within `tolerance` lines of a line the later fix
changed. Metrics are per tool, with 95% bootstrap intervals over PRs:

- recall: share of later-fixed bugs the tool pointed at (unbiased: no
  reviewer caused those fixes);
- hit_rate: share of the tool's comments that point at a later-fixed bug,
  matched one-to-one so extra comments around one bug count as noise; a
  lower bound on precision (a correct comment about a bug nobody fixed
  later does not count);
- precision: from hand labels, when a labelled sample exists;
- comments_per_pr.

With judged verdicts (arena.verdicts), only bugs judged real are counted and a
finding hits a bug only if it was judged to describe it: proximity alone
credits comments that land near a bug but are about something else.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

DEFAULT_TOLERANCE = 3
BOOTSTRAP_ROUNDS = 1000

SOURCE = re.compile(r"\.(?:[cm]?[jt]sx?|py|go|rs|java|kt|rb|cs|php|swift|scala|vue|svelte)$")
TEST = re.compile(r"(?:^|/)(?:__tests__|__mocks__|tests?|e2e|fixtures?)/|\.(?:test|spec)\.[a-z]+$")


def in_scope(path: str, scope: str) -> bool:
    """`code`: non-test source files, where bugs ship; `all`: every file (docs, config, tests)."""
    return scope == "all" or (bool(SOURCE.search(path)) and not TEST.search(path))


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    #: Hand label when sampled: True correct, False wrong, None unlabelled.
    correct: bool | None = None
    #: Stable id within the tool's results for the PR (a comment id, a rule and location).
    id: str = ""
    message: str = ""


@dataclass
class PrResult:
    pr: str
    #: Each bug: (path, lines in the merge commit[, fixing commit]).
    bugs: list[tuple] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


@dataclass
class Metric:
    value: float | None
    low: float | None
    high: float | None


@dataclass
class ToolScore:
    prs: int
    bugs: int
    comments: int
    recall: Metric
    hit_rate: Metric
    precision: Metric
    comments_per_pr: Metric


#: (PR, finding, bug) -> whether the finding describes that bug.
Describes = Callable[[str, Finding, tuple], bool]


def hits(finding: Finding, bug: tuple, tolerance: int = DEFAULT_TOLERANCE) -> bool:
    path, lines = bug[0], bug[1]
    return finding.path == path and any(abs(finding.line - line) <= tolerance for line in lines)


def _counts(result: PrResult, tolerance: int, describes: Describes | None) -> tuple[int, int, int, int, int, int]:
    """(bugs, bugs hit, comments, comments hitting a bug, labelled, labelled correct)."""
    caught = _matched(result, tolerance, describes)
    labelled = [f for f in result.findings if f.correct is not None]
    return len(result.bugs), caught, len(result.findings), caught, len(labelled), sum(1 for f in labelled if f.correct)


def _matched(result: PrResult, tolerance: int, describes: Describes | None = None) -> int:
    """Bugs matched one-to-one: each bug is claimed by at most one comment, each comment
    claims at most one bug, so a burst of comments around one bug earns one hit."""
    used: set[int] = set()
    matched = 0
    for bug in result.bugs:
        for index, finding in enumerate(result.findings):
            if index not in used and hits(finding, bug, tolerance) and (describes is None or describes(result.pr, finding, bug)):
                used.add(index)
                matched += 1
                break
    return matched


def _ratios(rows: list[tuple[int, int, int, int, int, int]]) -> tuple[float | None, float | None, float | None, float]:
    bugs, caught, comments, on_bug, labelled, correct = (sum(col) for col in zip(*rows)) if rows else (0,) * 6
    ratio = lambda n, d: n / d if d else None  # noqa: E731
    return ratio(caught, bugs), ratio(on_bug, comments), ratio(correct, labelled), comments / len(rows) if rows else 0.0


def score(results: list[PrResult], tolerance: int = DEFAULT_TOLERANCE, seed: int = 7,
          describes: Describes | None = None) -> ToolScore:
    rows = [_counts(r, tolerance, describes) for r in results]
    point = _ratios(rows)
    rng = random.Random(seed)
    samples = [_ratios([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(BOOTSTRAP_ROUNDS)] if rows else []
    metrics = [_interval(point[i], [s[i] for s in samples]) for i in range(4)]
    return ToolScore(
        prs=len(results), bugs=sum(r[0] for r in rows), comments=sum(r[2] for r in rows),
        recall=metrics[0], hit_rate=metrics[1], precision=metrics[2], comments_per_pr=metrics[3],
    )


def _interval(value: float | None, draws: list[float | None]) -> Metric:
    values = sorted(d for d in draws if d is not None)
    if value is None or not values:
        return Metric(value, None, None)
    return Metric(value, values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))])
