"""Confusion-matrix metrics and a publish gate that a useless model cannot pass.

A model that answers "drift" to everything has perfect recall, and one that
answers "no drift" has zero false positives; accuracy alone hid both. The gate
uses balanced accuracy, caps the false-positive rate, rejects constant answers,
and counts unparseable replies as wrong.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Prediction:
    item_id: str
    expected: bool
    #: None when the reply could not be parsed.
    predicted: bool | None


@dataclass(frozen=True)
class Thresholds:
    min_balanced_accuracy: float = 0.6
    max_false_positive_rate: float = 0.2
    max_unparseable_rate: float = 0.1


@dataclass(frozen=True)
class Report:
    total: int
    tp: int
    fp: int
    tn: int
    fn: int
    unparseable: int
    precision: float | None
    recall: float | None
    false_positive_rate: float | None
    balanced_accuracy: float
    constant_answer: bool

    def to_dict(self) -> dict:
        return asdict(self)


def report(predictions: list[Prediction]) -> Report:
    tp = sum(1 for p in predictions if p.expected and p.predicted is True)
    fn = sum(1 for p in predictions if p.expected and p.predicted is not True)
    tn = sum(1 for p in predictions if not p.expected and p.predicted is False)
    fp = sum(1 for p in predictions if not p.expected and p.predicted is not False)
    ratio = lambda n, d: n / d if d else None  # noqa: E731
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    answered = {p.predicted for p in predictions if p.predicted is not None}
    return Report(
        total=len(predictions), tp=tp, fp=fp, tn=tn, fn=fn,
        unparseable=sum(1 for p in predictions if p.predicted is None),
        precision=ratio(tp, tp + fp), recall=recall, false_positive_rate=ratio(fp, fp + tn),
        balanced_accuracy=((recall or 0.0) + (specificity or 0.0)) / 2,
        constant_answer=len(answered) <= 1,
    )


def gate(r: Report, thresholds: Thresholds = Thresholds()) -> list[str]:
    """Reasons the model must not be published; empty when it passes."""
    failures = []
    if r.constant_answer:
        failures.append("gives the same answer to every item")
    if r.balanced_accuracy < thresholds.min_balanced_accuracy:
        failures.append(f"balanced accuracy {r.balanced_accuracy:.2f} < {thresholds.min_balanced_accuracy:.2f}")
    if r.false_positive_rate is None or r.false_positive_rate > thresholds.max_false_positive_rate:
        failures.append(f"false-positive rate {_pct(r.false_positive_rate)} > {thresholds.max_false_positive_rate:.0%}")
    if r.total and r.unparseable / r.total > thresholds.max_unparseable_rate:
        failures.append(f"{r.unparseable}/{r.total} replies unparseable")
    return failures


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"
