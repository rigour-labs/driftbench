"""AI-consensus labels (docs/LABELLING.md, "AI-consensus labels").

A point gets a class only when two independent labellers from different
model families agree: a Claude labeller's blind labels (another labels
directory) and the non-Claude model's suggestions (bench/labels/model_file.py).
Where they disagree, or one has no class, the point stays unclassified. The
label file records both sources, their agreement and kappa, and how many
points each class lost to disagreement, so the report can show it.
"""
from __future__ import annotations

from collections import Counter

from bench.labels.agreement import cohens_kappa
from bench.labels.store import LabelError, confirm, empty_labels, text_sha256

CLAUDE_LABELLER_ROLE = "a Claude model working as DriftBench's builder, which knows the benchmark and its guide"


def is_consensus(labels: dict) -> bool:
    return bool(labels.get("consensus"))


def dropped_by_class(first: dict[str, str], second: dict[str, str]) -> dict[str, dict[str, int]]:
    """Per class, how many disagreed points each labeller had put there."""
    disagreed = [pid for pid in set(first) & set(second) if first[pid] != second[pid]]
    return {"first": dict(sorted(Counter(first[pid] for pid in disagreed).items())),
            "second": dict(sorted(Counter(second[pid] for pid in disagreed).items()))}


def build_consensus(repo: str, first: dict, model_data: dict, sample_ids: list[str],
                    texts: dict[str, str | None]) -> dict:
    """Label file holding only the classes both labellers gave; `first` is the Claude labeller's label file."""
    sampled = set(sample_ids)
    claude = {pid: e["label"] for pid, e in first["points"].items()
              if e.get("label") and pid in sampled and texts.get(pid) is not None
              and text_sha256(texts[pid]) == e.get("text_sha256")}
    model = {pid: e["suggested"] for pid, e in model_data["points"].items() if e.get("suggested") and pid in sampled}
    labellers = sorted({e["labeller"] for e in first["points"].values() if e.get("label")})
    if len(labellers) != 1:
        raise LabelError(f"{repo}: the Claude labels need exactly one labeller, found {labellers or 'none'}")
    name = f"consensus: {labellers[0]} + {model_data['model']}"
    labels = empty_labels(repo)
    for pid in sorted(set(claude) & set(model)):
        if claude[pid] == model[pid]:
            labels = confirm(labels, pid, claude[pid], name, {"text_sha256": text_sha256(texts[pid]), "blind": True})
    agreement = cohens_kappa(claude, model)
    return {**labels, "consensus": {
        "sources": {"first": labellers[0], "second": model_data["model"]},
        "first_role": CLAUDE_LABELLER_ROLE,
        "both_labelled": agreement["points"], "agreed": len(labels["points"]), "kappa": agreement["kappa"],
        "dropped_by_class": dropped_by_class(claude, model),
    }}
