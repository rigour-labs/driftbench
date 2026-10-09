import pytest

from bench.labels.consensus import CLAUDE_LABELLER_ROLE, build_consensus, is_consensus
from bench.labels.store import LabelError, confirm, empty_labels, text_sha256

TEXTS = {f"p{i}": f"comment {i}" for i in range(6)}
CLAUDE = {"p0": "mechanical", "p1": "judgment", "p2": "claim/contract", "p3": "performance", "p4": "judgment"}
MODEL = {"p0": "mechanical", "p1": "mechanical", "p2": "claim/contract", "p3": "judgment", "p5": "user journey"}


def claude_labels(labeller="claude-opus-5-5", texts=TEXTS):
    labels = empty_labels("o/r")
    for pid, label in CLAUDE.items():
        labels = confirm(labels, pid, label, labeller, {"text_sha256": text_sha256(texts[pid]), "blind": True})
    return labels


def model_file():
    return {"model": "example-org/example-model", "points": {pid: {"suggested": c} for pid, c in MODEL.items()}}


def test_only_agreed_points_get_a_class_and_the_disagreements_are_counted_by_class():
    labels = build_consensus("o/r", claude_labels(), model_file(), list(TEXTS), TEXTS)
    assert {pid: e["label"] for pid, e in labels["points"].items()} == {"p0": "mechanical", "p2": "claim/contract"}
    entry = labels["points"]["p0"]
    assert entry["labeller"] == "consensus: claude-opus-5-5 + example-org/example-model" and entry["blind"] is True
    c = labels["consensus"]
    assert (c["both_labelled"], c["agreed"]) == (4, 2) and c["first_role"] == CLAUDE_LABELLER_ROLE
    assert c["dropped_by_class"] == {"first": {"judgment": 1, "performance": 1},
                                     "second": {"judgment": 1, "mechanical": 1}}
    assert is_consensus(labels) and not is_consensus(empty_labels("o/r"))


def test_points_outside_the_sample_or_with_changed_text_are_left_out():
    changed = {**TEXTS, "p0": "edited since"}
    labels = build_consensus("o/r", claude_labels(), model_file(), ["p0", "p2", "p3"], changed)
    assert set(labels["points"]) == {"p2"} and labels["consensus"]["both_labelled"] == 2


def test_the_claude_side_must_be_one_labeller():
    mixed = confirm(claude_labels(), "p4", "judgment", "someone-else", {"text_sha256": text_sha256("comment 4"),
                                                                       "blind": True})
    with pytest.raises(LabelError, match="exactly one labeller"):
        build_consensus("o/r", mixed, model_file(), list(TEXTS), TEXTS)
