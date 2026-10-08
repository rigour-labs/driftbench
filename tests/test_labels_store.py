import pytest

from bench.labels.store import (LabelError, confirm, effective_labels, empty_labels, label_status, labels_path,
                                mark_skipped, read_labels, text_sha256, write_labels)
from bench.labels.suggestions import (confirm_time_suggestions, merge_suggestions, read_suggestions,
                                      suggestions_path, write_suggestions)

BLIND = {"text_sha256": text_sha256("A leaks"), "blind": True}


def test_label_file_never_holds_suggestions(tmp_path):
    labels = confirm(empty_labels("o/r"), "a", "claim/contract", "maintainer", BLIND)
    write_labels(mark_skipped(labels, "b"), labels_path(tmp_path, "o/r"))
    raw = labels_path(tmp_path, "o/r").read_text()
    assert "suggest" not in raw and "claim/contract" in raw and "skipped: true" in raw


def test_suggestions_keep_the_confirm_time_value(tmp_path):
    path = suggestions_path(tmp_path, "o/r")
    suggestions = merge_suggestions(read_suggestions(path, "o/r"), {"a": "judgment", "b": None}, labelled=set())
    later = merge_suggestions(suggestions, {"a": "mechanical", "b": "performance"}, labelled={"a"})
    assert later["points"]["a"] == {"suggested": "judgment", "suggested_now": "mechanical"}
    assert later["points"]["b"] == {"suggested": "performance"}
    write_suggestions(later, path)
    assert confirm_time_suggestions(read_suggestions(path, "o/r")) == {"a": "judgment", "b": "performance"}
    assert "suggested_now" not in suggestions["points"]["a"]  # inputs aren't mutated


def test_confirm_skip_and_validation(tmp_path):
    with pytest.raises(LabelError, match="unknown class"):
        confirm(empty_labels("o/r"), "a", "security", "maintainer", BLIND)
    labels = confirm(empty_labels("o/r"), "a", "user journey", "m", BLIND)
    assert mark_skipped(labels, "a") == labels          # skipping never removes a label
    path = labels_path(tmp_path, "o/r")
    write_labels(labels, path)
    assert read_labels(path, "o/r") == labels
    with pytest.raises(LabelError, match="not a label file"):
        read_labels(path, "x/y")
    path.write_text(path.read_text().replace("user journey", "vibes"))
    with pytest.raises(LabelError, match="unknown class"):
        read_labels(path, "o/r")


def test_label_is_used_only_while_the_text_is_the_one_read():
    labels = confirm(empty_labels("o/r"), "a", "claim/contract", "m", BLIND)
    assert effective_labels(labels, {"a": "A leaks"}) == {"a": "claim/contract"}
    assert effective_labels(labels, {"a": "a different paragraph"}) == {}
    assert effective_labels(labels, {"a": None}) == {} and effective_labels(labels, {}) == {}


def test_status_counts_stale_and_blind_agreement_only():
    labels = empty_labels("o/r")
    for pid, label, blind in (("a", "judgment", True), ("b", "claim/contract", True), ("c", "judgment", False),
                              ("d", "mechanical", True)):
        labels = confirm(labels, pid, label, "m", {"text_sha256": text_sha256(f"t{pid}"), "blind": blind})
    suggested = {"a": "judgment", "b": "mechanical", "c": "judgment", "d": None}
    status = label_status(labels, {"a": "ta", "b": "tb", "c": "tc", "d": "changed"}, suggested)
    assert status == {"points": 4, "confirmed": 3, "stale": 1, "by_class": {"claim/contract": 1, "judgment": 2},
                      "blind_suggestion_agreement": "1/2", "dropped_ids": 0}
