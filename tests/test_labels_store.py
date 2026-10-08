import pytest

from bench.labels.store import (LabelError, confirm, empty_labels, label_status, labels_path, merge_suggestions,
                                read_labels, write_labels)


def test_merge_adds_refreshes_and_never_touches_confirmed():
    labels = merge_suggestions(empty_labels("o/r"), {"a": "judgment", "b": None})
    labels = confirm(labels, "a", "claim/contract", "maintainer")
    merged = merge_suggestions(labels, {"a": "mechanical", "b": "performance", "c": None})
    assert merged["points"]["a"] == {"suggested": "mechanical", "label": "claim/contract", "labeller": "maintainer"}
    assert merged["points"]["b"]["suggested"] == "performance" and merged["points"]["b"]["label"] is None
    assert set(merged["points"]) == {"a", "b", "c"}
    assert labels["points"]["a"]["suggested"] == "judgment"  # inputs aren't mutated


def test_confirm_rejects_unknown_class_or_point():
    labels = merge_suggestions(empty_labels("o/r"), {"a": None})
    with pytest.raises(LabelError, match="unknown class"):
        confirm(labels, "a", "security", "maintainer")
    with pytest.raises(LabelError, match="not in the label file"):
        confirm(labels, "zzz", "judgment", "maintainer")


def test_round_trip_and_validation(tmp_path):
    path = labels_path(tmp_path, "o/r")
    assert path.name == "o__r.yaml" and read_labels(path, "o/r")["points"] == {}
    labels = confirm(merge_suggestions(empty_labels("o/r"), {"a": "judgment"}), "a", "user journey", "m")
    write_labels(labels, path)
    assert read_labels(path, "o/r") == labels
    with pytest.raises(LabelError, match="not a label file"):
        read_labels(path, "x/y")
    path.write_text(path.read_text().replace("user journey", "vibes"))
    with pytest.raises(LabelError, match="unknown class"):
        read_labels(path, "o/r")


def test_status_counts_agreement_and_stale_points():
    labels = merge_suggestions(empty_labels("o/r"), {"a": "judgment", "b": "mechanical", "c": None})
    labels = confirm(confirm(labels, "a", "judgment", "m"), "b", "claim/contract", "m")
    status = label_status(labels, current_ids={"a", "b"})
    assert status == {"points": 3, "confirmed": 2, "by_class": {"claim/contract": 1, "judgment": 1},
                      "suggestion_agreement": "1/2", "stale": 1}
