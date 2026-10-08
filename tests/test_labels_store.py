import pytest

from bench.labels.store import (LabelError, confirm, effective_labels, empty_labels, label_status, labels_path,
                                merge_suggestions, read_labels, text_sha256, write_labels)

BLIND = {"text_sha256": text_sha256("A leaks"), "blind": True}


def test_merge_adds_refreshes_and_never_moves_a_confirmed_entry():
    labels = merge_suggestions(empty_labels("o/r"), {"a": "judgment", "b": None})
    labels = confirm(labels, "a", "claim/contract", "maintainer", BLIND)
    merged = merge_suggestions(labels, {"a": "mechanical", "b": "performance", "c": None})
    assert merged["points"]["a"]["suggested"] == "judgment"          # as it was at confirm time
    assert merged["points"]["a"]["suggested_now"] == "mechanical"
    assert merged["points"]["a"]["label"] == "claim/contract"
    assert merged["points"]["b"]["suggested"] == "performance" and merged["points"]["b"]["label"] is None
    assert set(merged["points"]) == {"a", "b", "c"}
    assert "suggested_now" not in labels["points"]["a"]  # inputs aren't mutated


def test_confirm_rejects_unknown_class_or_point():
    labels = merge_suggestions(empty_labels("o/r"), {"a": None})
    with pytest.raises(LabelError, match="unknown class"):
        confirm(labels, "a", "security", "maintainer", BLIND)
    with pytest.raises(LabelError, match="not in the label file"):
        confirm(labels, "zzz", "judgment", "maintainer", BLIND)


def test_round_trip_and_validation(tmp_path):
    path = labels_path(tmp_path, "o/r")
    assert path.name == "o__r.yaml" and read_labels(path, "o/r")["points"] == {}
    labels = confirm(merge_suggestions(empty_labels("o/r"), {"a": "judgment"}), "a", "user journey", "m", BLIND)
    write_labels(labels, path)
    assert read_labels(path, "o/r") == labels
    with pytest.raises(LabelError, match="not a label file"):
        read_labels(path, "x/y")
    path.write_text(path.read_text().replace("user journey", "vibes"))
    with pytest.raises(LabelError, match="unknown class"):
        read_labels(path, "o/r")


def test_label_is_used_only_while_the_text_is_the_one_read():
    labels = confirm(merge_suggestions(empty_labels("o/r"), {"a": None}), "a", "claim/contract", "m", BLIND)
    assert effective_labels(labels, {"a": "A leaks"}) == {"a": "claim/contract"}
    assert effective_labels(labels, {"a": "a different paragraph"}) == {}
    assert effective_labels(labels, {"a": None}) == {} and effective_labels(labels, {}) == {}


def test_status_counts_stale_and_blind_agreement_only():
    labels = merge_suggestions(empty_labels("o/r"), {"a": "judgment", "b": "mechanical", "c": "judgment", "d": None})
    labels = confirm(labels, "a", "judgment", "m", {"text_sha256": text_sha256("ta"), "blind": True})
    labels = confirm(labels, "b", "claim/contract", "m", {"text_sha256": text_sha256("tb"), "blind": True})
    labels = confirm(labels, "c", "judgment", "m", {"text_sha256": text_sha256("tc"), "blind": False})
    labels = confirm(labels, "d", "mechanical", "m", {"text_sha256": text_sha256("td"), "blind": True})
    status = label_status(labels, {"a": "ta", "b": "tb", "c": "tc", "d": "changed"})
    assert status == {"points": 4, "confirmed": 3, "stale": 1,
                      "by_class": {"claim/contract": 1, "judgment": 2},
                      "blind_suggestion_agreement": "1/2", "dropped_ids": 0}
