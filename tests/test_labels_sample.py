import pytest

from bench.labels.agreement import cohens_kappa
from bench.labels.sample import SampleError, allocate, draw_sample, read_sample, sample_path, write_sample


def points_file(inline=30, body=15, conversation=5, unscorable=10):
    pts = [{"id": f"i{n}", "kind": "inline", "scorable": True} for n in range(inline)]
    pts += [{"id": f"b{n}", "kind": "body", "scorable": True} for n in range(body)]
    pts += [{"id": f"c{n}", "kind": "conversation", "scorable": True} for n in range(conversation)]
    pts += [{"id": f"x{n}", "kind": "inline", "scorable": False} for n in range(unscorable)]
    return {"repo": "o/r", "points": pts}


def test_allocation_is_proportional_and_exact():
    assert allocate({"inline": 30, "body": 15, "conversation": 5}, 20) == {"inline": 12, "body": 6, "conversation": 2}
    assert allocate({"inline": 2, "body": 1, "conversation": 0}, 50) == {"inline": 2, "body": 1, "conversation": 0}
    assert sum(allocate({"inline": 7, "body": 7, "conversation": 7}, 10).values()) == 10


def test_draw_is_reproducible_stratified_and_scorable_only():
    first = draw_sample(points_file(), 20, seed=11)
    assert first["point_ids"] == draw_sample(points_file(), 20, seed=11)["point_ids"]
    assert first["point_ids"] != draw_sample(points_file(), 20, seed=12)["point_ids"]
    assert first["by_kind"] == {"inline": 12, "body": 6, "conversation": 2} and first["size"] == 20
    assert not any(pid.startswith("x") for pid in first["point_ids"])
    changed = points_file()
    changed["points"][0]["kind"] = "body"
    assert draw_sample(changed, 20, seed=11)["points_sha256"] != first["points_sha256"]


def test_never_replaced_silently(tmp_path):
    path = sample_path(tmp_path, "o/r")
    write_sample(draw_sample(points_file(), 5, seed=1), path, replace=False)
    with pytest.raises(SampleError, match="--replace"):
        write_sample(draw_sample(points_file(), 5, seed=2), path, replace=False)
    write_sample(draw_sample(points_file(), 5, seed=2), path, replace=True)
    assert read_sample(path)["seed"] == 2 and read_sample(tmp_path / "none.yaml") is None


def test_cohens_kappa():
    a = {"1": "mechanical", "2": "judgment", "3": "mechanical", "4": "judgment"}
    assert cohens_kappa(a, a)["kappa"] == 1.0
    b = {"1": "mechanical", "2": "mechanical", "3": "judgment", "4": "judgment"}
    assert cohens_kappa(a, b) == {"points": 4, "observed": 0.5, "kappa": 0.0}
    assert cohens_kappa(a, {"9": "x"})["kappa"] is None
    assert cohens_kappa({"1": "m"}, {"1": "m"})["kappa"] is None  # no variation: undefined
