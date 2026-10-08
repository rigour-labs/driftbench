import pytest

from bench.labels.agreement import cohens_kappa
from bench.labels.sample import SampleError, draw_sample, read_sample, sample_path, write_sample


def inline(pid, scorable=True, side="RIGHT", line=5):
    return {"id": pid, "kind": "inline", "scorable": scorable,
            "anchor": {"path": "a.py", "line": line, "side": side, "commit_sha": "c"}}


def points_file(inline_points=30):
    pts = [inline(f"i{n}") for n in range(inline_points)]
    pts += [{"id": f"b{n}", "kind": "body", "scorable": True, "anchor": None} for n in range(15)]
    pts += [inline("unscorable", scorable=False), inline("left", side="LEFT"), inline("lineless", line=None)]
    return {"repo": "o/r", "points": pts}


def test_draw_is_reproducible_and_location_scorable_only():
    first = draw_sample(points_file(), 20, seed=11)
    assert first["point_ids"] == draw_sample(points_file(), 20, seed=11)["point_ids"]
    assert first["point_ids"] != draw_sample(points_file(), 20, seed=12)["point_ids"]
    assert first["size"] == 20 and first["eligible"] == 30
    assert all(pid.startswith("i") for pid in first["point_ids"])
    assert draw_sample(points_file(5), 50, seed=1)["size"] == 5
    changed = points_file()
    changed["points"][0]["anchor"]["line"] = 6
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
