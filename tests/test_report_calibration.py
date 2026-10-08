import pytest

from bench.report.calibration import (CalibrationError, draw, read_calibration, summarise_calibration,
                                      write_calibration)


def ledger_rows():
    rows = []
    for tool in ("no-tool", "every-hunk", "alpha", "beta"):
        for n in range(40):
            rows.append({"tool": tool, "repo": "o/r", "point": f"p{n}", "pr": 1, "head_sha": "h1", "finding": 0,
                         "distance": n % 6})
    return rows


def points():
    pts = []
    for n in range(30):
        basis = "direct" if n < 20 else "ancestor"
        pts.append({"id": f"p{n}", "repo": "o/r", "pr": 1, "acted_basis": basis, "acted_on": n % 2 == 0})
    pts.append({"id": "rebased", "repo": "o/r", "pr": 1, "acted_basis": "rebased", "acted_on": None})
    return pts


def test_draw_is_reproducible_stratified_and_skips_baselines():
    first, second = draw(ledger_rows(), points(), seed=7), draw(ledger_rows(), points(), seed=7)
    assert first == second and draw(ledger_rows(), points(), seed=8) != first
    location = [e for e in first["entries"] if e["kind"] == "location"]
    assert {e["tool"] for e in location} == {"alpha", "beta"} and len(location) == 50
    assert all(e["tool"] not in ("no-tool", "every-hunk") for e in location)
    acted = [(e["basis"], e["acted_on"]) for e in first["entries"] if e["kind"] == "acted_on"]
    assert acted.count(("direct", True)) == 10 and acted.count(("direct", False)) == 5
    assert sum(1 for basis, _ in acted if basis == "ancestor") == 5


def test_round_trip_validation_and_summary(tmp_path):
    path = tmp_path / "calibration.yaml"
    assert read_calibration(path) is None and summarise_calibration(None)["validated"] is False
    calibration = draw(ledger_rows(), points(), seed=1)
    for entry in calibration["entries"]:
        entry["verdict"] = "yes" if entry["kind"] == "location" else "no"
    write_calibration(calibration, path)
    summary = summarise_calibration(read_calibration(path))
    assert summary["validated"] is True and summary["location"]["alpha"] == {"yes": 25}
    assert summary["acted_on"]["direct"] == {"disagree": 15}
    path.write_text(path.read_text().replace("verdict: 'yes'", "verdict: maybe", 1).replace(
        "verdict: yes", "verdict: maybe", 1), encoding="utf-8")
    with pytest.raises(CalibrationError, match="maybe"):
        read_calibration(path)


def test_partial_sample_is_not_validated():
    calibration = draw(ledger_rows(), points(), seed=1)
    calibration["entries"][0]["verdict"] = "partly"
    summary = summarise_calibration(calibration)
    assert summary["validated"] is False and summary["location"] == {calibration["entries"][0]["tool"]: {"partly": 1}}
