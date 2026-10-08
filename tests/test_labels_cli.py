import hashlib

import pytest

from bench.__main__ import main
from bench.labels.store import labels_path, read_labels
from bench.points.points_file import write_points
from tests.github_fakes import FakeClient

TEXT = "This leaks the handle.\n\n- nit: rename `f`"


def point(pid: str, kind: str, span, dropped=None) -> dict:
    return {"id": pid, "pr": 7, "kind": kind, "source_id": 10, "span": span, "dropped": dropped, "anchor": None,
            "body_sha256": hashlib.sha256(TEXT.encode()).hexdigest()}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    points = [point("7-body-10-0", "body", [0, 22]), point("7-body-10-1", "body", [24, 41]),
              point("7-body-10-2", "body", [0, 1], dropped="ACK-1")]
    write_points({"schema": 1, "repo": "o/r", "points": points}, tmp_path / "points")
    client = FakeClient({}, {"repos/o/r/pulls/7/reviews": [{"id": 10, "body": TEXT}]})
    monkeypatch.setattr("bench.labels.cli.GitHubClient", lambda cache: client)
    return ["label", "--points", str(tmp_path / "points"), "--labels", str(tmp_path / "labels")], tmp_path


def test_suggest_show_set_status(workspace, capsys):
    base, tmp_path = workspace
    assert main([*base, "suggest"]) == 0
    labels = read_labels(labels_path(tmp_path / "labels", "o/r"), "o/r")
    assert {k: v["suggested"] for k, v in labels["points"].items()} == {
        "7-body-10-0": "claim/contract", "7-body-10-1": "mechanical"}  # dropped point not labelled
    assert main([*base, "show", "--repo", "o/r"]) == 0
    assert "This leaks the handle." in capsys.readouterr().out
    assert main([*base, "set", "--repo", "o/r", "7-body-10-0", "claim/contract", "--labeller", "m"]) == 0
    assert main([*base, "status"]) == 0
    assert "'confirmed': 1" in capsys.readouterr().out


def test_errors_return_1(workspace, tmp_path):
    base, _ = workspace
    assert main([*base, "set", "--repo", "o/r", "missing-id", "judgment", "--labeller", "m"]) == 1
    assert main(["label", "--points", str(tmp_path / "nothing"), "status"]) == 1
