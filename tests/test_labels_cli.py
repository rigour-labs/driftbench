import hashlib

import pytest

from bench.__main__ import main
from bench.labels.store import labels_path, read_labels
from bench.points.points_file import write_points
from tests.github_fakes import FakeClient

TEXT = "This leaks the handle.\n\n- nit: rename `f`\n\n- and add a test"
FIRST, SECOND, THIRD = [0, 22], [24, 41], [43, 59]


def point(pid: str, span, dropped=None) -> dict:
    return {"id": pid, "pr": 7, "kind": "body", "source_id": 10, "span": span, "dropped": dropped, "anchor": None,
            "body_sha256": hashlib.sha256(TEXT.encode()).hexdigest()}


def write(tmp_path, points):
    write_points({"schema": 1, "repo": "o/r", "points": points}, tmp_path / "points")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    write(tmp_path, [point("7-body-10-0", FIRST), point("7-body-10-1", SECOND),
                     point("7-body-10-2", [0, 1], dropped="ACK-1")])
    client = FakeClient({}, {"repos/o/r/pulls/7/reviews": [{"id": 10, "body": TEXT}]})
    monkeypatch.setattr("bench.labels.cli.GitHubClient", lambda cache: client)
    return ["label", "--points", str(tmp_path / "points"), "--labels", str(tmp_path / "labels")], tmp_path


def test_suggest_show_set_status(workspace, capsys):
    base, tmp_path = workspace
    assert main([*base, "suggest"]) == 0
    labels = read_labels(labels_path(tmp_path / "labels", "o/r"), "o/r")
    assert {k: v["suggested"] for k, v in labels["points"].items()} == {
        "7-body-10-0": "claim/contract", "7-body-10-1": "mechanical"}  # dropped point not labelled
    capsys.readouterr()
    assert main([*base, "show", "--repo", "o/r"]) == 0
    shown = capsys.readouterr().out
    assert "This leaks the handle." in shown and "suggested" not in shown  # blind by default
    assert main([*base, "show", "--repo", "o/r", "--with-suggestion"]) == 0
    assert "suggested: claim/contract" in capsys.readouterr().out
    assert main([*base, "set", "--repo", "o/r", "7-body-10-0", "claim/contract", "--labeller", "m"]) == 0
    entry = read_labels(labels_path(tmp_path / "labels", "o/r"), "o/r")["points"]["7-body-10-0"]
    assert entry["blind"] is True and entry["text_sha256"] == hashlib.sha256(b"This leaks the handle.").hexdigest()
    assert main([*base, "status"]) == 0
    assert "'confirmed': 1" in capsys.readouterr().out


def test_a_resplit_makes_the_label_stale_not_reused(workspace, capsys):
    base, tmp_path = workspace
    main([*base, "suggest"])
    main([*base, "set", "--repo", "o/r", "7-body-10-1", "mechanical", "--labeller", "m", "--saw-suggestion"])
    write(tmp_path, [point("7-body-10-0", FIRST), point("7-body-10-1", THIRD)])  # index 1 now another paragraph
    capsys.readouterr()
    assert main([*base, "status"]) == 0
    status = capsys.readouterr().out
    assert "'confirmed': 0" in status and "'stale': 1" in status


def test_errors_return_1(workspace, tmp_path):
    base, _ = workspace
    main([*base, "suggest"])
    assert main([*base, "set", "--repo", "o/r", "missing-id", "judgment", "--labeller", "m"]) == 1
    assert main([*base, "set", "--repo", "o/r", "7-body-10-2", "judgment", "--labeller", "m"]) == 1  # dropped
    assert main(["label", "--points", str(tmp_path / "nothing"), "status"]) == 1
