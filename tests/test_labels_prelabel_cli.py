import hashlib
from pathlib import Path

import pytest

from bench.__main__ import main
from bench.labels.model_file import model_path, read_model_file
from bench.labels.openrouter import chat
from bench.labels.sample import write_sample
from bench.labels.store import labels_path, read_labels
from bench.points.points_file import write_points
from tests.fixture_files import load_json
from tests.github_fakes import FakeClient

FIXTURE = load_json("openrouter-chat.json")
COMMENTS = [{"id": 100 + n, "body": f"comment {n}", "diff_hunk": f"@@ hunk {n}"} for n in range(12)]


def point(n: int) -> dict:
    body = COMMENTS[n]["body"]
    return {"id": f"7-inline-{100 + n}-0", "pr": 7, "kind": "inline", "source_id": 100 + n, "span": None,
            "dropped": None, "anchor": {"path": "a.py", "line": n + 1, "commit_sha": "h"},
            "body_sha256": hashlib.sha256(body.encode()).hexdigest()}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    write_points({"schema": 1, "repo": "o/r", "points": [point(n) for n in range(12)]}, tmp_path / "points")
    sample = {"repo": "o/r", "seed": 1, "size": 10, "point_ids": [point(n)["id"] for n in range(10)]}
    write_sample(sample, tmp_path / "labels" / "o__r.sample.yaml", replace=False)
    client = FakeClient({}, {"repos/o/r/pulls/7/comments": COMMENTS})
    monkeypatch.setattr("bench.labels.prelabel_cli.GitHubClient", lambda cache: client)
    sent = []

    def fake_chat(model, messages, max_tokens):
        sent.append(messages[1]["content"])
        return chat(model, messages, max_tokens, transport=lambda body: FIXTURE)
    monkeypatch.setattr("bench.labels.prelabel_cli.chat", fake_chat)
    base = ["label", "--points", str(tmp_path / "points"), "--labels", str(tmp_path / "labels")]
    return base, tmp_path, sent


def prelabel(base, *extra):
    return main([*base, "prelabel", "--repo", "o/r", *extra])


def test_suggests_the_label_sample_only_and_leaves_labels_alone(workspace, capsys):
    base, tmp_path, sent = workspace
    assert prelabel(base, "--model", "example-org/example-model", "--max-usd", "2") == 0
    data = read_model_file(model_path(tmp_path / "labels", "o/r"), "o/r")
    assert set(data["points"]) == {point(n)["id"] for n in range(10)}      # points 10 and 11 are not in the sample
    assert len(sent) == 10 and "comment 3" in sent[3] and "@@ hunk 3" in sent[3]
    assert all(e["suggested"] == "performance" for e in data["points"].values())
    assert data["model"] == "example-org/example-model" and len(data["blind_ids"]) == 2 and data["prompt_sha256"]
    assert data["spent_usd"] == round(10 * 0.000412, 6)
    assert read_labels(labels_path(tmp_path / "labels", "o/r"), "o/r")["points"] == {}
    assert "10 of 10 suggested" in capsys.readouterr().out


@pytest.mark.parametrize("extra, message", [
    (["--model", "anthropic/some-model", "--max-usd", "2"], "outside the Claude family"),
    (["--model", "example-org/example-model", "--max-usd", "0"], "approved cap"),
])
def test_refuses_a_claude_model_or_no_cap(workspace, capsys, extra, message):
    base, _, sent = workspace
    assert prelabel(base, *extra) == 1 and message in capsys.readouterr().err and sent == []


def test_refuses_to_mix_models_in_one_file(workspace, capsys):
    base, _, sent = workspace
    assert prelabel(base, "--model", "example-org/example-model", "--max-usd", "2") == 0
    assert prelabel(base, "--model", "example-org/other-model", "--max-usd", "2") == 1
    assert "model differs" in capsys.readouterr().err and len(sent) == 10
