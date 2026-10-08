import json
import subprocess

import pytest

from bench.collect.github import GitHubClient, GitHubError


class ScriptedRunner:
    """Returns queued (returncode, stdout, stderr) results and records the argv."""

    def __init__(self, *results: tuple[int, str, str]):
        self.results = list(results)
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(args)
        code, out, err = self.results.pop(0)
        return subprocess.CompletedProcess(args, code, out, err)


def make_client(runner: ScriptedRunner, cache=None) -> tuple[GitHubClient, list[float]]:
    sleeps: list[float] = []
    return GitHubClient(cache, runner=runner, sleep=sleeps.append), sleeps


def test_get_passes_params_and_parses_json():
    runner = ScriptedRunner((0, '{"a": 1}', ""))
    client, _ = make_client(runner)
    assert client.get("repos/o/r/pulls", {"state": "closed"}) == {"a": 1}
    assert runner.calls[0][:5] == ["gh", "api", "-X", "GET", "repos/o/r/pulls"]
    assert "state=closed" in runner.calls[0]


def test_get_all_flattens_pages():
    runner = ScriptedRunner((0, json.dumps([[1, 2], [3]]), ""))
    client, _ = make_client(runner)
    assert client.get_all("repos/o/r/pulls/1/reviews") == [1, 2, 3]
    assert "--paginate" in runner.calls[0] and "per_page=100" in runner.calls[0]


def test_cache_serves_repeat_calls(tmp_path):
    runner = ScriptedRunner((0, "[1]", ""))
    client, _ = make_client(runner, cache=tmp_path)
    assert client.get("x") == [1]
    assert make_client(ScriptedRunner(), cache=tmp_path)[0].get("x") == [1]
    assert len(runner.calls) == 1


def test_rate_limit_is_retried_with_growing_waits():
    runner = ScriptedRunner((1, "", "API rate limit exceeded"), (1, "", "HTTP 429"), (0, "[]", ""))
    client, sleeps = make_client(runner)
    assert client.get("x") == []
    assert [s for s in sleeps if s >= 60] == [60, 120]


def test_other_errors_raise_without_retry():
    runner = ScriptedRunner((1, "", "HTTP 404: Not Found"))
    client, _ = make_client(runner)
    with pytest.raises(GitHubError, match="404"):
        client.get("x")
    assert len(runner.calls) == 1


def test_invalid_json_raises_github_error():
    client, _ = make_client(ScriptedRunner((0, "<html>", "")))
    with pytest.raises(GitHubError, match="not valid JSON"):
        client.get("x")
