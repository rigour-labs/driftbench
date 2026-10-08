import json
import subprocess

import pytest

from bench.collect.github import GitHubClient, GitHubError, require_gh


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
    runner = ScriptedRunner((1, "", "You have exceeded a secondary rate limit"), (1, "", "HTTP 429"), (0, "[]", ""))
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


@pytest.mark.parametrize("result, message", [
    ((1, "", "command not found"), "is required"),
    ((0, "gh version 2.40.1 (2023-12-13)", ""), "too old"),
])
def test_require_gh_rejects_missing_or_old(result, message):
    with pytest.raises(GitHubError, match=message):
        require_gh(ScriptedRunner(result))


def test_require_gh_accepts_current():
    runner = ScriptedRunner((0, "gh version 2.102.0 (2026-09-30)", ""))
    assert require_gh(runner) is None
    assert runner.calls == [["gh", "--version"]]


def test_get_optional_returns_none_on_404_and_raises_on_others():
    client, _ = make_client(ScriptedRunner((1, "", "HTTP 404: Not Found"), (1, "", "HTTP 500")))
    assert client.get_optional("repos/o/r/commits/gone") is None
    with pytest.raises(GitHubError, match="500"):
        client.get_optional("repos/o/r/commits/x")


def test_hourly_limit_waits_until_the_reset():
    runner = ScriptedRunner((1, "", "API rate limit exceeded for installation"),
                            (0, json.dumps({"resources": {"core": {"reset": 1_000_600}}}), ""),
                            (0, "[1]", ""))
    sleeps: list[float] = []
    client = GitHubClient(None, runner=runner, sleep=sleeps.append, clock=lambda: 1_000_000)
    assert client.get("x") == [1]
    assert 605 in sleeps and runner.calls[1] == ["gh", "api", "rate_limit"]


def test_reset_wait_is_capped_and_survives_a_bad_answer():
    runner = ScriptedRunner((1, "", "API rate limit exceeded"), (0, "not json", ""), (0, "[]", ""))
    sleeps: list[float] = []
    GitHubClient(None, runner=runner, sleep=sleeps.append, clock=lambda: 50.0).get("x")
    assert 65.0 in sleeps
    runner = ScriptedRunner((1, "", "API rate limit exceeded"),
                            (0, json.dumps({"resources": {"core": {"reset": 10**9}}}), ""), (0, "[]", ""))
    sleeps = []
    GitHubClient(None, runner=runner, sleep=sleeps.append, clock=lambda: 0.0).get("x")
    assert 3700 in sleeps
