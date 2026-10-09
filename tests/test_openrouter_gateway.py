import io
import json

import pytest

from bench.__main__ import main
from bench.adapters import PaidSettings, select_adapters
from bench.adapters.claude_cli import check_model_id, paid_env, provider_env_names
from bench.harness.types import AdapterError
from bench.labels import openrouter
from bench.score.spend import openrouter_billed, spend_notes
from tests.test_score_spend import record

MODEL = "anthropic/claude-example-4.5-20260901"


class Request:
    env = {"HOME": "/tmp/h", "PATH": "/bin"}


def test_openrouter_sets_the_gateway_and_empties_the_anthropic_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(AdapterError, match="OPENROUTER_API_KEY is not set"):
        paid_env(Request(), "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    env = paid_env(Request(), "openrouter")
    assert env == {**Request.env, "ANTHROPIC_BASE_URL": "https://openrouter.ai/api", "ANTHROPIC_AUTH_TOKEN": "sk-or-test",
                   "ANTHROPIC_API_KEY": ""}
    assert set(env) - set(Request.env) == set(provider_env_names("openrouter"))


def test_both_paid_entrants_get_the_same_provider_and_variables():
    paid = PaidSettings(model=MODEL, max_usd=5, head_bounds={}, provider="openrouter")
    adapters = select_adapters(["rigour-reviewer", "claude-code-review"], paid)
    assert {a.provider for a in adapters} == {"openrouter"}
    assert {a.env_extra for a in adapters} == {("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")}
    default = select_adapters(["claude-code-review"], PaidSettings(model="claude-x", max_usd=5))
    assert default[0].env_extra == ("ANTHROPIC_API_KEY",)


@pytest.mark.parametrize("model", ["anthropic/claude-latest", "openai/gpt-6.1-sol", "anthropic/claude-sonnet"])
def test_openrouter_needs_a_pinned_anthropic_model(model):
    with pytest.raises(ValueError, match="full versioned anthropic/"):
        check_model_id(model, "openrouter")
    check_model_id(MODEL, "openrouter")
    with pytest.raises(ValueError, match="unknown provider"):
        check_model_id(MODEL, "bedrock")


def test_run_json_records_the_provider(tmp_path):
    from bench.harness.cli import read_manifest
    out = tmp_path / "run"
    assert main(["manifest", "--entrants", "claude-code-review", "--out", str(out), "--labels", str(tmp_path),
                 "--model", MODEL, "--max-usd", "5", "--head-bound", "claude-code-review=0.8",
                 "--provider", "openrouter"]) == 0
    assert read_manifest(out / "run.json")["paid"]["provider"] == "openrouter"


KEY_REPLY = {"data": {"label": "sk-or-v1-abc...xyz", "usage": 12.5, "limit": 50, "is_free_tier": False}}


def stub_urlopen(monkeypatch, reply=None, raises=None):
    """Replaces urllib.request.urlopen with its real signature, so a body passed by position is caught."""
    seen = []

    def urlopen(url, data=None, timeout=None, *, context=None):
        seen.append({"method": url.get_method(), "data": data, "body": url.data, "timeout": timeout,
                     "auth": url.get_header("Authorization"), "url": url.full_url})
        if raises:
            raise raises
        return io.BytesIO(reply if isinstance(reply, bytes) else json.dumps(reply).encode())
    monkeypatch.setattr(openrouter.urllib.request, "urlopen", urlopen)
    return seen


def test_key_usage_is_a_get_with_no_body_and_reads_data_usage(monkeypatch):
    monkeypatch.setenv(openrouter.KEY_NAME, "sk-or-test")
    seen = stub_urlopen(monkeypatch, KEY_REPLY)
    assert openrouter.key_usage() == 12.5
    assert seen == [{"method": "GET", "data": None, "body": None, "timeout": openrouter.TIMEOUT_S,
                     "auth": "Bearer sk-or-test", "url": "https://openrouter.ai/api/v1/key"}]


@pytest.mark.parametrize("reply, raises, reason", [
    (b"not json", None, "JSONDecodeError"),
    ({"data": {}}, None, "no numeric data.usage"),
    (None, openrouter.urllib.error.URLError("down"), "URLError"),
    (None, TimeoutError(), "TimeoutError"),
])
def test_any_failure_to_read_usage_is_an_openrouter_error(monkeypatch, reply, raises, reason):
    monkeypatch.setenv(openrouter.KEY_NAME, "sk-or-test")
    stub_urlopen(monkeypatch, reply, raises)
    with pytest.raises(openrouter.OpenRouterError, match=reason) as caught:
        openrouter.key_usage()
    assert "sk-or-test" not in str(caught.value)


def test_an_unreadable_usage_is_recorded_unavailable_and_the_run_goes_on(monkeypatch, capsys):
    import argparse
    from bench.harness.cli import gateway_usage
    monkeypatch.setenv(openrouter.KEY_NAME, "sk-or-test")
    stub_urlopen(monkeypatch, raises=TimeoutError())
    adapters = select_adapters(["claude-code-review"], PaidSettings(model=MODEL, max_usd=5, provider="openrouter"))
    reading = gateway_usage(argparse.Namespace(provider="openrouter"), adapters)
    assert reading["usage_usd"] is None and "TimeoutError" in reading["error"]
    assert "OpenRouter usage unavailable" in capsys.readouterr().err
    billed = openrouter_billed([{"start": reading, "end": reading}])
    assert billed["billed_usd"] is None and "TimeoutError" in billed["reason"]
    assert "billed by OpenRouter: unavailable (reading the key's usage failed: TimeoutError)" in spend_notes({}, billed)


def test_billed_is_the_window_across_every_job_and_fills_the_notes(tmp_path, capsys):
    usages = [{"start": {"at": "T1", "usage_usd": 10.0}, "end": {"at": "T4", "usage_usd": 13.5}},
              {"start": {"at": "T2", "usage_usd": 10.4}, "end": {"at": "T3", "usage_usd": 12.0}}]
    assert openrouter_billed(usages) == {"billed_usd": 3.5, "from": "T1", "to": "T4"}
    assert openrouter_billed([{"start": {"at": "T1", "usage_usd": None}, "end": None}])["billed_usd"] is None
    record(tmp_path, "claude-code-review", "h1", cost_usd=3.2, charged="reported")
    (tmp_path / "budget-o__r.json").write_text('{"estimate_per_head": {}, "largest_per_head": {}}')
    (tmp_path / "openrouter-usage-o__r.json").write_text(json.dumps(usages[0]))
    assert main(["spend", "--run", str(tmp_path), "--openrouter"]) == 0
    out = capsys.readouterr().out
    assert "estimated $3.20; billed by OpenRouter $3.50 (the key's usage from T1 to T4" in out
    assert "$____" not in out
    assert "unavailable" in spend_notes({}, openrouter_billed([]))
