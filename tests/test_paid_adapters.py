import copy
import re
import subprocess
from pathlib import Path

import pytest

from bench.adapters import claude_cli, claude_code, rigour_reviewer
from bench.harness.types import AdapterError, ReviewInput

FIXTURES = Path(__file__).parent / "fixtures"

# Synthetic shapes, written from the CLIs' sources (rigour 6.8.1/6.9.0 review --json `reviewer` section;
# Claude Code stream-json events). No real tool output and no repository data.
REVIEWER_REPORT = {
    "status": "PASS", "failures": [],
    "reviewer": {"outcome": "findings", "items": [{"file": "a.py", "line": 7, "class": "contract", "issue": "x"}],
                 "advisory": [{"file": "b.py", "line": 2, "class": "style", "issue": "y"}], "notes": [],
                 "cost_usd": 0.42, "tokens": {"input": 1000, "output": 200}, "cached": False, "pr": None,
                 "record": {"judges": [{"reviewer": "claude", "model": "m", "cost_usd": 0.42}],
                            "reported": {"human_reviews": 0}}},
}


def test_rigour_reviewer_items_block_and_advisory_does_not():
    out = rigour_reviewer.to_output(REVIEWER_REPORT)
    assert out.verdict == "fail" and out.cost_usd == 0.42 and out.model_runs == 1 and out.leak_signals == 0
    assert [(f.path, f.line, f.blocking) for f in out.findings] == [("a.py", 7, True), ("b.py", 2, False)]


def test_rigour_reviewer_leak_cached_and_unavailable():
    seen = copy.deepcopy(REVIEWER_REPORT)
    seen["reviewer"]["record"]["reported"]["human_reviews"] = 3
    seen["reviewer"]["pr"] = {"number": 9}
    assert rigour_reviewer.to_output(seen).leak_signals == 4
    cached = copy.deepcopy(REVIEWER_REPORT)
    cached["reviewer"].update(cached=True, cost_usd=None)
    assert rigour_reviewer.to_output(cached).model_runs == 0
    down = {"status": "PASS", "reviewer": {"outcome": "unavailable", "reason": "no agent"}}
    with pytest.raises(AdapterError, match="unavailable: no agent"):
        rigour_reviewer.to_output(down)
    with pytest.raises(AdapterError, match="no reviewer section"):
        rigour_reviewer.to_output({"status": "PASS"})


def test_rigour_reviewer_command_and_config():
    adapter = rigour_reviewer.RigourReviewer("model-x", orchestrated=True)
    request = ReviewInput("/w", "b" * 40, "h" * 40, "/d", None, 600, {"HOME": "/home"})
    assert adapter.command(request, "/home/rigour-bench.yml")[-5:] == ["--reviewer", "--single", "-c",
                                                                       "/home/rigour-bench.yml", "--orchestrator"]
    config = rigour_reviewer.reviewer_config("model-x", 600)["review"]["reviewer"]
    assert config["models"] == {"claude": "model-x"} and config["timeout_ms"] == 600_000 and config["mode"] == "single"


def stream(result: dict, tools: list[dict]) -> list[dict]:
    calls = [{"type": "assistant", "message": {"content": [{"type": "tool_use", **t}]}} for t in tools]
    return [{"type": "system"}, *calls, {"type": "result", **result}]


RESULT = {"result": "- `src/app.py:12` leaks the handle\n- see src/app.py:L20-L24 and other.py:3",
          "total_cost_usd": 0.31, "num_turns": 6,
          "usage": {"input_tokens": 100, "cache_read_input_tokens": 900, "output_tokens": 50}}


def test_claude_code_citations_cost_and_no_blocking():
    out = claude_code.to_output(stream(RESULT, [{"name": "Read", "input": {"file_path": "src/app.py"}}]),
                                {"src/app.py"})
    assert [(f.path, f.line, f.end_line, f.blocking) for f in out.findings] == [("src/app.py", 12, None, False),
                                                                               ("src/app.py", 20, 24, False)]
    assert (out.cost_usd, out.input_tokens, out.output_tokens, out.model_runs) == (0.31, 1000, 50, 6)
    assert out.leak_signals == 0 and out.verdict == "pass"


def test_claude_code_leaks_and_errors():
    leaky = [{"name": "Bash", "input": {"command": "gh pr view 7 --comments"}},
             {"name": "WebFetch", "input": {"url": "https://example.test"}},
             {"name": "Bash", "input": {"command": "git log --oneline"}}]
    assert claude_code.to_output(stream(RESULT, leaky), set()).leak_signals == 2
    with pytest.raises(AdapterError, match="no result"):
        claude_code.to_output([{"type": "system"}], set())
    with pytest.raises(AdapterError, match="reported an error"):
        claude_code.to_output(stream({"is_error": True, "result": "boom"}, []), set())


def test_claude_code_command_blocks_web_and_github():
    command = claude_code.ClaudeCodeReview("model-x", max_usd_per_review=1.5).command()
    assert command[:3] == ["claude", "-p", "/code-review"] and "--strict-mcp-config" in command
    assert {"WebFetch", "WebSearch", "Bash(gh:*)"} <= set(command) and command[-2:] == ["--max-budget-usd", "1.50"]
    assert command[command.index("--model") + 1] == "model-x"


def test_paid_env_and_cli_version(monkeypatch):
    request = ReviewInput("/w", "b", "h", "/d", None, 5, {"HOME": "/home", "PATH": "/bin"})
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(AdapterError, match="ANTHROPIC_API_KEY is not set"):
        claude_cli.paid_env(request)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GH_TOKEN", "should-not-pass")
    env = claude_cli.paid_env(request)
    assert env == {"HOME": "/home", "PATH": "/bin", "ANTHROPIC_API_KEY": "test-key"}
    monkeypatch.setattr(claude_cli.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, "2.1.999 (Claude Code)\n", ""))
    with pytest.raises(AdapterError, match="expected 2.1.285"):
        claude_cli.require_claude_cli(env)
    monkeypatch.setattr(claude_cli.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, "2.1.285 (Claude Code)\n", ""))
    assert claude_cli.require_claude_cli(env) is None


def test_claude_code_rejects_an_unreadable_transcript():
    with pytest.raises(AdapterError, match="unreadable transcript"):
        claude_code.events('{"type": "system"}\nnot json\n')


def test_every_flag_passed_to_claude_exists_in_the_pinned_cli():
    """tests/fixtures/claude-code-help.txt is `claude --help` from the pinned version (flag lines only)."""
    help_text = (FIXTURES / "claude-code-help.txt").read_text()
    assert claude_cli.CLAUDE_CODE_VERSION in help_text.splitlines()[0]
    flags = [part for part in claude_code.ClaudeCodeReview("m", 1.0).command() if part.startswith("-")]
    missing = [flag for flag in flags if not re.search(rf"(^|[ ,]){re.escape(flag)}([ ,]|$)", help_text, re.M)]
    assert missing == []


def test_failed_reviews_report_what_they_spent():
    with pytest.raises(AdapterError) as failed:
        claude_code.to_output(stream({"is_error": True, "result": "boom", "total_cost_usd": 0.9}, []), set())
    assert failed.value.cost_usd == 0.9
    down = {"status": "PASS", "reviewer": {"outcome": "unavailable", "reason": "x", "spent_usd": 1.2}}
    with pytest.raises(AdapterError) as unavailable:
        rigour_reviewer.to_output(down)
    assert unavailable.value.cost_usd == 1.2


def test_rigour_cost_prefers_spent_usd_over_cost_usd():
    newer = copy.deepcopy(REVIEWER_REPORT)
    newer["reviewer"]["spent_usd"] = 0.97          # every run, failed passes, retries (6.9.0)
    assert rigour_reviewer.to_output(newer).cost_usd == 0.97
    assert rigour_reviewer.to_output(REVIEWER_REPORT).cost_usd == 0.42   # older versions: cost_usd


def test_code_review_gets_exactly_rigours_tool_access_and_isolation():
    from bench.adapters import tool_access
    command = claude_code.ClaudeCodeReview("model-x").command()
    allowed = command[command.index("--allowedTools") + 1:command.index("--disallowedTools")]
    denied = command[command.index("--disallowedTools") + 1:]
    assert tuple(allowed) == tool_access.READ_ONLY_TOOLS
    assert tuple(denied) == tool_access.DENIED_TOOLS + tool_access.NETWORK_DENIED
    assert not any(t.startswith("Bash") and "git" not in t for t in allowed)      # no general shell
    for flag in ("--strict-mcp-config", "--setting-sources", "--settings", "--max-turns", "--mcp-config"):
        assert flag in command


def test_rigours_pinned_tool_lists_match_ours():
    """tests/fixtures/rigour-6.9.0-claude-tools.txt: the two lines from Rigour 6.9.0's reviewer/adapters.js."""
    from bench.adapters import tool_access
    source = (FIXTURES / "rigour-6.9.0-claude-tools.txt").read_text()
    assert tool_access.parse_rigour_tools(source) == (tool_access.READ_ONLY_TOOLS, tool_access.DENIED_TOOLS)
    tool_access.check_parity(source)
    widened = source.replace("'Bash(git grep:*)'", "'Bash(git grep:*)', 'Bash'")
    with pytest.raises(tool_access.ToolAccessError, match="differs"):
        tool_access.check_parity(widened)
    with pytest.raises(tool_access.ToolAccessError, match="not found"):
        tool_access.parse_rigour_tools("nothing here")


def test_code_review_env_disables_memory_files(monkeypatch, tmp_path):
    from bench.adapters import tool_access
    seen = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(claude_code, "require_claude_cli", lambda env: None)

    def fake_run(args, **kwargs):
        seen.update(kwargs["env"])
        return subprocess.CompletedProcess(args, 0, '{"type": "result", "total_cost_usd": 0.1, "num_turns": 1}\n', "")
    monkeypatch.setattr(claude_code.subprocess, "run", fake_run)
    diff = tmp_path / "x.diff"
    diff.write_text("")
    claude_code.ClaudeCodeReview("m").review(ReviewInput(tmp_path, "b", "h", diff, None, 5, {"HOME": "/h", "PATH": "/b"}))
    assert all(seen[k] == v for k, v in tool_access.ISOLATION_ENV.items()) and seen["ANTHROPIC_API_KEY"] == "test-key"
