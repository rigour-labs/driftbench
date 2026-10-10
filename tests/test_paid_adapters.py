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
                 "blind": True,
                 "record": {"judges": [{"reviewer": "claude", "model": "m", "cost_usd": 0.42}], "blind": True,
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
    assert adapter.command(request, "/home/rigour-bench.yml")[-6:] == ["--reviewer", "--single", "--blind", "-c",
                                                                       "/home/rigour-bench.yml", "--orchestrator"]
    config = rigour_reviewer.reviewer_config("model-x", 600)["review"]["reviewer"]
    assert config["models"] == {"claude": "model-x"} and config["timeout_ms"] == 600_000 and config["mode"] == "single"


def test_a_review_not_marked_blind_is_a_leak():
    """--blind means no pull request context; a report that doesn't say so can't be trusted to be time-correct."""
    for change in ({"blind": False}, {"blind": None}, {"record": {**REVIEWER_REPORT["reviewer"]["record"], "blind": None}}):
        report = copy.deepcopy(REVIEWER_REPORT)
        report["reviewer"].update(change)
        assert rigour_reviewer.to_output(report).leak_signals == 1
    no_record = copy.deepcopy(REVIEWER_REPORT)
    del no_record["reviewer"]["record"]
    assert rigour_reviewer.to_output(no_record).leak_signals == 0          # e.g. nothing to review: no record


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
    """tests/fixtures/rigour-claude-tools.txt: the two lines from the pinned Rigour's reviewer/adapters.js."""
    from bench.adapters import tool_access
    source = (FIXTURES / "rigour-claude-tools.txt").read_text()
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


def test_the_pinned_rigour_reports_spent_usd_and_it_is_read():
    """tests/fixtures/rigour-reviewer-json-keys.txt: the reviewer JSON lines from the pinned Rigour's dist.
    A recorded reviewer report needs a paid run; this checks the key exists and is the one we read first."""
    keys = (FIXTURES / "rigour-reviewer-json-keys.txt").read_text()
    assert "spent_usd: result.spentUsd" in keys and "cost_usd: result.costUsd" in keys
    assert "blind: !!result.blind" in keys                         # the pinned reviewer reports --blind
    report = copy.deepcopy(REVIEWER_REPORT)
    report["reviewer"].update(spent_usd=1.37, cost_usd=0.42)       # all runs vs the verdict's judges only
    assert rigour_reviewer.to_output(report).cost_usd == 1.37
    cached = copy.deepcopy(REVIEWER_REPORT)
    cached["reviewer"].update(cached=True, spent_usd=0, cost_usd=0.42)   # a cached verdict re-reports old cost
    out = rigour_reviewer.to_output(cached)
    assert out.cost_usd == 0 and out.model_runs == 0                 # honest $0, nothing ran


def test_claude_code_tokens_come_from_model_usage_and_diagnostics_are_recorded():
    result = {"type": "result", "is_error": False, "result": "Issue at `src/a.py:3`", "total_cost_usd": 0.3,
              "num_turns": 5, "usage": {"input_tokens": 0, "output_tokens": 0},
              "modelUsage": {"m1": {"inputTokens": 100, "outputTokens": 20, "cacheReadInputTokens": 1000,
                                    "cacheCreationInputTokens": 50},
                             "m2": {"inputTokens": 10, "outputTokens": 5}}}
    out = claude_code.to_output([result], {"src/a.py"})
    assert (out.input_tokens, out.output_tokens, out.model_runs) == (1160, 25, 5)
    assert out.diagnostics == {"result_chars": 21, "citation_like": 1, "link_like": 0, "cited_changed": 1,
                               "files_named": 1, "num_turns": 5}
    plain = claude_code.to_output([{**result, "modelUsage": None, "usage": {"input_tokens": 7, "output_tokens": 3}}],
                                  {"src/a.py"})
    assert (plain.input_tokens, plain.output_tokens) == (7, 3)


def test_claude_code_with_zero_turns_but_output_counts_as_a_model_run():
    """/code-review reported num_turns 0 while its agents ran (re-smoke 38027582432): it must never pass as free."""
    from bench.harness.paid import unreported
    result = {"type": "result", "is_error": False, "result": "review text", "total_cost_usd": 0, "num_turns": 0,
              "modelUsage": {"m": {"inputTokens": 500, "outputTokens": 40}}}
    out = claude_code.to_output([result], {"src/a.py"})
    assert out.model_runs == 1 and unreported(out)               # a $0 report from a model that ran: charged the bound
    idle = claude_code.to_output([{**result, "modelUsage": {}, "usage": {}}], {"src/a.py"})
    assert idle.model_runs == 0


def test_what_the_rigour_reviewer_held_back_is_kept_with_full_messages():
    """tests/fixtures/rigour-reviewer-held-back.json (synthetic): dropped, unverified, disputed and dismissed
    findings, and the shown tally, are kept so "saw it and filtered it" can be told from "never saw it"."""
    from tests.fixture_files import load_json
    report = load_json("rigour-reviewer-held-back.json")
    out = rigour_reviewer.to_output(report)
    assert [f.message for f in out.findings] == ["the retry loop never stops after shutdown"]   # served only
    held = out.held_back
    assert held["counts"] == {"dropped": 1, "unverified": 1, "disputed": 1, "dismissed": 0}
    assert held["dropped"][0]["issue"] == "off-by-one in the window check" and held["dropped"][0]["why"]
    assert held["shown"] == {"blocking": 0, "should_fix": 1, "folded": 3}
    bare = rigour_reviewer.to_output({"status": "PASS", "reviewer": {**report["reviewer"], "dropped": None,
                                                                      "shown": "n/a"}})
    assert bare.held_back["counts"]["dropped"] == 0 and bare.held_back["shown"] is None
