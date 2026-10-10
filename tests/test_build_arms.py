"""The build track's workspace, arms and agent record: equal setups but for Rigour, and nothing after the parent."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bench.buildtrack import arms
from bench.buildtrack.agent import run_agent, transcript_numbers
from bench.buildtrack.toolchains import GO, ToolchainError, packages, toolchain
from bench.buildtrack.workspace import FIXED, final_diff, parent, snapshot
from bench.harness.gitrepo import RepoCheckout


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(repo), **FIXED}).stdout


@pytest.fixture()
def history(tmp_path):
    """main: base -> later; a branch from base: first -> second (the pull request's heads)."""
    src = tmp_path / "src"
    subprocess.run(["git", "init", "-q", "-b", "main", str(src)], check=True)
    shas = {}
    for name, branch in (("base", "main"), ("later", "main")):
        (src / "a.go").write_text(f"package a // {name}\n")
        git(src, "add", "-A")
        git(src, "commit", "-qm", name)
        shas[name] = git(src, "rev-parse", "HEAD").strip()
    git(src, "checkout", "-q", "-b", "pr", shas["base"])
    for name in ("first", "second"):
        (src / "b.go").write_text(f"package a // {name}\n")
        git(src, "add", "-A")
        git(src, "commit", "-qm", name)
        shas[name] = git(src, "rev-parse", "HEAD").strip()
    checkout = RepoCheckout(str(src), tmp_path / "clone", blobless=False)
    checkout.ensure_clone()
    return checkout, shas, tmp_path


def test_the_agent_starts_at_the_parent_with_no_history_no_remote_and_nothing_later(history):
    checkout, shas, tmp = history
    start = parent(checkout, {"pr": 1, "base_sha": shas["later"], "first_head": shas["first"]})
    assert start == shas["base"]
    repo = snapshot(checkout, start, tmp / "task")
    assert git(repo, "rev-list", "--all").strip().count("\n") == 0          # one commit, nothing else
    assert git(repo, "remote").strip() == "" and git(repo, "branch", "--format=%(refname:short)").split() == ["main"]
    assert (repo / "a.go").read_text() == "package a // base\n" and not (repo / "b.go").exists()


def test_the_final_diff_counts_committed_uncommitted_and_new_files(history):
    checkout, shas, tmp = history
    repo = snapshot(checkout, shas["base"], tmp / "task")
    (repo / "a.go").write_text("package a // changed\n")
    git(repo, "commit", "-qam", "agent commit")
    (repo / "new_test.go").write_text("package a\n")
    diff = final_diff(repo)
    assert "+package a // changed" in diff and "new_test.go" in diff


def test_both_arms_run_the_same_agent_and_differ_only_by_rigour():
    alone = arms.agent_command("alone", "anthropic/m-1", GO, 60, 1.0, "task text")
    rigour = arms.agent_command("rigour", "anthropic/m-1", GO, 60, 1.0, "task text")
    assert set(alone) ^ set(rigour) == {"--strict-mcp-config", "--mcp-config", arms.NO_MCP, arms.RIGOUR_MCP_TOOLS}
    assert alone[alone.index("--max-budget-usd") + 1] == "1.00" and "user,project" in alone
    assert "Bash(go test:*)" in alone and "WebFetch" in alone[alone.index("--disallowedTools"):]
    with pytest.raises(arms.ArmError):
        arms.agent_command("other", "m", GO, 1, 1, "t")
    env = arms.agent_env({"HOME": "/h"}, GO, None)
    assert env["GOPROXY"] == "off" and env["npm_config_prefer_offline"] == "true" and "RIGOUR_REVIEW_LESSONS" not in env
    assert arms.agent_env({"HOME": "/h"}, GO, Path("/s.json"))["RIGOUR_REVIEW_LESSONS"] == "/s.json"


def test_what_rigour_setup_installed_is_recorded(tmp_path):
    home, repo = tmp_path / "home", tmp_path / "repo"
    (home / ".claude").mkdir(parents=True)
    (repo / ".git").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text(json.dumps({"hooks": {"Stop": [{"hooks": []}]}}))
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"rigour": {"command": "npx"}}, "userID": "x"}))
    (repo / ".git" / "rigour-enabled").write_text("")
    assert arms.installed(home, repo) == {"hooks": {"Stop": [{"hooks": []}]}, "mcp_servers": {"rigour": {"command": "npx"}},
                                          "switched_on": True}


def test_hidden_tests_run_per_go_package_and_unknown_repos_refuse():
    assert packages(["net/dns/a_test.go", "net/dns/b_test.go", "x_test.go"]) == ["./net/dns", "."]
    assert GO.test_command(["net/dns/a_test.go"]) == ["go", "test", "-count=1", "-json", "./net/dns"]
    with pytest.raises(ToolchainError):
        toolchain("o/unknown")


RESULT = {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.42, "num_turns": 7,
          "modelUsage": {"m": {"inputTokens": 100, "cacheReadInputTokens": 900, "outputTokens": 50}}}
CALL = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Edit", "input": {}},
                                                     {"type": "tool_use", "name": "WebFetch", "input": {}}]}}


def test_the_record_keeps_cost_tokens_turns_tool_use_and_leaks():
    out = transcript_numbers([CALL, RESULT])
    assert out == {"cost_usd": 0.42, "input_tokens": 1000, "output_tokens": 50, "turns": 7, "stop": "success",
                   "is_error": False, "tool_calls": {"Edit": 1, "WebFetch": 1}, "leak_signals": 1}


def test_an_agent_run_records_its_diff_even_when_the_transcript_is_unreadable(history):
    checkout, shas, tmp = history
    repo = snapshot(checkout, shas["base"], tmp / "task")
    script = f"printf '%s\\n' '{json.dumps(RESULT)}'; echo 'package a' > c.go"
    ok = run_agent(["sh", "-c", script], repo, {"PATH": "/usr/bin:/bin"}, timeout_s=30)
    assert ok["cost_usd"] == 0.42 and "c.go" in ok["diff"] and ok["timed_out"] is False
    broken = run_agent(["sh", "-c", "echo not-json"], repo, {"PATH": "/usr/bin:/bin"}, timeout_s=30)
    assert "unreadable transcript" in broken["error"] and "c.go" in broken["diff"]
