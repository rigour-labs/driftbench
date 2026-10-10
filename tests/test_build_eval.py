"""The build track's checks: hidden tests that must discriminate, Rigour's hooks as Claude Code runs them, its log."""
from __future__ import annotations

import json
import os
import shutil

import pytest

from bench.buildtrack import hooks
from bench.buildtrack.evaluate import discriminates, outcome, rigour_events, run_tests, write_files
from bench.buildtrack.toolchains import GO

ENV = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def test_the_outcome_reads_go_test_events():
    run = lambda action, test="TestA": {"Action": action, "Test": test}
    assert outcome([run("pass"), run("pass", "TestB")]) == {"outcome": "pass", "pass": 2, "fail": 0, "skip": 0}
    assert outcome([run("pass"), run("fail", "TestB")])["outcome"] == "fail"
    assert outcome([run("skip")])["outcome"] == "no-tests"                       # skipped tests check nothing
    assert outcome([{"Action": "output", "Output": "FAIL\tx [build failed]\n"}])["outcome"] == "no-build"
    assert outcome([{"Action": "build-fail", "ImportPath": "x"}])["outcome"] == "no-build"


def test_only_tests_that_fail_at_the_parent_and_pass_merged_discriminate():
    assert discriminates({"outcome": "no-build"}, {"outcome": "pass"})
    assert discriminates({"outcome": "fail"}, {"outcome": "pass"})
    assert not discriminates({"outcome": "pass"}, {"outcome": "pass"})          # passes either way
    assert not discriminates({"outcome": "fail"}, {"outcome": "no-tests"})      # skips on this runner


@pytest.mark.skipif(shutil.which("go") is None, reason="go is not installed")
def test_hidden_tests_run_for_real_on_a_tiny_module(tmp_path):
    write_files(tmp_path, {"go.mod": "module example.com/a\n\ngo 1.21\n",
                           "a/a.go": "package a\n\nfunc Add(x, y int) int { return x + y }\n"})
    env = {**ENV, "HOME": str(tmp_path), "GOCACHE": str(tmp_path / "cache"), "GOPATH": str(tmp_path / "gopath")}
    test = {"a/a_test.go": "package a\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) { if Add(1, 2) != 3 { t.Fatal() } }\n"}
    write_files(tmp_path, test)
    assert run_tests(GO, list(test), tmp_path, env)["outcome"] == "pass"
    write_files(tmp_path, {"a/a.go": "package a\n\nfunc Add(x, y int) int { return x - y }\n"})
    assert run_tests(GO, list(test), tmp_path, env)["outcome"] == "fail"
    write_files(tmp_path, {"a/a.go": "package a\n"})                             # Add missing: the test can't build
    assert run_tests(GO, list(test), tmp_path, env)["outcome"] == "no-build"


def settings(home, script: str) -> None:
    """Installed hooks that run `script`, the shape `rigour setup` writes."""
    (home / ".claude").mkdir(parents=True)
    hook = [{"type": "command", "command": script}]
    (home / ".claude" / "settings.json").write_text(json.dumps({"hooks": {
        "PostToolUse": [{"matcher": "Write|Edit|MultiEdit", "hooks": hook}],
        "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "exit 2"}]}],
        "Stop": [{"hooks": hook}]}}))


BLOCKS_SECRETS = "if grep -rqs AKIA . ; then echo '{\"decision\":\"block\",\"reason\":\"secret\"}'; fi; exit 0"


def test_the_pre_pilot_check_passes_only_when_both_hooks_block_uncommitted_work(tmp_path):
    for script, ok in ((BLOCKS_SECRETS, True), ("exit 0", False)):
        home, repo = tmp_path / f"home-{ok}", tmp_path / f"repo-{ok}"
        repo.mkdir()
        settings(home, script)
        found = hooks.hookcheck(repo, home, ENV)
        assert found["ok"] is ok and found["edit_blocked"] is ok and found["stop_blocked"] is ok
        assert "const awsKey = \"AKIA" in (repo / "rigour_hookcheck.go").read_text()


def test_hooks_are_found_by_event_and_tool_and_block_by_exit_or_answer(tmp_path):
    settings(tmp_path, "true")
    assert hooks.commands(tmp_path, "PostToolUse", "Write") == ["true"] and hooks.commands(tmp_path, "PostToolUse", "Read") == []
    assert hooks.commands(tmp_path, "Stop") == ["true"]
    assert hooks.blocked(2, "") and hooks.blocked(0, 'info\n{"continue": false}') and hooks.blocked(0, '{"decision": "block"}')
    assert not hooks.blocked(0, "") and not hooks.blocked(1, '{"decision": "approve"}')
    payload = hooks.write_payload(tmp_path, "a/b.go", "x", "s")
    assert payload["tool_input"]["file_path"] == str(tmp_path / "a/b.go") and payload["hook_event_name"] == "PostToolUse"


def test_each_planted_key_pair_is_fresh_and_in_aws_format():
    import re
    first, second = hooks.planted(), hooks.planted()
    assert first != second and re.search(r'"AKIA[A-Z0-9]{16}"', first) and re.search(r'"[A-Za-z0-9/+]{40}"', first)


def test_rigours_event_log_is_summarised_as_numbers(tmp_path):
    (tmp_path / ".rigour").mkdir()
    events = ({"type": "lessons_served", "lessons": ["l1", "l2"]},
              {"type": "hook_check", "blocked": True, "files": ["a.go"], "findings": [{"gate": "security-patterns"}]},
              {"type": "stop_review", "blocked": True, "against": "uncommitted work"},
              {"type": "hook_check", "blocked": False, "files": ["a.go"]},
              {"type": "stop_review", "blocked": False})
    (tmp_path / ".rigour" / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\nnot json\n")
    out = rigour_events(tmp_path)
    assert out["events"] == 5 and out["lessons_served"] == 2 and out["gates"] == {"security-patterns": 1}
    assert out["by_type"]["hook_check"] == {"events": 2, "blocked": 1} and out["by_type"]["stop_review"]["blocked"] == 1
    assert out["after_block"] == {"files_blocked": 1, "files_fixed": 1, "stop_blocked": True, "stop_cleared": True}
    assert rigour_events(tmp_path / "none")["events"] == 0


def test_every_block_on_the_approved_change_is_listed_as_a_false_block(tmp_path):
    home, repo = tmp_path / "home", tmp_path / "repo"
    repo.mkdir()
    settings(home, BLOCKS_SECRETS)
    clean = hooks.false_blocks(repo, home, ENV, {"a.go": "package a\n"})
    assert clean == {"false_blocks": 0, "listed": [], "files": 1}
    flagged = hooks.false_blocks(repo, home, ENV, {"b.go": hooks.planted()})
    assert flagged["false_blocks"] == 2 and flagged["listed"][0].startswith("b.go: ") and flagged["listed"][1].startswith("stop: ")
