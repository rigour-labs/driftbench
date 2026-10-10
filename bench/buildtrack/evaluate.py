"""What a task's result is checked against (docs/BUILD_TRACK.md, "Measures").

- Hidden tests: the pull request's own test files at its merged head, written
  into a tree, then the repository's test command on their packages. The
  outcome is pass, fail, no-build or no-tests (every test skipped or none ran).
- Discriminating: the hidden tests fail or don't build at the parent and pass
  on the merged head; otherwise the task checks nothing and is passed over.
- Rigour's own event log (arm B): what it did during the run (brief and lessons served, edit checks,
  blocks, stop reviews) and what the agent fixed after a block.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from bench.buildtrack.toolchains import Toolchain
from bench.harness.gitrepo import RepoCheckout

TEST_TIMEOUT_S = 1200
BUILD_FAILED = ("[build failed]", "[setup failed]")
EVENTS = Path(".rigour") / "events.jsonl"


def files_at(checkout: RepoCheckout, sha: str, paths: list[str]) -> dict[str, str]:
    return {path: checkout.git("show", f"{sha}:{path}").stdout for path in paths}


def write_files(repo: Path, files: dict[str, str]) -> None:
    for path, content in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(content, encoding="utf-8")


def outcome(events: list[dict]) -> dict:
    """pass / fail / no-build / no-tests, with per-test counts, from `go test -json` events."""
    counts = {"pass": 0, "fail": 0, "skip": 0}
    no_build = False
    for e in events:
        action = e.get("Action")
        if action == "build-fail" or any(m in str(e.get("Output") or "") for m in BUILD_FAILED):
            no_build = True
        elif e.get("Test") and action in counts:
            counts[action] += 1
    result = ("no-build" if no_build else "fail" if counts["fail"] else "no-tests" if not counts["pass"]
              else "pass")
    return {"outcome": result, **counts}


def parse_json_lines(text: str, source: str) -> list[dict]:
    """The JSON objects in a tool's line output; any other line is reported and skipped."""
    parsed = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            print(f"{source}: skipped a non-JSON line: {line[:120]}", file=sys.stderr)
            continue
        if isinstance(item, dict):
            parsed.append(item)
    return parsed


def run_tests(toolchain: Toolchain, test_files: list[str], repo: Path, env: dict[str, str]) -> dict:
    result = subprocess.run(toolchain.test_command(test_files), cwd=repo, env={**env, **toolchain.env},
                            capture_output=True, text=True, timeout=TEST_TIMEOUT_S, check=False)
    found = outcome(parse_json_lines(result.stdout, "go test"))
    if found["outcome"] == "no-tests" and result.returncode != 0:
        found["outcome"] = "no-build"
    return found


def discriminates(at_parent: dict, at_merged: dict) -> bool:
    return at_parent["outcome"] in ("fail", "no-build") and at_merged["outcome"] == "pass"


def fixed_after_block(lines: list[dict]) -> dict:
    """Blocks the agent then cleared: an edit check that blocked a file and a later one on that file that passed;
    a stop review that blocked and a later one that didn't."""
    open_files: set[str] = set()
    fixed_files: set[str] = set()
    stop_blocked = stop_cleared = False
    for e in lines:
        if e.get("type") == "hook_check":
            files = set(e.get("files") or [])
            if e.get("blocked"):
                open_files |= files
            else:
                fixed_files |= files & open_files
        elif e.get("type") == "stop_review":
            stop_blocked = stop_blocked or bool(e.get("blocked"))
            stop_cleared = stop_cleared or (stop_blocked and not e.get("blocked"))
    return {"files_blocked": len(open_files), "files_fixed": len(fixed_files),
            "stop_blocked": stop_blocked, "stop_cleared": stop_cleared}


def rigour_events(repo: Path) -> dict:
    """Arm B: what Rigour did during the run, from its own event log, numbers only: every event by type and how
    many blocked, the checks that fired (by gate), lessons served, and what the agent fixed after a block."""
    path = repo / EVENTS
    lines = parse_json_lines(path.read_text(encoding="utf-8"), str(EVENTS)) if path.exists() else []
    by_type: dict[str, dict[str, int]] = {}
    gates: dict[str, int] = {}
    for e in lines:
        entry = by_type.setdefault(str(e.get("type")), {"events": 0, "blocked": 0})
        entry["events"] += 1
        entry["blocked"] += 1 if e.get("blocked") else 0
        for finding in e.get("findings") or []:
            if isinstance(finding, dict):
                gates[str(finding.get("gate"))] = gates.get(str(finding.get("gate")), 0) + 1
    served = sum(len(e.get("lessons") or []) for e in lines if e.get("type") == "lessons_served")
    return {"events": len(lines), "by_type": by_type, "gates": gates, "lessons_served": served,
            "after_block": fixed_after_block(lines)}
