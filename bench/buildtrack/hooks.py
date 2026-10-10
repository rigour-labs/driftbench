"""Rigour's hooks as Claude Code runs them, outside an agent (docs/BUILD_TRACK.md).

The commands come from what `rigour setup` installed in the sandbox HOME, and
each gets the payload Claude Code would send on stdin. A hook blocks when it
exits 2, or answers `{"decision": "block"}` or `{"continue": false}`.

Used twice:
- the pre-pilot check: a proven issue (a fresh fake AWS key pair) planted, uncommitted, in a task's
  snapshot must be blocked by both the Stop hook and the edit hook (the
  agent cannot commit, so a hook that checks only commits would check
  nothing), or the pilot does not run;
- false blocks: the real merged change, which maintainers approved, written
  into the parent's snapshot. Anything the hooks block there is a false block.
"""
from __future__ import annotations

import json
import re
import secrets
import string
import subprocess
from pathlib import Path

from bench.buildtrack.arms import read_json
from bench.buildtrack.evaluate import parse_json_lines

HOOK_TIMEOUT_S = 900
BLOCK_EXIT = 2
KEY_ALPHABET = string.ascii_uppercase + string.digits
SECRET_ALPHABET = string.ascii_letters + string.digits + "/+"


def planted() -> str:
    """A Go file with a fake AWS key pair in AWS's format, made fresh for each check: no real credential, nothing
    secret-shaped in this source, and not AWS's published example pair, which detectors may allow."""
    rng = secrets.SystemRandom()
    key = "AKIA" + "".join(rng.choice(KEY_ALPHABET) for _ in range(16))
    secret = "".join(rng.choice(SECRET_ALPHABET) for _ in range(40))
    return f'package main\n\nconst awsKey = "{key}"\nconst awsSecret = "{secret}"\n'


def commands(home: Path, event: str, tool: str | None = None) -> list[str]:
    """The installed commands for a hook event, for `tool` when the event matches on tools."""
    found = []
    for entry in read_json(home / ".claude" / "settings.json").get("hooks", {}).get(event, []):
        matcher = entry.get("matcher")
        if tool is None or not matcher or re.fullmatch(matcher, tool):
            found += [h["command"] for h in entry.get("hooks", []) if h.get("type") == "command"]
    return found


def stop_payload(repo: Path, session: str) -> dict:
    return {"hook_event_name": "Stop", "session_id": session, "cwd": str(repo), "stop_hook_active": False,
            "transcript_path": "/dev/null"}


def write_payload(repo: Path, path: str, content: str, session: str) -> dict:
    return {"hook_event_name": "PostToolUse", "session_id": session, "cwd": str(repo), "tool_name": "Write",
            "tool_input": {"file_path": str(repo / path), "content": content}, "tool_response": {"success": True}}


def blocked(code: int, stdout: str) -> bool:
    """Exit 2, or a last JSON answer that blocks."""
    if code == BLOCK_EXIT:
        return True
    answers = parse_json_lines(stdout, "hook")
    return bool(answers) and (answers[-1].get("decision") == "block" or answers[-1].get("continue") is False)


def run_hook(command: str, payload: dict, repo: Path, env: dict[str, str]) -> dict:
    """{blocked, exit, message}: one hook run the way Claude Code runs it."""
    result = subprocess.run(["sh", "-c", command], input=json.dumps(payload), cwd=repo, capture_output=True,
                            text=True, timeout=HOOK_TIMEOUT_S, check=False,
                            env={**env, "CLAUDE_PROJECT_DIR": str(repo)})
    message = (result.stdout or "").strip() or (result.stderr or "").strip()
    return {"blocked": blocked(result.returncode, result.stdout or ""), "exit": result.returncode,
            "message": message[-300:]}


def on_files(repo: Path, home: Path, env: dict[str, str], files: dict[str, str], session: str) -> dict:
    """Write `files` into the working tree, uncommitted, then run every edit hook on each and the Stop hook once."""
    for path, content in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(content, encoding="utf-8")
    edits = {path: [run_hook(c, write_payload(repo, path, content, session), repo, env)
                    for c in commands(home, "PostToolUse", "Write")] for path, content in files.items()}
    stops = [run_hook(c, stop_payload(repo, session), repo, env) for c in commands(home, "Stop")]
    return {"edits": edits, "stop": stops}


def hookcheck(repo: Path, home: Path, env: dict[str, str]) -> dict:
    """The pre-pilot check: a planted, uncommitted secret must be blocked by the Stop hook and an edit hook."""
    found = on_files(repo, home, env, {"rigour_hookcheck.go": planted()}, "hookcheck")
    edit = any(r["blocked"] for runs in found["edits"].values() for r in runs)
    stop = any(r["blocked"] for r in found["stop"])
    return {**found, "edit_blocked": edit, "stop_blocked": stop, "ok": edit and stop}


def false_blocks(repo: Path, home: Path, env: dict[str, str], merged_files: dict[str, str]) -> dict:
    """The real merged change written into the parent's snapshot: every hook block on it is a false block."""
    found = on_files(repo, home, env, merged_files, "reference")
    listed = [f"{path}: {r['message']}" for path, runs in found["edits"].items() for r in runs if r["blocked"]]
    listed += [f"stop: {r['message']}" for r in found["stop"] if r["blocked"]]
    return {"false_blocks": len(listed), "listed": listed, "files": len(merged_files)}
