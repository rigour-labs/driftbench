"""The agent's shell, enforced (docs/BUILD_TRACK.md, "Arms"): a PreToolUse hook both arms get equally.

Claude Code's allow-list matches a Bash command by its prefix, so `go build
./... && cat > new.go <<EOF` passes as `go build`. In the pilot, agents wrote
files that way with no Edit or Write call. This hook allows one plain command
from the designed list and nothing else: no chaining, pipes, redirects,
substitution, and no `gofmt -w`. It exits 2 with the reason, which Claude Code
shows the agent, so it uses Edit and Write for files.

    python3 shell_guard.py < hook payload
"""
from __future__ import annotations

import json
import re
import shlex
import sys

ALLOWED = (("go", "build"), ("go", "test"), ("go", "vet"), ("gofmt",), ("git", "diff"), ("git", "status"),
           ("git", "log"), ("git", "show"))
FORBIDDEN = re.compile(r"&&|\|\||[;|<>`]|\$\(")


def refusal(command: str) -> str | None:
    """Why the command is not allowed, or None."""
    if FORBIDDEN.search(command):
        return "one plain command only: no &&, ||, ;, pipes, redirects or substitution; write files with Edit or Write"
    try:
        words = shlex.split(command)
    except ValueError as exc:
        print(f"bench shell: unparseable command: {exc}", file=sys.stderr)
        return "the command could not be parsed"
    if not words or not any(tuple(words[:len(a)]) == a for a in ALLOWED):
        return "allowed: go build, go test, go vet, gofmt (without -w), git diff/status/log/show"
    if words[0] == "gofmt" and any(w == "-w" or (w.startswith("-") and "w" in w[1:]) for w in words[1:]):
        return "gofmt may not write files (-w); edit files with Edit or Write"
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except ValueError as exc:                       # not a tool payload: nothing to guard
        print(f"bench shell: unreadable hook payload ({exc}); allowed", file=sys.stderr)
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    reason = refusal(str((payload.get("tool_input") or {}).get("command") or ""))
    if reason:
        print(f"bench shell: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
