"""Identical, read-only tool access for every paid entrant (docs/ENTRANTS.md, "Tool access").

A paid review runs with a real key in its environment while it reads
untrusted public pull request content, so nothing in it may run arbitrary
shell, write files, or reach the network. Rigour's reviewer runs Claude Code
with an explicit read-only allow-list; Claude Code's /code-review gets exactly
the same list, the same isolation (no MCP servers, no hooks or user settings,
no memory files, a turn limit), plus an explicit deny of web and GitHub tools.
Before a paid run, the list is read from the pinned Rigour CLI and must match
ours, or the run stops.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

READ_ONLY_TOOLS = ("Read", "Grep", "Glob", "Bash(git diff:*)", "Bash(git show:*)", "Bash(git log:*)",
                   "Bash(git grep:*)")
DENIED_TOOLS = ("Edit", "Write", "NotebookEdit", "Bash(git push:*)", "Bash(git commit:*)")
NETWORK_DENIED = ("WebFetch", "WebSearch", "Bash(gh:*)", "Bash(curl:*)", "Bash(wget:*)")
ISOLATION_ARGS = ("--max-turns", "80", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                  "--setting-sources", "project", "--settings", '{"hooks":{},"outputStyle":"default"}')
ISOLATION_ENV = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}

ALLOWED_RE = re.compile(r"READ_ONLY_TOOLS\s*=\s*\[([^\]]*)\]")
DENIED_RE = re.compile(r"'--disallowedTools',\s*((?:'[^']*',?\s*)+)")
QUOTED_RE = re.compile(r"'([^']*)'")
ADAPTERS_JS = Path("node_modules/@rigour-labs/core/dist/review/reviewer/adapters.js")


class ToolAccessError(RuntimeError):
    pass


def tool_record() -> dict:
    """What run.json records about tool access for the paid entrants."""
    return {"allowed": list(READ_ONLY_TOOLS), "denied": list(DENIED_TOOLS), "network_denied": list(NETWORK_DENIED)}


def parse_rigour_tools(source: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(allowed, denied) from the text of Rigour's reviewer/adapters.js."""
    allowed, denied = ALLOWED_RE.search(source), DENIED_RE.search(source)
    if not allowed or not denied:
        raise ToolAccessError("Rigour's Claude Code tool lists were not found in its adapters.js")
    return tuple(QUOTED_RE.findall(allowed.group(1))), tuple(QUOTED_RE.findall(denied.group(1)))


def check_parity(source: str) -> None:
    allowed, denied = parse_rigour_tools(source)
    if allowed != READ_ONLY_TOOLS or denied != DENIED_TOOLS:
        raise ToolAccessError(f"Rigour's reviewer tool access differs from DriftBench's: allowed {allowed}, "
                              f"denied {denied}; update bench/adapters/tool_access.py before a paid run")


def pinned_rigour_source(version: str, npm_cache: Path) -> str:
    """adapters.js of the pinned Rigour CLI, fetched into `npm_cache` through npx."""
    env = {"PATH": os.environ.get("PATH", os.defpath), "npm_config_cache": str(npm_cache), "HOME": str(npm_cache)}
    subprocess.run(["npx", "--yes", f"@rigour-labs/cli@{version}", "--version"], capture_output=True, text=True,
                   check=False, env=env)
    for package in sorted(npm_cache.glob("_npx/*/node_modules/@rigour-labs/cli/package.json")):
        try:
            found = json.loads(package.read_text(encoding="utf-8")).get("version")
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolAccessError(f"unreadable {package}: {exc}") from exc
        if found == version:
            return (package.parents[3] / ADAPTERS_JS).read_text(encoding="utf-8")
    raise ToolAccessError(f"Rigour CLI {version} not found in {npm_cache} to check its tool access")
