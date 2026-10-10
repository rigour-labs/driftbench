"""The two arms (docs/BUILD_TRACK.md, "Arms"): the same agent alone, and with Rigour set up around it.

Both arms run the same pinned Claude Code, model, provider, tools, turn limit,
timeout and dollar bound, with user and project settings and the
repository's own CLAUDE.md / AGENTS.md. The one intended difference is
Rigour: in arm B, `rigour setup` (its defaults) has written its hooks and its
MCP server into the sandbox HOME, the agent may call Rigour's MCP tools, and
the lesson store for the task's cutoff is served through
RIGOUR_REVIEW_LESSONS. Arm A's HOME is empty and it has no MCP servers.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from bench.adapters.tool_access import NETWORK_DENIED
from bench.buildtrack.toolchains import Toolchain

ARMS = ("alone", "rigour")
RIGOUR_VERSION = "7.0.0-rc.2"  # rigour next f5c9452: phase-1 fixes, change-scoped file-size in hooks, camelCase secrets
AGENT_TOOLS = ("Read", "Grep", "Glob", "Edit", "Write", "MultiEdit", "TodoWrite",
               "Bash(git diff:*)", "Bash(git status:*)", "Bash(git log:*)", "Bash(git show:*)")
AGENT_DENIED = (*NETWORK_DENIED, "Bash(git push:*)", "Bash(git remote:*)", "Bash(git fetch:*)", "Bash(npm:*)",
                "Bash(npx:*)", "Bash(pip:*)")
RIGOUR_MCP_TOOLS = "mcp__rigour__*"
NO_MCP = '{"mcpServers":{}}'
SHARED_ENV = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "npm_config_prefer_offline": "true"}


class ArmError(RuntimeError):
    pass


def agent_command(arm: str, model: str, toolchain: Toolchain, max_turns: int, max_usd: float, prompt: str) -> list[str]:
    if arm not in ARMS:
        raise ArmError(f"unknown arm {arm!r}; known: {', '.join(ARMS)}")
    allowed = [*AGENT_TOOLS, *toolchain.commands, *([RIGOUR_MCP_TOOLS] if arm == "rigour" else [])]
    mcp = [] if arm == "rigour" else ["--strict-mcp-config", "--mcp-config", NO_MCP]
    return ["claude", "-p", prompt, "--model", model, "--output-format", "stream-json", "--verbose",
            "--max-turns", str(max_turns), "--max-budget-usd", f"{max_usd:.2f}", "--setting-sources", "user,project",
            "--no-session-persistence", *mcp, "--allowedTools", *allowed, "--disallowedTools", *AGENT_DENIED]


def agent_env(base: dict[str, str], toolchain: Toolchain, lessons: Path | None) -> dict[str, str]:
    """The sandbox env, the toolchain's offline settings, and arm B's lesson store."""
    return {**base, **SHARED_ENV, **toolchain.env, **({"RIGOUR_REVIEW_LESSONS": str(lessons)} if lessons else {})}


def setup_rigour(repo: Path, env: dict[str, str], version: str) -> dict:
    """`rigour setup` with its defaults, then what it installed: hooks, MCP servers, the switch-on marker."""
    result = subprocess.run(["npx", "--yes", f"@rigour-labs/cli@{version}", "setup"], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=900, check=False)
    if result.returncode != 0:
        raise ArmError(f"rigour setup failed: {(result.stderr or result.stdout).strip()[:300]}")
    return installed(Path(env["HOME"]), repo)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError) as exc:
        raise ArmError(f"unreadable {path}: {exc}") from exc


def hook_summary(hooks: dict) -> dict:
    """Per event, each installed hook's matcher, its command's sha256 and its first 120 characters."""
    return {event: [{"matcher": entry.get("matcher"),
                     "commands": [{"sha256": hashlib.sha256(str(h.get("command")).encode()).hexdigest(),
                                   "head": str(h.get("command"))[:120]} for h in entry.get("hooks", [])]}
                    for entry in entries] for event, entries in hooks.items()}


def installed(home: Path, repo: Path) -> dict:
    """What a run records about arm B's setup (never secrets: these files hold commands, not keys)."""
    return {"hooks": hook_summary(read_json(home / ".claude" / "settings.json").get("hooks") or {}),
            "mcp_servers": read_json(home / ".claude.json").get("mcpServers") or {},
            "switched_on": (repo / ".git" / "rigour-enabled").exists()}


def prompt_for(statement: str) -> str:
    """The task as the agent sees it: the statement, and nothing about the pull request that followed."""
    return ("Make the change this task describes in this repository. Write or update tests where they belong, "
            "and make sure the code builds.\n\nTask:\n" + statement)
