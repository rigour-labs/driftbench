"""What the paid entrants share: one pinned Claude Code CLI, one model, one key.

Rigour's reviewer and Claude Code's /code-review both drive the `claude` CLI
on PATH. For a fair comparison they use the same CLI version (checked before
every review), the same model (pinned by full ID) and the same timeout. The
only variable added to the harness's sandbox env is the model provider's key.
"""
from __future__ import annotations

import os
import subprocess

from bench.harness.types import AdapterError, ReviewInput

CLAUDE_CODE_VERSION = "2.1.285"  # its --help is recorded in tests/fixtures; a bump is its own pull request
KEY_NAME = "ANTHROPIC_API_KEY"


def paid_env(request: ReviewInput) -> dict[str, str]:
    """The sandbox env plus the provider key, and nothing else from this machine."""
    key = os.environ.get(KEY_NAME)
    if not key:
        raise AdapterError(f"{KEY_NAME} is not set; a paid entrant can't run without it")
    return {**request.env, KEY_NAME: key}


def require_claude_cli(env: dict[str, str]) -> None:
    """Refuse to review unless `claude` on PATH is the pinned version."""
    try:
        result = subprocess.run(["claude", "--version"], env=env, capture_output=True, text=True, timeout=60,
                                check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AdapterError(f"the claude CLI could not run: {exc}") from exc
    found = (result.stdout or "").split()
    if not found or found[0] != CLAUDE_CODE_VERSION:
        raise AdapterError(f"claude CLI is {' '.join(found) or 'missing'}; expected {CLAUDE_CODE_VERSION}")
