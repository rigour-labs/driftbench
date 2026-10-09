"""What the paid entrants share: one pinned Claude Code CLI, one model, one provider.

Rigour's reviewer and Claude Code's /code-review both drive the `claude` CLI
on PATH. For a fair comparison they use the same CLI version (checked before
every review), the same model (pinned by full ID), the same timeout and the
same provider. The only variables added to the harness's sandbox env are the
provider's:
- anthropic: ANTHROPIC_API_KEY;
- openrouter: Claude Code's gateway settings, ANTHROPIC_BASE_URL pointing at
  OpenRouter, ANTHROPIC_AUTH_TOKEN from OPENROUTER_API_KEY, and
  ANTHROPIC_API_KEY set empty so no Anthropic key can be used instead.
"""
from __future__ import annotations

import os
import subprocess

from bench.harness.types import AdapterError, ReviewInput

CLAUDE_CODE_VERSION = "2.1.285"  # its --help is recorded in tests/fixtures; a bump is its own pull request
KEY_NAME = "ANTHROPIC_API_KEY"
PROVIDERS = ("anthropic", "openrouter")
OPENROUTER_KEY = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
GATEWAY_NAMES = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", KEY_NAME)


def provider_env_names(provider: str) -> tuple[str, ...]:
    """The variables an entrant adds to the sandbox env for `provider` (recorded with every head)."""
    return GATEWAY_NAMES if provider == "openrouter" else (KEY_NAME,)


def check_model_id(model: str, provider: str) -> None:
    """Through OpenRouter, a pinned Anthropic model: `anthropic/<name>` with a version, never an alias."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; known: {', '.join(PROVIDERS)}")
    if provider == "openrouter" and not (model.startswith("anthropic/") and any(c.isdigit() for c in model)
                                         and "latest" not in model):
        raise ValueError(f"{model!r}: through OpenRouter, give a full versioned anthropic/ model ID, not an alias")


def paid_env(request: ReviewInput, provider: str = "anthropic") -> dict[str, str]:
    """The sandbox env plus the provider's variables, and nothing else from this machine."""
    if provider == "openrouter":
        key = os.environ.get(OPENROUTER_KEY)
        if not key:
            raise AdapterError(f"{OPENROUTER_KEY} is not set; a paid entrant can't run through OpenRouter without it")
        return {**request.env, "ANTHROPIC_BASE_URL": OPENROUTER_BASE_URL, "ANTHROPIC_AUTH_TOKEN": key, KEY_NAME: ""}
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
