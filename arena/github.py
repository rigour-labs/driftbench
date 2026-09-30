"""GitHub REST reads through the `gh` CLI, with backoff for rate limits.

Search and secondary rate limits are hit quickly when mining many PRs, so
every call is spaced out, and a rate-limited call is retried with growing
waits instead of failing the run.
"""
from __future__ import annotations

import json
import subprocess
import time

MIN_INTERVAL_S = 0.8
MAX_RETRIES = 6
_last_call = 0.0


class GitHubError(RuntimeError):
    pass


def api(path: str, paginate: bool = False, params: dict[str, str] | None = None) -> list | dict:
    """GET `path`; with `paginate`, every page concatenated into one list."""
    args = ["gh", "api", "-X", "GET", path, "-H", "Accept: application/vnd.github+json"]
    for key, value in (params or {}).items():
        args += ["-f", f"{key}={value}"]
    if paginate:
        args += ["--paginate", "--slurp"]
    for attempt in range(MAX_RETRIES):
        _throttle()
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            data = json.loads(result.stdout or "null")
            return [item for page in data for item in page] if paginate else data
        if not _rate_limited(result.stderr + result.stdout):
            raise GitHubError(f"gh api {path} failed: {(result.stderr or result.stdout).strip()[:300]}")
        time.sleep(60 * (attempt + 1))
    raise GitHubError(f"gh api {path}: still rate limited after {MAX_RETRIES} attempts")


def _throttle() -> None:
    global _last_call
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _rate_limited(text: str) -> bool:
    lowered = text.lower()
    return "rate limit" in lowered or "abuse detection" in lowered or "http 429" in lowered
