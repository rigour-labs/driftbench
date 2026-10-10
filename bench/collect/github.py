"""GitHub REST reads through the `gh` CLI: cached on disk, spaced out, retried on rate limits.

The cache makes a collection resumable and lets a run be repeated without
calling the API again. It holds raw API responses, which include comment text,
so it lives under `work/` and is never published.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

MIN_INTERVAL_S = 0.8
MAX_RETRIES = 6
GRAPHQL = "graphql"
RATE_LIMIT_MARKERS = ("rate limit", "abuse detection", "http 429")
PRIMARY_LIMIT_MARKER = "api rate limit exceeded"  # hourly quota; wait for its reset
MAX_RESET_WAIT_S = 3700
MIN_GH_VERSION = (2, 48, 0)  # first release with `gh api --slurp`

Runner = Callable[[list[str]], subprocess.CompletedProcess]


class GitHubError(RuntimeError):
    pass


def parse_json(text: str, source: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GitHubError(f"{source}: not valid JSON ({exc})") from exc


def run_gh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def require_gh(runner: Runner = run_gh) -> None:
    """Fail early, with the fix, when `gh` is missing or too old for `--slurp`."""
    result = runner(["gh", "--version"])
    match = re.search(r"gh version (\d+)\.(\d+)\.(\d+)", result.stdout or "")
    if result.returncode != 0 or not match:
        raise GitHubError("the GitHub CLI `gh` is required: https://cli.github.com")
    found = tuple(int(part) for part in match.groups())
    if found < MIN_GH_VERSION:
        wanted = ".".join(map(str, MIN_GH_VERSION))
        raise GitHubError(f"gh {'.'.join(map(str, found))} is too old; upgrade to {wanted} or later")


class GitHubClient:
    def __init__(self, cache_dir: Path | None, runner: Runner = run_gh, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time):
        self.cache_dir = cache_dir
        self.runner = runner
        self.sleep = sleep
        self.clock = clock
        self.last_call = 0.0

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        """One GET, one page."""
        return self.cached(path, params or {}, paginate=False)

    def get_optional(self, path: str) -> Any:
        """One GET, or None when GitHub answers 404 (e.g. a commit that no longer exists)."""
        return self.cached(path, {}, paginate=False, allow_missing=True)

    def graphql(self, query: str) -> Any:
        """One GraphQL query (POST), cached like a GET; GitHub's errors raise."""
        data = self.cached(GRAPHQL, {"query": query}, paginate=False)
        if not isinstance(data, dict) or data.get("errors") or "data" not in data:
            raise GitHubError(f"graphql: {str((data or {}).get('errors') if isinstance(data, dict) else data)[:300]}")
        return data["data"]

    def get_all(self, path: str, params: dict[str, str] | None = None) -> list:
        """Every page of a list endpoint, concatenated."""
        pages = self.cached(path, {"per_page": "100", **(params or {})}, paginate=True)
        return [item for page in pages for item in page]

    def cached(self, path: str, params: dict[str, str], paginate: bool, allow_missing: bool = False) -> Any:
        file = self.cache_file(path, params, paginate)
        if file and file.exists():
            return parse_json(file.read_text(encoding="utf-8"), f"cache file {file}")
        data = self.fetch(path, params, paginate, allow_missing)
        if file:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(data), encoding="utf-8")
        return data

    def cache_file(self, path: str, params: dict[str, str], paginate: bool) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps([path, sorted(params.items()), paginate])
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def fetch(self, path: str, params: dict[str, str], paginate: bool, allow_missing: bool) -> Any:
        args = (["gh", "api", GRAPHQL] if path == GRAPHQL
                else ["gh", "api", "-X", "GET", path, "-H", "Accept: application/vnd.github+json"])
        for key, value in params.items():
            args += ["-f", f"{key}={value}"]
        if paginate:
            args += ["--paginate", "--slurp"]
        for attempt in range(MAX_RETRIES):
            self.throttle()
            result = self.runner(args)
            if result.returncode == 0:
                return parse_json(result.stdout or "null", f"gh api {path}")
            output = (result.stderr or "") + (result.stdout or "")
            if allow_missing and "http 404" in output.lower():
                return None
            if not any(marker in output.lower() for marker in RATE_LIMIT_MARKERS):
                raise GitHubError(f"gh api {path} failed: {output.strip()[:300]}")
            if PRIMARY_LIMIT_MARKER in output.lower():
                self.wait_for_reset()
            else:
                self.sleep(60 * (attempt + 1))
        raise GitHubError(f"gh api {path}: still rate limited after {MAX_RETRIES} attempts")

    def wait_for_reset(self) -> None:
        """Sleep until the hourly quota resets (from `gh api rate_limit`), at most MAX_RESET_WAIT_S."""
        result = self.runner(["gh", "api", "rate_limit"])
        try:
            reset = int(json.loads(result.stdout)["resources"]["core"]["reset"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            reset = int(self.clock()) + 60
        self.sleep(max(5.0, min(reset - self.clock() + 5, MAX_RESET_WAIT_S)))

    def throttle(self) -> None:
        wait = MIN_INTERVAL_S - (time.monotonic() - self.last_call)
        if wait > 0:
            self.sleep(wait)
        self.last_call = time.monotonic()
