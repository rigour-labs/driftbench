"""GitHub REST reads through the `gh` CLI: cached on disk, spaced out, retried on rate limits.

The cache makes a collection resumable and lets a run be repeated without
calling the API again. It holds raw API responses, which include comment text,
so it lives under `work/` and is never published.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

MIN_INTERVAL_S = 0.8
MAX_RETRIES = 6
RATE_LIMIT_MARKERS = ("rate limit", "abuse detection", "http 429")

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


class GitHubClient:
    def __init__(self, cache_dir: Path | None, runner: Runner = run_gh, sleep: Callable[[float], None] = time.sleep):
        self.cache_dir = cache_dir
        self.runner = runner
        self.sleep = sleep
        self.last_call = 0.0

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        """One GET, one page."""
        return self.cached(path, params or {}, paginate=False)

    def get_all(self, path: str, params: dict[str, str] | None = None) -> list:
        """Every page of a list endpoint, concatenated."""
        pages = self.cached(path, {"per_page": "100", **(params or {})}, paginate=True)
        return [item for page in pages for item in page]

    def cached(self, path: str, params: dict[str, str], paginate: bool) -> Any:
        file = self.cache_file(path, params, paginate)
        if file and file.exists():
            return parse_json(file.read_text(encoding="utf-8"), f"cache file {file}")
        data = self.fetch(path, params, paginate)
        if file:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(data), encoding="utf-8")
        return data

    def cache_file(self, path: str, params: dict[str, str], paginate: bool) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps([path, sorted(params.items()), paginate])
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def fetch(self, path: str, params: dict[str, str], paginate: bool) -> Any:
        args = ["gh", "api", "-X", "GET", path, "-H", "Accept: application/vnd.github+json"]
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
            if not any(marker in output.lower() for marker in RATE_LIMIT_MARKERS):
                raise GitHubError(f"gh api {path} failed: {output.strip()[:300]}")
            self.sleep(60 * (attempt + 1))
        raise GitHubError(f"gh api {path}: still rate limited after {MAX_RETRIES} attempts")

    def throttle(self) -> None:
        wait = MIN_INTERVAL_S - (time.monotonic() - self.last_call)
        if wait > 0:
            self.sleep(wait)
        self.last_call = time.monotonic()
