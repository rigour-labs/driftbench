"""What a tool runs in: a fresh home, a minimal environment, a repo that can't see the future.

- Environment: built here, never inherited. PATH, a fresh HOME (so no tool
  reads this machine's settings, stored answers or profiles), telemetry
  off, a shared npm cache. No tokens or API keys; a paid adapter adds only
  its own key, explicitly.
- Repository: a new repo that borrows the clone's objects (git alternates)
  and has no refs at all, so there is no `origin/main`, no other branch and
  no tag pointing at commits after the reviewed head. Only the head is
  checked out.
Both are deleted after the run.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Sandbox:
    home: Path
    repo: Path
    env: dict[str, str]


def adapter_env(home: Path, npm_cache: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(home),
        "TMPDIR": str(home / "tmp"),
        "LANG": "C.UTF-8",
        "RIGOUR_HOME": str(home / ".rigour"),
        "RIGOUR_PROFILES": str(home / "no-profiles.json"),
        "RIGOUR_TELEMETRY": "0",
        "DO_NOT_TRACK": "1",
        "npm_config_cache": str(npm_cache),
        "npm_config_update_notifier": "false",
    }
    env.update(extra or {})
    return env


def env_keys(extra: dict[str, str] | None = None) -> list[str]:
    """The variable names every adapter run gets (recorded in each run record; values are not)."""
    return sorted(adapter_env(Path("/"), Path("/"), extra))


@contextlib.contextmanager
def sandbox(scratch_dir: Path, npm_cache: Path) -> Iterator[Sandbox]:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=scratch_dir))
    try:
        home = root / "home"
        (home / "tmp").mkdir(parents=True)
        npm_cache.mkdir(parents=True, exist_ok=True)
        yield Sandbox(home=home, repo=root / "repo", env=adapter_env(home, npm_cache))
    finally:
        shutil.rmtree(root, ignore_errors=True)
