"""Were the labels fixed before the run started (docs/LABELLING.md, "Order")?

Per-class results are published only if every label and sample file was last
committed at or before the run's `run_started_at` and has no uncommitted
changes. Otherwise the labeller could have seen tool output first.
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


def git_out(path: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(path.parent), *args, "--", path.name],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def labels_precede_run(paths: list[Path], run_started_at: str) -> tuple[bool, str]:
    """(ok, reason). Missing files are fine: they hold no labels."""
    if not run_started_at:
        return False, "the run has no recorded start time"
    started = datetime.fromisoformat(run_started_at)
    for path in (p for p in paths if p.exists()):
        if git_out(path, "status", "--porcelain"):
            return False, f"{path.name} has uncommitted changes"
        committed = git_out(path, "log", "-1", "--format=%cI")
        if not committed:
            return False, f"{path.name} is not committed"
        if datetime.fromisoformat(committed) > started:
            return False, f"{path.name} was committed at {committed}, after the run started at {run_started_at}"
    return True, ""
