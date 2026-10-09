"""The adapter boundary (CONTRIBUTING.md, "Adding a tool").

The harness owns checkout, the diff, what history is visible, timing and
scoring. An adapter only turns a ReviewInput into a ReviewOutput.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class Finding:
    path: str | None
    line: int | None  # head side; None when the finding has no line
    blocking: bool    # the tool's own semantics: this finding fails its check
    message: str
    rule: str = ""
    end_line: int | None = None  # last line when a finding covers a range; None for a single line


@dataclasses.dataclass(frozen=True)
class ReviewInput:
    workdir: Path        # checkout at head_sha; treat as read-only
    base_sha: str        # merge base of the pull request's base and head_sha
    head_sha: str
    diff_path: Path      # unified diff base_sha...head_sha
    history: dict | None  # only for adapters with reads_history; None otherwise
    timeout_s: int
    env: dict[str, str]  # the only environment a tool process may get (bench/harness/sandbox.py)


@dataclasses.dataclass(frozen=True)
class ReviewOutput:
    findings: list[Finding]
    verdict: str              # "pass", "fail" (the tool blocks) or "error"
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str = ""
    model_runs: int | None = None  # model calls the tool made; 0 = honestly free (nothing to review, cached)
    leak_signals: int = 0          # signs the tool saw the PR's human reviews or fetched PR data
    charged: str = ""              # paid reviews: "reported" (the tool's cost) or "bound" (the per-head bound)


class AdapterError(RuntimeError):
    """An adapter couldn't produce a review; `cost_usd` is what the tool reported spending, if it said."""

    def __init__(self, message: str, cost_usd: float | None = None):
        super().__init__(message)
        self.cost_usd = cost_usd


class Adapter(Protocol):
    name: str
    version: str
    paid: bool
    reads_history: bool
    env_extra: tuple[str, ...]  # names of variables the adapter adds to the sandbox env (its own key only)

    def review(self, request: ReviewInput) -> ReviewOutput: ...
