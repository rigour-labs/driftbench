"""Entrants. Each is a single class, behind the same interface (bench/harness/types.py)."""
from __future__ import annotations

import dataclasses
from collections.abc import Callable

from bench.adapters.baselines import EveryHunk, NoTool
from bench.adapters.claude_code import ClaudeCodeReview
from bench.adapters.rigour import RigourDeterministic
from bench.adapters.rigour_reviewer import RigourReviewer
from bench.harness.types import Adapter


@dataclasses.dataclass(frozen=True)
class PaidSettings:
    """What every paid entrant shares in a run: one model, and a dollar cap (enforced by the harness)."""
    model: str
    max_usd: float


FREE: dict[str, type] = {cls.name: cls for cls in (NoTool, EveryHunk, RigourDeterministic)}
PAID: dict[str, Callable[[PaidSettings], Adapter]] = {
    "rigour-reviewer": lambda s: RigourReviewer(s.model),
    "rigour-reviewer-orchestrated": lambda s: RigourReviewer(s.model, orchestrated=True),
    "claude-code-review": lambda s: ClaudeCodeReview(s.model),
}
ADAPTERS = {**FREE, **{name: None for name in PAID}}


def expand(names: list[str]) -> list[str]:
    chosen: list[str] = []
    for name in names:
        expanded = list(FREE) if name == "free" else [name]
        chosen += [n for n in expanded if n not in chosen]
    unknown = [n for n in chosen if n not in FREE and n not in PAID]
    if unknown:
        raise ValueError(f"unknown entrant(s): {', '.join(unknown)}; known: {', '.join([*FREE, *PAID])}")
    return chosen


def select_adapters(names: list[str], paid: PaidSettings | None = None) -> list[Adapter]:
    """`free` expands to every free entrant; a paid one must be named and needs PaidSettings (model + cap)."""
    chosen = expand(names)
    wanted_paid = [n for n in chosen if n in PAID]
    if wanted_paid and (paid is None or paid.max_usd <= 0 or not paid.model):
        raise ValueError(f"paid entrant(s) {', '.join(wanted_paid)} need --model and an approved --max-usd")
    return [FREE[n]() if n in FREE else PAID[n](paid) for n in chosen]
