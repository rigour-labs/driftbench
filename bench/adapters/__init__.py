"""Entrants. Each is one class behind the same interface (bench/harness/types.py)."""
from __future__ import annotations

from bench.adapters.baselines import EveryHunk, NoTool
from bench.adapters.rigour import RigourDeterministic
from bench.harness.types import Adapter

ADAPTERS: dict[str, type] = {cls.name: cls for cls in (NoTool, EveryHunk, RigourDeterministic)}


def select_adapters(names: list[str]) -> list[Adapter]:
    """`free` expands to every adapter that can't cost money; paid ones must be named and need a cap."""
    chosen: list[str] = []
    for name in names:
        expanded = [n for n, cls in ADAPTERS.items() if not cls.paid] if name == "free" else [name]
        chosen += [n for n in expanded if n not in chosen]
    unknown = [n for n in chosen if n not in ADAPTERS]
    if unknown:
        raise ValueError(f"unknown entrant(s): {', '.join(unknown)}; known: {', '.join(ADAPTERS)}")
    paid = [n for n in chosen if ADAPTERS[n].paid]
    if paid:
        raise ValueError(f"paid entrant(s) {', '.join(paid)} need an approved dollar cap; not supported in this version")
    return [ADAPTERS[n]() for n in chosen]
