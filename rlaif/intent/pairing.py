"""Labels from a fix: which awaited calls' failures should be tolerated.

Sites come from `rigour export-training-sites` on the file before and after a
fix commit, and are paired by (function, callee, occurrence).

- tolerate: unhandled before, handled after. The fix made this call's failure survivable.
- propagate: unhandled before and after, in a function where the same fix made
  another call tolerant. The author looked at this function and left this
  failure to propagate: a hard negative.

Nothing else is labelled: a call the fix never looked at says nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

TOLERATE = "tolerate"
PROPAGATE = "propagate"


@dataclass(frozen=True)
class Example:
    file: str
    function: str
    callee: str
    ordinal: int
    line: int
    label: str
    #: The enclosing function before the fix: what the model reads.
    source: str


def _key(site: dict) -> tuple[str, str, int]:
    return site["function"], site["callee"], site["ordinal"]


def label_fix(before: list[dict], after: list[dict]) -> list[Example]:
    """Examples from one file's sites before and after a fix."""
    after_by_key = {_key(s): s for s in after}
    paired = [(b, after_by_key.get(_key(b))) for b in before if b["handledBy"] is None]
    tolerant_functions = {b["function"] for b, a in paired if a is not None and a["handledBy"] is not None}
    examples = []
    for b, a in paired:
        if a is None or b["function"] not in tolerant_functions:
            continue
        label = TOLERATE if a["handledBy"] is not None else PROPAGATE
        examples.append(Example(b["file"], b["function"], b["callee"], b["ordinal"], b["line"], label, b["source"]))
    return examples
