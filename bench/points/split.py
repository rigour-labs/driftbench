"""Split a review body or conversation comment into points (docs/SPEC.md, "Review points").

Rule SPLIT-1: a point is a paragraph (separated by blank lines) or a list item
(a line starting with "-", "*", "+" or "1."). Fenced code blocks stay with
the paragraph they belong to, blank lines inside them included.
Rule QUOTE-1: a paragraph whose lines all start with ">" quotes earlier text
and is not a point.
"""
from __future__ import annotations

import re

LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+")
FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")


def split_spans(text: str) -> list[tuple[int, int]]:
    """Character spans `[start, end)` of each paragraph or list item, in order."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    in_fence = False
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if FENCE_RE.match(line):
            in_fence = not in_fence
        boundary = not in_fence and (not stripped or LIST_ITEM_RE.match(line))
        if boundary and start is not None:
            spans.append(trim(text, start, offset))
            start = None
        if stripped and start is None:
            start = offset
        offset += len(line)
    if start is not None:
        spans.append(trim(text, start, offset))
    return [span for span in spans if span[1] > span[0]]


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def is_quote(span_text: str) -> bool:
    lines = [line for line in span_text.splitlines() if line.strip()]
    return bool(lines) and all(line.lstrip().startswith(">") for line in lines)
