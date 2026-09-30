"""Remove answer-revealing text from benchmark patches before a model sees them.

Most drift patches were written with comments that name the defect
(`# SECURITY DRIFT: SQL Injection vulnerability!`); golden patches were not.
A model that reads comments would score near 100% without judging the code.
The same rules apply to golden and drift patches, so neither side keeps a
tell the other lacks: comment-only lines and docstrings are removed from
added code (a golden docstring saying "prevents SQL injection" is as much a
tell as a drift comment), and trailing comments are cut from added lines.
"""
from __future__ import annotations

import re

# Comment-only lines: Python/shell, C-family, block comment bodies, HTML/JSX.
_COMMENT_LINE = re.compile(r"^\s*(#|//|/\*|\*|\*/|<!--|\{/\*)")
# A trailing comment after code. Deliberately conservative: only `  # ...` and
# `  // ...` preceded by whitespace, so URLs ("http://") and "#" inside strings survive.
_TRAILING = re.compile(r"\s+(#|//)\s.*$")
_QUOTES = re.compile(r"(['\"]).*?\1")
_DOCSTRING_START = re.compile(r'^\s*[rbuRBU]?("""|\'\'\')')


def sanitize_patch(patch: str) -> str:
    """The patch without comment-only lines or docstrings in added code, and with trailing comments cut.

    Hunk line counts are not recomputed: the result is for reading, not for `git apply`.
    """
    out = []
    docstring: str | None = None
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            keep, docstring = _added_line(line[1:], docstring)
            if keep is None:
                continue
            line = "+" + keep
        out.append(line)
    return "\n".join(out) + ("\n" if patch.endswith("\n") else "")


def _added_line(body: str, docstring: str | None) -> tuple[str | None, str | None]:
    """(text to keep or None to drop, the docstring quote still open after this line)."""
    if docstring:
        return None, (None if docstring in body else docstring)
    opening = _DOCSTRING_START.match(body)
    if opening:
        quote = opening.group(1)
        return None, (None if quote in body[opening.end():] else quote)
    if _COMMENT_LINE.match(body):
        return None, None
    return _strip_trailing_comment(body), None


def _strip_trailing_comment(code: str) -> str:
    # Mask string literals so a "#" or "//" inside a string is not taken for a comment.
    masked = _QUOTES.sub(lambda m: "x" * len(m.group(0)), code)
    match = _TRAILING.search(masked)
    return code[: match.start()].rstrip() if match else code
