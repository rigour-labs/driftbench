"""What of a tool's output is kept: rule, location and a short message without code.

Tools quote the reviewed code in their messages. The projects' code is
never republished, so every finding's message is reduced before it is
written: anything after "Found:" and anything inside quotes or backticks
is removed, whitespace is collapsed, and the result is cut to
MESSAGE_LIMIT characters.

One exception, for a paid entrant's own output: its whole answer and its
findings with full messages are kept under `paid_output` in the run's
records, which go to the run's release tarball only, never to main, so a
later issue-level comparison needs no second paid run. Code quoted there
belongs to its project, under its licence.
"""
from __future__ import annotations

import re

MESSAGE_LIMIT = 80
QUOTED_RE = re.compile(r"`[^`]*`|\"[^\"]*\"|'[^'\n]*'|“[^”]*”")
FOUND_RE = re.compile(r"\bfound:.*", re.IGNORECASE | re.DOTALL)


def short_message(message: str) -> str:
    text = FOUND_RE.sub("", message or "")
    text = QUOTED_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MESSAGE_LIMIT].rstrip()


PAID_OUTPUT = "paid_output"


def published_finding(finding: dict) -> dict:
    return {**finding, "message": short_message(finding.get("message", ""))}


def paid_output(review_text: str, findings: list[dict], held_back: dict | None = None,
                trace: dict | None = None) -> dict:
    """A paid entrant's answer and findings as the tool wrote them, what it held back, and how it ran
    (release tarball only)."""
    return {"review_text": review_text, "findings": findings, **({"held_back": held_back} if held_back else {}),
            **({"trace": trace} if trace else {})}
