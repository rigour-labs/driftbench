"""What of a tool's output is kept: rule, location and a short message without code.

Tools quote the reviewed code in their messages. The projects' code is
never republished, so every finding's message is reduced before it is
written: anything after "Found:" and anything inside quotes or backticks
is removed, whitespace is collapsed, and the result is cut to
MESSAGE_LIMIT characters. A tool's raw output is not stored at all.
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


def published_finding(finding: dict) -> dict:
    return {**finding, "message": short_message(finding.get("message", ""))}
