"""Text rules shared by selection and point splitting (docs/SPEC.md, "Review points").

Rule ACK-1: a text is an acknowledgement, not a review point, when it has at
most six words and every word is in ACK_WORDS ("LGTM", "Thank you!",
"Looks good to me, thanks", ":shipit:"). Emoji and punctuation are ignored.
ACK-1 assumes English-script reviews: a text with no Latin letters has no
words and counts as an acknowledgement.

Rule CMD-1: a text is a command, not a review point, when its first non-blank
line starts with `/` ("/retest") or mentions a bot first ("@dependabot rebase":
a login ending in "bot" or "[bot]").
"""
from __future__ import annotations

import re

ACK_MAX_WORDS = 6
ACK_WORDS = frozenset(
    "lgtm thanks thank you ty +1 looks look good great nice ship shipit it approved approve "
    "now to me all done much lg sgtm cool awesome perfect".split()
)
WORD_RE = re.compile(r"\+1|[a-z]+")
COMMAND_RE = re.compile(r"^(?:/\w|@[\w-]*bot(?:\[bot\])?(?![\w-]))", re.IGNORECASE)


def is_acknowledgement(text: str | None) -> bool:
    words = WORD_RE.findall((text or "").lower())
    return not words or (len(words) <= ACK_MAX_WORDS and all(word in ACK_WORDS for word in words))


def is_command(text: str | None) -> bool:
    first = next((line.strip() for line in (text or "").splitlines() if line.strip()), "")
    return bool(COMMAND_RE.match(first))


def is_review_point(text: str | None) -> bool:
    """Neither an acknowledgement (ACK-1) nor a command (CMD-1)."""
    return not is_acknowledgement(text) and not is_command(text)
