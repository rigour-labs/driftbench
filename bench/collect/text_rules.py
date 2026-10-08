"""Text rules shared by selection and point splitting (docs/SPEC.md, "Review points").

Rule ACK-1: a text is an acknowledgement, not a review point, when it has at
most six words and every word is in ACK_WORDS ("LGTM", "Thank you!",
"Looks good to me, thanks", ":shipit:"). Emoji and punctuation are ignored.
"""
from __future__ import annotations

import re

ACK_MAX_WORDS = 6
ACK_WORDS = frozenset(
    "lgtm thanks thank you ty +1 looks look good great nice ship shipit it approved approve "
    "now to me all done much lg sgtm cool awesome perfect".split()
)
WORD_RE = re.compile(r"\+1|[a-z]+")


def is_acknowledgement(text: str | None) -> bool:
    words = WORD_RE.findall((text or "").lower())
    return not words or (len(words) <= ACK_MAX_WORDS and all(word in ACK_WORDS for word in words))
