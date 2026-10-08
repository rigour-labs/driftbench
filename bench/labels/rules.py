"""The rules pass: suggest a review point's class from its text (docs/LABELLING.md).

A suggestion is never a label. It is shown to the human labeller, who
confirms or corrects it; an unconfirmed point is reported as `unclassified`.

Keywords are matched in the prose only: fenced code blocks (including
GitHub "suggestion" blocks) and inline `code` are removed first, so an
identifier or import path can't decide the class.

Order (first match wins), following the guideline's decision order:
1. a point that starts with "nit" is mechanical (the reviewer said so);
2. claim/contract, user journey, performance, mechanical, judgment, by keyword;
3. no keyword: no suggestion.
"""
from __future__ import annotations

import re

RULES_VERSION = 1
CLASSES = ("mechanical", "performance", "claim/contract", "user journey", "judgment")

NIT_RE = re.compile(r"^\W*nit\b", re.IGNORECASE)
CODE_RE = re.compile(r"```.*?(?:```|\Z)|~~~.*?(?:~~~|\Z)|`[^`\n]*`", re.DOTALL)
KEYWORDS = (
    ("claim/contract", r"docs?|docstring|returns?|contract|signature|types?|nil|null|edge case|invariant"
                       r"|race|deadlock|bug|wrong|incorrect|panic|crash\w*|leak\w*|breaks?|regression|backward\w*"),
    ("user journey", r"users?|ui|ux|screen|button|click\w*|page|dialog|modal|toast|tooltip|shown|displayed"
                     r"|error message|onboarding|accessib\w*"),
    ("performance", r"perf|performance|slow(?:er)?|faster|latency|allocat\w*|o\(n\S*\)|quadratic|hot path"
                    r"|cache[sd]?|memory|cpu|benchmark\w*|expensive"),
    ("mechanical", r"typo|spelling|naming|rename|format\w*|lint\w*|unused|imports?|whitespace|indent\w*"
                   r"|style|gofmt|prettier|capitali[sz]\w*"),
    ("judgment", r"why not|consider|prefer|i'd|i would|maybe|perhaps|design|approach|abstraction|simpler"
                 r"|cleaner|refactor|readab\w*"),
)
KEYWORD_RES = tuple((label, re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)) for label, pattern in KEYWORDS)


def suggest(text: str) -> str | None:
    prose = CODE_RE.sub(" ", text)
    if NIT_RE.match(prose):
        return "mechanical"
    return next((label for label, pattern in KEYWORD_RES if pattern.search(prose)), None)
