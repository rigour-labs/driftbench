"""The question a model is asked, and strict parsing of its answer."""
from __future__ import annotations

import json
import re

from evalgate.scenarios import EvalItem

_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def build_prompt(item: EvalItem) -> str:
    """Intent and the sanitized patch only: nothing says whether, or what kind of, drift is present."""
    return (
        "You are reviewing a code change against its stated intent.\n\n"
        f"Intent:\n{item.intent}\n\n"
        f"Patch:\n```diff\n{item.patch}```\n\n"
        "Does this change introduce drift: a defect, security problem, or departure from the intent "
        "or the project's established patterns?\n"
        'Reply with JSON only: {"is_drift": true or false, "confidence": 0.0-1.0}'
    )


def parse_answer(text: str) -> bool | None:
    """True/False from the first JSON object with a boolean `is_drift`; None otherwise (never a default)."""
    for match in _OBJECT.finditer(text):
        try:
            value = json.loads(match.group(0)).get("is_drift")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(value, bool):
            return value
    return None
