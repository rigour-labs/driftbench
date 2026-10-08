"""Turn each frozen PR record into review points (docs/SPEC.md, "Review points").

A point keeps IDs, a character span and an anchor, never text. A dropped
point stays in the output with the rule that dropped it. A point is
`scorable` when it isn't dropped and has a round (a reviewed head to run a
tool on).
"""
from __future__ import annotations

from collections.abc import Callable

from bench.collect.text_rules import is_acknowledgement, is_command
from bench.points.split import is_quote, split_spans
from bench.points.texts import TextSource

RULES_VERSION = 1
Acted = Callable[[dict | None], tuple[bool | None, str | None]]


def point(base: dict, index: int, span: tuple[int, int] | None, dropped: str | None) -> dict:
    return {
        **base,
        "id": f"{base['pr']}-{base['kind']}-{base['source_id']}-{index}",
        "span": list(span) if span else None,
        "dropped": dropped,
        "scorable": dropped is None and base["round"] is not None,
    }


def text_rule(text: str) -> str | None:
    if is_acknowledgement(text):
        return "ACK-1"
    return "CMD-1" if is_command(text) else None


def split_points(base: dict, text: str | None) -> list[dict]:
    """One point per paragraph or list item (SPLIT-1), each checked by QUOTE-1, ACK-1, CMD-1."""
    if text is None:
        return [point(base, 0, None, "TEXT-1")]
    points = []
    for index, (start, end) in enumerate(split_spans(text)):
        piece = text[start:end]
        points.append(point(base, index, (start, end), "QUOTE-1" if is_quote(piece) else text_rule(piece)))
    return points


def inline_points(pr: dict, texts: TextSource, acted: Acted) -> list[dict]:
    points = []
    for comment in pr["comments"]:
        anchor = {key: comment[key] for key in ("path", "line", "start_line", "side", "commit_sha")}
        base = {"pr": pr["number"], "kind": "inline", "source_id": comment["id"], "round": comment["round"],
                "head_sha": comment["commit_sha"], "anchor": anchor, "acted_on": None, "acted_basis": None}
        dropped = "AUTHOR-1" if comment["by_author"] else "THREAD-1" if comment["in_reply_to"] else None
        if not dropped:
            text = texts.verified_text(pr["number"], "inline", comment)
            dropped = "TEXT-1" if text is None else text_rule(text)
        if not dropped:
            base["acted_on"], base["acted_basis"] = acted(anchor)
        points.append(point(base, 0, None, dropped))
    return points


def body_points(pr: dict, texts: TextSource) -> list[dict]:
    points = []
    for review in pr["reviews"]:
        if review["body_chars"]:
            base = {"pr": pr["number"], "kind": "body", "source_id": review["id"], "round": review["round"],
                    "head_sha": review["commit_sha"], "anchor": None, "acted_on": None}
            points += split_points(base, texts.verified_text(pr["number"], "review", review))
    return points


def conversation_points(pr: dict, texts: TextSource) -> list[dict]:
    points = []
    for comment in pr["conversation"]:
        base = {"pr": pr["number"], "kind": "conversation", "source_id": comment["id"], "round": comment["round"],
                "head_sha": comment["head_sha"], "head_source": comment["head_source"], "anchor": None,
                "acted_on": None}
        if comment["by_author"]:
            points.append(point(base, 0, None, "AUTHOR-1"))
        else:
            points += split_points(base, texts.verified_text(pr["number"], "conversation", comment))
    return points


def points_for_pr(pr: dict, texts: TextSource, acted: Acted) -> list[dict]:
    return inline_points(pr, texts, acted) + body_points(pr, texts) + conversation_points(pr, texts)
