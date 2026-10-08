"""`bench label next`: label the sample one point at a time, blind, resumable.

Each answer is saved immediately through the same confirm path as `set`
(text hash, blind: true, the labeller's name), so quitting at any point loses
nothing. A skipped point stays unconfirmed and is marked `skipped`.
"""
from __future__ import annotations

from collections.abc import Callable

from bench.labels.rules import CLASSES
from bench.labels.store import confirm, mark_skipped, text_sha256

CHOICES = {str(index): label for index, label in enumerate(CLASSES, start=1)}
PROMPT = "  ".join(f"{key}={label}" for key, label in CHOICES.items()) + "  s=skip  q=quit > "


def links(repo: str, point: dict) -> list[str]:
    pr_url = f"https://github.com/{repo}/pull/{point['pr']}"
    anchor = point.get("anchor")
    if anchor and anchor.get("line"):
        lines = f"L{anchor['start_line']}-L{anchor['line']}" if anchor.get("start_line") else f"L{anchor['line']}"
        return [f"code: https://github.com/{repo}/blob/{anchor['commit_sha']}/{anchor['path']}#{lines}", f"PR: {pr_url}"]
    fragment = {"body": "pullrequestreview", "conversation": "issuecomment"}.get(point["kind"])
    return [f"PR: {pr_url}#{fragment}-{point['source_id']}" if fragment else f"PR: {pr_url}"]


def pending(sample_ids: list[str], labels: dict, include_skipped: bool) -> list[str]:
    entries = labels["points"]
    return [pid for pid in sample_ids
            if not entries.get(pid, {}).get("label") and (include_skipped or not entries.get(pid, {}).get("skipped"))]


def answer_for(ask: Callable[[str], str]) -> str:
    while True:
        reply = ask(PROMPT).strip().lower()
        if reply in CHOICES or reply in ("s", "q"):
            return reply


def record(labels: dict, point_id: str, reply: str, text: str, labeller: str) -> dict:
    if reply == "s":
        return mark_skipped(labels, point_id)
    return confirm(labels, point_id, CHOICES[reply], labeller, {"text_sha256": text_sha256(text), "blind": True})


def run_session(context: dict, labels: dict, ask: Callable[[str], str], show: Callable[[str], None]) -> dict:
    """`context`: repo, points (by id), sample_ids, text (point -> text or None), labeller, save, include_skipped."""
    queue = pending(context["sample_ids"], labels, context.get("include_skipped", False))
    for position, point_id in enumerate(queue, start=1):
        point = context["points"][point_id]
        text = context["text"](point)
        if text is None:
            show(f"[{point_id}] text changed since the freeze (TEXT-1); skipping it")
            continue
        show(f"\n[{position}/{len(queue)}] {point_id}\n" + "\n".join(links(context["repo"], point)) + f"\n\n{text}\n")
        reply = answer_for(ask)
        if reply == "q":
            break
        labels = record(labels, point_id, reply, text, context["labeller"])
        context["save"](labels)
    return labels
