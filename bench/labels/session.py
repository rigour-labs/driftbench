"""`bench label next`: label the sample one point at a time, resumable.

Each answer is saved immediately through the same confirm path as `set`
(text hash, blind, the labeller's name), so quitting at any point loses
nothing. A skipped point stays unconfirmed and is marked `skipped`.

With model suggestions (bench/labels/model_file.py), the seeded blind subset
comes first and is shown without any suggestion. Other points show the
model's class and reason; Enter accepts it, and the label records whether it
was accepted or overridden. Without them, every point is blind.
"""
from __future__ import annotations

from collections.abc import Callable

from bench.labels.model_file import shown_suggestion
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


def pending(sample_ids: list[str], labels: dict, include_skipped: bool, first: list[str] = ()) -> list[str]:
    """Unlabelled sample points, with `first` (the blind subset) ahead of the rest."""
    entries = labels["points"]
    todo = [pid for pid in sample_ids
            if not entries.get(pid, {}).get("label") and (include_skipped or not entries.get(pid, {}).get("skipped"))]
    return sorted(todo, key=lambda pid: pid not in first)


def answer_for(ask: Callable[[str], str], suggested: str | None) -> str:
    prompt = PROMPT if suggested is None else f"Enter={suggested}  " + PROMPT
    while True:
        reply = ask(prompt).strip().lower()
        if reply == "" and suggested is not None:
            return next(key for key, label in CHOICES.items() if label == suggested)
        if reply in CHOICES or reply in ("s", "q"):
            return reply


def record(labels: dict, point_id: str, reply: str, text: str, labeller: str, suggested: str | None) -> dict:
    if reply == "s":
        return mark_skipped(labels, point_id)
    label = CHOICES[reply]
    evidence = {"text_sha256": text_sha256(text), "blind": suggested is None}
    if suggested is not None:
        evidence["suggestion"] = "accepted" if label == suggested else "overridden"
    return confirm(labels, point_id, label, labeller, evidence)


def run_session(context: dict, labels: dict, ask: Callable[[str], str], show: Callable[[str], None]) -> dict:
    """`context`: repo, points (by id), sample_ids, text (point -> text or None), labeller, save,
    include_skipped, and model (the model suggestions file, or None)."""
    model = context.get("model")
    first = model["blind_ids"] if model else []
    queue = pending(context["sample_ids"], labels, context.get("include_skipped", False), first)
    for position, point_id in enumerate(queue, start=1):
        point = context["points"][point_id]
        text = context["text"](point)
        if text is None:
            show(f"[{point_id}] text changed since the freeze (TEXT-1); skipping it")
            continue
        hint = shown_suggestion(model, point_id)
        note = f"model suggests: {hint['suggested']} ({hint['reason']})\n" if hint else ""
        show(f"\n[{position}/{len(queue)}] {point_id}\n" + "\n".join(links(context["repo"], point))
             + f"\n\n{text}\n\n{note}")
        suggested = hint["suggested"] if hint else None
        reply = answer_for(ask, suggested)
        if reply == "q":
            break
        labels = record(labels, point_id, reply, text, context["labeller"], suggested)
        context["save"](labels)
    return labels
