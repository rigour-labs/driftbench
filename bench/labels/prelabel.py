"""Model pre-labels for the label samples (docs/LABELLING.md, "Model suggestions").

Each sampled point is sent as its review comment plus the diff hunk it is
anchored to, with the classes section of the labelling guide, and the model
answers one class from the fixed list and a one-line reason. An answer
outside the list is no suggestion. Spending is capped: before each call the
run's Budget must allow the entrant's per-call bound, and a point that can't
be afforded stays unsuggested, with the reason recorded.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from bench.harness.budget import Budget
from bench.labels.openrouter import OpenRouterError
from bench.labels.rules import CLASSES

ENTRANT = "prelabel"
MAX_TOKENS = 200
HUNK_LINES = 40
REASON_CHARS = 200
GUIDE_START, GUIDE_END = "## Classes", "Questions count by what they ask for"
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class ReplyError(ValueError):
    pass


def guide_excerpt(guide: Path) -> str:
    """The classes table, decision order and boundaries, exactly as the human labeller reads them."""
    text = guide.read_text(encoding="utf-8")
    start, end = text.find(GUIDE_START), text.find(GUIDE_END)
    if start < 0 or end <= start:
        raise ValueError(f"{guide}: the classes section was not found")
    return text[start:end].strip()


def system_prompt(excerpt: str) -> str:
    return ("You label one code review comment with the kind of problem the reviewer raised, following this "
            f"guide.\n\n{excerpt}\n\nReply with JSON only: {{\"class\": one of {json.dumps(list(CLASSES))}, "
            "\"reason\": one short sentence}.")


def prompt_sha256(system: str) -> str:
    return hashlib.sha256(system.encode("utf-8")).hexdigest()


def messages_for(system: str, text: str, hunk: str) -> list[dict]:
    user = f"Review comment:\n{text}\n\nCode it is anchored to (diff hunk):\n{hunk or '(none)'}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_reply(content: str | None) -> tuple[str, str]:
    """(class, one-line reason); ReplyError unless the reply names exactly one known class."""
    match = JSON_RE.search(content or "")
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError as exc:
        raise ReplyError("invalid output: not JSON") from exc
    if not isinstance(data, dict) or data.get("class") not in CLASSES:
        raise ReplyError("invalid output: no known class")
    return data["class"], " ".join(str(data.get("reason") or "").split())[:REASON_CHARS]


def anchored_hunk(comments: list[dict], source_id: int) -> str:
    hunk = next((c.get("diff_hunk") or "" for c in comments if c.get("id") == source_id), "")
    return "\n".join(hunk.splitlines()[-HUNK_LINES:])


def suggest_point(target: dict, call: Callable[[list[dict]], dict], budget: Budget) -> tuple[dict | None, str]:
    """One point: (suggestion entry or None, why not). Every call is charged, failures at the bound."""
    if target["text"] is None:
        return None, "text changed since the freeze (TEXT-1)"
    if not budget.allows_review(ENTRANT):
        return None, (f"budget: ${budget.spent:.4f} spent of ${budget.max_usd:.2f}; "
                      f"the next call could cost up to ${budget.per_head_bound(ENTRANT):.4f}")
    try:
        reply = call(target["messages"])
    except OpenRouterError as exc:  # the run outlives a failed call; it reports no cost, so the bound is charged
        print(f"warning: {target['id']}: {exc}", file=sys.stderr)
        reply = {"content": None, "cost_usd": None, "served_model": None, "error": str(exc)}
    if reply["cost_usd"] is None:
        budget.add_cost(ENTRANT, budget.per_head_bound(ENTRANT))
        return None, reply.get("error") or "OpenRouter reported no cost; charged at the bound"
    budget.add_cost(ENTRANT, reply["cost_usd"])
    entry = {"suggested": None, "reason": "", "cost_usd": reply["cost_usd"], "served_model": reply["served_model"]}
    try:
        entry["suggested"], entry["reason"] = parse_reply(reply["content"])
    except ReplyError as exc:  # paid for, but no suggestion
        print(f"warning: {target['id']}: {exc}", file=sys.stderr)
        return {**entry, "reason": str(exc)}, str(exc)
    return entry, ""


def run_prelabels(data: dict, targets: list[dict], call: Callable[[list[dict]], dict], budget: Budget,
                  save: Callable[[dict], None]) -> dict:
    """Suggest every target not already paid for; save after each point so a stop loses nothing."""
    for target in targets:
        pid = target["id"]
        if pid in data["points"]:
            continue
        entry, why = suggest_point(target, call, budget)
        unsuggested = {k: v for k, v in data["unsuggested"].items() if k != pid}
        points = data["points"]
        if entry is not None:
            points = {**points, pid: {"suggested_by": data["model"], **entry}}
            data = {**data, "spent_usd": round(data["spent_usd"] + entry["cost_usd"], 6)}
        if why:
            unsuggested[pid] = why
        data = {**data, "points": points, "unsuggested": unsuggested}
        save(data)
    return data
