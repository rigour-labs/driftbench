"""The blind issue judge: one call per human point, both entrants' reviews as A and B.

The judge is a model outside the Claude family (Rigour's reviewer runs on
Claude). Per point it sees the human comment, its file and line, and two
reviews labelled A and B, in a seeded random order per point, with nothing
naming the entrant. For each it answers whether the review raises the same
issue as the comment, judged at file level and by meaning: yes, partly or
no, with a one-line reason. A review an entrant didn't give (it reviewed
none of the point's heads) is never sent and is recorded as missing.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections.abc import Callable

from bench.harness.budget import Budget
from bench.labels.openrouter import OpenRouterError

ENTRANT = "issue-judge"
MAX_TOKENS = 300
VERDICTS = ("yes", "partly", "no")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
SYSTEM = ("You compare code reviews with a human reviewer's comment on the same pull request. For each review, "
          "decide whether it raises the same issue as the human comment, judged at file level and by meaning, "
          "not by exact line or wording: yes (the same issue), partly (related, or only part of it), no (not "
          "raised). Ignore everything else the review says. Reply with JSON only: "
          '{"A": {"verdict": "yes|partly|no", "reason": one short sentence}, "B": {...}}, '
          "with a key only for each review you were given.")


class JudgeError(ValueError):
    pass


def prompt_sha256() -> str:
    return hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()


def order(point_id: str, entrants: list[str], seed: int) -> list[str]:
    """The entrants in the order shown as A, B for this point: seeded, so anyone can redo it."""
    shown = sorted(entrants)
    random.Random(f"{seed}:{point_id}").shuffle(shown)
    return shown


def messages_for(point: dict, comment: str, reviews: dict[str, str]) -> list[dict]:
    """`reviews`: label (A, B) -> review text, already in the shown order."""
    anchor = point["anchor"]
    parts = [f"Human comment on {anchor['path']} line {anchor['line']}:\n{comment}"]
    parts += [f"Review {label}:\n{text}" for label, text in reviews.items()]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def parse(content: str | None, labels: list[str]) -> dict[str, dict]:
    match = JSON_RE.search(content or "")
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError as exc:
        raise JudgeError("invalid output: not JSON") from exc
    if not isinstance(data, dict):
        raise JudgeError("invalid output: not an object")
    out = {}
    for label in labels:
        item = data.get(label)
        if not isinstance(item, dict) or item.get("verdict") not in VERDICTS:
            raise JudgeError(f"invalid output: no verdict for review {label}")
        out[label] = {"verdict": item["verdict"], "reason": " ".join(str(item.get("reason") or "").split())[:200]}
    return out


def judge_point(point: dict, comment: str, by_entrant: dict[str, str | None], seed: int,
                call: Callable[[list[dict]], dict], budget: Budget) -> dict:
    """{order, verdicts: {entrant: {verdict, reason} | missing}, cost_usd} or {error}; every call charged."""
    shown = [e for e in order(point["id"], list(by_entrant), seed) if by_entrant[e] is not None]
    labels = {entrant: "AB"[i] for i, entrant in enumerate(shown)}
    record = {"order": shown, "verdicts": {e: {"verdict": "missing"} for e in by_entrant if e not in labels}}
    if not shown:
        return record
    if not budget.allows_review(ENTRANT):
        return {**record, "error": f"budget: ${budget.spent:.4f} spent of ${budget.max_usd:.2f}"}
    reviews = {labels[e]: by_entrant[e] for e in shown}
    try:
        reply = call(messages_for(point, comment, reviews))
    except OpenRouterError as exc:  # the run outlives a failed call; it reports no cost, so the bound is charged
        print(f"warning: {point['id']}: {exc}", file=sys.stderr)
        reply = {"content": None, "cost_usd": None, "served_model": None, "error": str(exc)}
    if reply["cost_usd"] is None:
        budget.add_cost(ENTRANT, budget.per_head_bound(ENTRANT))
        return {**record, "error": reply.get("error") or "OpenRouter reported no cost; charged at the bound"}
    budget.add_cost(ENTRANT, reply["cost_usd"])
    record["cost_usd"] = reply["cost_usd"]
    try:
        parsed = parse(reply["content"], list(reviews))
    except JudgeError as exc:  # paid for, but no verdict
        print(f"warning: {point['id']}: {exc}", file=sys.stderr)
        return {**record, "error": str(exc)}
    record["verdicts"].update({e: parsed[labels[e]] for e in shown})
    return record
