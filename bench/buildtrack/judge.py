"""Did the agent's version repeat what the human reviewer pointed out? (docs/BUILD_TRACK.md, "Measures" 1)

The same non-Claude judge as the issue comparison (docs/ISSUES.md), blind:
per acted-on human point, it sees the point (its file, line and text, from
the real pull request) and both arms' final diffs as A and B, in a seeded
order per point, with nothing naming an arm or Rigour. For each diff it
answers:
- repeated: the diff has the problem the reviewer pointed out;
- avoided: the diff touches what the point is about and does not have it;
- not_applicable: the diff has no code the point is about.
A diff longer than DIFF_CHARS is cut, and the cut is recorded.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable

from bench.harness.budget import Budget
from bench.issues.judge import JSON_RE, JudgeError, order
from bench.labels.openrouter import OpenRouterError

ENTRANT = "build-judge"
MAX_TOKENS = 300
DIFF_CHARS = 30000
VERDICTS = ("repeated", "avoided", "not_applicable")
SYSTEM = ("A human reviewer commented on a real pull request. The same change was later made twice more, "
          "independently, from the same starting code; those diffs are A and B. For each diff, decide whether it has the problem the reviewer "
          "pointed out: repeated (the diff has that problem), avoided (the diff contains the code the comment is "
          "about and does not have the problem), not_applicable (the diff contains no code the comment is about). "
          "Judge by meaning, not by file names or line numbers, which may differ. Reply with JSON only: "
          '{"A": {"verdict": "repeated|avoided|not_applicable", "reason": one short sentence}, "B": {...}}.')


def prompt_sha256() -> str:
    return hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()


def cut(diff: str) -> tuple[str, bool]:
    return (diff, False) if len(diff) <= DIFF_CHARS else (diff[:DIFF_CHARS] + "\n[diff cut]", True)


def messages_for(point: dict, comment: str, diffs: dict[str, str]) -> list[dict]:
    anchor = point["anchor"]
    parts = [f"Reviewer's comment on {anchor['path']} line {anchor['line']}:\n{comment}"]
    parts += [f"Diff {label}:\n{text or '(no change)'}" for label, text in diffs.items()]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def parse(content: str | None, labels: list[str]) -> dict[str, dict]:
    match = JSON_RE.search(content or "")
    try:
        data = json.loads(match.group(0)) if match else None
    except ValueError as exc:
        raise JudgeError("invalid output: not JSON") from exc
    if not isinstance(data, dict):
        raise JudgeError("invalid output: not an object")
    out = {}
    for label in labels:
        item = data.get(label)
        if not isinstance(item, dict) or item.get("verdict") not in VERDICTS:
            raise JudgeError(f"invalid output: no verdict for diff {label}")
        out[label] = {"verdict": item["verdict"], "reason": " ".join(str(item.get("reason") or "").split())[:200]}
    return out


def ask(call: Callable[[list[dict]], dict], messages: list[dict], point_id: str) -> dict:
    try:
        return call(messages)
    except OpenRouterError as exc:  # the run outlives a failed call; it reports no cost, so the bound is charged
        print(f"warning: {point_id}: {exc}", file=sys.stderr)
        return {"content": None, "cost_usd": None, "error": str(exc)}


def judge_point(point: dict, comment: str, diffs_by_arm: dict[str, str], seed: int,
                call: Callable[[list[dict]], dict], budget: Budget) -> dict:
    """{order, cut, verdicts: {arm: {verdict, reason}}, cost_usd} or with {error}; every call charged."""
    shown = order(point["id"], list(diffs_by_arm), seed)
    labels = {arm: "AB"[i] for i, arm in enumerate(shown)}
    cuts = {arm: cut(diffs_by_arm[arm]) for arm in shown}
    record = {"order": shown, "cut": [arm for arm in shown if cuts[arm][1]], "verdicts": {}}
    if not budget.allows_review(ENTRANT):
        return {**record, "error": f"budget: ${budget.spent:.4f} spent of ${budget.max_usd:.2f}"}
    reply = ask(call, messages_for(point, comment, {labels[a]: cuts[a][0] for a in shown}), point["id"])
    if reply.get("cost_usd") is None:
        budget.add_cost(ENTRANT, budget.per_head_bound(ENTRANT))
        return {**record, "error": reply.get("error") or "OpenRouter reported no cost; charged at the bound"}
    budget.add_cost(ENTRANT, reply["cost_usd"])
    record["cost_usd"] = reply["cost_usd"]
    try:
        parsed = parse(reply["content"], list(labels.values()))
    except JudgeError as exc:  # paid for, but no verdict
        print(f"warning: {point['id']}: {exc}", file=sys.stderr)
        return {**record, "error": str(exc)}
    record["verdicts"] = {arm: parsed[labels[arm]] for arm in shown}
    return record
