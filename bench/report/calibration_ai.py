"""AI verdicts for the calibration sample (docs/SPEC.md, "Calibration").

- acted-on entries check DriftBench's own scorer, not an entrant, so they
  use AI consensus: a verdict stands only where the Claude labeller's blind
  verdict and the non-Claude model's agree; otherwise the entry is disputed
  and left out, and the count is shown.
- location entries judge an entrant's finding, and Rigour's reviewer runs
  on Claude, so a Claude labeller never takes part: the non-Claude model
  decides alone.

A verdict a human already wrote into the sample always wins.
"""
from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable

from bench.harness.budget import Budget
from bench.labels.openrouter import OpenRouterError
from bench.report.calibration import ACTED_VERDICTS, LOCATION_VERDICTS, CalibrationError

ENTRANT = "calibration"
MAX_TOKENS = 200
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
QUESTIONS = {
    "acted_on": ("Did the change near the anchor respond to the review comment? Answer yes only if the code "
                 "changed in a way that addresses what the reviewer asked.", ACTED_VERDICTS),
    "location": ("Does the tool's finding raise the same issue as the review comment? yes: the same issue; "
                 "partly: related, or only part of it; no: a different issue at the same spot.", LOCATION_VERDICTS),
}


class VerdictError(CalibrationError):
    pass


def entry_key(entry: dict) -> str:
    return f"{entry['kind']}:{entry['point']}:{entry.get('tool') or ''}:{entry.get('finding', '')}"


def messages_for(entry: dict, evidence: str) -> list[dict]:
    question, allowed = QUESTIONS[entry["kind"]]
    system = (f"You check one item of a code review benchmark. {question} Reply with JSON only: "
              f"{{\"verdict\": one of {json.dumps(list(allowed))}, \"reason\": one short sentence}}.")
    return [{"role": "system", "content": system}, {"role": "user", "content": evidence}]


def parse_verdict(content: str | None, kind: str) -> tuple[str, str]:
    allowed = QUESTIONS[kind][1]
    match = JSON_RE.search(content or "")
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError as exc:
        raise VerdictError("invalid output: not JSON") from exc
    if not isinstance(data, dict) or data.get("verdict") not in allowed:
        raise VerdictError(f"invalid output: verdict not one of {allowed}")
    return data["verdict"], " ".join(str(data.get("reason") or "").split())[:200]


def judge(entry: dict, evidence: str, call: Callable[[list[dict]], dict], budget: Budget) -> tuple[dict | None, str]:
    """One model verdict: (entry or None, why not). Every call is charged; failures at the bound."""
    if not budget.allows_review(ENTRANT):
        return None, f"budget: ${budget.spent:.4f} spent of ${budget.max_usd:.2f}"
    try:
        reply = call(messages_for(entry, evidence))
    except OpenRouterError as exc:  # the run outlives a failed call; it reports no cost, so the bound is charged
        print(f"warning: {entry['point']}: {exc}", file=sys.stderr)
        reply = {"content": None, "cost_usd": None, "served_model": None, "error": str(exc)}
    if reply["cost_usd"] is None:
        budget.add_cost(ENTRANT, budget.per_head_bound(ENTRANT))
        return None, reply.get("error") or "OpenRouter reported no cost; charged at the bound"
    budget.add_cost(ENTRANT, reply["cost_usd"])
    result = {"verdict": None, "reason": "", "cost_usd": reply["cost_usd"], "served_model": reply["served_model"]}
    try:
        result["verdict"], result["reason"] = parse_verdict(reply["content"], entry["kind"])
    except VerdictError as exc:  # paid for, but no verdict
        print(f"warning: {entry['point']}: {exc}", file=sys.stderr)
        return {**result, "reason": str(exc)}, str(exc)
    return result, ""


def merge_verdicts(calibration: dict, claude: dict, model: dict) -> dict:
    """Write verdicts into the sample. `claude`: {labeller, verdicts: {key: verdict}} for acted-on entries only;
    `model`: {model, verdicts: {key: {verdict, ...}}}. Human verdicts are kept as they are."""
    if any(key.startswith("location:") for key in claude.get("verdicts", {})):
        raise VerdictError("a Claude labeller must not judge location entries: Rigour's reviewer runs on Claude")
    entries = []
    for entry in calibration["entries"]:
        if entry.get("verdict") is not None and entry.get("verdict_by", "human") == "human":
            entries.append({**entry, "verdict_by": "human"})
            continue
        key = entry_key(entry)
        mine = (model.get("verdicts", {}).get(key) or {}).get("verdict")
        if entry["kind"] == "location":
            entries.append({**entry, "verdict": mine, "verdict_by": f"model: {model['model']}" if mine else None})
            continue
        theirs = claude.get("verdicts", {}).get(key)
        agreed = mine is not None and mine == theirs
        disputed = mine is not None and theirs is not None and mine != theirs
        by = f"consensus: {claude.get('labeller')} + {model['model']}" if agreed else None
        entries.append({**entry, "verdict": mine if agreed else None, "verdict_by": by,
                        **({"disputed": True} if disputed else {})})
    return {**calibration, "entries": entries}
