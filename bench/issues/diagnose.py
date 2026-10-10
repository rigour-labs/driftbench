"""Diagnosis: did Rigour's reviewer consider an issue and hold it back, or never raise it?

For each target point, every finding the reviewer wrote on the point's
eligible heads in a diagnostic run is numbered and shown to the non-Claude
judge: the served findings, and the held-back ones (dropped, unverified,
disputed, dismissed), mixed and with nothing saying which list a finding
came from, so the judge can't favour either. It names the findings that
raise the same issue as the human point (by meaning, at file level), and
the diagnosis maps them back to their lists. A point ends in one bucket:
`served` (the reviewer raised it this time), `held_back:<list>` (it
considered the issue and filtered it out), or `absent` (never raised).
"""
from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable

from bench.adapters.rigour_reviewer import HELD_BACK
from bench.harness.budget import Budget
from bench.issues.inputs import redact
from bench.labels.openrouter import OpenRouterError
from bench.score.match import eligible_heads

ENTRANT = "diagnosis"
MAX_TOKENS = 300
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
SYSTEM = ("You compare a human reviewer's comment with a numbered list of findings from an automated code review "
          "of the same change. Name every finding that raises the same issue as the human comment, judged at file "
          "level and by meaning, not by exact line or wording; name findings that raise it only in part "
          'separately. Reply with JSON only: {"same": [numbers], "partly": [numbers], "reason": one short sentence}.')


class DiagnosisError(ValueError):
    pass


def candidates(records: list[dict]) -> list[dict]:
    """Every finding the reviewer wrote on these heads, with the list it came from (served or held back)."""
    found = []
    for record in records:
        output = record.get("paid_output") or {}
        for f in output.get("findings") or []:
            found.append({"list": "served", "path": f.get("path"), "line": f.get("line"), "text": f.get("message", "")})
        held = output.get("held_back") or {}
        for name in HELD_BACK:
            for e in held.get(name) or []:
                found.append({"list": name, "path": e.get("file"), "line": e.get("line"),
                              "text": " ".join(str(e.get(k) or "") for k in ("issue", "why")).strip()})
    return found


def point_records(point: dict, rounds: list[dict], by_head: dict[str, dict]) -> list[dict]:
    return [by_head[h] for h in eligible_heads(point, rounds) if h in by_head]


def messages_for(point: dict, comment: str, findings: list[dict]) -> list[dict]:
    anchor = point["anchor"]
    listed = "\n".join(f"{i}. {f['path']}:{f['line']}: {redact(f['text'])}" for i, f in enumerate(findings, 1))
    user = f"Human comment on {anchor['path']} line {anchor['line']}:\n{comment}\n\nFindings:\n{listed}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(content: str | None, count: int) -> tuple[set[int], set[int], str]:
    match = JSON_RE.search(content or "")
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError as exc:
        raise DiagnosisError("invalid output: not JSON") from exc
    if not isinstance(data, dict):
        raise DiagnosisError("invalid output: not an object")
    picked = []
    for key in ("same", "partly"):
        values = data.get(key) or []
        if not isinstance(values, list) or not all(isinstance(v, int) and 1 <= v <= count for v in values):
            raise DiagnosisError(f"invalid output: {key} is not a list of finding numbers")
        picked.append(set(values))
    return picked[0], picked[1], " ".join(str(data.get("reason") or "").split())[:200]


def bucket(findings: list[dict], same: set[int], partly: set[int]) -> str:
    """served if a served finding raises it; else the first held-back list that does; else absent."""
    lists = [findings[i - 1]["list"] for i in sorted(same | partly)]
    if "served" in lists:
        return "served"
    return f"held_back:{lists[0]}" if lists else "absent"


def diagnose_point(point: dict, comment: str, findings: list[dict], call: Callable[[list[dict]], dict],
                   budget: Budget) -> dict:
    """{bucket, same, partly, lists, reason, cost_usd} or {error}; no findings means absent, with no call."""
    if not findings:
        return {"bucket": "absent", "findings": 0}
    if not budget.allows_review(ENTRANT):
        return {"error": f"budget: ${budget.spent:.4f} spent of ${budget.max_usd:.2f}", "findings": len(findings)}
    try:
        reply = call(messages_for(point, comment, findings))
    except OpenRouterError as exc:  # the run outlives a failed call; it reports no cost, so the bound is charged
        print(f"warning: {point['id']}: {exc}", file=sys.stderr)
        reply = {"content": None, "cost_usd": None, "error": str(exc)}
    if reply["cost_usd"] is None:
        budget.add_cost(ENTRANT, budget.per_head_bound(ENTRANT))
        return {"error": reply.get("error") or "OpenRouter reported no cost; charged at the bound",
                "findings": len(findings)}
    budget.add_cost(ENTRANT, reply["cost_usd"])
    result = {"findings": len(findings), "cost_usd": reply["cost_usd"]}
    try:
        same, partly, reason = parse(reply["content"], len(findings))
    except DiagnosisError as exc:  # paid for, but no answer
        print(f"warning: {point['id']}: {exc}", file=sys.stderr)
        return {**result, "error": str(exc)}
    return {**result, "bucket": bucket(findings, same, partly), "same": sorted(same), "partly": sorted(partly),
            "lists": sorted({findings[i - 1]["list"] for i in same | partly}), "reason": reason}
