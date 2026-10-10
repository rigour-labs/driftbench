"""Rigour with its reviewer: `rigour review --base <mb> --json --reviewer --single --blind -c <config>` (paid).

Runs the deterministic gates and then the reviewer, which drives the pinned
Claude Code CLI with the run's model (set through a config file written into
the sandbox HOME, so the reviewed repository is untouched; Rigour otherwise
runs with its defaults and no tuning on this corpus). `--orchestrator` makes
the orchestrated variant.

Cost is every dollar the review spent: `spent_usd` where the version reports
it (all runs, failed passes, fallback, retries), else `cost_usd`.

Blocking follows Rigour: gate failures and reviewer `items` block;
`advisory` and `notes` don't. A reviewer that reports `unavailable` is an
error, not a block.

Time-correctness: `--blind` makes the reviewer review the change alone, with
no pull request lookup, description or human reviews, so it needs no `gh`,
and the sandbox gives it none. Every head checks the reviewer's own report:
a review not marked blind (in the reviewer section and in its record), any
human review seen, or a pull request found is a leak signal, and the head is
not scored.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from bench.adapters.claude_cli import paid_env, provider_env_names, require_claude_cli
from bench.adapters.rigour import VERSION, load_report, parse_report
from bench.harness.types import AdapterError, Finding, ReviewInput, ReviewOutput

TIMEOUT_MARGIN_S = 120


def reviewer_config(model: str, timeout_s: int) -> dict:
    return {"review": {"reviewer": {"enabled": True, "reviewers": ["claude"], "mode": "single",
                                    "models": {"claude": model}, "timeout_ms": timeout_s * 1000}}}


def reviewer_findings(reviewer: dict) -> list[Finding]:
    def item(entry: dict, blocking: bool) -> Finding:
        return Finding(entry.get("file"), entry.get("line"), blocking, str(entry.get("issue") or ""),
                       f"reviewer:{entry.get('class') or ''}")
    return ([item(e, True) for e in reviewer.get("items") or [] if isinstance(e, dict)]
            + [item(e, False) for e in (reviewer.get("advisory") or []) + (reviewer.get("notes") or [])
               if isinstance(e, dict)])


def leak_signals(reviewer: dict) -> int:
    """Signs the review could have seen pull request context: not blind, human reviews seen, a pull request found."""
    record = reviewer.get("record") or {}
    seen = int(((record.get("reported") or {}).get("human_reviews")) or 0)
    not_blind = reviewer.get("blind") is not True or (bool(record) and record.get("blind") is not True)
    return seen + (1 if reviewer.get("pr") else 0) + (1 if not_blind else 0)


def model_runs(reviewer: dict) -> int | None:
    if reviewer.get("cached"):
        return 0
    judges = (reviewer.get("record") or {}).get("judges")
    return len(judges) if isinstance(judges, list) else None


def spent_usd(reviewer: dict) -> float | None:
    """Every dollar the review spent (all runs, failed passes, retries) where the version reports it;
    otherwise the older `cost_usd`."""
    for key in ("spent_usd", "spentUsd", "cost_usd"):
        if isinstance(reviewer.get(key), (int, float)):
            return float(reviewer[key])
    return None


HELD_BACK = ("dropped", "unverified", "disputed", "dismissed")


def held_back(reviewer: dict) -> dict:
    """What the reviewer considered but didn't serve: each list with its entries as written (full messages),
    the `shown` tally (including `folded`), and a count per list. Kept so "saw it and filtered it out" can be
    told from "never saw it"."""
    lists = {key: [e for e in (reviewer.get(key) or []) if isinstance(e, dict)] for key in HELD_BACK}
    return {**lists, "shown": reviewer.get("shown") if isinstance(reviewer.get("shown"), dict) else None,
            "counts": {key: len(entries) for key, entries in lists.items()}}


TRACE = ("record", "mode", "tokens", "passes", "turns", "num_turns", "tool_calls", "cached")


def trace(reviewer: dict) -> dict:
    """How the review ran, as the reviewer reports it: its record (judges, lessons served), its mode (what was asked
    and what ran, passes with hunks, chars and reads beyond the slice), tokens and any turn counts. Kept so an empty
    verdict can be told from a review that stopped early or ran out of room."""
    return {key: reviewer[key] for key in TRACE if key in reviewer}


def to_output(report: dict) -> ReviewOutput:
    reviewer = report.get("reviewer")
    if not isinstance(reviewer, dict):
        raise AdapterError("the report has no reviewer section")
    if reviewer.get("outcome") == "unavailable":
        raise AdapterError(f"reviewer unavailable: {reviewer.get('reason') or 'no reason given'}",
                           cost_usd=spent_usd(reviewer))
    gates = parse_report(report)
    blocks = report.get("status") == "FAIL" or bool(reviewer.get("items"))
    tokens = reviewer.get("tokens") or {}
    return ReviewOutput(findings=gates + reviewer_findings(reviewer), verdict="fail" if blocks else "pass",
                        cost_usd=spent_usd(reviewer), input_tokens=tokens.get("input"),
                        output_tokens=tokens.get("output"), model_runs=model_runs(reviewer),
                        leak_signals=leak_signals(reviewer), held_back=held_back(reviewer),
                        trace=trace(reviewer))


class RigourReviewer:
    version = VERSION
    paid = True
    reads_history = False
    def __init__(self, model: str, orchestrated: bool = False, provider: str = "anthropic"):
        self.model = model
        self.orchestrated = orchestrated
        self.provider = provider
        self.env_extra = provider_env_names(provider)
        self.name = "rigour-reviewer-orchestrated" if orchestrated else "rigour-reviewer"

    def command(self, request: ReviewInput, config: Path) -> list[str]:
        return ["npx", "--yes", f"@rigour-labs/cli@{VERSION}", "review", "--base", request.base_sha, "--json",
                "--reviewer", "--single", "--blind", "-c", str(config), *(["--orchestrator"] if self.orchestrated else [])]

    def review(self, request: ReviewInput) -> ReviewOutput:
        env = paid_env(request, self.provider)
        require_claude_cli(env)
        config = Path(env["HOME"]) / "rigour-bench.yml"
        config.write_text(yaml.safe_dump(reviewer_config(self.model, request.timeout_s)), encoding="utf-8")
        try:
            result = subprocess.run(self.command(request, config), cwd=request.workdir, env=env, capture_output=True,
                                    text=True, timeout=request.timeout_s + TIMEOUT_MARGIN_S, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AdapterError(f"timed out after {request.timeout_s + TIMEOUT_MARGIN_S}s") from exc
        return to_output(load_report(result.stdout, result.returncode))

