"""Rules for entrants that can spend money or reach the network (docs/SPEC.md, "Running a tool").

- Budget: before each paid review, the run's Budget must allow it; otherwise
  the head is recorded `not_scored` with the reason (bench/harness/budget.py).
- Usage: a paid review whose model ran (`model_runs` > 0) must report its
  cost; one that reports none is an `error`, never a silent $0. A review where
  no model ran (nothing to review, a cached verdict) honestly costs $0.
- Leakage: an adapter counts every sign that its tool saw the pull request's
  human reviews or fetched pull request data (`leak_signals`). Any sign makes
  the head `leaked`: its findings are dropped and it is not scored.
"""
from __future__ import annotations

import dataclasses

from bench.harness.budget import Budget
from bench.harness.types import Adapter, ReviewOutput


def not_scored(reason: str) -> dict:
    return {"base_sha": None, "changed_lines": None, "verdict": "not_scored", "findings": [], "wall_s": None,
            "cost_usd": None, "input_tokens": None, "output_tokens": None, "error": reason}


def paid_gate(adapter: Adapter, budget: Budget | None) -> dict | None:
    """A `not_scored` record when a paid review may not run, else None."""
    if not adapter.paid:
        return None
    if budget is None:
        return not_scored("budget: a paid entrant needs --max-usd")
    if not budget.allows_review(adapter.name):
        return not_scored(f"budget: ${budget.spent:.2f} spent of ${budget.max_usd:.2f}; "
                          f"this review could cost up to ${budget.per_head_bound(adapter.name):.2f}")
    return None


def charge_review(adapter: Adapter, output: ReviewOutput, budget: Budget | None) -> ReviewOutput:
    """Charge a paid review to the budget, failures included.

    What the tool reported, when it reported anything; otherwise, for a review
    that failed (a timeout burns its whole limit) or whose model ran without
    reporting a cost, the entrant's per-head bound, so neither can slip past
    the cap. The head records which.
    """
    if not adapter.paid or budget is None:
        return output
    if output.cost_usd is not None:
        budget.add_cost(adapter.name, output.cost_usd)
        return dataclasses.replace(output, charged="reported")
    ran_unreported = (output.model_runs or 0) > 0  # the model ran but no cost came back: assume the worst
    if output.verdict == "error" or ran_unreported:
        budget.add_cost(adapter.name, budget.per_head_bound(adapter.name))
        return dataclasses.replace(output, charged="bound")
    return output


def settle_review(adapter: Adapter, output: ReviewOutput, budget: Budget | None) -> ReviewOutput:
    """Charge the budget, then apply the leakage and usage rules to a finished review."""
    output = charge_review(adapter, output, budget)
    if output.leak_signals:
        return dataclasses.replace(output, findings=[], verdict="leaked",
                                   error=f"{output.leak_signals} sign(s) the tool saw pull request reviews or data")
    if adapter.paid and output.verdict in ("pass", "fail") and (output.model_runs or 0) > 0 and output.cost_usd is None:
        return dataclasses.replace(output, findings=[], verdict="error", error="the model ran but reported no usage")
    return output
