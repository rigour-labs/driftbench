import pytest

from bench.harness.budget import Budget, BudgetError
from bench.harness.paid import paid_gate, settle_review
from bench.harness.types import Finding, ReviewOutput


class Paid:
    name, version, paid, reads_history, env_extra = "p", "1", True, False, ("ANTHROPIC_API_KEY",)


class Free:
    name, version, paid, reads_history, env_extra = "f", "1", False, False, ()


def test_first_review_is_bounded_by_the_estimate_then_by_the_largest_real_cost():
    budget = Budget(max_usd=1.0, estimate_per_head={"p": 0.30})
    assert budget.per_head_bound("p") == 0.30 and budget.allows_review("p")
    budget.add_cost("p", 0.45)
    assert budget.per_head_bound("p") == 0.45                      # a real head cost more than the estimate
    budget.add_cost("p", 0.10)
    assert budget.per_head_bound("p") == 0.45 and budget.spent == 0.55


def test_stop_at_the_boundary_and_stay_stopped():
    budget = Budget(max_usd=1.0, estimate_per_head={"p": 0.5})
    assert budget.allows_review("p")
    budget.add_cost("p", 0.5)
    assert budget.allows_review("p")                              # 0.5 + 0.5 == 1.0 still fits
    budget.add_cost("p", 0.5)
    assert not budget.allows_review("p") and budget.exhausted
    budget.spent = 0.0                                     # never resumes once exhausted
    assert not budget.allows_review("p")
    assert budget.as_record()["exhausted"] is True


def test_a_paid_entrant_without_a_bound_is_refused():
    with pytest.raises(BudgetError, match="no per-head cost bound"):
        Budget(1.0, {}).per_head_bound("p")


def test_gate():
    assert paid_gate(Free(), None) is None
    assert paid_gate(Paid(), None)["verdict"] == "not_scored"
    budget = Budget(max_usd=0.2, estimate_per_head={"p": 0.3})
    record = paid_gate(Paid(), budget)
    assert record["verdict"] == "not_scored" and "budget" in record["error"]


def test_settle_leaks_usage_and_honest_zero():
    budget = Budget(max_usd=5, estimate_per_head={"p": 1})
    finding = Finding("a.py", 1, True, "m")
    leaked = settle_review(Paid(), ReviewOutput([finding], "fail", cost_usd=0.2, model_runs=1, leak_signals=2), budget)
    assert leaked.verdict == "leaked" and leaked.findings == [] and budget.spent == 0.2   # still paid for
    silent = settle_review(Paid(), ReviewOutput([finding], "fail", cost_usd=None, model_runs=3), budget)
    assert silent.verdict == "error" and "no usage" in silent.error
    free_ran = settle_review(Paid(), ReviewOutput([finding], "pass", cost_usd=None, model_runs=0), budget)
    assert free_ran.verdict == "pass" and free_ran.findings == [finding]               # nothing to review: $0
    assert settle_review(Free(), ReviewOutput([], "pass"), None).verdict == "pass"


def test_a_model_run_without_a_reported_cost_is_charged_the_bound_and_errors():
    budget = Budget(max_usd=5, estimate_per_head={"p": 0.7})
    out = settle_review(Paid(), ReviewOutput([Finding("a.py", 1, True, "m")], "fail", cost_usd=None, model_runs=2),
                        budget)
    assert (out.verdict, out.charged, budget.spent) == ("error", "bound", 0.7)
    honest = settle_review(Paid(), ReviewOutput([], "pass", cost_usd=None, model_runs=0), budget)
    assert (honest.verdict, honest.charged, budget.spent) == ("pass", "", 0.7)     # nothing ran: no charge


def test_a_model_run_reporting_zero_dollars_is_no_usage():
    budget = Budget(max_usd=5, estimate_per_head={"p": 0.6})
    out = settle_review(Paid(), ReviewOutput([], "pass", cost_usd=0.0, model_runs=4), budget)
    assert (out.verdict, out.charged, budget.spent) == ("error", "bound", 0.6)
    cached = settle_review(Paid(), ReviewOutput([], "pass", cost_usd=0.0, model_runs=0), budget)
    assert (cached.verdict, budget.spent) == ("pass", 0.6)           # nothing ran: $0 is honest
