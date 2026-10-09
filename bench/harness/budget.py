"""The hard dollar stop for paid entrants (docs/SPEC.md, "Running a tool").

One cap for the whole run, across every paid entrant. Before each paid
review, the harness asks whether the review could still fit: the money spent
so far plus that entrant's per-head bound. The bound is the larger of the
run estimate's upper bound per head (given up front, so the first review is
bounded too) and the most the entrant has actually cost on one head so far.
Once a review doesn't fit, the run is out of budget: that review and every
later paid one is recorded `not_scored` with the reason, never skipped
silently.
"""
from __future__ import annotations

import dataclasses


class BudgetError(ValueError):
    pass


@dataclasses.dataclass
class Budget:
    max_usd: float
    estimate_per_head: dict[str, float]  # entrant -> upper bound per head, from the run estimate
    spent: float = 0.0
    largest: dict[str, float] = dataclasses.field(default_factory=dict)
    exhausted: bool = False

    def per_head_bound(self, entrant: str) -> float:
        if entrant not in self.estimate_per_head:
            raise BudgetError(f"no per-head cost bound for paid entrant {entrant}; give it from the run estimate")
        return max(self.estimate_per_head[entrant], self.largest.get(entrant, 0.0))

    def allows_review(self, entrant: str) -> bool:
        """Whether one more review by `entrant` fits; once one doesn't, nothing paid does."""
        if not self.exhausted and self.spent + self.per_head_bound(entrant) > self.max_usd:
            self.exhausted = True
        return not self.exhausted

    def add_cost(self, entrant: str, cost_usd: float) -> None:
        self.spent = round(self.spent + cost_usd, 6)
        self.largest[entrant] = max(self.largest.get(entrant, 0.0), cost_usd)

    def as_record(self) -> dict:
        return {"max_usd": self.max_usd, "spent_usd": self.spent, "estimate_per_head": self.estimate_per_head,
                "largest_per_head": self.largest, "exhausted": self.exhausted}
