# Paid entrants

Paid entrants run under the same sandbox, with exactly one variable added:
the model provider's key. For a fair comparison they share one model (by
full ID), one timeout and one pinned Claude Code CLI, all recorded in
`run.json`. Rigour runs with its defaults and no tuning on this corpus.

- **Hard dollar stop.** One cap for the invocation. Before each paid review
  the harness checks that the money spent plus that entrant's per-head bound
  still fits. The bound is the larger of the run estimate's upper bound and
  the most the entrant has cost on one head so far. Once a review doesn't
  fit, it and every later paid review is recorded `not_scored`, counted on
  the page, never left out silently. A review that fails (a timeout, the
  tool unavailable, an unreadable transcript) is charged what it reported
  spending, or, if it reported nothing, the entrant's per-head bound; each
  head records which (`charged: reported | bound`). Tools that can stop
  themselves also get the per-head bound (Claude Code's `--max-budget-usd`)
  as a second line of defence.
- **What the dollars are.** For both tool families the cost is Claude
  Code's own figure: a list-price estimate from token counts, not a bill.
  The page labels it so. After each paid run the maintainer reads the
  provider's billed amount for the run window, and the run notes record
  estimate and billed side by side (`bench spend`). The provider is a direct
  Anthropic key, so the model ID is native and the estimate tracks the bill;
  a spend limit on the key's console workspace is the external backstop.
- **Usage.** Rigour's cost is every dollar the review spent (`spent_usd`
  where the version reports it: all runs, failed passes, retries; else
  `cost_usd`). A review whose model ran must report a cost above $0; one
  reporting none, or $0, is charged the per-head bound and recorded as an
  error. A review where no model ran (nothing to review, a
  cached verdict) honestly costs $0.
- **Leakage.** Tools with a reviewer mode may look up the pull request and
  its human reviews, the very answers being scored. The sandbox blocks that
  (no token, no login, no remote, and the tools' web and GitHub access
  turned off). Each head is also checked: Rigour's own record of human
  reviews seen, and any web or GitHub call in Claude Code's transcript.
  Any sign makes the head `leaked`; it is not scored.
- **Cold start.** Every entrant runs cold: no learned lessons, team memory
  or past reviews. Rigour's reviewer learns from a team's history, and this
  benchmark doesn't measure that; a later run with time-ordered learning
  (lessons only from pull requests merged before each head) would.
- **Claude Code `/code-review`** reports findings in prose; each `path:line`
  it cites in a changed file becomes a finding. It has no blocking findings,
  so its blocking-only catch rate and false blocks are shown as "n/a", not
  as 0.

The method these rules belong to is in [docs/SPEC.md](SPEC.md).
