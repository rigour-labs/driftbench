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
- **Tool access, identical and read-only.** A paid review has a real key in
  its environment while it reads untrusted public pull request content, so
  nothing in it may run arbitrary shell, write files, or reach the network.
  Both families run Claude Code with the same explicit allow-list
  (`Read`, `Grep`, `Glob`, `git diff/show/log/grep`), the same denials
  (`Edit`, `Write`, `NotebookEdit`, `git push/commit`), and the same
  isolation (no MCP servers, no hooks or user settings, no memory files, at
  most 80 turns). Claude Code's `/code-review` also has web and GitHub tools
  denied. The list is recorded in `run.json`; before a paid run, Rigour's
  list is read from the pinned CLI and the run stops if it differs. If
  `/code-review` needs more tools to work, that is decided from a smoke run,
  not by widening the list in advance.
- **The key.** A dedicated Anthropic workspace key with a spend limit on
  that workspace, used only for these runs and rotated after every paid run.
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
