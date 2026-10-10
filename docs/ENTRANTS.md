# Paid entrants

Paid entrants run under the same sandbox, with only the model provider's
variables added. For a fair comparison they share one model (by full ID),
one timeout, one provider and one pinned Claude Code CLI, all recorded in
`run.json`. Rigour runs with its defaults and no tuning on this corpus.

- **Provider.** `--provider anthropic` (the default) adds `ANTHROPIC_API_KEY`.
  `--provider openrouter` sends Claude Code through OpenRouter's Anthropic
  gateway instead: `ANTHROPIC_BASE_URL=https://openrouter.ai/api`,
  `ANTHROPIC_AUTH_TOKEN` from `OPENROUTER_API_KEY`, and `ANTHROPIC_API_KEY`
  set empty so no Anthropic key can be used. Both entrants get the same
  three variables. The model must then be a full versioned `anthropic/` ID,
  never an alias such as `latest`.

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
- **Smoke runs.** Before a paid run, `--max-heads N` (run.yml `max_heads`,
  with `repos` to pick one repo) reviews only the first N heads per entrant
  per repo, in corpus order, whose diff has at least 20 changed lines, so
  the cost per head is realistic. The cut is fixed in `run.json` (`smoke`),
  other heads get no record, `bench score` refuses the run, and its draft
  release is tagged `smoke-…` with notes saying it is not a result.
- **What the dollars are.** For both tool families the cost is Claude
  Code's own figure: a list-price estimate from token counts, not a bill.
  The page labels it so. The run notes record estimate and billed side by
  side (`bench spend`). Through Anthropic, the maintainer reads the billed
  amount for the run window from the console. Through OpenRouter it comes
  from data (`bench spend --openrouter`): each run job reads the key's
  cumulative usage before and after, and billed is the latest end minus the
  earliest start, exact when the key is used only for the run. Either way a
  spend limit on the key is the external backstop.
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
- **Instructions.** Each paid run records which review instructions each
  entrant ran, by hash only. Claude Code's `/code-review` is built into its
  proprietary native binary: its text is never extracted, copied or
  published. `run.json` records the native package, version and npm
  integrity for Claude Code, and the core package, its integrity and its
  `PROMPT_VERSION` for Rigour. Each review job adds the sha256 of the
  installed files that hold them (`instructions-<repo>.json`). Anyone can
  install the same versions and compare.
- **Claude Code `/code-review`** reports findings in prose; each `path:line`
  it cites in a changed file becomes a finding. It has no blocking findings,
  so its blocking-only catch rate and false blocks are shown as "n/a", not
  as 0.

The method these rules belong to is in [docs/SPEC.md](SPEC.md).
