# DriftBench

**A neutral benchmark of AI code reviewers and quality gates, scored against
real, human-reviewed open-source pull requests.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Status: rebuilding.** No results are published yet. The previous version
> (arena, Kaggle runs, RLAIF data, evalgate, API) is preserved under the
> [`v1`](https://github.com/rigour-labs/driftbench/tree/v1) tag and the
> `v1-archive` branch.

## What it measures

For each repository, per tool:

| Measure | Meaning |
|---|---|
| **Flagged the same spot, before the human** | Share of human review points where the tool flagged the same file and lines (±3) using only code from that review round or earlier |
| **False blocks on approved code** | Blocking findings on the head the maintainers approved and merged. A usable gate has close to zero |
| **Noise** | Findings per 100 changed lines, shown next to every catch rate |
| **Cost and time** | Dollars per review when reported (else tokens), and wall time |
| **By class** | Mechanical, performance, claim/contract, user journey, judgment, where a human confirmed the label |

The full method, with its known limits, is in [docs/SPEC.md](docs/SPEC.md).
In short:

- **Time-correct.** A tool reviewing round k sees the code at that round's
  head and, optionally, earlier comments. It never sees the review it is
  being compared to, or anything after it.
- **Location, calibrated.** "Same spot" isn't proof of "same issue". A
  hand-checked sample of matches per release gives each tool's
  location-to-issue rate. An every-hunk baseline shows the score that
  flagging everything would get.
- **Small samples aren't reported.** A repository needs at least 20 acted-on
  points and 10 approved heads, or it shows *insufficient data*.

## Entrants

| Entrant | Cost | Notes |
|---|---|---|
| No tool | Free | Baseline: zero findings |
| Every hunk | Free | Baseline: flags every changed hunk, the ceiling for spraying |
| Rigour `review` (deterministic) | Free | `@rigour-labs/cli`, pinned version |
| Rigour `review --reviewer` | Paid | Runs only with an approved dollar cap |
| Claude Code `/code-review` | Paid | Runs only with an approved dollar cap |
| CodeRabbit | n/a | **Not run**: its CLI needs an account sign-in. A vendor-run adapter is welcome |

**Disclosure:** this repository is maintained by Rigour Labs, which makes one
of the entrants. Rigour gets no special handling: it is one adapter behind the
same interface as the others, and the scoring code doesn't know which tool
produced a finding. Every raw output is published so anyone can check this.

## Repositories

Pinned in [repos.yaml](repos.yaml): tailscale/tailscale (BSD-3-Clause),
zulip/zulip (Apache-2.0), immich-app/immich (AGPL-3.0); then apache/superset
(Apache-2.0) and logto-io/logto (MPL-2.0). No project's code is copied here.
The corpus stores pull request numbers, SHAs, comment IDs and spans.

## Running it

Runs need a GitHub token (through the `gh` CLI) and are meant for GitHub
Actions, because some repositories are over 1 GB.

```bash
python -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m bench repos          # validate and list the pinned repos
.venv/bin/python -m bench collect        # freeze the corpus of reviewed PRs (GitHub API, free)
.venv/bin/python -m bench points         # split reviews into points; check which were acted on
.venv/bin/python -m bench label suggest  # suggest classes; a human confirms (docs/LABELLING.md)
.venv/bin/python -m pytest               # harness and scoring tests
```

The collect, run and score commands are added stage by stage (see
[docs/ARCH.md](docs/ARCH.md)). When they land, one command re-runs every free
entrant from the frozen corpus:

```bash
.venv/bin/python -m bench run --entrants free
```

## Results

Summary tables go in `results/` on `main`. Raw outputs (verdicts, match
ledgers, logs, the frozen corpus) are published as GitHub Release assets
tagged `run-YYYY-MM-DD`.

## Adding a tool

One adapter module and one recorded-output test. See
[CONTRIBUTING.md](CONTRIBUTING.md#adding-a-tool-for-any-vendor-or-user).

## Licence

MIT for this repository's code, method and labels. Each benchmarked repository
keeps its own licence (listed in `repos.yaml`).
