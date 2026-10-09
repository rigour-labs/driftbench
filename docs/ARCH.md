# Architecture

The pipeline has five stages. Each stage reads the previous stage's files and
writes its own, so any stage can be re-run alone and every intermediate
result can be published.

```
repos.yaml
   │
   ▼
collect ──► work/corpus/<repo>.json      frozen GitHub data: PRs, rounds, reviews (no text)
   │
   ▼
points  ──► work/points/<repo>.json      review points, drop rules, acted-on (no text)
   │
   ▼
label   ──► labels/<repo>.yaml           committed; point IDs, suggested and confirmed class (no text)
   │
   ▼
run     ──► work/runs/<date>/<tool>/<repo>/<pr>/<head>.json
   │         one record per (tool, reviewed head): cases on that head, findings,
   │         verdict, changed lines, wall time, usage, tool version; resumable
   ▼
score   ──► results/<date>/summary.md    committed tables
            work/runs/<date>/ledger.jsonl  every match decision, published as a release asset
```

## Modules (`bench/`)

| Module | Job | Status |
|---|---|---|
| `repos.py` | Load and validate `repos.yaml` | PR 1 |
| `name_guard.py` | Refuse to publish blocked names (pre-push, pre-release) | PR 1 |
| `collect/` | GitHub reads (cached), PR selection, rounds, the frozen record | PR 2 |
| `points/` | Point splitting, drop rules, acted-on | PR 3 |
| `labels/` | Label schema, rules pass, merge with confirmations | PR 4 |
| `harness/` | Adapter interface, time-correct checkout, runner | PR 5 |
| `adapters/` | One module per entrant | PR 5, 8 |
| `score/` | Location matching, noise, false blocks, ledger | PR 6 |
| `report/` | Markdown tables, per-class results, calibration samples | PR 7 |

## Adapter boundary

The harness owns everything that could favour one tool: checkout, the diff,
what history is visible, timing and scoring. An adapter only turns a
`ReviewInput` into a `ReviewOutput` (see `CONTRIBUTING.md`). No scoring code
knows which tool produced a finding.

## Where runs happen

Real runs execute in GitHub Actions (or other remote compute), never on a
contributor's machine: clones of these repositories (some over 1 GB) plus
package caches don't belong on a laptop. Clones are blobless
(`--filter=blob:none`). `work/` from a run is uploaded as a workflow artifact
and attached to a **draft** release; nothing is published from CI.

Local development uses synthetic fixtures (`tests/fixtures/`, `tests/*_fixtures.py`)
and at most one small repository.

The order of a release, every step remote except labelling and the checks:

1. **Collect** (`collect.yml`, dispatched by hand): `bench collect` and
   `bench points` with the default `GITHUB_TOKEN` (1,000 API requests an
   hour; the client waits for each reset). The API cache stays in the Actions
   cache, never in an artifact or release, because it holds review text.
   A workflow run by a pull request from a fork can restore caches made on
   the base branch, so this cache must only ever hold data that is already
   public (here: public review text from public repositories), never
   anything private. A
   draft release `corpus-<date>-<run id>` gets the corpus and points files.
2. A maintainer downloads that draft, runs `bench guard` and
   `python -m bench.release_check` on it, and publishes it.
3. **Label**: the maintainer draws the sample from that corpus's points file
   and labels it locally (`bench label sample`, `bench label next`); only the
   label files are committed, before step 4.
4. **Run** (`run.yml`, input: the corpus tag): one job writes the run
   record once (`bench manifest`: start time, entrant versions, label commit
   and blob hashes) and every later job reuses that file, one job per repository runs the free
   entrants, one job scores, draws the calibration sample and writes the
   report (it still runs when one repository's job fails, and the release
   notes name that repository); a draft release `run-<date>-<run id>` gets
   the run records, ledger,
   summary and calibration sample, with `run_started_at` and the corpus tag in
   its notes. The diffs handed to tools contain the projects' code and are
   never uploaded.
5. A maintainer runs `bench guard` and the release check on the draft's
   assets, publishes it, fills the calibration verdicts, and opens a pull
   request adding `results/<date>/`.

The workflows are dispatched by hand and least-privilege: `contents: read`
everywhere, except the job that creates a draft release (`contents: write`).
Actions are pinned by commit SHA, and no personal token is used.

## Paid entrants

Adapters declare `paid = True`. The `free` entrant set excludes them; naming
one requires `--model` (one model for every paid entrant, by full ID) and
`--max-usd` (the hard cap for that invocation; a run split across N jobs
gives each job its share). Each paid entrant also needs `--head-bound
NAME=USD`, its per-head upper bound from the run estimate, so the very first
review is bounded (bench/harness/budget.py, bench/harness/paid.py).

In `run.yml`, a paid run takes the `model`, `max_usd` and `head_bounds`
inputs. The start job records them in `run.json`; each repo job gets an even
share of the cap; the pinned Claude Code CLI is installed and the
`ANTHROPIC_API_KEY` secret is exposed only in the paid review step; the
score job writes `spend.md` (`bench spend`) into the draft release notes.

Paid entrants: `rigour-reviewer`, `rigour-reviewer-orchestrated` and
`claude-code-review`. All drive the same pinned Claude Code CLI with the same
model and timeout, recorded in `run.json` under `paid`.
