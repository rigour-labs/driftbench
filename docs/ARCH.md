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
run     ──► work/runs/<date>/<tool>/<repo>/<pr>/<head>.json (+ .raw.txt)
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

The workflow is least-privilege: `contents: read` everywhere, except the
release job that uploads run assets (`contents: write`). Third-party actions
are pinned by commit SHA.

## Paid entrants

Adapters declare `paid = True`. The `free` entrant set excludes them, and
running a paid adapter requires `--max-usd`. The harness stops before a
round's estimated cost would exceed it.
