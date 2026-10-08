# Architecture

The pipeline has five stages. Each stage reads the previous stage's files and
writes its own, so any stage can be re-run alone and every intermediate
result can be published.

```
repos.yaml
   │
   ▼
collect ──► work/corpus/<repo>.json      frozen GitHub data: PRs, rounds, reviews, points
   │
   ▼
label   ──► labels/<repo>.yaml           committed; IDs, spans, classes (no comment text)
   │
   ▼
run     ──► work/runs/<date>/<tool>/<repo>/<pr>/<round>.json
   │         one file per (tool, round): findings, usage, wall time, tool version
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
| `points/` | Point splitting, acted-on | PR 3 |
| `labels/` | Label schema, rules pass, merge with confirmations | PR 3 |
| `harness/` | Adapter interface, time-correct checkout, runner | PR 4 |
| `adapters/` | One module per entrant | PR 4, 7 |
| `score/` | Location matching, noise, false blocks, per-class tables | PR 5 |

## Adapter boundary

The harness owns everything that could favour one tool: checkout, the diff,
what history is visible, timing and scoring. An adapter only turns a
`ReviewInput` into a `ReviewOutput` (see `CONTRIBUTING.md`). No scoring code
knows which tool produced a finding.

## Where runs happen

Runs execute in GitHub Actions with blobless clones (`--filter=blob:none`).
Some repositories are over 1 GB, and nothing large is downloaded to a
contributor's machine by default. Local development uses small recorded
fixtures under `tests/fixtures/`.

The workflow is least-privilege: `contents: read` everywhere, except the
release job that uploads run assets (`contents: write`). Third-party actions
are pinned by commit SHA.

## Paid entrants

Adapters declare `paid = True`. The `free` entrant set excludes them, and
running a paid adapter requires `--max-usd`. The harness stops before a
round's estimated cost would exceed it.
