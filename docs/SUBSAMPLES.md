# Subsamples

A paid run that can't afford the whole corpus reviews a **subsample**: a
selection of whole pull requests, drawn once, committed, and recorded by
every run that uses it (`bench subsample`, `bench/subsample.py`).

## The unit is the pull request

A selection never splits a pull request. A point on a later round is
matched against earlier heads of the same pull request (time-correct
matching), and false blocks are judged on its approved head; a sample of
single heads would break both.

## Per repository, one rule

- **full corpus**: every pull request;
- **sampled**: whole pull requests until about N heads. Pull requests are
  grouped by their largest diff (changed lines: 0-50, 50-200, 200-600, 600+),
  each group gets a share of N in proportion to its share of the
  repository's heads (largest remainder), and pull requests within a group
  are taken in a seeded random order;
- **not run in this round**, with the reason; the page says so, and earlier
  runs' results for that repository stand.

Diff sizes come from an earlier run's records (every entrant sees the same
diff) and the selection file names that source as published, never a local
path. The seed is chosen before the draw and never re-chosen to fit a
budget; a selection is never redrawn once a run has used it.

## What a run and the page record

`run.json` keeps the file, its SHA-256, the seed, and each repository's rule
with its counts of pull requests and heads; scoring refuses a selection file
that has changed since. Scoring and the page use only the selected pull
requests, so every count, minimum and interval is over the subsample; heads
left out are not "unavailable", they were never in scope. Each repository's
section opens with its rule and counts.

## Budget

A paid run's cap is split across the repository jobs in proportion to their
selected heads (recorded in `run.json` as `cap_shares`), so the larger
sample isn't stopped early by an even split. A review that doesn't fit is
recorded `not_scored` and counted, as always.

## Diagnostic runs

A diagnostic run answers a question about an earlier result. It is never a
result itself. Its selection lists exact heads with its purpose
(`bench subsample --heads-from FILE --purpose ...`): only those heads run,
not their whole pull requests. `--diagnostic PURPOSE` (run.yml
`diagnostic`) records the purpose and its limit in `run.json`. `bench score`
refuses the run, and its draft release is tagged `diagnostic-<name>` with a
first line saying it is not a result. The limit, stated in its notes: the
entrants are language models, so a re-run can serve different findings than
the earlier run on the same head. It shows what they consider and filter on
those heads in general, not exactly what they did then.
