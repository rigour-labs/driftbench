# Issue-level comparison

The headline metric is a location match: a finding within N lines of a
human point. A review that explains a problem without citing file:line
can't match anything, whatever it says. Claude Code's /code-review, run
headless, mostly writes that way (run 2: no citation of any shape on 68 of
72 heads). So next to the location metric the page shows an issue-level
comparison, from the output the run already kept (`paid_output`, release
tarball only): no second paid run.

## The question

For each location-scorable human point P of the reviewed pull requests,
and each paid entrant E: *does E's review raise the same issue as P,
judged at file level and by meaning?* yes / partly / no, with a one-line
reason. E's review is its output on P's eligible heads (rounds up to P's
own: time-correct, as for location matches).

## The judge

- A model outside the Claude family, since Rigour's reviewer runs on
  Claude. The model ID and the prompt's SHA-256 are recorded.
- One call per point: the human comment, its file and line, and both
  entrants' reviews as **A** and **B**, in a seeded random order per
  point. Nothing names the entrant, and words that would (a tool's name, a
  CLAUDE.md check, a "generated with" footer) are redacted. The two
  entrants write differently (prose versus file:line bullets), and a
  judge could tell them apart by format; that limit stays.
- A seeded sample of points (10 by default) is asked again, and the page
  reports how often the judge repeated its verdict.
- Paid, under a hard cap (`--max-usd`), with failures charged at the
  per-call bound, like every paid step.

## What the page shows

Per repository and entrant: same issue (yes) over judged points with a
95% Wilson interval, plus counts of partly, not raised, and points whose
heads the entrant didn't review; labelled "AI judge, non-Claude, blind to
entrant". The verdicts and reasons are in `issue-judgments.yaml`, next to
the run's results. It is an AI judgment, not a human one, and is reported
as such.

```bash
OPENROUTER_API_KEY=... python -m bench issues judge --run work/runs/<name> --results results/<name> \
    --model <non-Claude OpenRouter model ID> --max-usd 2
```

## Diagnosis: held back, or never raised?

For the points one entrant raised and another didn't, a diagnostic run
(docs/SUBSAMPLES.md) re-reviews the heads behind them, keeping what
Rigour's reviewer held back (dropped, unverified, disputed, dismissed).
`bench issues diagnose` shows the same non-Claude judge, per point, every
finding the reviewer wrote on the point's eligible heads, served and held
back mixed and numbered, with nothing saying which list each came from.
It names the findings that raise the point's issue. The point then lands
in one bucket: `served` (raised this time), `held_back:<list>` (considered
and filtered out), or `absent` (never raised). It is a diagnosis, never a
result.

An `absent` point on a head where the reviewer served nothing needs one
more question: did it read the change and find nothing, or stop early?
Each paid record keeps the reviewer's own account of how it ran under
`paid_output.trace` (its record, with judges and lessons served; its
mode, with the passes, their hunks and characters, and reads beyond the
slice; tokens; turn counts where reported), release tarball only.

```bash
OPENROUTER_API_KEY=... python -m bench issues diagnose --run work/runs/<diagnostic> \
    --judgments results/<earlier run>/issue-judgments.yaml --out diagnosis.yaml --model <non-Claude ID> --max-usd 1
```
