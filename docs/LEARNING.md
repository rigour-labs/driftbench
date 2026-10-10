# The learning run: does Rigour's reviewer get better with time-correct lessons?

Run 4 reviews every head twice with `rigour-reviewer` (blind, same model):
cold, with no lessons, and learned, with the lessons Rigour's own learner
(`rigour learn-reviews`) takes from review comments that existed before the
head's pull request was opened. The metric is issue-level recall from the
non-Claude judge (docs/ISSUES.md), cold against learned, on the same heads.

This page covers the free steps. The paid arms need a cap like every paid run.

## Cutoff

One cutoff per pull request: its `created_at`. Every head of the pull request
shares one store. Nothing from the pull request itself is ever in it, not even
an earlier round. A blind review treats the pull request's human reviews as a
leak, and so does this run.

## Building a store

`bench learning prepare` lists the repository's merged pull requests once and
fetches the review comments and reviews of every pull request any store needs.
Each store learns from the 100 pull requests merged last before its cutoff,
which is the learner's own default (`--limit 100`). The crawl orders them by
merge time; the learner's live listing orders by last update, which only today
can see.

`bench/learning/stores.mjs` then runs the learner (`learnFromReviews` from
`@rigour-labs/core`, at an exact version) once per cutoff, building a fresh
store each time with no model (no `--rules`). It reads the crawl through an
injected fetch that serves only:

- pull requests merged before the cutoff;
- the comments and reviews on them written before the cutoff.

People comment on merged pull requests too. The learner's own listing keeps
those later comments when given `--until`, which is a Rigour bug, so the bench
filters them itself. The learner also runs git in a full clone, reading the
commits a comment was written on and the outcomes on main before the cutoff
(`git log --before`, by committer date).

The crawl and the stores hold review text. They stay in the Actions cache and
on the runner, never in an artifact or a release.

## Leakage

`bench learning check` checks every piece of evidence in every store against
the crawl:

- it is never from the pull request under review;
- a review point's pull request merged before the cutoff, and its comment or
  review was written before the cutoff;
- an outcome on main (a later fix, a revert) is dated before the cutoff.

A store with any leak has every head marked as an error, never served, and the
workflow fails after uploading the record. Lessons resting on a comment edited
after the cutoff are counted, since only today's text exists.

## Variants

Nobody accepts lessons here, because there is no person in the loop. Both
variants are modes Rigour ships:

- **V, `verified`** (the default): a lesson is served once its point recurs in
  at least 2 pull requests by 2 authors, raised by 2 reviewers or in 2
  wordings. A lesson only review bots raised is never served.
- **A, `all`**: every candidate a person raised.

In real use, people would also promote lessons. Neither variant has that.

## The pre-check

For each head and each variant, the pre-check lists the lessons the reviewer
would be served. It uses `lessonsForDiff` with the judge's limits, read from
the installed reviewer and refused if the reviewer's call has changed. A head
served no lesson gets exactly the cold arm's input, so only heads with at least
one lesson can differ. The paid learned arms run on those heads alone, and
every other head reuses the cold answer, labelled as such.

```bash
python -m bench learning prepare --repo tailscale/tailscale --subsample subsamples/run-2.yaml --out work/learning
node bench/learning/stores.mjs <core dir> work/learning/crawl.json work/repos/tailscale__tailscale \
    work/learning/heads.json work/learning
python -m bench learning check --out work/learning
```

The `learning` workflow runs all three per repo, with Rigour's learner at an
exact version, and uploads `precheck-<repo>.json` (numbers and lesson ids).
