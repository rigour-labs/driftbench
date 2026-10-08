# Method

This is the method the scripts implement. A change to the method changes this
file in the same pull request, and every published table names the method
version it was produced with.

**Method version: 1 (draft; no run published yet).**

## The question

For pull requests that real maintainers reviewed and merged:

1. **Flagged the same spot, before the human.** For each point a human
   reviewer raised, did the tool flag the same spot, seeing only the code that
   existed when the human raised it (or earlier), and none of that review or
   anything after it?
2. **False blocks on approved code.** On the head the maintainers approved and
   merged, how many blocking findings did the tool raise? A usable gate has
   close to zero.
3. **Cost and time.** Dollars per review when the tool reports them, otherwise
   tokens; wall time per review either way.
4. **By class,** where points carry a confirmed label.

## Corpus

### Repositories

`repos.yaml` lists each repository with its licence and a pinned default-branch
commit and timestamp. Only pull requests merged at or before `pinned_at` are
eligible. Repositories are chosen for real human review history and
open-source licences. One is not permissive (immich, AGPL-3.0). We redistribute
no repository's code, only identifiers (see Storage).

### Pull requests

- Listed with the GitHub API, sorted by `updated` and then by merge date, so long-lived,
  multi-round pull requests aren't crowded out by quick ones.
- Kept only if merged and **substantively reviewed**: at least one review by
  someone other than the author that is `CHANGES_REQUESTED`, has a non-empty
  body, or has inline code comments. Bare approvals don't count. Bot accounts
  don't count as reviewers.

### Rounds

Every review records the commit it was made on (`commit_id`). **Round k** is
the k-th distinct reviewed head. A tool reviewing round k sees:

- the repository at that head, and the diff from the merge base to it;
- if the adapter declares that it reads history: the pull request title and
  body, and review comments posted *before* round k's first review.

It never sees round k's reviews or anything later. This is what makes the
comparison time-correct.

### Review points

- Each inline review comment is one point, anchored to its file and line on
  the commit it was posted on.
- Each review body is split into points on paragraphs and list items. Body
  points have no anchor.
- Replies by the pull request author, and acknowledgements with no request
  ("thanks", "LGTM"), are dropped by a documented rule and kept in the corpus as
  dropped, with the rule that dropped them.
- **Acted on:** a point is acted on if a later commit in the same pull request
  changed one of the lines it is anchored to (±3 lines) before merge.

### Must-not-block cases

The merged head of every pull request in the corpus. The maintainers approved
it, so a blocking finding on it is counted as a false block. Reviewers
sometimes approve code with real problems, so this is an upper bound on the
true false-block rate, not an exact count.

## Labels

Each point carries a class: **mechanical** (style, naming, lint-like),
**performance**, **claim/contract** (code doesn't do what its name, docs,
types or callers promise), **user journey** (behaviour a user sees),
**judgment** (design or taste), or **unclassified**.

The labelling is a reproducible step, and the label files are part of every release:

1. A rules pass (`bench label suggest`) proposes a class from the point text
   and the files it touches. Its rules are in the repository and versioned.
2. A human confirms or corrects each suggestion in `labels/<repo>.yaml`,
   recording the labeller and the guideline version (`docs/LABELLING.md`).
3. Unconfirmed points are reported as `unclassified`, never under a guessed
   class.

Label files hold comment IDs, character spans and classes, never the comment
text. The text is fetched from GitHub when a run needs it.

**Disclosed limit:** the labeller for the first releases is the maintainer of
this repository, who also maintains one entrant (Rigour). Labels are confirmed
without looking at any tool's output, and the label files are published so
anyone can audit or relabel them.

## Matching a finding to a point

The headline metric is a **location match**, named for what it measures:

> **Flagged the same spot**: the tool raised a finding in the same file,
> within N lines of the point's anchor, on the point's head or an earlier
> round's head (line positions carried forward through the diff).

A location match doesn't prove that the tool found the same problem. Three
guards keep it honest:

1. **Noise next to every number.** Every catch rate is published alongside
   *findings per 100 changed lines* and *blocks per approved head*.
2. **An every-hunk baseline.** A pseudo-tool that flags every changed hunk
   sets the ceiling a spraying tool could reach. An entrant whose catch rate
   and finding volume come close to that ceiling is labelled **noise** in the
   tables, whatever its catch rate.
3. **Calibration.** A random sample of location matches (target: 50 per
   release, stratified by entrant) is checked by hand: *same issue: yes /
   partly / no*. The resulting location→issue rate per entrant is published
   alongside the headline. Until a release has a calibration sample, its
   headline is marked **unvalidated**.

**N = 3** is the headline. A sensitivity table repeats the headline at N = 0
and N = 10.

Body points have no anchor, so location matching can't score them. They are
counted and reported as *not location-scorable*. An LLM judge for body points
and semantic matches is a planned **optional, paid** column. It runs only with
explicit approval and a dollar cap, and is never part of the headline.

## Reporting rules

- Per tool, per repository. Pooled numbers only alongside per-repo ones.
- A repository is reported only with **at least 20 acted-on points and at
  least 10 approved heads**. Below that, the table says *insufficient data*
  instead of a number.
- Both denominators are reported: **all substantive points** and **acted-on
  points**.
- Each run records the tool version, settings and flags, and these appear
  with each result.
- Every number is reported, including where any entrant (Rigour included)
  loses.

## Known limits

- **Acted-on misses fixes made elsewhere.** A point fixed by changing other
  lines (a caller, a test, a config) isn't counted as acted on. Reported
  alongside *all substantive points* for this reason.
- **Location isn't meaning.** See Calibration above.
- **Approved isn't correct.** Reviewers miss things, so false blocks are an
  upper bound.
- **Human reviewers are the reference.** A tool can find a real problem no
  human raised. That finding is counted as noise here, not credited.
- **Training contamination.** The pull requests are public; a model may have
  seen them. Rounds and dates are published so this can be studied.
- **Small samples.** The minimums above prevent reporting tiny samples. They
  don't make larger samples representative of all software.

## Storage

- `main`: the method, scripts, `repos.yaml`, label files, and summary tables
  under `results/`.
- GitHub Release `run-YYYY-MM-DD` per run: the frozen corpus (API snapshots),
  raw tool outputs, verdict ledgers, logs and the calibration sample.
- One command re-runs everything from a frozen corpus.
