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

- Closed pull requests are listed most recently updated first (the API's only
  useful order), up to `--max-listed`. Those merged at or before the pin are
  kept, then sorted by updated date and then merge date. `merged_candidates`
  in the corpus records how many survived the filter. With an old pin, raise
  `--max-listed` or report the survivor count.
- The listing order changes as pull requests get new activity, so selection
  is reproducible from the frozen corpus, not by re-listing.
- Kept only if merged and **substantively reviewed**: before the merge,
  someone other than the author either
  - submitted a review that is `CHANGES_REQUESTED`, or whose body or inline
    comments contain a review point; or
  - posted a conversation-tab comment that is a review point.
- A **review point** is a text that is neither an acknowledgement (ACK-1) nor
  a command (CMD-1). Bare approvals don't count. Bot accounts don't count as
  reviewers.
- **Rule ACK-1 (acknowledgement):** at most six words, every one of them from
  a fixed list ("LGTM", "Thank you!", "Looks good to me, thanks"). ACK-1
  assumes English-script reviews: a review with no Latin letters counts as an
  acknowledgement and is dropped.
- **Rule CMD-1 (command):** the first non-blank line starts with `/`
  ("/retest") or with a mention of a bot account ("@dependabot rebase").
- Both rules are in `bench/collect/text_rules.py`.
- Every human, non-author, pre-merge review is recorded, substantive or not,
  with a `substantive` flag. Only substantive reviews define rounds. Bare
  approvals are kept so that the approved head is known.

### Rounds

Every review records the commit it was made on (`commit_id`), checked by
**rule COMMIT-1**. GitHub can report a review's commit as one created after
the review was submitted: an approval appears to move forward onto a later
force-pushed head. A commit can't be reviewed before it exists, so the
reported commit is trusted only if its committer date is at or before the
review. Otherwise the commit that review's own inline comments were written
on is used, if it predates the review. Otherwise the review is marked
`untrusted`: it stays in the record but defines no round and is not an
approved head. In a 10-PR sample of tailscale/tailscale, 3 of 26 human
reviews (all approvals) needed this rule.

COMMIT-1 is one-sided. A committer date before the review shows the commit
*could* have been reviewed, not that it had been pushed: a commit made at
10:00 and pushed at 15:00 passes for a 12:00 review. Committer clocks can
also be wrong, so a few `untrusted` reviews may be genuine. The untrusted
count is reported per repository next to the headline.

**Round k** is the k-th distinct trusted head that received a substantive
review. A tool reviewing
round k sees:

- the repository at that head, and the diff from the merge base to it;
- if the adapter declares that it reads history: the pull request title and
  body, and review comments posted *before* round k's first review.

It never sees round k's reviews or anything later. This is what makes the
comparison time-correct.

**Conversation comments** carry no commit, so each one is tied to the head the
pull request had when it was written. The **head history** comes from the
issue timeline:
- force-push events give the new head and the push time (`push`, exact);
- `committed` events give the committer date, which can be earlier than the
  push (`commit_date`, approximate);
- if the timeline has neither, the commit list's committer dates are used
  (`commit_date`);
- the timeline only lists commits still in the pull request, so heads that
  were rebased away are missing; every review with a trusted commit adds
  that commit at the review's time (`review`).

The comment's round is the round reviewed on that head. If no substantive
review was made on that head, the point is counted in the corpus but not
scored: there is no reviewed head to compare a tool on. That count is
reported. Points are reported by head source (`push`, `review`,
`commit_date`); `commit_date` is approximate.

### Running a tool

`bench run` gives every entrant the same input for every distinct reviewed
head (each round's head and the must-not-block head; a head shared by two
cases is reviewed once):

- a fresh repository at that head that borrows the clone's objects but has
  **no refs**: no `origin/main`, no other branch, no tag. A tool can't
  reach commits made after the head through any ref (they exist only as
  unreferenced objects). It is deleted after the run, so nothing a tool
  writes (caches, its own state) reaches the next run;
- a **minimal environment** built by the harness, never inherited: `PATH`,
  a fresh `HOME` (so no tool reads the machine's settings, stored answers
  or profiles), telemetry off (`RIGOUR_TELEMETRY=0`, `DO_NOT_TRACK=1`), a
  shared package cache, and no tokens or API keys. A paid adapter adds only
  its own key. The variable names (not values) are in every run record;
- the diff from the merge base of the pull request's recorded base commit
  and the head;
- a time limit (default 900 s).

No free entrant reads history, and this version gives history to no
adapter; an adapter that declares it reads history is refused until a
time-correct history builder exists. A head whose commit GitHub no longer
serves is recorded as `unavailable` and reported, not dropped. The two
baselines: `no-tool` (no findings) and `every-hunk` (one non-blocking
finding on the first added line of every changed hunk).

### Review points

`bench points` turns each frozen record into points. A point holds IDs, a
character span and an anchor, never text. Text is fetched by ID and must match
the frozen hash, or the point is dropped (TEXT-1). On the collector's own warm
cache TEXT-1 can't fire, since the text comes from the same cached lists; it
protects a re-run by someone else, without that cache.

- Each inline review comment is one point, anchored to its file and line on
  the commit it was written on.
- Each review body and each conversation comment is split into points
  (SPLIT-1): paragraphs and list items. Fenced code stays with its paragraph.
  These points have no anchor.

Dropped points stay in the output, marked with the rule that dropped them:

| Rule | Drops |
|---|---|
| AUTHOR-1 | comments by the pull request author |
| THREAD-1 | inline replies, including other reviewers' replies: a thread is one point, its first comment |
| ACK-1, CMD-1 | acknowledgements and commands (see Pull requests) |
| QUOTE-1 | paragraphs that only quote earlier text (`>`) |
| TEXT-1 | text deleted, or edited since the freeze |

A point is **scorable** when it isn't dropped and has a round. Kept points
with no round (a conversation comment on an unreviewed head) are counted and
reported, but not scored.

**Acted on (ACTED-1)**, for inline points on the new side of the diff: yes if
the code within 3 lines of the anchor changed between the anchor commit and
the merged head, or the file was removed. The basis is recorded:
- `ancestor`: the anchor commit is an ancestor of the merged head, so the
  compare patch is exactly the later change (when GitHub omits the file's
  patch, or truncates the file list at 300, the file at the two commits is
  diffed instead);
- `direct`: the branch was amended or force-pushed in place (it gained no
  upstream commits), so the file at the two commits is diffed directly;
- `rebased`: the merged head gained upstream commits, by a rebase onto a
  newer base or by merging the base branch in; a direct diff would mix in
  upstream changes, so acted-on is unknown. The test is the compare's
  `ahead_by` against the pull request's own commit count.

In a 10-PR tailscale sample, 26 of 35 kept inline points got an answer (24
yes); 8 were `rebased` and 1 was on the old side.

### Must-not-block cases

One per pull request: the **approved head**, the trusted commit (COMMIT-1)
of the last approval before the merge (`approved_head_sha`). It often differs
from the merged head, because commits can land after the approval. If a pull
request has no trusted approval, the merged head is used instead, and these
cases are counted separately.

If a reviewer requested changes (on a trusted commit) after that approval,
the record sets `approval_overridden`. The later points may apply to the
approved head, so these cases are left out of the false-block denominator
and their count is reported.

Reviewers sometimes approve code with real problems, so this is an upper bound
on the true false-block rate, not an exact count. In particular, an approval
that came with comments can sit on a head that is also a round head with
review points; a blocking finding there still counts as a false block,
because the human approved that head anyway.

## Labels

Each point carries a class: **mechanical** (style, naming, lint-like),
**performance**, **claim/contract** (code doesn't do what its name, docs,
types or callers promise), **user journey** (behaviour a user sees),
**judgment** (design or taste), or **unclassified**.

The labelling is a reproducible step, and the label files are part of every release:

1. A rules pass (`bench label suggest`) proposes a class from the point text
   by keyword (`bench/labels/rules.py`, versioned). It is only a suggestion.
2. A human confirms or corrects each suggestion in `labels/<repo>.yaml`,
   recording the labeller and the guideline version (`docs/LABELLING.md`).
3. Unconfirmed points are reported as `unclassified`, never under a guessed
   class.

Label files hold point IDs, the suggested class, the confirmed class, the
labeller, whether the label was made blind to the suggestion, and the
SHA-256 of the text the labeller read, never the text itself. A label is used
only while the point's text still has that hash; otherwise it is stale and
the point counts as unclassified. The share of blind labels that match the
suggestion is reported.

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
2. **An every-hunk baseline.** A pseudo-tool that flags every changed hunk,
   each finding spanning the hunk's added lines, sets the ceiling a spraying
   tool could reach. An entrant whose catch rate
   and finding volume come close to that ceiling is labelled **noise** in the
   tables, whatever its catch rate.
3. **Calibration.** A random sample of location matches (target: 50 per
   release, split evenly across entrants, baselines excluded) is checked by
   hand: *same issue: yes / partly / no*. The resulting location→issue rate
   per entrant is published alongside the headline. The same sample checks
   acted-on decisions (10 `direct` yes, 5 `direct` no, 5 `ancestor`): *did
   the change near the anchor respond to the point: yes / no*, with
   agreement reported per basis. If `direct` agrees clearly less often than
   `ancestor`, acted-on numbers are reported per basis. The sample is drawn
   with a recorded seed (`bench calibrate draw`), never redrawn silently,
   and committed with its verdicts. It draws only from reportable
   repositories. When a pool is smaller than its target, the shortfall
   ("short: 31 of 50") is recorded and printed with the status. Until every
   entry has a verdict, the page opens with the headline marked
   **unvalidated**.

**N = 3** is the headline. A sensitivity table repeats the headline at N = 0
and N = 10.

How a match is decided (`bench score`):
- Only location-scorable points count: kept, with a round, inline, with a
  line on the new side.
- A tool may use its findings on the head of any round from 1 up to the
  point's own round; never a later one.
- A finding in the point's file is carried from its head to the commit the
  point was written on. The file at the two commits is diffed: an unchanged
  line keeps its place, and a line in a changed region maps to where that
  region starts. Its distance to the anchored line (or range) is measured,
  and the closest finding decides. A finding that covers a range of lines
  (every-hunk spans each hunk's added lines) is at distance 0 when the
  ranges overlap. A file that can't be read at either commit gives no match.
- Catch rates are reported twice, with the same windows and denominators:
  for **any finding**, and for **blocking findings only** ("would it have
  stopped this"). An advisory note at the spot is weaker than a block
  there.
- A head the tool failed on (`error`) or couldn't get (`unavailable`) gives
  it no findings, so it counts as a miss. The counts of `error` and
  `unavailable` heads are reported per entrant next to its scores, so
  failures can't pass for silence.
- Every decision, with the closest finding and its distance, is written to
  the match ledger (a release asset).

**Rule NOISE-1:** an entrant is labelled **noise** when its N = 3 catch rate is
at least 75% of every-hunk's and its findings per 100 changed lines are at
least 50% of every-hunk's.

Body points have no anchor, so location matching can't score them. They are
counted and reported as *not location-scorable*. An LLM judge for body points
and semantic matches is a planned **optional, paid** column. It runs only with
explicit approval and a dollar cap, and is never part of the headline.

## Reporting rules

- Per tool, per repository. Pooled numbers only alongside per-repo ones.
- A repository is reported only with **at least 20 acted-on points and at
  least 10 approved heads** (trusted approvals, not overridden; counted from
  the corpus). Below that, the page says *insufficient data*, the summary
  committed in `results/` carries only the counts and the reason, the
  repository is left out of the calibration sample, and its full metrics stay
  only in the run's release assets.
- Ratios (blocks per approved head, findings per 100 changed lines) are
  printed with their n. False blocks are also broken down: merged-head
  fallbacks, overridden approvals left out, and heads not scored.
- False blocks count must-not-block heads with at least one blocking
  finding, over approved heads. Heads that fell back to the merged head are
  reported separately; overridden approvals and heads the tool failed on are
  counted and left out.
- Both denominators are reported: **all substantive points** and **acted-on
  points**.
- Every rate is printed with its n and a 95% Wilson interval; no rate is
  printed without its n. With 20 to 60 points per repository, the
  intervals are wide.
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

- `main`: the method, scripts, `repos.yaml`, label files, and under
  `results/<date>/` the published summaries, `summary.md` and the calibration
  sample with its verdicts (added by pull request after each run).
- GitHub Release `run-YYYY-MM-DD` per run: the frozen corpus, raw tool
  outputs, verdict ledgers, logs and the calibration sample.
- The frozen corpus holds IDs, SHAs, anchors and timestamps, and a SHA-256
  and length for each review or comment text, never the text itself. A run
  fetches text by ID and checks it against the hash. Text edited or deleted
  since the freeze is reported, not silently re-scored.
- The raw API cache (`work/cache/`) contains text and is never published.
- One command re-runs everything from a frozen corpus.
