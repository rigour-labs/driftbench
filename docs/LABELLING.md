# Labelling guideline

**Guideline version: 1.** Label files record the version they were made
under. Changing a definition below bumps the version, and labels made under
an older version are reported with that version.

## What gets a label

A **random sample** of 50 points per repository, not every point. The sample
is drawn from **location-scorable** points only (inline comments with a line
on the new side, kept and with a round), because per-class catch rates are
computed over those points and no others:

```bash
python -m bench label sample --repo zulip/zulip --size 50 --seed 20261008
```

The seed, the size and the SHA-256 of the points file are written to
`labels/<owner>__<repo>.sample.yaml` and committed, so anyone can redraw the
same sample and see it wasn't hand-picked. A second draw needs `--replace`,
and the old draw stays in the file's history. The report refuses a sample
whose points file differs from the one the run scored.

Each sampled point gets exactly one class: the main thing the reviewer asked
for. If a point asks for two things, label the one the reviewer spent more
words on; if they are equal, label the earlier one. Per-class results are
computed from the labelled sample only; every other point is unclassified.

## Classes

| Class | The reviewer is saying | Examples (made up) |
|---|---|---|
| **mechanical** | The code works, but its form should change: naming, formatting, typos, unused code, import order, lint-like style. | "nit: `cfg` → `config`"; "typo in the comment"; "this import is unused"; "unused outside tests: no production path passes this flag" |
| **performance** | The code works, but costs too much: time, memory, allocations, calls, network, queries. | "this allocates on every packet"; "O(n²) over all peers"; "cache this lookup"; "this is an N+1: one query per album"; "this scans every row, then filters in Python"; "redundant read: these rows were loaded above" |
| **claim/contract** | The code doesn't do what its name, docs, types, callers or tests promise, or it breaks in a case it should handle. Bugs, races, leaks, error handling, edge cases, compatibility. | "this returns nil on timeout but callers expect an error"; "the doc says inclusive, the loop is exclusive"; "this leaks the file handle"; "this limit is stale: the batch size it was sized for is now 50" |
| **security/privacy** | The code exposes or trusts what it shouldn't: personal data or secrets in logs or responses, missing authorization, injection, unsanitized input. | "this logs the email address of every caller"; "the API token ends up in the error response"; "this builds SQL from request input, an injection risk" |
| **user journey** | What a user sees or does is wrong or worse: UI, messages, flows, defaults, CLI output, accessibility. | "the error toast shows the raw exception"; "this flag now needs a restart, which users won't expect" |
| **judgment** | A design or taste call with no single right answer: structure, abstraction, where code lives, whether to do this at all. | "I'd keep this in the handler, not a new package"; "could we avoid the extra interface?"; "this duplicates the helper in the utils module" |

## Decision order

When a point fits more than one class, the first match wins:

1. **claim/contract** over everything else: if the code is wrong, it's a
   contract problem, even when the reviewer phrases it as style or design.
2. **security/privacy** over user journey and the rest: exposure of data or
   trust in input, even when the code otherwise does what it promises.
3. **user journey** over performance and the rest: what the user sees first.
4. **performance** over mechanical and judgment.
5. **mechanical** over judgment: a rename request is mechanical even when it
   is a matter of taste.
6. **judgment** for what remains.

Two boundaries that come up often:

- **Unused code is mechanical**, including a parameter, default or branch
  that no production path uses and that exists only for tests. It is
  judgment only when the reviewer opens an explicit API design debate.
- **Duplication is judgment** ("extract a helper", "this duplicates X"),
  unless the copies have already diverged into a bug; then it is
  claim/contract.

Questions count by what they ask for: "why not return an error here?" is
claim/contract if the missing error is a bug, judgment if both are valid.

The rules pass (`bench/labels/rules.py`) follows the same order using
keywords, except that a point starting with "nit" is suggested as
mechanical. Keywords miss context, so it only makes a suggestion, and this
guideline wins wherever the two disagree.

## Blind labels and agreement

`bench label show` hides the suggestion by default, so the labeller isn't
anchored to it. Each confirmed label records `blind: true` unless the
labeller passes `--saw-suggestion`. The agreement between labels and
suggestions (`bench label status`) is counted over blind labels only; for
labels made after seeing the suggestion it would partly measure anchoring.

A confirmed entry keeps the suggestion it had when it was confirmed. If the
rules change later, the new suggestion is stored as `suggested_now`, so the
agreement figure doesn't shift after the fact.

## Labels are pinned to the text that was read

A point ID (`<pr>-<kind>-<source id>-<index>`) depends on how text is split.
If the splitting rules or the comment change, the same ID can name different
text. So `set` stores `text_sha256`, the hash of the exact text the labeller
read. A label is used only while the point's current text has that hash;
otherwise it is **stale**, reported as unclassified, and never carried over.
`bench label status` counts stale labels. `stale` includes labels whose point
no longer exists; `dropped_ids` counts those again on their own, so don't add
the two together.

## How to label

```bash
python -m bench label next --repo zulip/zulip --labeller maintainer
```

`next` shows one unlabelled sampled point at a time: its text, a link to the
code at the commit the comment was written on and a link to the pull request.
It never shows the rule suggestion; it writes each sampled point's suggestion
to the separate `labels/<owner>__<repo>.suggest.yaml`, unseen, so blind
agreement can be measured. Answer `1` to `5` for
the class, `s` to skip (the point stays unconfirmed and is marked skipped),
or `q` to quit. Every answer is saved at once, so you can stop and resume
anywhere; `--include-skipped` revisits skipped points.

```bash
python -m bench label status                          # sampled, labelled, skipped, stale, per class
python -m bench label agreement --repo zulip/zulip --a labels --b labels-second   # Cohen's kappa
```

The older commands still work: `suggest` (the rules pass over all points),
`show` (blind by default) and `set` (one point by ID).

## Order: labels are fixed at run start

Labels should exist before any tool output does. `bench run` records, at its
start, the repository's HEAD commit and the git blob hash of every file
under `labels/` (in `run.json`, and in the draft release notes, so the public
copy fixes them). `bench report` publishes per-class results only if each
repository's label and sample files are committed and still have exactly
those hashes; otherwise the page says why they are withheld. This fixes the
labels by content at run start. It doesn't prove when they were written, and
it doesn't rely on dates, which git lets anyone set.

## A second labeller

`--labeller` records who confirmed each label. A second labeller works in
their own labels directory (`--labels labels-second`), and `bench label
agreement` reports Cohen's kappa over the points both labelled.

Rules for the labeller:

- **Label before looking at any tool's output** for that repository. Label
  files are committed before the run that scores them, and the commit order
  shows this.
- Use the point's text and the surrounding pull request on GitHub. Read
  the code if the text alone doesn't say what's wrong.
- When unsure, leave the point unconfirmed. It is reported as `unclassified`;
  a wrong label is worse than none.
- Record who you are with `--labeller`. The first releases are labelled by
  the repository's maintainer, who also maintains one entrant (Rigour); that
  is a disclosed limit (docs/SPEC.md, "Labels").

## Model suggestions

To spare the labeller, a model can suggest a class for each sampled point
first (`bench label prelabel`). The suggestion is never a label: the human
confirms or overrides it, and the label records which.

- **The model** is outside the Claude family, since both paid entrants run
  on Claude. It is reached through OpenRouter, which reports the real cost of
  every call. Its full ID is pinned in the run and written to the
  suggestions file, with the SHA-256 of the prompt.
- **What it reads:** the review comment and the diff hunk it is anchored to,
  plus this guide's classes table, decision order and boundaries. It answers
  one class from the fixed list and a one-line reason; any other answer is
  no suggestion.
- **Honest agreement:** a seeded random 20% of each sample (`blind_ids`,
  redrawn by anyone from the sample's seed) is labelled first, with no
  suggestion shown. Model-human agreement is computed on that blind subset
  only, as a rate from 10 points up. Where a suggestion was on screen, the
  report counts accepted and overridden labels separately and calls them
  anchored, never agreement.
- **Hard cap:** the run needs `--max-usd` and the maintainer's go. Before each
  call it checks that the money spent so far (OpenRouter's reported cost)
  plus a per-call bound still fits; a failed call, or one that reports no
  cost, is charged at the bound. A point that can't be afforded stays
  unsuggested and is listed with the reason, never skipped silently.
- **Calibration stays human:** the calibration sample (location and acted-on
  checks) is never sent to the model and never shown a suggestion.

```bash
OPENROUTER_API_KEY=... python -m bench label prelabel --repo immich-app/immich --repo tailscale/tailscale \
    --repo zulip/zulip --model <full OpenRouter model ID> --max-usd 2
python -m bench label next --repo immich-app/immich --labeller <name>   # blind subset first, then Enter accepts
```

`bench label next` shows the blind subset first. On the other points it shows
the model's class and reason; Enter accepts it, a number overrides it.

## What the label files contain

- `labels/<owner>__<repo>.yaml`: for each labelled point, the class, the
  labeller, `blind`, `suggestion` (accepted or overridden, when a model
  suggestion was shown) and the SHA-256 of the text that was read; skipped
  points are marked `skipped`. No review text and no suggested classes, so
  opening the file can't anchor a labeller.
- `labels/<owner>__<repo>.sample.yaml`: the seed, size, points-file hash and
  sampled IDs.
- `labels/<owner>__<repo>.suggest.yaml`: rule suggestions, kept apart. A
  labelled point keeps the suggestion it had when it was labelled; a later
  rules change only adds `suggested_now`.
- `labels/<owner>__<repo>.model.yaml`: model suggestions, kept apart: the
  model ID, prompt hash, blind subset, reported spend, each point's class,
  reason and cost, and every unsuggested point with why. It is fixed at run
  start with the labels.

Anyone can relabel a point in a pull request by changing its entry and naming
the rule above they followed.
