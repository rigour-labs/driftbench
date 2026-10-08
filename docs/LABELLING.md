# Labelling guideline

**Guideline version: 1.** Label files record the version they were made
under. Changing a definition below bumps the version, and labels made under
an older version are reported with that version.

## What gets a label

Every kept review point (not dropped by a rule in docs/SPEC.md, "Review
points"). Each point gets exactly one class: the main thing the reviewer asked
for. If a point asks for two things, label the one the reviewer spent more
words on; if they are equal, label the earlier one.

## Classes

| Class | The reviewer is saying | Examples (made up) |
|---|---|---|
| **mechanical** | The code works, but its form should change: naming, formatting, typos, unused code, import order, lint-like style. | "nit: `cfg` → `config`"; "typo in the comment"; "this import is unused" |
| **performance** | The code works, but costs too much: time, memory, allocations, calls, network. | "this allocates on every packet"; "O(n²) over all peers"; "cache this lookup" |
| **claim/contract** | The code doesn't do what its name, docs, types, callers or tests promise, or it breaks in a case it should handle. Bugs, races, leaks, error handling, edge cases, compatibility. | "this returns nil on timeout but callers expect an error"; "the doc says inclusive, the loop is exclusive"; "this leaks the file handle" |
| **user journey** | What a user sees or does is wrong or worse: UI, messages, flows, defaults, CLI output, accessibility. | "the error toast shows the raw exception"; "this flag now needs a restart, which users won't expect" |
| **judgment** | A design or taste call with no single right answer: structure, abstraction, where code lives, whether to do this at all. | "I'd keep this in the handler, not a new package"; "could we avoid the extra interface?" |

## Decision order

When a point fits more than one class, the first match wins:

1. **claim/contract** over everything else: if the code is wrong, it's a
   contract problem, even when the reviewer phrases it as style or design.
2. **user journey** over performance and the rest: what the user sees first.
3. **performance** over mechanical and judgment.
4. **mechanical** over judgment: a rename request is mechanical even when it
   is a matter of taste.
5. **judgment** for what remains.

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
`bench label status` counts stale labels.

## How to label

```bash
python -m bench label suggest                        # rules pass; never overwrites a confirmed label
python -m bench label show --repo zulip/zulip         # unconfirmed points with their text, locally (blind)
python -m bench label set --repo zulip/zulip 123-inline-456-0 claim/contract --labeller maintainer
python -m bench label status                         # confirmed, stale, per class, blind agreement
```

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

## What the label files contain

`labels/<owner>__<repo>.yaml`: for each point ID, the suggested class, the
confirmed class, the labeller, `blind`, and the SHA-256 of the text that was
read. No review text. Anyone can relabel a point
in a pull request by changing its entry and naming the rule above they
followed.
