# Run 3: an agent alone, or the same agent with Rigour around it?

**Status: design, fixed before the run.** This page states the hypothesis,
the arms, the measures and the targets. The run is reported against them
whatever it shows, including a loss. A change to this page after the run
starts is a new run.

## Hypothesis

Claude Code with Rigour around it (Rigour's deterministic checks, the team's
time-correct lessons, and Rigour's review using Claude Code as its judge)
catches more of the points human reviewers raised than Claude Code alone at a
similar cost, without more noise and without false blocks on approved code.

## Why a control arm

If the Rigour arm is Claude Code's review plus Rigour's, it can never catch
fewer points than Claude Code alone, because a union of two reviews cannot
lose recall. A recall win alone would show only that more model calls find
more. So the headline compares the Rigour arm with Claude Code reviewing
twice at a similar spend, and puts noise and cost beside recall.

## Arms

Every arm reviews the same heads (whole pull requests, a seeded sample;
docs/SUBSAMPLES.md), blind, with the same model, the same pinned Claude Code
CLI, the same read-only tools and the same isolation (docs/ENTRANTS.md).

| Arm | What runs per head | Answer |
|---|---|---|
| 1, agent alone | `claude -p "/code-review"` | its review |
| 1×2, agent twice (control) | arm 1's run, and a second independent `/code-review` | the union of both reviews |
| 2, agent with Rigour | arm 1's run, reused as-is, and Rigour (below) | the union of arm 1's review and Rigour's output |

Arm 2 reuses arm 1's actual answer on each head, not a fresh Claude Code run.
The difference between arm 2 and arm 1 is then exactly what Rigour adds, with
no run-to-run noise from Claude Code.

### What "Rigour around the agent" runs

Three parts, each recorded per head:

- **(a) Deterministic gates:** `rigour review` without the reviewer. Free.
- **(b) The team's lessons:** a store built by Rigour's own learner, with
  only review comments from pull requests merged before the head's pull
  request was opened (docs/LEARNING.md: cutoff, main pinned before it, leak
  assertions). It runs in the shipped default mode, `verified`, with `all`
  reported as a second variant. With core 6.12.4, `verified` served no lesson
  on run 2's sample and `all` served at least one on 42 of 72 heads. The
  pre-check is repeated on the pinned version before the run.
- **(c) Rigour's review, with Claude Code as its judge.** This part plugs
  into one of two modes, and the mode is fixed before the run:
  - **independent:** `rigour review --reviewer --single --blind`, which
    reviews the change on its own;
  - **agent input:** Rigour takes the agent's review (arm 1's answer on the
    head) as input and verifies, extends or rejects its points. This mode
    does not exist yet. When it does, the agent's answer is passed exactly as
    arm 1 produced it, and nothing else about the head changes.

Either way, the instruction hashes (docs/ENTRANTS.md, "Instructions"), the
pinned versions and the mode are recorded in `run.json`.

## Measures

Every judgement is made by the same non-Claude judge, blind A/B with a seeded
order (docs/ISSUES.md).

- **Human points caught.** For each acted-on point, whether the arm's answer
  raises its issue. Reported per arm with Wilson intervals. For each pair of
  arms: points only the first raised, only the second, and both. **The
  headline pair is arm 2 against arm 1×2.**
- **False blocks.** Claude Code has no blocking findings, so arms 1 and 1×2
  show "n/a", not 0. In arm 2, a block from Rigour on an approved head is a
  false block, and each is listed.
- **Unmatched issues (noise).** One more judge question per head and arm:
  list the distinct issues the answer raises, and mark which match a human
  point on that head. Issues that match none are reported as **unmatched**,
  per head. They are not called false, because humans miss things too.
- **Cost per pull request.** For each arm, the sum over the pull request's
  heads of the reported per-head cost (Claude Code's estimate, Rigour's
  `spent_usd`; the gates and the lesson builder cost $0). Each run's
  OpenRouter-billed total is shown beside it. Reported as the mean per pull
  request and per head, and as points caught per dollar.

## Targets (proposed; fixed when this page merges)

Arm 2 supports the hypothesis when, against arm 1×2 on the same heads:

1. it catches more human points, with arm-2-only points outnumbering
   arm-1×2-only points;
2. its mean cost per pull request is at most arm 1×2's plus 10%;
3. its unmatched issues per head are at most arm 1×2's;
4. it has 0 false blocks on approved heads.

Missing any target is reported as a miss on that target, with the numbers.

## Not in scope

- No pull request text or human review reaches any arm; lessons come only
  from before each cutoff.
- No tuning on this sample. The pinned Rigour version, the lesson mode and
  the mode of (c) are fixed before the run.
- Cost estimates and the cap are set before any paid run, as for every paid
  run.
