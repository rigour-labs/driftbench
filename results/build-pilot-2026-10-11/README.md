# Build-track pilot, 2026-10-11 (run 38109978788)

**A pilot, not a result.** It shows the pipeline working end to end and
sizes a full run. It decides nothing about Rigour.

The same agent (Claude Code 2.1.285, anthropic/claude-sonnet-5.5 through
OpenRouter) did 5 real tailscale tasks twice:
- **alone**, with the repository's own instructions;
- **with Rigour 7.0.0-rc.3** set up by its defaults: hooks, the brief, its MCP
  tools.

Each task starts at the pull request's parent commit, from the task as first
written (`tasks/build-pilot.yaml`, docs/BUILD_TRACK.md). Corpus
`corpus-2026-10-09-37884111000`.

**Spend:**
- OpenRouter billed **$5.15** for the agents (key usage over the run).
- The judge (openai/gpt-6.1-sol) reported **$0.48**.
- Total about **$5.63** of a **$10** cap.

The agents' own estimate was $3.01 alone and $3.22 with Rigour, about $0.60
and $0.64 per task.

## What it found

**No measurable difference between the arms.**

| | alone | with Rigour |
|---|---|---|
| Human review points repeated (of those that apply) | 4 of 10 (40%, 95% CI 17–69%) | 4 of 10 (40%, 95% CI 17–69%) |
| Points the diff didn't touch | 4 | 4 |
| Hidden tests: API agreement with the human PR | 1 of 5 pass | 1 of 5 pass |
| Turns, all 5 tasks | 106 | 121 |
| Cost, all 5 tasks (agent estimate) | $3.01 | $3.22 |

- **Paired by point** (applicable in both arms): 2 repeated by alone only, 2 by
  Rigour only, 2 by both, 4 by neither.
- **The judge's verdicts:** 14 of 16 points got a valid verdict. Two answers
  weren't the asked JSON and weren't retried at the time. A retry is in the
  method now.

### What Rigour did during the Rigour arm

- **Activity:** 36 events across the 5 tasks:
  - 25 edit checks, 0 blocked;
  - 5 stop reviews, 0 blocked;
  - 3 MCP calls (`rigour_recall` once, `rigour_check` twice).
- **Blocks:** none, so the agent had nothing to fix after a block.
- **False blocks:** 0, on the 5 approved merged changes (checked before the
  run).
- **Brief:** the hook was installed, but the event log has no brief entry, so
  this pilot can't show whether a brief was served.
- **No team lessons.** Rigour's default (`verified`) mode served 0 lessons:
  on public history, no person promotes a lesson and none recurred.
  `all` mode would have served 95 lessons on 22 of 29 heads; the pilot didn't
  run it.

### Why the hidden tests didn't build on 4 of 5 tasks

`bench build retest` re-ran them on the recorded diffs ([retest.json](retest.json)).
Every "no-build", in both arms, came from the human pull request's tests:

| task | first error (both arms) |
|---|---|
| tailscale#21366 | `undefined: derp.LoanedBytes`, a type the human PR added |
| tailscale#21485 | `c2b.packetsRecv undefined`, `bytesRecv undefined`: fields the human PR added |
| tailscale#21615 | `undefined: HookExitNodeHeartbeatStateLocked`, `HookExitNodeCheckResponsiveness` |
| tailscale#21672 | `TestWriteLinuxDHCPClientSetup redeclared`: the agent's own new test has the same name as the PR's |
| tailscale#21369 | passes in both arms (184 tests) |

The hidden tests encode the human's exact API. An agent making the same
change independently names things differently, so this outcome measures
**API agreement** with the human, not whether the change is right. It is a
design limit, reported as such. tailscale#21672's check is thin anyway: on
the runner, 65 of its 73 hidden tests skip (VM tests).

### The agents wrote files through the shell

In 3 of the 10 agent runs, the agent changed files with no Edit or Write call:
- tailscale#21366 with Rigour;
- tailscale#21615 in both arms.

Claude Code matches an allowed command by its prefix, so a command like
`go build ./... && cat > new.go` ran as `go build`. In those runs, Rigour's
edit hook, which watches Edit and Write, never saw the writes.

The bench now enforces the designed shell in both arms equally, with a hook
that allows one plain command. That guard is a harness control. In real use,
agents do write through the shell, and Rigour 7.0.0 now checks those writes
too (F7).

### Also

`secret_assignment`, Rigour's new check for secrets in plain assignments,
fired **0 times on 180 approved public heads** (325 heads in all). That holds
on 6.14.0-rc.1, 7.0.0-rc.2 and 7.0.0-rc.3, with defaults and with
`security.block`.

## Limits

- 5 tasks and 10 applicable points per arm: far too few to tell the arms
  apart. The intervals above overlap completely.
- One repository, Go only.
- No team lessons: Rigour ran as a brief, hooks and tools, without the
  learning it is built around.
- The hidden-test outcome measures API agreement, not correctness.
- The shell leak let some writes bypass Rigour's edit hook. It is closed now,
  so it doesn't apply to later runs.

## What the next run needs

- **Team lessons in play:** a store with promoted lessons (`all` mode as a
  stated arm, or lessons a person accepts), so the Rigour arm has its main
  input.
- **More tasks:** enough applicable points to see a real difference. At this
  pilot's ~2 applicable points and ~$1.25 per task (both arms), 50 tasks is
  about **$65** plus the judge.
- **Tasks the agent completes:** a primary outcome that doesn't hinge on the
  human's API:
  - the agent's own tests must pass;
  - plus the judge's per-point verdicts.
  The hidden tests are reported apart, as API agreement.
- **The shell guard** in both arms (done), and Rigour checking shell writes
  (7.0.0, F7).

Records: the run's artifact `build-2026-10-11-38109978788` (task records
with diffs, judgments, summary). This folder keeps the numbers:
[summary.json](summary.json) and [retest.json](retest.json).
