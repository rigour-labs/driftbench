# The build track: does the same agent write better code with Rigour around it?

**Status: design, fixed before the pilot.** This page states the question,
the tasks, the arms, what each arm can see, the measures and the pilot. The
pilot is reported against it whatever it shows. A change to this page after a
paid run starts is a new run.

## Question

Given a real task from a merged pull request, does the same coding agent with
the same model produce a change that human reviewers would have had less to
say about, and that passes the pull request's own tests, when Rigour is set
up around it (a brief before it writes, checks while it writes, the team's
lessons), compared with the agent alone? And at what cost?

## Tasks

A task is a merged pull request from the frozen corpus that:

- changed at least one test file, which is used as the hidden check;
- had at least one acted-on human review point (docs/LABELLING.md), which
  is what "less to say" is measured against;
- has a task statement:
  - the linked issue's text as it stood when the pull request was opened;
  - else the pull request's description as first written, from its edit
    history (`userContentEdits`), never a later edit, which can describe the
    review's outcome.

Statements that quote the diff (a `diff` block, or three or more diff-like
lines) or are under 80 characters are excluded, and each exclusion is listed.
A pull request that changed more files than a run's limit is left out, so a
task fits the per-task bound.

`bench build tasks` writes every eligible pull request in a seeded order
(`tasks/*.yaml`: IDs, SHAs, test paths and the statement's hash, never its
text). A run takes the first `count` in that order whose hidden tests
discriminate: on the runner, before any agent spends, they must fail or not
build at the parent and pass on the merged change. A test that skips or
cannot run there checks nothing, and its task is passed over and listed.

The agent starts at the pull request's **parent commit**: the merge base of
its first head with its base branch. Starting there gives it the code as it
stood before the author began, with the history truncated so that no later
commit, ref or branch exists in its checkout.

## Arms

Both arms run the same pinned Claude Code CLI and model, through the same
provider, with the same agent tools: edit, write, read, search, and shell
limited to the repository's own build and test commands. Both have the same
isolation (no web, no GitHub, no network beyond a pre-filled dependency
cache) and the same per-task turn limit, timeout and dollar bound.

| Arm | Setup in the task's checkout |
|---|---|
| A, agent alone | the repository as it was at the parent commit, including its own `CLAUDE.md` / `AGENTS.md` (what a team already has) |
| B, agent with Rigour | the same files, plus exactly what `rigour setup` installs by default: hooks, the brief, and Rigour's MCP server; plus the team's lessons |

**The one intended difference** is Rigour itself:

- its hooks, which Claude Code runs on its own, outside the agent's tool
  permissions;
- its MCP tools (brief, recall, remember, context scope), which arm B's
  agent has as extra tools;
- its lessons.

Arm A has no MCP servers. Everything else is identical.

In arm B, Rigour works as follows:

- **Version:** pinned to an exact release that routes Claude Code's hooks
  correctly. The pre-tool data-loss check and the first-edit brief must run
  as Claude Code hooks; an earlier release candidate sent them to another
  tool's mode, where they passed everything.
- **Configuration:** its defaults only, with no tuning on these tasks. The
  installed hooks and `.mcp.json` are recorded per task.
- **Lessons:** time-correct, built by the run-4 machinery
  (docs/LEARNING.md) from pull requests merged before the task's pull
  request was opened. The cutoff is the same, main is pinned before it,
  and the leak assertions are the same. The store is served in the shipped
  default mode, `verified`.
- **Brief:** as installed by default, with the task statement as the goal.
- **Hooks:** run as installed. Every check they ran, everything they
  flagged, and whether the agent changed the flagged code afterwards are
  recorded from Rigour's own event log.
- **Semantic search:** installed from a pre-filled cache. It is never
  downloaded during a task, and if it cannot be used, the run records that.

**Cost:** every model call Rigour makes during a task counts toward arm B's
cost and its per-task bound, which is the same bound as arm A's.

## Measures

Every judgement is made by the same non-Claude judge, blind, in a seeded
order (docs/ISSUES.md).

1. **Human points repeated.** For each acted-on human review point on the
   real pull request, the judge sees the point and the agent's final diff
   and answers one question: does this diff have the problem the reviewer
   pointed out? Reported per arm as points repeated out of points that
   apply. A point about code the agent never wrote is "not applicable" and
   counted apart. **Lower is better.** The judge sees both arms' diffs for
   the point as A and B in a seeded order, nothing naming an arm, and
   answers repeated, avoided or not_applicable for each; a diff over 30,000
   characters is cut, and the cut is recorded (`bench build judge`, then
   `bench build report`).
2. **Tests.** The pull request's own changed test files, hidden from the
   agent, are applied to the agent's version, and the repository's test
   command runs on them. The result is pass, fail or does not build. The
   real merged pull request runs the same way as the reference.
3. **What Rigour caught during the work (arm B).** The checks that flagged
   something, how many flags the agent then fixed, and how many it left.
4. **False blocks (arm B).** Rigour's hooks and gates also run, unchanged,
   on the real merged pull request's change at the same parent. Anything
   they block there is a false block, because maintainers approved and
   merged that code. Each one is listed.
5. **Cost and time per task.** The reported cost (Claude Code's estimate,
   plus Rigour's `spent_usd`), each run's OpenRouter-billed total, turns and
   wall time.

Arm A and arm B are compared on the same tasks, paired: tasks where only B
repeated fewer points, only A did, or neither; and tasks where the tests pass
in one arm only.

## Leakage

- **The agent never sees:** the pull request's diff, its later commits, its
  review comments, its tests, or anything dated after the parent commit.
  The checkout has no remote and no refs past the parent; web and GitHub
  tools are denied, and a transcript showing any such call marks the task
  leaked and unscored.
- **The task statement:** its edit history is checked, and the version
  dated before the first review is used.
- **Lessons:** they come only from before the cutoff, and every store passes
  the leak assertions.
- **The judge:** it sees the agent's diff and the human point, never which
  arm produced the diff.

## Before the pilot: do Rigour's hooks see the agent's work?

The agent cannot commit, so its work stays uncommitted. A hook that checks
only commits would check nothing in arm B, and the run would wrongly show
Rigour adding nothing. So before any agent spends, `bench build hookcheck`
runs on one task's snapshot with the pinned Rigour set up by default. It
plants a proven issue (a fake AWS key pair in AWS's format, made fresh each time), uncommitted, then runs the
installed Stop hook and edit hook exactly as Claude Code does: same commands,
same payloads on stdin, `CLAUDE_PROJECT_DIR` set. The pilot runs only if both
hooks block, and the result is kept in the run record.

## Pilot

- **Scope:** 5 tasks, from one repository whose tests build and run on the
  standard runner (tailscale, Go), drawn by a seeded rule. Both arms run on
  every task, and the whole pilot is capped at $10.
- **Purpose:** to show that everything works end to end before the real run:
  - the sandbox, the hidden tests and the event log;
  - the leak checks;
  - the judge;
  - cost per task.
  The pilot is never scored as a result.
- **Afterwards:** the full run's estimate (tasks, repositories, dollars)
  comes from the pilot's measured cost per task, and is fixed before its cap
  is asked for.
