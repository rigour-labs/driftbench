# Review Arena

Scores code-review tools on the same real pull requests, with ground truth
that no reviewer influenced.

## Why not "the comment was acted on"?

On historical PRs developers only saw the bot's comments, so they edited the
lines it flagged. A correct finding from another tool on a line nobody
touched would count as wrong. That signal is kept (`acted_on`, bot only) but
it is not the headline.

## Ground truth: bugs fixed after merge (SZZ)

For every later first-parent commit whose subject marks a defect fix, the
lines it removes or replaces are blamed just before the fix. Lines written
by the PR mean the PR introduced a bug that was fixed later. No reviewer
caused those fixes, so every tool is judged the same way. `arena/szz.py`
documents the filters (defect keywords, non-defect keywords, fix size cap,
trivial lines).

## Metrics (`arena/score.py`)

Every location is moved onto the PR's merge commit. Each metric has a 95%
bootstrap interval over PRs.

| Metric | Meaning |
|:---|:---|
| recall | share of later-fixed bugs a tool pointed at (±3 lines, same file) |
| hit rate | share of a tool's comments on a later-fixed bug, matched one-to-one; a **lower bound** on precision |
| precision | from hand labels on a sampled set of comments, same rules for every tool |
| comments/PR | noise |

Scopes: `code` (non-test source files, the default) and `all`.

## Corpus rules

- A PR counts only if the bot **submitted a review** on it. A "review skipped"
  notice is not a review.
- Only PRs into the default branch, merged at least `--min-age-days` (default 60) before the snapshot, so fixes have had time to land.
- Stored: PR numbers, commits, comment locations, category and severity, and links. Comment text is **not** stored.

## Run

```bash
export GH_TOKEN=...                 # read-only access to public repos
export RIGOUR_CLI=/path/to/rigour/packages/rigour-cli/dist/cli.js   # or RIGOUR_VERSION for npx
python -m arena mine  --repo TanStack/router --limit 300
python -m arena label --repo TanStack/router
python -m arena run   --repo TanStack/router --tool coderabbit
python -m arena run   --repo TanStack/router --tool rigour-semantic
python -m arena score --repo TanStack/router --scope code
python -m pytest arena/tests
```

Clones live in `~/.cache/driftbench/arena` (`ARENA_CACHE`). Rigour runs in
throwaway worktrees, with `HOME` set to an arena directory.

## Known limits

- SZZ misses bugs whose fix did not say "fix", and credits some refactors as bugs. Hand-checked samples measure this.
- A tool's comment about a real bug that nobody fixed later lowers its hit rate. That's why hit rate is a lower bound and precision comes from hand labels.
- Bot comments are made on an earlier PR commit and moved to the merge commit. A comment on lines rewritten later lands on the rewritten hunk.
