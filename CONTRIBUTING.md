# Contributing

## Adding a tool (for any vendor or user)

The benchmark is only neutral if adding a tool is easy and nothing about
scoring depends on which tool it is. An entrant is one Python module under
`bench/adapters/` that turns a review request into findings. The harness does
everything else: it checks out the code, controls what history is visible,
times the run and scores the result. Adapters don't see scoring.

### The interface

```python
class Adapter(Protocol):
    name: str            # stable id used in tables, e.g. "acme-review"
    version: str         # exact tool version, pinned (no "latest")
    paid: bool           # True if a run can cost money; excluded from free runs
    reads_history: bool  # True to receive the PR title/body and earlier comments
    env_extra: tuple[str, ...]  # names of variables you add to request.env (a paid tool's own key only)

    def review(self, request: ReviewInput) -> ReviewOutput: ...
```

`ReviewInput` gives you:

| Field | Meaning |
|---|---|
| `workdir` | A fresh repository at the round's head with no refs (no branches, no tags), deleted after the run |
| `base_sha`, `head_sha` | Merge base and the reviewed head |
| `diff_path` | Unified diff `base_sha...head_sha` |
| `history` | Only if `reads_history`: PR title, body, and comments posted before this round |
| `timeout_s` | Hard limit; the harness kills the process after it |
| `env` | The only environment your tool's processes may get: pass it as `env=` to every subprocess. It has a fresh `HOME` and no tokens |

`ReviewOutput` returns:

| Field | Meaning |
|---|---|
| `findings` | List of `{path, line, blocking, message, rule, end_line}`. `line` is on the head side, or `None` if the finding has no line; `end_line` only when one finding covers a range |
| `verdict` | `"fail"` if the tool would block the change, `"pass"` if not, `"error"` if it couldn't review |
| `error` | Why, when `verdict` is `"error"` |
| `cost_usd` | Dollars if the tool reports them, else `None` |
| `input_tokens`, `output_tokens` | If reported, else `None` |
| `model_runs` | Paid tools: model calls made (0 when nothing needed reviewing, so $0 is honest) |
| `leak_signals` | Count of signs your tool saw the PR's human reviews or fetched PR data; any makes the head `leaked` |

**`blocking`** must follow the tool's own semantics: the finding fails the
check, or the tool marks it as must-fix. Don't map severities to "blocking"
just to look stricter or quieter. The mapping is reviewed in the pull request.

### Rules for adapters

- Pin the exact tool version. A version bump is its own pull request.
- Never read anything outside `request`: no network lookups of the pull
  request, its reviews or later commits. The harness already gives you the
  history you are allowed to see.
- Run every tool process with `env=request.env`, never the inherited
  environment. A paid adapter adds only its own key to a copy of it.
- Keep keys in environment variables. Never commit them, never log them.
- Paid adapters must report usage: a review whose model ran without a cost
  is recorded as an error. They must also count leak signals (see
  docs/SPEC.md, "Paid entrants").
- Add a test with a recorded tool output under `tests/fixtures/` that checks
  the parsing, so CI never runs a paid tool.

### Getting in the tables

Open a pull request with the adapter and its test. A maintainer runs the
free entrants on the next scheduled run. Paid entrants run when someone funds
the run. Either way the run records are published, so you can check how your
tool was invoked. Each finding is kept as rule, location and a short message:
the harness strips quoted code and doesn't keep raw output.

## Other contributions

- **Repositories:** add an entry to `repos.yaml` with the licence and a pinned
  SHA. Repositories need public human review history.
- **Labels:** relabelling is welcome. Change `labels/<repo>.yaml` and give the
  guideline rule you followed.
- **Method:** changes go through `docs/SPEC.md` in the same pull request.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest
```

Before pushing, run the name guard. It checks the tree and the outgoing commit
messages against a blocked-name list kept outside the repository, and fails if
the list is missing:

```bash
python -m bench guard --range origin/main..HEAD
```

Pull requests stay small: one stage or one adapter each, with tests. The body
says what the PR does, how it's tested, and what it doesn't do.
