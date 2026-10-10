"""Run one adapter over a repository's frozen corpus, one record per reviewed head.

Records are written as they finish and an existing record is never re-run,
so an interrupted run resumes where it stopped. An adapter that raises is
recorded as an error; it never stops the run.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

from bench.harness.cases import ReviewCase, cases_for_pr, heads_to_run
from bench.harness.diffstat import changed_lines, parse_hunks
from bench.harness.budget import Budget
from bench.harness.gitrepo import GitError, RepoCheckout
from bench.harness.paid import paid_gate, settle_review
from bench.harness.publish import PAID_OUTPUT, paid_output, published_finding
from bench.harness.sandbox import env_keys, sandbox
from bench.harness.types import Adapter, ReviewInput, ReviewOutput
from bench.repos import slug_of

RECORD_SCHEMA = 1
SMOKE_MIN_CHANGED_LINES = 20


@dataclasses.dataclass(frozen=True)
class RunConfig:
    out_dir: Path      # records: <out>/<tool>/<repo>/<pr>/<head>.json
    scratch_dir: Path  # diffs and per-run sandboxes; outside every checkout
    npm_cache: Path    # shared by every sandbox so npx stays fast
    timeout_s: int = 900
    run_started_at: str = ""  # harness clock at the start of `bench run`; proves labels came first
    budget: Budget | None = None  # the hard dollar stop shared by every paid entrant
    max_heads: int = 0  # smoke runs only: review this many heads per entrant per repo (0 = every head)
    min_changed_lines: int = SMOKE_MIN_CHANGED_LINES  # smoke runs only: skip heads smaller than this
    only_heads: frozenset[str] | None = None  # an explicit selection: run these heads and no others


def record_path_for(run_dir: Path, tool: str, repo: str, pr: int, head: str) -> Path:
    return run_dir / tool / slug_of(repo) / str(pr) / f"{head[:12]}.json"


def record_path(config: RunConfig, adapter: Adapter, repo: str, pr: int, head: str) -> Path:
    return record_path_for(config.out_dir, adapter.name, repo, pr, head)


def safe_review(adapter: Adapter, request: ReviewInput) -> tuple[ReviewOutput, float]:
    """Run the adapter; any failure becomes an `error` output and a warning on stderr."""
    started = time.monotonic()
    try:
        output = adapter.review(request)
    except Exception as exc:  # the harness must outlive any adapter failure
        print(f"warning: {adapter.name} failed on {request.head_sha[:12]}: {exc}", file=sys.stderr)
        output = ReviewOutput(findings=[], verdict="error", error=f"{type(exc).__name__}: {exc}",
                              cost_usd=getattr(exc, "cost_usd", None))
    return output, round(time.monotonic() - started, 3)


def prepare(checkout: RepoCheckout, pr: dict, head: str, config: RunConfig) -> tuple[str, Path]:
    """Fetch and check out `head`, write the diff from the merge base; GitError if a commit is unavailable."""
    for sha in (head, pr["base_sha"]):
        if not checkout.ensure_commit(sha, pr["number"]):
            raise GitError(f"commit {sha} is not available from the remote")
    base = checkout.merge_base(pr["base_sha"], head)
    checkout.checkout(head)
    diff_path = config.scratch_dir / f"{pr['number']}-{head[:12]}.diff"
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(checkout.diff(base, head), encoding="utf-8")
    return base, diff_path


def unavailable(exc: GitError) -> dict:
    return {"base_sha": None, "changed_lines": None, "verdict": "unavailable", "findings": [],
            "wall_s": None, "error": str(exc)}


def run_head(adapter: Adapter, checkout: RepoCheckout, pr: dict, head: str, config: RunConfig) -> dict:
    """Review one head in a fresh sandbox: minimal env, fresh home, a repo with no refs."""
    try:
        base, diff_path = prepare(checkout, pr, head, config)
        with sandbox(config.scratch_dir, config.npm_cache) as box:
            checkout.isolated_copy(head, box.repo)
            request = ReviewInput(box.repo, base, head, diff_path, None, config.timeout_s, box.env)
            output, wall_s = safe_review(adapter, request)
            output = settle_review(adapter, output, config.budget)
    except GitError as exc:
        print(f"warning: PR {pr['number']} head {head[:12]} unavailable: {exc}", file=sys.stderr)
        return unavailable(exc)
    return {
        "base_sha": base,
        "changed_lines": changed_lines(parse_hunks(diff_path.read_text(encoding="utf-8"))),
        "verdict": output.verdict,
        "findings": [published_finding(dataclasses.asdict(f)) for f in output.findings],
        "wall_s": wall_s,
        "cost_usd": output.cost_usd,
        "input_tokens": output.input_tokens,
        "output_tokens": output.output_tokens,
        "error": output.error[:300],
        "model_runs": output.model_runs,
        "leak_signals": output.leak_signals,
        **({"charged": output.charged} if output.charged else {}),
        **({"diagnostics": output.diagnostics} if output.diagnostics else {}),
        **({PAID_OUTPUT: paid_output(output.review_text, [dataclasses.asdict(f) for f in output.findings],
                                     output.held_back, output.trace)}
           if adapter.paid else {}),
    }


def write_record(path: Path, header: dict, result: dict) -> None:
    """The record only: findings already reduced (bench/harness/publish.py); no raw tool output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**header, **result}, indent=1, sort_keys=True) + "\n", encoding="utf-8")


class RecordError(ValueError):
    pass


def read_record(path: Path) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordError(f"cannot read run record {path}: {exc}") from exc
    if record.get("schema") != RECORD_SCHEMA:
        raise RecordError(f"{path}: expected run record schema {RECORD_SCHEMA}")
    return record


def smoke_eligible(checkout: RepoCheckout, pr: dict, head: str, config: RunConfig) -> bool:
    """A smoke head is a realistic one: its diff has at least `min_changed_lines` changed lines."""
    try:
        _, diff_path = prepare(checkout, pr, head, config)
    except GitError as exc:  # an unavailable head can't be a smoke head; try the next
        print(f"warning: PR {pr['number']} head {head[:12]} unavailable for the smoke cut: {exc}", file=sys.stderr)
        return False
    return changed_lines(parse_hunks(diff_path.read_text(encoding="utf-8"))) >= config.min_changed_lines


def run_corpus(adapter: Adapter, checkout: RepoCheckout, corpus: dict, config: RunConfig) -> dict:
    """Run every case of every PR; returns counts of records written, skipped and by verdict.

    A smoke run (`max_heads`) reviews only the first `max_heads` heads, in corpus order, whose diff has at
    least `min_changed_lines` changed lines; other heads get no record, so a smoke run is never scorable."""
    if adapter.reads_history:
        raise ValueError(f"{adapter.name} reads history, which this harness version doesn't provide yet")
    checkout.ensure_clone()
    counts = {"written": 0, "skipped": 0}
    for pr in corpus["prs"]:
        for head, cases in heads_to_run(cases_for_pr(pr)).items():
            if config.max_heads and counts["written"] + counts["skipped"] >= config.max_heads:
                return counts
            if config.only_heads is not None and head not in config.only_heads:
                continue
            path = record_path(config, adapter, corpus["repo"], pr["number"], head)
            if path.exists():
                counts["skipped"] += 1
                continue
            if config.max_heads and not smoke_eligible(checkout, pr, head, config):
                continue
            result = paid_gate(adapter, config.budget) or run_head(adapter, checkout, pr, head, config)
            header = record_header(adapter, corpus["repo"], pr["number"], head, cases)
            write_record(path, {**header, "run_started_at": config.run_started_at}, result)
            counts["written"] += 1
            counts[result["verdict"]] = counts.get(result["verdict"], 0) + 1
    return counts


def record_header(adapter: Adapter, repo: str, pr: int, head: str, cases: list[ReviewCase]) -> dict:
    return {
        "schema": RECORD_SCHEMA,
        "tool": adapter.name,
        "tool_version": adapter.version,
        "repo": repo,
        "pr": pr,
        "head_sha": head,
        "cases": [dataclasses.asdict(c) for c in cases],
        "env_keys": sorted({*env_keys(), *getattr(adapter, "env_extra", ())}),
        "blocking_semantics": getattr(adapter, "has_blocking", True),
    }
