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
from bench.harness.gitrepo import GitError, RepoCheckout
from bench.harness.types import Adapter, AdapterError, ReviewInput, ReviewOutput
from bench.repos import slug_of

RECORD_SCHEMA = 1


@dataclasses.dataclass(frozen=True)
class RunConfig:
    out_dir: Path      # records: <out>/<tool>/<repo>/<pr>/<head>.json
    scratch_dir: Path  # diffs handed to tools; outside every checkout
    timeout_s: int = 900


def record_path(config: RunConfig, adapter: Adapter, repo: str, pr: int, head: str) -> Path:
    return config.out_dir / adapter.name / slug_of(repo) / str(pr) / f"{head[:12]}.json"


def safe_review(adapter: Adapter, request: ReviewInput) -> tuple[ReviewOutput, float]:
    """Run the adapter; any failure becomes an `error` output and a warning on stderr."""
    started = time.monotonic()
    try:
        output = adapter.review(request)
    except Exception as exc:  # the harness must outlive any adapter failure
        print(f"warning: {adapter.name} failed on {request.head_sha[:12]}: {exc}", file=sys.stderr)
        raw = exc.raw if isinstance(exc, AdapterError) else ""
        output = ReviewOutput(findings=[], verdict="error", raw=raw, error=f"{type(exc).__name__}: {exc}")
    return output, round(time.monotonic() - started, 3)


def prepare(checkout: RepoCheckout, pr: dict, head: str, config: RunConfig) -> ReviewInput:
    """Check out `head` and write the diff from the merge base; GitError if a commit is unavailable."""
    for sha in (head, pr["base_sha"]):
        if not checkout.ensure_commit(sha, pr["number"]):
            raise GitError(f"commit {sha} is not available from the remote")
    base = checkout.merge_base(pr["base_sha"], head)
    checkout.checkout(head)
    diff_path = config.scratch_dir / f"{pr['number']}-{head[:12]}.diff"
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(checkout.diff(base, head), encoding="utf-8")
    return ReviewInput(checkout.path, base, head, diff_path, None, config.timeout_s)


def run_head(adapter: Adapter, checkout: RepoCheckout, pr: dict, head: str, config: RunConfig) -> dict:
    try:
        request = prepare(checkout, pr, head, config)
    except GitError as exc:
        print(f"warning: PR {pr['number']} head {head[:12]} unavailable: {exc}", file=sys.stderr)
        return {"base_sha": None, "changed_lines": None, "verdict": "unavailable", "findings": [],
                "wall_s": None, "error": str(exc), "raw": ""}
    output, wall_s = safe_review(adapter, request)
    return {
        "base_sha": request.base_sha,
        "changed_lines": changed_lines(parse_hunks(request.diff_path.read_text(encoding="utf-8"))),
        "verdict": output.verdict,
        "findings": [dataclasses.asdict(f) for f in output.findings],
        "wall_s": wall_s,
        "cost_usd": output.cost_usd,
        "input_tokens": output.input_tokens,
        "output_tokens": output.output_tokens,
        "error": output.error,
        "raw": output.raw,
    }


def write_record(path: Path, header: dict, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = result.pop("raw", "")
    path.with_suffix(".raw.txt").write_text(raw, encoding="utf-8")
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


def run_corpus(adapter: Adapter, checkout: RepoCheckout, corpus: dict, config: RunConfig) -> dict:
    """Run every case of every PR; returns counts of records written, skipped and by verdict."""
    if adapter.reads_history:
        raise ValueError(f"{adapter.name} reads history, which this harness version doesn't provide yet")
    checkout.ensure_clone()
    counts = {"written": 0, "skipped": 0}
    for pr in corpus["prs"]:
        for head, cases in heads_to_run(cases_for_pr(pr)).items():
            path = record_path(config, adapter, corpus["repo"], pr["number"], head)
            if path.exists():
                counts["skipped"] += 1
                continue
            result = run_head(adapter, checkout, pr, head, config)
            write_record(path, record_header(adapter, corpus["repo"], pr["number"], head, cases), result)
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
    }
