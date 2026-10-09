"""Synthetic corpus, points and run records for scoring tests. No real data."""
from __future__ import annotations

from pathlib import Path

from bench.harness.runner import write_record
from bench.repos import slug_of


class SameFileVersions:
    """Every file is identical at every commit: lines carry over unchanged."""

    def carry(self, path, line, from_sha, to_sha):
        return line


def pr_record(number: int, approved: str | None = "h2", overridden: bool = False) -> dict:
    return {"number": number, "head_sha": "h2", "approved_head_sha": approved, "approval_overridden": overridden,
            "rounds": [{"index": 1, "head_sha": "h1"}, {"index": 2, "head_sha": "h2"}],
            "reviews": [{"commit_check": "ok"}]}


def inline_point(pid: str, pr: int, rnd: int, line: int, acted: bool | None, start: int | None = None) -> dict:
    return {"id": pid, "pr": pr, "kind": "inline", "round": rnd, "scorable": True, "dropped": None,
            "acted_on": acted, "anchor": {"path": "app.py", "line": line, "start_line": start, "side": "RIGHT",
                                          "commit_sha": f"h{rnd}"}}


def finding(line: int | None, blocking: bool = False, path: str = "app.py") -> dict:
    return {"path": path, "line": line, "blocking": blocking, "message": "m", "rule": "r"}


def write_run(run_dir: Path, tool: str, pr: int, head: str, findings: list, cases: list, verdict: str = "pass",
              changed: int = 100, blocking_semantics: bool = True) -> None:
    header = {"schema": 1, "tool": tool, "tool_version": "1", "repo": "o/r", "pr": pr, "head_sha": head,
              "cases": cases, "env_keys": [], "blocking_semantics": blocking_semantics}
    result = {"base_sha": "b", "changed_lines": changed, "verdict": verdict, "findings": findings, "wall_s": 1.5,
              "cost_usd": None, "input_tokens": None, "output_tokens": None, "error": "", "raw": ""}
    write_record(run_dir / tool / slug_of("o/r") / str(pr) / f"{head}.json", header, result)


def round_case(pr: int, index: int) -> dict:
    return {"kind": "round", "case_id": f"{pr}-round-{index}", "round": index, "source": "",
            "approval_overridden": False}


def mnb_case(pr: int, source: str = "approved", overridden: bool = False) -> dict:
    return {"kind": "must_not_block", "case_id": f"{pr}-must-not-block", "round": None, "source": source,
            "approval_overridden": overridden}
