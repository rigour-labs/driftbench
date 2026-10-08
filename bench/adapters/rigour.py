"""Rigour, deterministic mode: `rigour review --base <merge base> --json`, no model, no key.

Blocking follows Rigour's own verdict: `status: FAIL` (exit 1) exactly when
its `failures` list (findings on changed lines) is non-empty, so each of
those is blocking. Advisory, file-level and context findings are reported
as non-blocking. `--base` is used rather than `--diff`: on the same change,
6.8.1 reported findings with `--base` and none with `--diff`.
"""
from __future__ import annotations

import json
import subprocess

from bench.harness.types import AdapterError, Finding, ReviewInput, ReviewOutput

VERSION = "6.8.1"
NON_BLOCKING_LISTS = ("advisory", "file_findings", "context_findings")


def command(request: ReviewInput) -> list[str]:
    return ["npx", "--yes", f"@rigour-labs/cli@{VERSION}", "review", "--base", request.base_sha, "--json"]


def to_finding(item: dict, blocking: bool) -> Finding:
    rule = str(item.get("id") or item.get("rule") or "")
    return Finding(item.get("file"), item.get("line"), blocking, str(item.get("message") or item.get("reason") or ""), rule)


def load_report(stdout: str, returncode: int) -> dict:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"exit {returncode}: no JSON report", raw=stdout) from exc


def parse_report(report: dict) -> list[Finding]:
    findings = [to_finding(item, True) for item in report.get("failures") or []]
    for key in NON_BLOCKING_LISTS:
        findings += [to_finding(item, False) for item in report.get(key) or [] if isinstance(item, dict)]
    return findings


class RigourDeterministic:
    name = "rigour"
    version = VERSION
    paid = False
    reads_history = False

    def review(self, request: ReviewInput) -> ReviewOutput:
        try:
            result = subprocess.run(command(request), cwd=request.workdir, capture_output=True, text=True,
                                    timeout=request.timeout_s, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AdapterError(f"timed out after {request.timeout_s}s") from exc
        report = load_report(result.stdout, result.returncode)
        verdict = "fail" if report.get("status") == "FAIL" else "pass"
        return ReviewOutput(findings=parse_report(report), verdict=verdict, raw=result.stdout)
