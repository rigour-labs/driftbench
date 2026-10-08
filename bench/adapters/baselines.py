"""Two baselines that bracket every tool (docs/SPEC.md, "Matching a finding to a point").

- `no-tool`: never says anything. Zero catches, zero false blocks.
- `every-hunk`: one non-blocking finding on every changed hunk. It is the
  ceiling a tool could reach by commenting everywhere; a tool close to it
  in catches and volume is reported as noise.
"""
from __future__ import annotations

from bench.harness.diffstat import parse_hunks
from bench.harness.types import Finding, ReviewInput, ReviewOutput


class NoTool:
    name = "no-tool"
    version = "1"
    paid = False
    reads_history = False

    def review(self, request: ReviewInput) -> ReviewOutput:
        return ReviewOutput(findings=[], verdict="pass")


class EveryHunk:
    name = "every-hunk"
    version = "1"
    paid = False
    reads_history = False

    def review(self, request: ReviewInput) -> ReviewOutput:
        hunks = parse_hunks(request.diff_path.read_text(encoding="utf-8"))
        findings = [Finding(h.path, h.first_line, False, "changed hunk", "every-hunk") for h in hunks]
        return ReviewOutput(findings=findings, verdict="pass")
