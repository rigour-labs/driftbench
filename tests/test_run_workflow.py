"""The run workflow's own guarantees, read from .github/workflows/run.yml (it can't run in tests)."""
import re
from pathlib import Path

import yaml

WORKFLOW = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github" / "workflows" / "run.yml").read_text())
JOBS = WORKFLOW["jobs"]


def steps(job: str) -> list[dict]:
    return JOBS[job]["steps"]


def scripts() -> list[str]:
    return [step.get("run", "") for job in JOBS.values() for step in job["steps"]]


def test_paid_args_are_only_appended_after_the_smoke_cut():
    """A reassignment after `paid_args=(--max-heads ...)` once dropped the smoke cut (Actions 37960124662)."""
    using = [s for s in scripts() if "paid_args" in s]
    assert len(using) == 2
    for script in using:
        assignments = re.findall(r"paid_args(\+?)=\(", script)
        assert assignments[0] == "" and "--max-heads" in script.split("paid_args=(", 1)[1].split(")")[0]
        assert all(op == "+" for op in assignments[1:]), "paid_args is reassigned after --max-heads"


def test_the_review_job_runs_node_22_and_the_pinned_cli_must_start():
    node = next(s for s in steps("review") if "setup-node" in s.get("uses", ""))
    assert node["with"]["node-version"] == "22"
    install = next(s for s in steps("review") if "claude-code@" in s.get("run", ""))
    assert "claude --version" in install["run"] and "exit 1" in install["run"]


def test_records_are_uploaded_even_when_the_check_fails():
    review = steps("review")
    check = next(i for i, s in enumerate(review) if "bench.release_check" in s.get("run", ""))
    upload = next(i for i, s in enumerate(review) if "upload-artifact" in s.get("uses", ""))
    assert check < upload and review[upload]["if"] == "always()"
    assert "if python -m bench.release_check" in review[check]["run"] and "else" in review[check]["run"]


def test_records_that_failed_the_check_are_kept_three_days_and_passing_ones_ninety():
    review = steps("review")
    check = next(s for s in review if "bench.release_check" in s.get("run", ""))
    upload = next(s for s in review if "upload-artifact" in s.get("uses", ""))
    assert check["id"] == "check" and "passed=true" in check["run"] and "passed=false" in check["run"]
    assert upload["with"]["retention-days"] == "${{ steps.check.outputs.passed == 'true' && 90 || 3 }}"
    notes = next(s for s in steps("draft-release") if "notes.md" in s.get("run", ""))
    assert "3-day retention" in notes["run"]


def test_a_failed_check_marks_the_draft_and_never_fails_the_score_job():
    check = next(s for s in steps("score") if "bench.release_check" in s.get("run", ""))
    assert "if ! python -m bench.release_check" in check["run"] and "check-failed.txt" in check["run"]
    notes = next(s for s in steps("draft-release") if "notes.md" in s.get("run", ""))
    assert "CHECK FAILED: do not publish" in notes["run"]
