import subprocess
from pathlib import Path

import pytest

from bench.adapters import PaidSettings, select_adapters
from bench.adapters import rigour
from bench.harness.types import AdapterError, ReviewInput

FIXTURE = Path(__file__).parent / "fixtures" / "rigour-review-fail.json"


def request(tmp_path) -> ReviewInput:
    return ReviewInput(tmp_path, "b" * 40, "h" * 40, tmp_path / "x.diff", None, 5, {"PATH": "/bin", "HOME": "/sandbox"})


def test_rigour_blocking_follows_its_failures_list():
    findings = rigour.parse_report(rigour.load_report(FIXTURE.read_text(), 1))
    blocking = [(f.path, f.line, f.rule) for f in findings if f.blocking]
    assert blocking == [("calc.py", 9, "security-patterns"), ("calc.py", 9, "deprecated-apis")]
    assert all(not f.blocking for f in findings[2:])


def test_rigour_command_is_pinned_and_uses_the_merge_base(tmp_path):
    assert rigour.command(request(tmp_path)) == [
        "npx", "--yes", "@rigour-labs/cli@6.8.1", "review", "--base", "b" * 40, "--json"]


def test_rigour_review_verdicts(tmp_path, monkeypatch):
    def fake_run(args, **kwargs):
        assert kwargs["cwd"] == tmp_path and kwargs["timeout"] == 5
        assert kwargs["env"] == {"PATH": "/bin", "HOME": "/sandbox"}  # only the harness's env
        return subprocess.CompletedProcess(args, 1, FIXTURE.read_text(), "")
    monkeypatch.setattr(rigour.subprocess, "run", fake_run)
    output = rigour.RigourDeterministic().review(request(tmp_path))
    assert output.verdict == "fail" and len([f for f in output.findings if f.blocking]) == 2

    monkeypatch.setattr(rigour.subprocess, "run", lambda args, **k: subprocess.CompletedProcess(args, 2, "oops", "err"))
    with pytest.raises(AdapterError, match="no JSON report"):
        rigour.RigourDeterministic().review(request(tmp_path))

    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 5)
    monkeypatch.setattr(rigour.subprocess, "run", timeout)
    with pytest.raises(AdapterError, match="timed out"):
        rigour.RigourDeterministic().review(request(tmp_path))


def test_select_free_unknown_and_paid():
    assert [a.name for a in select_adapters(["free"])] == ["no-tool", "every-hunk", "rigour"]
    assert [a.name for a in select_adapters(["rigour", "free"])] == ["rigour", "no-tool", "every-hunk"]
    with pytest.raises(ValueError, match="unknown"):
        select_adapters(["nope"])
    assert all(not a.paid for a in select_adapters(["free"]))          # free never includes a paid entrant
    for missing in (None, PaidSettings(model="", max_usd=5), PaidSettings(model="m", max_usd=0)):
        with pytest.raises(ValueError, match="--max-usd"):
            select_adapters(["claude-code-review"], missing)
    paid = select_adapters(["rigour-reviewer", "rigour-reviewer-orchestrated", "claude-code-review"],
                           PaidSettings(model="model-x", max_usd=5))
    assert [a.name for a in paid] == ["rigour-reviewer", "rigour-reviewer-orchestrated", "claude-code-review"]
    assert all(a.paid and a.model == "model-x" and a.env_extra == ("ANTHROPIC_API_KEY",) for a in paid)
