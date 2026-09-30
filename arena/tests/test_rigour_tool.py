import json
import subprocess
from pathlib import Path

from arena.tools import rigour


def _fake(monkeypatch, stdout: str):
    monkeypatch.setattr(rigour, "_cli", lambda: ["true"])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout, ""))


def test_a_run_that_did_not_review_is_an_error_not_silence(monkeypatch, tmp_path: Path):
    _fake(monkeypatch, json.dumps({"error": "CONFIG_ERROR", "message": "Config file not found"}))
    run = rigour._run(tmp_path, tmp_path / "pr.diff", rigour.RigourConfig("x"))
    assert run.status == "error" and "CONFIG_ERROR" in run.error and run.findings == []


def test_reads_findings_from_a_completed_review(monkeypatch, tmp_path: Path):
    _fake(monkeypatch, json.dumps({"status": "FAIL", "failures": [{"file": "src/a.ts", "line": 7}, {"file": "", "line": 1}]}))
    run = rigour._run(tmp_path, tmp_path / "pr.diff", rigour.RigourConfig("x"))
    assert run.status == "FAIL" and [(f.path, f.line) for f in run.findings] == [("src/a.ts", 7)]


def test_a_run_that_times_out_is_a_recorded_error_with_a_configurable_limit(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(rigour, "_cli", lambda: ["true"])
    monkeypatch.setenv("ARENA_RIGOUR_TIMEOUT_S", "1800")
    seen = {}

    def slow(*args, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", slow)
    run = rigour._run(tmp_path, tmp_path / "pr.diff", rigour.RigourConfig("x"))
    assert seen["timeout"] == 1800
    assert (run.status, run.error, run.findings) == ("error", "timed out after 1800s", [])
