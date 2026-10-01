import json
import subprocess
import sys
from pathlib import Path

import pytest

from kaggle_arena import build, report

ENV = {"KAGGLE_USERNAME": "rigourlabs", "MODELS": "qwen-coder-1.5b", "PER_REPO": "1",
       "REPOS_OVERRIDE": "trpc/trpc", "RIGOUR_REF": "main", "DRIFTBENCH_REF": "fix/x"}


def test_the_kernel_carries_its_parameters_and_runs_as_a_private_gpu_script(tmp_path: Path):
    out = build.build(tmp_path / "kernel", ENV)
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert (meta["id"], meta["enable_gpu"], meta["enable_internet"], meta["is_private"]) == ("rigourlabs/rigour-arena-gpu", True, True, True)
    plan = subprocess.run([sys.executable, str(out / "run.py")], capture_output=True, text=True, check=True).stdout
    params = json.loads(plan.split("plan:", 1)[1])
    assert [m["name"] for m in params["models"]] == ["qwen-coder-1.5b"]
    assert (params["repos"], params["per_repo"], params["driftbench_ref"]) == (["trpc/trpc"], 1, "fix/x")


def test_an_unknown_model_name_is_refused():
    with pytest.raises(SystemExit, match="unknown model"):
        build.params({**ENV, "MODELS": "gpt-9"})


def test_the_report_shows_pre_emption_and_the_funnel_per_model(tmp_path: Path):
    entry = {"status": "FAIL", "seconds": 120, "commit": "c",
             "targets": [{"id": "1", "path": "a.ts", "start": 10, "end": 10, "severity": "🟠 Major"}],
             "findings": [{"path": "a.ts", "line": 11, "id": "r", "message": "m"}],
             "deep": {"findings_proposed": 3, "findings_withdrawn": 2, "findings_count": 1, "chunks_failed": 0}}
    target = tmp_path / "qwen-coder-7b" / "trpc__trpc" / "pre-pr"
    target.mkdir(parents=True)
    (target / "kaggle-qwen-coder-7b.json").write_text(json.dumps({"prs": {"1": entry, "2": {**entry, "status": "error", "deep": None}}}))
    [row] = report.summarize(tmp_path)
    assert (row["model"], row["prs"], row["errors"], row["targets"], row["raised"], row["minutes_per_pr"]) == ("qwen-coder-7b", 2, 1, 1, 1, 2.0)
    assert (row["findings_proposed"], row["findings_withdrawn"], row["findings_count"]) == (3, 2, 1)


def test_the_newest_release_of_the_node_line_is_chosen(monkeypatch):
    import io
    from kaggle_arena import run
    index = json.dumps([{"version": "v26.1.0"}, {"version": "v24.21.0"}, {"version": "v24.20.0"}, {"version": "v22.23.3"}])
    monkeypatch.setattr(run.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(index.encode()))
    assert run.latest_node("v24") == "v24.21.0"
