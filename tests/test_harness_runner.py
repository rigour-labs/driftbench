import pytest

from bench.adapters.baselines import EveryHunk, NoTool
from bench.harness.gitrepo import RepoCheckout
from bench.harness.runner import RecordError, RunConfig, read_record, record_path, run_corpus
from bench.harness.types import AdapterError, Finding, ReviewOutput
from tests.git_fixture import make_origin


class StatefulTool:
    """Leaves state in the checkout, and reports if it ever finds state from an earlier run."""
    name, version, paid, reads_history = "stateful", "1", False, False

    def review(self, request):
        state = request.workdir / ".tool-state"
        leaked = state.exists()
        state.write_text(request.head_sha)
        return ReviewOutput([Finding("app.py", 1, leaked, "leaked" if leaked else "clean")], "pass")


class CrashingTool:
    name, version, paid, reads_history = "crashing", "1", False, False

    def review(self, request):
        raise RuntimeError("boom")


class FailingTool:
    name, version, paid, reads_history = "failing", "1", False, False

    def review(self, request):
        raise AdapterError("no report", raw="partial output")


@pytest.fixture
def setup(tmp_path):
    origin = make_origin(tmp_path)
    pr = {"number": 5, "base_sha": origin["base"], "head_sha": origin["head2"], "approved_head_sha": origin["head2"],
          "approval_overridden": False,
          "rounds": [{"index": 1, "head_sha": origin["head1"]}, {"index": 2, "head_sha": origin["head2"]}]}
    corpus = {"repo": "o/r", "prs": [pr]}
    checkout = RepoCheckout(origin["path"], tmp_path / "clone", blobless=False)
    config = RunConfig(out_dir=tmp_path / "runs", scratch_dir=tmp_path / "scratch", timeout_s=30)
    return corpus, checkout, config, origin


def read(config: RunConfig, adapter, head: str) -> dict:
    return read_record(record_path(config, adapter, "o/r", 5, head))


def test_one_record_per_head_with_cases_diff_and_timing(setup):
    corpus, checkout, config, origin = setup
    assert run_corpus(EveryHunk(), checkout, corpus, config) == {"written": 2, "skipped": 0, "pass": 2}
    first = read(config, EveryHunk(), origin["head1"])
    assert first["base_sha"] == origin["base"] and first["changed_lines"] == 6  # 1 removed, 5 added
    assert [f["line"] for f in first["findings"]] == [2]
    assert [c["case_id"] for c in first["cases"]] == ["5-round-1"]
    second = read(config, EveryHunk(), origin["head2"])
    assert [c["case_id"] for c in second["cases"]] == ["5-round-2", "5-must-not-block"]  # shared head, run once
    assert second["cases"][1]["source"] == "approved" and second["wall_s"] >= 0
    assert record_path(config, EveryHunk(), "o/r", 5, origin["head2"]).with_suffix(".raw.txt").exists()


def test_no_state_survives_between_heads(setup):
    corpus, checkout, config, origin = setup
    run_corpus(StatefulTool(), checkout, corpus, config)
    assert [read(config, StatefulTool(), origin[h])["findings"][0]["message"] for h in ("head1", "head2")] == [
        "clean", "clean"]


def test_crash_is_recorded_and_runs_resume(setup):
    corpus, checkout, config, origin = setup
    assert run_corpus(CrashingTool(), checkout, corpus, config)["error"] == 2
    assert "RuntimeError: boom" in read(config, CrashingTool(), origin["head1"])["error"]
    assert run_corpus(CrashingTool(), checkout, corpus, config) == {"written": 0, "skipped": 2}
    run_corpus(FailingTool(), checkout, corpus, config)
    path = record_path(config, FailingTool(), "o/r", 5, origin["head1"])
    assert path.with_suffix(".raw.txt").read_text() == "partial output"


def test_unavailable_commit_and_history_adapters(setup):
    corpus, checkout, config, _ = setup
    corpus["prs"][0]["rounds"] = [{"index": 1, "head_sha": "f" * 40}]
    counts = run_corpus(NoTool(), checkout, corpus, config)
    assert counts["unavailable"] == 1 and read(config, NoTool(), "f" * 40)["verdict"] == "unavailable"

    class HistoryTool(NoTool):
        reads_history = True
    with pytest.raises(ValueError, match="reads history"):
        run_corpus(HistoryTool(), checkout, corpus, config)


def test_read_record_rejects_bad_files(tmp_path):
    bad = tmp_path / "r.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(RecordError, match="cannot read"):
        read_record(bad)
    bad.write_text('{"schema": 0}', encoding="utf-8")
    with pytest.raises(RecordError, match="schema"):
        read_record(bad)
