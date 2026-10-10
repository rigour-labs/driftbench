import dataclasses
import subprocess
from pathlib import Path

import pytest

from bench.adapters.baselines import EveryHunk, NoTool
from bench.harness.gitrepo import RepoCheckout
from bench.harness.runner import RecordError, RunConfig, read_record, record_path, run_corpus
from bench.harness.types import AdapterError, Finding, ReviewOutput
from tests.git_fixture import commit_file, git, make_origin


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
        raise AdapterError("no report")


@pytest.fixture
def setup(tmp_path):
    origin = make_origin(tmp_path)
    pr = {"number": 5, "base_sha": origin["base"], "head_sha": origin["head2"], "approved_head_sha": origin["head2"],
          "approval_overridden": False,
          "rounds": [{"index": 1, "head_sha": origin["head1"]}, {"index": 2, "head_sha": origin["head2"]}]}
    corpus = {"repo": "o/r", "prs": [pr]}
    checkout = RepoCheckout(origin["path"], tmp_path / "clone", blobless=False)
    config = RunConfig(out_dir=tmp_path / "runs", scratch_dir=tmp_path / "scratch",
                       npm_cache=tmp_path / "npm-cache", timeout_s=30)
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
    assert not record_path(config, EveryHunk(), "o/r", 5, origin["head2"]).with_suffix(".raw.txt").exists()


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
    record = read(config, FailingTool(), origin["head1"])
    assert record["error"] == "AdapterError: no report" and "raw" not in record


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


class EnvTool:
    """Keeps the environment a child process of the tool actually gets, per head."""
    name, version, paid, reads_history = "env", "1", False, False

    def __init__(self):
        self.seen: dict[str, str] = {}

    def review(self, request):
        dump = subprocess.run(["env"], env=request.env, capture_output=True, text=True, check=True).stdout
        self.seen[request.head_sha] = dump
        return ReviewOutput([], "pass")


class RefsTool:
    """Keeps every ref and every commit reachable from any ref in its workdir, per head."""
    name, version, paid, reads_history = "refs", "1", False, False

    def __init__(self):
        self.seen: dict[str, str] = {}

    def review(self, request):
        def git_out(*args):
            return subprocess.run(["git", "-C", str(request.workdir), *args], capture_output=True, text=True).stdout
        self.seen[request.head_sha] = git_out("for-each-ref") + "|" + git_out("log", "--all", "--format=%H")
        return ReviewOutput([], "pass")


def test_tools_get_a_minimal_env_and_a_fresh_home(setup, monkeypatch):
    corpus, checkout, config, origin = setup
    for name in ("RIGOUR_TEAM_ID", "GH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "planted-secret")
    tool = EnvTool()
    run_corpus(tool, checkout, corpus, config)
    record = read(config, tool, origin["head1"])
    dump = tool.seen[origin["head1"]]
    assert "planted-secret" not in dump
    env = dict(line.split("=", 1) for line in dump.splitlines() if "=" in line)
    assert env["HOME"] != str(Path.home()) and env["RIGOUR_TELEMETRY"] == "0" and env["DO_NOT_TRACK"] == "1"
    assert not Path(env["HOME"]).exists()  # deleted after the run
    assert "GH_TOKEN" not in record["env_keys"] and "HOME" in record["env_keys"]


def test_tools_cannot_see_refs_or_commits_after_the_head(setup):
    corpus, checkout, config, origin = setup
    future = commit_file(Path(origin["path"]), "app.py", "def f():\n    return 99\n", "future on main")
    git(Path(origin["path"]), "tag", "v9", future)
    tool = RefsTool()
    run_corpus(tool, checkout, corpus, config)
    refs, reachable = tool.seen[origin["head1"]].split("|")
    assert refs == ""                                   # no origin/main, no branches, no tags
    assert future not in reachable and origin["head2"] not in reachable
    assert set(reachable.split()) == {origin["base"], origin["head1"]}


def test_run_manifest_keeps_the_first_start_and_fixes_the_labels(tmp_path):
    from bench.harness.cli import run_manifest
    from tests.git_fixture import commit_file
    repo = tmp_path / "repo"
    (repo / "labels").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    commit_file(repo, "labels/o__r.yaml", "points: {}\n", "labels")
    first = run_manifest(tmp_path / "run", [NoTool()], repo / "labels")
    assert first["entrants"] == {"no-tool": "1"} and set(first["labels"]["files"]) == {"o__r.yaml"}
    assert len(first["labels"]["commit"]) == 40
    assert run_manifest(tmp_path / "run", [EveryHunk()], repo / "labels") == first   # resume keeps it


def test_manifest_command_writes_once_for_split_jobs(tmp_path, capsys):
    from bench.__main__ import main
    out = tmp_path / "run"
    assert main(["manifest", "--entrants", "free", "--out", str(out), "--labels", str(tmp_path / "none")]) == 0
    written = (out / "run.json").read_text()
    assert main(["manifest", "--entrants", "no-tool", "--out", str(out), "--labels", str(tmp_path / "none")]) == 0
    assert (out / "run.json").read_text() == written          # the first record is kept
    assert '"rigour": "6.12.1"' in written and '"files": {}' in written


def test_a_smoke_run_reviews_only_the_first_large_enough_heads(setup):
    corpus, checkout, config, origin = setup
    smoke = dataclasses.replace(config, max_heads=1, min_changed_lines=5)
    assert run_corpus(EveryHunk(), checkout, corpus, smoke) == {"written": 1, "skipped": 0, "pass": 1}
    assert read(smoke, EveryHunk(), origin["head1"])["changed_lines"] == 6
    assert not record_path(smoke, EveryHunk(), "o/r", 5, origin["head2"]).exists()   # beyond the cut: no record
    tiny = dataclasses.replace(config, out_dir=config.out_dir.parent / "tiny", max_heads=1, min_changed_lines=1000)
    assert run_corpus(EveryHunk(), checkout, corpus, tiny) == {"written": 0, "skipped": 0}


def test_smoke_runs_are_recorded_and_never_scored(tmp_path, capsys):
    from bench.__main__ import main
    from bench.harness.cli import read_manifest
    out = tmp_path / "run"
    assert main(["manifest", "--entrants", "free", "--out", str(out), "--labels", str(tmp_path), "--max-heads", "1"]) == 0
    smoke = read_manifest(out / "run.json")["smoke"]
    assert smoke["max_heads"] == 1 and smoke["min_changed_lines"] == 20 and "corpus order" in smoke["rule"]
    assert main(["score", "--run", str(out), "--out", str(tmp_path / "results")]) == 1
    assert "never scored" in capsys.readouterr().err
    full = tmp_path / "full"
    assert main(["manifest", "--entrants", "free", "--out", str(full), "--labels", str(tmp_path)]) == 0
    assert "smoke" not in read_manifest(full / "run.json")


def test_a_failing_entrant_in_a_smoke_run_makes_one_attempt(setup):
    corpus, checkout, config, origin = setup
    smoke = dataclasses.replace(config, max_heads=1, min_changed_lines=5)
    assert run_corpus(FailingTool(), checkout, corpus, smoke) == {"written": 1, "skipped": 0, "error": 1}
    assert not record_path(smoke, FailingTool(), "o/r", 5, origin["head2"]).exists()


def test_records_keep_model_runs_leak_signals_and_numeric_diagnostics(setup):
    corpus, checkout, config, origin = setup

    class Reporting:
        name, version, paid, reads_history = "reporting", "1", False, False

        def review(self, request):
            return ReviewOutput([], "pass", model_runs=3, leak_signals=0,
                                diagnostics={"result_chars": 120, "citation_like": 0})
    run_corpus(Reporting(), checkout, corpus, config)
    record = read(config, Reporting(), origin["head1"])
    assert record["model_runs"] == 3 and record["leak_signals"] == 0
    assert record["diagnostics"] == {"result_chars": 120, "citation_like": 0}


def test_a_paid_entrants_whole_answer_is_kept_and_free_entrants_get_none(setup):
    corpus, checkout, config, origin = setup

    class PaidTool:
        name, version, paid, reads_history = "paid-tool", "1", True, False

        def review(self, request):
            return ReviewOutput([Finding("app.py", 1, False, 'Use "x" here: Found: secret code', "r")], "pass",
                                cost_usd=0.1, model_runs=1, review_text="Full answer, quoting `code()` at length.")
    from bench.harness.budget import Budget
    paid = dataclasses.replace(config, budget=Budget(5.0, {"paid-tool": 1.0}))   # a paid entrant needs a budget
    run_corpus(PaidTool(), checkout, corpus, paid)
    record = read(paid, PaidTool(), origin["head1"])
    assert record["findings"][0]["message"] == "Use here:"                          # reduced, as before
    assert record["paid_output"]["review_text"] == "Full answer, quoting `code()` at length."
    assert record["paid_output"]["findings"][0]["message"] == 'Use "x" here: Found: secret code'
    run_corpus(EveryHunk(), checkout, corpus, config)
    assert "paid_output" not in read(config, EveryHunk(), origin["head1"])


def test_an_explicit_selection_runs_only_its_heads(setup):
    corpus, checkout, config, origin = setup
    only = dataclasses.replace(config, only_heads=frozenset({origin["head2"]}))
    assert run_corpus(EveryHunk(), checkout, corpus, only)["written"] == 1
    assert record_path(only, EveryHunk(), "o/r", 5, origin["head2"]).exists()
    assert not record_path(only, EveryHunk(), "o/r", 5, origin["head1"]).exists()


def test_a_diagnostic_run_states_its_purpose_and_limit_and_is_never_scored(tmp_path, capsys):
    from bench.__main__ import main
    from bench.harness.cli import read_manifest
    out = tmp_path / "run"
    assert main(["manifest", "--entrants", "free", "--out", str(out), "--labels", str(tmp_path),
                 "--diagnostic", "tell filtered from unseen"]) == 0
    diagnostic = read_manifest(out / "run.json")["diagnostic"]
    assert diagnostic["purpose"] == "tell filtered from unseen" and "not what they did" in diagnostic["limit"]
    assert main(["score", "--run", str(out), "--out", str(tmp_path / "results")]) == 1
    assert "never scored" in capsys.readouterr().err
