import dataclasses
import subprocess

import pytest

from bench.harness.runner import run_corpus
from bench.harness.types import ReviewOutput
from tests.test_harness_runner import read, setup  # noqa: F401  (pytest fixture)


class PaidFake:
    name, version, paid, reads_history, env_extra = "paid-fake", "1", True, False, ("ANTHROPIC_API_KEY",)

    def review(self, request):
        return ReviewOutput([], "pass", cost_usd=0.4, model_runs=1)


def test_budget_stop_marks_every_remaining_paid_head_not_scored(setup):
    from bench.harness.budget import Budget
    corpus, checkout, config, origin = setup
    budget = Budget(max_usd=0.5, estimate_per_head={"paid-fake": 0.4})
    counts = run_corpus(PaidFake(), checkout, corpus, dataclasses.replace(config, budget=budget))
    assert counts == {"written": 2, "skipped": 0, "pass": 1, "not_scored": 1}
    stopped = read(config, PaidFake(), origin["head2"])
    assert stopped["verdict"] == "not_scored" and stopped["error"].startswith("budget:")
    assert budget.spent == 0.4 and "ANTHROPIC_API_KEY" in stopped["env_keys"]


def test_inside_the_sandbox_gh_cannot_reach_pull_request_data(setup):
    """The lookup Rigour's reviewer makes (gh api repos/{owner}/{repo}/commits/<head>/pulls) fails in the sandbox."""
    import shutil
    if shutil.which("gh") is None:
        pytest.skip("gh is not installed")
    corpus, checkout, config, origin = setup

    class GhProbe:
        name, version, paid, reads_history, env_extra = "gh-probe", "1", False, False, ()
        seen: dict = {}

        def review(self, request):
            result = subprocess.run(["gh", "api", f"repos/{{owner}}/{{repo}}/commits/{request.head_sha}/pulls"],
                                    cwd=request.workdir, env=request.env, capture_output=True, text=True)
            self.seen[request.head_sha] = result.returncode
            return ReviewOutput([], "pass")
    probe = GhProbe()
    run_corpus(probe, checkout, corpus, config)
    assert probe.seen and all(code != 0 for code in probe.seen.values())


def test_bench_run_refuses_paid_entrants_without_cap_model_or_bound(tmp_path, capsys):
    from bench.__main__ import main
    base = ["run", "--entrants", "claude-code-review", "--corpus", str(tmp_path), "--out", str(tmp_path / "out")]
    assert main(base) == 1 and "--max-usd" in capsys.readouterr().err
    assert main([*base, "--max-usd", "5", "--model", "m"]) == 1
    assert "no per-head cost bound" in capsys.readouterr().err


class PaidTimeout:
    name, version, paid, reads_history, env_extra = "paid-timeout", "1", True, False, ("ANTHROPIC_API_KEY",)

    def review(self, request):
        from bench.harness.types import AdapterError
        raise AdapterError("timed out after 900s")


def test_a_failed_paid_review_is_charged_its_bound_and_the_next_is_refused(setup):
    from bench.harness.budget import Budget
    corpus, checkout, config, origin = setup
    budget = Budget(max_usd=0.5, estimate_per_head={"paid-timeout": 0.4})
    counts = run_corpus(PaidTimeout(), checkout, corpus, dataclasses.replace(config, budget=budget))
    assert counts == {"written": 2, "skipped": 0, "error": 1, "not_scored": 1}
    first = read(config, PaidTimeout(), origin["head1"])
    assert first["verdict"] == "error" and first["charged"] == "bound" and budget.spent == 0.4
    assert read(config, PaidTimeout(), origin["head2"])["verdict"] == "not_scored"


def test_a_paid_review_with_a_reported_cost_records_it(setup):
    from bench.harness.budget import Budget
    corpus, checkout, config, origin = setup
    budget = Budget(max_usd=5, estimate_per_head={"paid-fake": 0.4})
    run_corpus(PaidFake(), checkout, corpus, dataclasses.replace(config, budget=budget))
    assert read(config, PaidFake(), origin["head1"])["charged"] == "reported"


def test_run_json_records_the_paid_settings_and_tool_access(tmp_path, monkeypatch):
    from bench.__main__ import main
    from bench.harness import prompts
    monkeypatch.setattr(prompts, "run_text", lambda args: "sha512-native")
    from bench.adapters.tool_access import tool_record
    from bench.harness.cli import read_manifest
    out = tmp_path / "run"
    assert main(["manifest", "--entrants", "claude-code-review", "--out", str(out), "--labels", str(tmp_path),
                 "--model", "model-x", "--max-usd", "5", "--head-bound", "claude-code-review=0.8"]) == 0
    paid = read_manifest(out / "run.json")["paid"]
    assert paid["model"] == "model-x" and paid["max_usd"] == 5.0 and paid["tools"] == tool_record()
    assert paid["claude_code"] == "2.1.285"
    assert paid["prompts"]["claude-code-review"]["integrity"] == "sha512-native"
