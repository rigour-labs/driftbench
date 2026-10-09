from bench.__main__ import main
from bench.harness.runner import write_record
from bench.score.spend import spend_by_tool, spend_notes


def record(run, tool, head, **result):
    header = {"schema": 1, "tool": tool, "tool_version": "1", "repo": "o/r", "pr": 1, "head_sha": head, "cases": []}
    body = {"verdict": "pass", "findings": [], "cost_usd": None, **result}
    write_record(run / tool / "o__r" / "1" / f"{head}.json", header, body)


def test_spend_counts_reported_costs_and_bound_charges(tmp_path):
    record(tmp_path, "claude-code-review", "h1", cost_usd=0.40, charged="reported")
    record(tmp_path, "claude-code-review", "h2", verdict="error", charged="bound")
    record(tmp_path, "claude-code-review", "h3", verdict="not_scored")
    record(tmp_path, "rigour", "h1")
    budgets = [{"estimate_per_head": {"claude-code-review": 0.5}, "largest_per_head": {"claude-code-review": 0.4}}]
    spend = spend_by_tool(tmp_path, budgets)
    assert spend["claude-code-review"]["estimated_usd"] == 0.9            # 0.40 reported + 1 head at the 0.50 bound
    assert spend["claude-code-review"]["heads"] == {"error": 1, "not_scored": 1, "pass": 1}
    notes = spend_notes(spend)
    assert "claude-code-review: estimated $0.90 (1 head(s) at the bound), not scored 1" in notes
    assert "billed: $____" in notes and "rigour:" not in notes          # free entrants aren't listed


def test_spend_command(tmp_path, capsys):
    record(tmp_path, "claude-code-review", "h1", cost_usd=0.25, charged="reported")
    (tmp_path / "budget-o__r.json").write_text('{"estimate_per_head": {}, "largest_per_head": {}}')
    assert main(["spend", "--run", str(tmp_path)]) == 0
    assert "estimated $0.25" in capsys.readouterr().out
