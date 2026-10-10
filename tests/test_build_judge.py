"""The build track's blind judge and its numbers."""
from __future__ import annotations

import json

from bench.buildtrack import judge
from bench.buildtrack.report import paired, repeated, summary
from bench.harness.budget import Budget

POINT = {"id": "7-inline-1-0", "anchor": {"path": "net/dns.go", "line": 12}}


def call_answering(verdicts: dict[str, str], cost: float = 0.002):
    seen = []

    def call(messages):
        seen.append(messages)
        return {"content": json.dumps({k: {"verdict": v, "reason": "r"} for k, v in verdicts.items()}),
                "cost_usd": cost}
    return call, seen


def test_the_judge_sees_both_diffs_blind_and_records_per_arm():
    call, seen = call_answering({"A": "repeated", "B": "avoided"})
    budget = Budget(1.0, {judge.ENTRANT: 0.01})
    out = judge.judge_point(POINT, "retry never stops", {"alone": "+a", "rigour": "+b"}, 7, call, budget)
    first, second = out["order"]
    assert out["verdicts"][first]["verdict"] == "repeated" and out["verdicts"][second]["verdict"] == "avoided"
    text = seen[0][1]["content"]
    assert "alone" not in text and "rigour" not in text.lower() and "Diff A:" in text and "Diff B:" in text
    assert budget.spent == 0.002 and out["cut"] == []


def test_long_diffs_are_cut_and_bad_answers_are_charged_but_not_counted():
    call, _ = call_answering({"A": "maybe"})
    budget = Budget(1.0, {judge.ENTRANT: 0.01})
    long = "+" * (judge.DIFF_CHARS + 5)
    out = judge.judge_point(POINT, "c", {"alone": long, "rigour": ""}, 7, call, budget)
    assert out["cut"] == ["alone"] and "invalid output" in out["error"] and budget.spent == 0.002
    unpriced = judge.judge_point(POINT, "c", {"alone": "x", "rigour": "y"}, 7, lambda m: {"content": None, "cost_usd": None},
                                 budget)
    assert "charged at the bound" in unpriced["error"] and budget.spent == 0.012


def verdicts(alone: str, rigour: str) -> dict:
    return {"verdicts": {"alone": {"verdict": alone}, "rigour": {"verdict": rigour}}}


JUDGED = {"p1": verdicts("repeated", "avoided"), "p2": verdicts("repeated", "repeated"),
          "p3": verdicts("avoided", "avoided"), "p4": verdicts("not_applicable", "repeated"), "p5": {"error": "x"}}


def test_points_repeated_are_counted_out_of_those_that_apply_and_paired():
    assert repeated(JUDGED, "alone")["repeated"] == 2 and repeated(JUDGED, "alone")["applicable"] == 3
    assert repeated(JUDGED, "alone")["not_applicable"] == 1 and repeated(JUDGED, "rigour")["repeated"] == 2
    assert paired(JUDGED) == {"alone_only": 1, "rigour_only": 0, "both": 1, "neither": 1}


def test_the_summary_counts_tests_cost_rigour_events_and_false_blocks():
    agent = lambda cost: {"cost_usd": cost, "turns": 5, "wall_s": 10}
    task = {"arms": {"alone": {"agent": agent(0.4), "tests": {"outcome": "pass"}},
                     "rigour": {"agent": agent(0.6), "tests": {"outcome": "fail"},
                                "rigour": {"events": {"events": 3, "by_type": {"stop_review": {"events": 2, "blocked": 1}}}}}},
            "reference": {"false_blocks": {"false_blocks": 1}}}
    out = summary([task, task], JUDGED)
    alone, rigour = out["arms"]["alone"], out["arms"]["rigour"]
    assert alone["tests"]["pass"] == 2 and alone["cost_per_task"] == 0.4 and "false_blocks" not in alone
    assert rigour["tests"]["fail"] == 2 and rigour["rigour_events"] == 6 and rigour["rigour_blocks"] == 2
    assert rigour["false_blocks"] == 2 and out["judge_errors"] == 1


def test_judge_and_report_commands_run_over_a_build_run_and_resume(tmp_path, monkeypatch):
    from bench.__main__ import main
    from bench.buildtrack import judge_cli
    run, points = tmp_path / "run", tmp_path / "points"
    (run / "tasks").mkdir(parents=True)
    points.mkdir()
    (points / "o__r.json").write_text(json.dumps({"schema": 1, "points": [{**POINT, "pr": 7}]}))
    arm = lambda diff: {"agent": {"paid_output": {"diff": diff}, "cost_usd": 0.5, "turns": 3, "wall_s": 9},
                        "tests": {"outcome": "pass"}}
    (run / "tasks" / "7.json").write_text(json.dumps({"repo": "o/r", "pr": 7, "points": [POINT["id"]],
                                                      "arms": {"alone": arm("+a"), "rigour": arm("+b")}}))
    monkeypatch.setattr(judge_cli, "point_text", lambda texts, point: "the retry never stops")
    calls = []
    monkeypatch.setattr(judge_cli, "chat", lambda model, messages, max_tokens: calls.append(1) or {
        "content": json.dumps({"A": {"verdict": "repeated"}, "B": {"verdict": "avoided"}}), "cost_usd": 0.003})
    args = ["build", "judge", "--run", str(run), "--model", "openai/gpt-x", "--max-usd", "1", "--points", str(points),
            "--cache", str(tmp_path / "cache")]
    assert main(args) == 0 and main(args) == 0 and len(calls) == 1                 # the rerun resumes
    assert main(["build", "judge", "--run", str(run), "--model", "anthropic/claude-x", "--max-usd", "1"]) == 1
    assert main(["build", "report", "--run", str(run)]) == 0
    out = judge_cli.read_json(run / "summary.json")
    assert out["paired_points"]["alone_only"] + out["paired_points"]["rigour_only"] == 1 and out["tasks"] == 1
