"""One agent run on one task in one arm, and the record it leaves (docs/BUILD_TRACK.md).

The record holds numbers and the final diff (the agent's own work, kept in
the run's release only), never the task statement: that is review-adjacent
text, fetched at run time and checked against its hash.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from bench.adapters.claude_code import events, leak_signals, model_runs, token_counts, tool_calls
from bench.buildtrack.workspace import final_diff
from bench.harness.types import AdapterError

TIMEOUT_MARGIN_S = 120


def transcript_numbers(stream: list[dict]) -> dict:
    """Cost, tokens, turns, tool use and leak signals from Claude Code's stream-json transcript."""
    result = next((e for e in reversed(stream) if e.get("type") == "result"), None)
    if result is None:
        raise AdapterError("no result in the transcript")
    inputs, outputs = token_counts(result)
    calls = tool_calls(stream)
    names: dict[str, int] = {}
    for call in calls:
        names[str(call.get("name"))] = names.get(str(call.get("name")), 0) + 1
    return {"cost_usd": result.get("total_cost_usd"), "input_tokens": inputs, "output_tokens": outputs,
            "turns": model_runs(result, outputs), "stop": result.get("subtype"), "is_error": bool(result.get("is_error")),
            "tool_calls": names, "leak_signals": leak_signals(stream)}


def run_agent(command: list[str], repo: Path, env: dict[str, str], timeout_s: int) -> dict:
    """Run the agent to its end (or the timeout) and record what it did and what it changed."""
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True,
                                timeout=timeout_s + TIMEOUT_MARGIN_S, check=False)
        stdout, timed_out = result.stdout, False
    except subprocess.TimeoutExpired as exc:
        print(f"agent: timed out after {timeout_s + TIMEOUT_MARGIN_S}s; recording what it left", file=sys.stderr)
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        timed_out = True
    record: dict = {"wall_s": round(time.monotonic() - started, 1), "timed_out": timed_out}
    try:
        record.update(transcript_numbers(events(stdout)))
    except AdapterError as exc:
        print(f"agent: {exc}", file=sys.stderr)
        record["error"] = str(exc)[:300]
    record["diff"] = final_diff(repo)
    return record
