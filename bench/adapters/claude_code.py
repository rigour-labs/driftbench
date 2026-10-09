"""Claude Code's `/code-review` at its default level (paid).

`claude -p "/code-review"` in the sandbox repository, with the run's model,
the pinned CLI, and exactly the read-only tool access and isolation Rigour's
reviewer uses (bench/adapters/tool_access.py), plus web and GitHub tools
denied and no session saved. The transcript (stream-json) gives the cost, the tokens, the number
of model turns, and every tool call.

/code-review reports findings in prose; each `path:line` (or `path:L-L`) it
cites for a file in the diff becomes a finding. It has no notion of a
blocking finding, so none are blocking and its false-block rate is 0 by
construction (the report says so).

Time-correctness: any web fetch, web search, or shell command reaching
GitHub (`gh`, api.github.com, curl, wget) in the transcript is a leak
signal, and the head is not scored. Those tools are also blocked up front.
"""
from __future__ import annotations

import json
import re
import subprocess

from bench.adapters.claude_cli import CLAUDE_CODE_VERSION, paid_env, provider_env_names, require_claude_cli
from bench.adapters.tool_access import DENIED_TOOLS, ISOLATION_ARGS, ISOLATION_ENV, NETWORK_DENIED, READ_ONLY_TOOLS
from bench.harness.diffstat import parse_hunks
from bench.harness.types import AdapterError, Finding, ReviewInput, ReviewOutput

LEAKY_TOOLS = {"WebFetch", "WebSearch"}
LEAKY_COMMAND_RE = re.compile(r"(?<![\w-])(?:gh|curl|wget)(?![\w-])|api\.github\.com|github\.com/.+/pull")
CITATION_RE = re.compile(r"`?([\w./-]+\.[\w]+):L?(\d+)(?:-L?(\d+))?`?")
TIMEOUT_MARGIN_S = 120


def events(stdout: str) -> list[dict]:
    """The stream-json transcript: one JSON object per line; anything else means the run is unreadable."""
    try:
        parsed = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise AdapterError(f"unreadable transcript line: {exc}") from exc
    return [item for item in parsed if isinstance(item, dict)]


def tool_calls(stream: list[dict]) -> list[dict]:
    calls = []
    for event in stream:
        content = ((event.get("message") or {}).get("content")) if event.get("type") == "assistant" else None
        calls += [block for block in content or [] if isinstance(block, dict) and block.get("type") == "tool_use"]
    return calls


def leak_signals(stream: list[dict]) -> int:
    count = 0
    for call in tool_calls(stream):
        command = str((call.get("input") or {}).get("command") or "")
        if call.get("name") in LEAKY_TOOLS or LEAKY_COMMAND_RE.search(command):
            count += 1
    return count


def citations(text: str, changed_paths: set[str]) -> list[Finding]:
    """Each file:line the review cites for a changed file, once."""
    seen: set[tuple[str, int, int | None]] = set()
    findings = []
    for line in text.splitlines():
        for match in CITATION_RE.finditer(line):
            path, first = match.group(1), int(match.group(2))
            last = int(match.group(3)) if match.group(3) else None
            if path in changed_paths and (path, first, last) not in seen:
                seen.add((path, first, last))
                findings.append(Finding(path, first, False, line.strip(), "code-review", end_line=last))
    return findings


def to_output(stream: list[dict], changed_paths: set[str]) -> ReviewOutput:
    result = next((e for e in reversed(stream) if e.get("type") == "result"), None)
    if result is None:
        raise AdapterError("no result in the transcript")
    if result.get("is_error"):
        raise AdapterError(f"claude reported an error: {str(result.get('result'))[:200]}",
                           cost_usd=result.get("total_cost_usd"))
    usage = result.get("usage") or {}
    inputs = sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_read_input_tokens",
                                                   "cache_creation_input_tokens"))
    return ReviewOutput(findings=citations(str(result.get("result") or ""), changed_paths), verdict="pass",
                        cost_usd=result.get("total_cost_usd"), input_tokens=inputs or None,
                        output_tokens=usage.get("output_tokens"), model_runs=result.get("num_turns"),
                        leak_signals=leak_signals(stream))


class ClaudeCodeReview:
    name = "claude-code-review"
    version = CLAUDE_CODE_VERSION
    paid = True
    reads_history = False
    has_blocking = False  # /code-review never blocks: its blocking-only numbers are n/a, not 0
    def __init__(self, model: str, max_usd_per_review: float | None = None, provider: str = "anthropic"):
        self.model = model
        self.max_usd_per_review = max_usd_per_review
        self.provider = provider
        self.env_extra = provider_env_names(provider)

    def command(self) -> list[str]:
        budget = ["--max-budget-usd", f"{self.max_usd_per_review:.2f}"] if self.max_usd_per_review else []
        return ["claude", "-p", "/code-review", "--model", self.model, "--output-format", "stream-json", "--verbose",
                *ISOLATION_ARGS, "--no-session-persistence", "--allowedTools", *READ_ONLY_TOOLS,
                "--disallowedTools", *DENIED_TOOLS, *NETWORK_DENIED, *budget]

    def review(self, request: ReviewInput) -> ReviewOutput:
        env = {**paid_env(request, self.provider), **ISOLATION_ENV}
        require_claude_cli(env)
        try:
            result = subprocess.run(self.command(), cwd=request.workdir, env=env, capture_output=True, text=True,
                                    timeout=request.timeout_s + TIMEOUT_MARGIN_S, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AdapterError(f"timed out after {request.timeout_s + TIMEOUT_MARGIN_S}s") from exc
        changed = {hunk.path for hunk in parse_hunks(request.diff_path.read_text(encoding="utf-8"))}
        return to_output(events(result.stdout), changed)
