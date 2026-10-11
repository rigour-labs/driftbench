"""The agent's shell: what it ran and wrote (recorded), and the guard both arms get (enforced)."""
from __future__ import annotations

import json
import subprocess
import sys

from bench.buildtrack import arms, shell_guard
from bench.buildtrack.evaluate import first_errors
from bench.buildtrack.shell import programs, shell_use, writes
from bench.buildtrack.toolchains import GO


def bash(command: str) -> dict:
    return {"type": "tool_use", "name": "Bash", "input": {"command": command}}


def test_shell_commands_are_recorded_by_program_and_by_how_they_write():
    """Pilot 38109978788: files changed with no Edit or Write call, through the shell."""
    assert programs("GOFLAGS=x go build ./... && cat > a.go <<'EOF'") == ["go build", "cat"]
    assert writes("go build ./... && cat > a.go <<'EOF'") == ["redirect", "heredoc"]
    assert writes("gofmt -l -w net/a.go") == ["gofmt -w"] and writes("sed -i 's/a/b/' x.go") == ["sed -i"]
    assert writes("go test ./net/... 2>&1") == [] and writes("go vet ./...") == []
    used = shell_use([bash("go test ./a"), bash("go build ./... && cat > b.go <<EOF"), {"name": "Edit", "input": {}}])
    assert used == {"commands": 2, "programs": {"go test": 1, "go build": 1, "cat": 1}, "writing": 1,
                    "how": {"redirect": 1, "heredoc": 1}}


def test_the_guard_allows_one_plain_designed_command_and_nothing_else():
    for allowed in ("go test ./net/dns/...", "go build ./...", "go vet ./ssh", "gofmt -l net/a.go", "git diff --stat"):
        assert shell_guard.refusal(allowed) is None, allowed
    for refused in ("go build ./... && cat > a.go", "go test ./... | tee out", "go test ./... > out.txt",
                    "gofmt -w a.go", "gofmt -lw a.go", "cat a.go", "echo $(id)", "python3 x.py", "go run ./cmd"):
        assert shell_guard.refusal(refused), refused


def test_the_guard_runs_as_a_hook_and_blocks_with_exit_2():
    run = lambda payload: subprocess.run([sys.executable, str(arms.GUARD)], input=json.dumps(payload), text=True,
                                         capture_output=True, check=False)
    blocked = run({"tool_name": "Bash", "tool_input": {"command": "go build ./... && cat > x.go"}})
    assert blocked.returncode == 2 and "Edit or Write" in blocked.stderr
    assert run({"tool_name": "Bash", "tool_input": {"command": "go test ./..."}}).returncode == 0
    assert run({"tool_name": "Edit", "tool_input": {}}).returncode == 0


def test_both_arms_get_the_same_guard():
    alone = arms.agent_command("alone", "anthropic/m-1", GO, 60, 1.0, "t")
    rigour = arms.agent_command("rigour", "anthropic/m-1", GO, 60, 1.0, "t")
    settings = json.loads(alone[alone.index("--settings") + 1])
    assert settings == json.loads(rigour[rigour.index("--settings") + 1])
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "Bash" and "shell_guard.py" in json.dumps(settings)


def test_the_first_error_lines_are_kept():
    events = [{"Action": "build-output", "Output": "net/dns/a_test.go:12:5: undefined: Load\n"},
              {"Action": "output", "Output": "ok   example 0.1s\n"}]
    assert first_errors(events, "FAIL\texample [build failed]\n") == ["net/dns/a_test.go:12:5: undefined: Load",
                                                                     "FAIL\texample [build failed]"]
