"""What an agent did through the shell (docs/BUILD_TRACK.md, "Arms").

The pilot (run 38109978788) changed files in 3 of 10 agent runs with no Edit or
Write call: the writes went through the shell, so Rigour's edit hook, which
matches Edit and Write, never saw them. Each shell command is recorded by its
programs and whether it writes files. The command text itself is not kept.
"""
from __future__ import annotations

import re

SEGMENT_RE = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
WRITES = (("redirect", re.compile(r"(?<![<>&\d])>{1,2}(?!&)\s*[^\s&|;]")), ("heredoc", re.compile(r"<<-?\s*['\"]?\w+")),
          ("tee", re.compile(r"(^|\s)tee(\s|$)")), ("sed -i", re.compile(r"(^|\s)sed\s+(-\w*i|--in-place)")),
          ("gofmt -w", re.compile(r"(^|\s)gofmt\s+(.*\s)?-w(\s|$)")), ("cp/mv", re.compile(r"(^|\s)(cp|mv|install)\s")),
          ("go generate", re.compile(r"(^|\s)go\s+generate(\s|$)")))


def programs(command: str) -> list[str]:
    """The program each segment of a compound command runs (`go test`, `gofmt`, `cat`, ...)."""
    found = []
    for segment in SEGMENT_RE.split(command.strip()):
        words = [w for w in segment.split() if "=" not in w.split("/")[0]]   # skip leading VAR=value
        if words:
            found.append(" ".join(words[:2]) if words[0] in ("go", "git") else words[0])
    return found


def writes(command: str) -> list[str]:
    """How a command writes files, if it does."""
    return [name for name, pattern in WRITES if pattern.search(command)]


def shell_use(calls: list[dict]) -> dict:
    """{commands, programs: {program: n}, writing: n, how: {kind: n}} over the transcript's Bash calls."""
    used: dict[str, int] = {}
    how: dict[str, int] = {}
    writing = 0
    for call in calls:
        if call.get("name") != "Bash":
            continue
        command = str((call.get("input") or {}).get("command") or "")
        for program in programs(command):
            used[program] = used.get(program, 0) + 1
        kinds = writes(command)
        writing += bool(kinds)
        for kind in kinds:
            how[kind] = how.get(kind, 0) + 1
    return {"commands": sum(1 for c in calls if c.get("name") == "Bash"), "programs": used, "writing": writing,
            "how": how}
