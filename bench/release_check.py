"""Refuse to upload corpus, points or run files that carry review text.

The frozen corpus and points files hold IDs, SHAs, anchors and text hashes.
A string under a key that names text, or any long string, means review text
leaked into them; the upload stops. (A count keyed "body", the point kind,
is fine.) Run records are checked the same way; a finding's `message` must
be the reduced form (bench/harness/publish.py), at most 80 characters, and
there is no raw tool output at all.

    python -m bench.release_check <file or directory>...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bench.harness.publish import MESSAGE_LIMIT

TEXT_KEYS = {"body", "text", "title", "comment", "diff", "patch", "content", "raw"}
MAX_STRING = 400


def problems(value: object, where: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TEXT_KEYS and isinstance(item, str):
                found.append(f"{where}.{key}: text-bearing key")
            elif key == "message" and isinstance(item, str) and len(item) > MESSAGE_LIMIT:
                found.append(f"{where}.message: {len(item)} characters (limit {MESSAGE_LIMIT})")
            else:
                found += problems(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += problems(item, f"{where}[{index}]")
    elif isinstance(value, str) and len(value) > MAX_STRING:
        found.append(f"{where}: {len(value)}-character string")
    return found


SCRATCH = "_scratch"  # sandboxes and diffs of the reviewed code; never uploaded, so never checked


def files_to_check(path: Path) -> list[Path]:
    if not path.is_dir():
        return [path]
    return sorted(f for f in path.rglob("*.json") if SCRATCH not in f.relative_to(path).parts)


def check_paths(paths: list[Path]) -> list[str]:
    """Every problem found; an unreadable or undecodable file is a problem, never a crash."""
    found: list[str] = []
    for path in paths:
        for file in files_to_check(path):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:  # UnicodeDecodeError and JSONDecodeError are ValueErrors
                found.append(f"{file}: unreadable ({type(exc).__name__})")
                continue
            found += problems(data, str(file))
    return found


def main(argv: list[str]) -> int:
    found = check_paths([Path(arg) for arg in argv])
    for line in found[:50]:
        print(line, file=sys.stderr)
    print(f"release check: {len(found)} problem(s)", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
