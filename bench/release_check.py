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


def check_paths(paths: list[Path]) -> list[str]:
    found: list[str] = []
    for path in paths:
        files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        for file in files:
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                found.append(f"{file}: unreadable ({exc})")
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
