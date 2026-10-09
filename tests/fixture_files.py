"""Recorded tool and API responses under tests/fixtures."""
from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def load_json(name: str) -> dict:
    try:
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssertionError(f"fixture {name} is missing or not valid JSON") from exc
