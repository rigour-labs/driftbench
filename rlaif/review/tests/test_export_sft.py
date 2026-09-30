import json
from pathlib import Path

import pytest

from rlaif.review.export_sft import export


def test_rows_are_prompt_completion_without_duplicate_prompts(tmp_path: Path):
    data = tmp_path / "a.jsonl"
    rows = [{"repo": "lucia-auth/lucia", "prompt": "p1", "target": '{"findings": []}'},
            {"repo": "lucia-auth/lucia", "prompt": "p1", "target": '{"findings": []}'},
            {"repo": "honojs/hono", "prompt": "p2", "target": '{"findings": [{"line": 3}]}'}]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert sorted(export([data]), key=lambda r: r["prompt"]) == [
        {"prompt": "p1", "completion": '{"findings": []}'}, {"prompt": "p2", "completion": '{"findings": [{"line": 3}]}'}]


def test_pairs_from_an_evaluation_repository_never_reach_training(tmp_path: Path):
    data = tmp_path / "stale.jsonl"
    data.write_text(json.dumps({"repo": "supabase/supabase", "prompt": "p", "target": "t"}) + "\n")
    with pytest.raises(SystemExit, match="evaluation repository"):
        export([data])
