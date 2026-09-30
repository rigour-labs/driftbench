"""Review pairs -> SFT data for `rlaif.finetune --max`.

Joins every mined repository's pairs (rlaif/review/data/*.jsonl), drops exact
duplicate prompts, and writes prompt/completion rows: TRL then trains on the
completion only, so the model learns the review, not the prompt.

    python -m rlaif.review.export_sft --out rlaif/data/review_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rlaif.fixtrees import refuse_eval_repo

DATA = Path(__file__).resolve().parent / "data"


def export(files: list[Path], seed: int = 7) -> list[dict]:
    rows: dict[str, dict] = {}
    for path in files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            pair = json.loads(line)
            refuse_eval_repo(pair["repo"])  # a stale file from an eval repo must never reach training
            rows.setdefault(pair["prompt"], {"prompt": pair["prompt"], "completion": pair["target"]})
    shuffled = list(rows.values())
    random.Random(seed).shuffle(shuffled)
    return shuffled


def main() -> None:
    parser = argparse.ArgumentParser(description="Review pairs to SFT prompt/completion rows")
    parser.add_argument("--out", default="rlaif/data/review_sft.jsonl")
    args = parser.parse_args()
    rows = export(sorted(DATA.glob("*.jsonl")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} SFT rows -> {out}")


if __name__ == "__main__":
    main()
