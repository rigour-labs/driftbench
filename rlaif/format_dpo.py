"""
DPO (Direct Preference Optimization) Data Formatter

Reads the SQLite training database and outputs JSONL files for fine-tuning:
  - sft_data.jsonl:  Supervised fine-tuning pairs (facts → verified findings)
  - dpo_data.jsonl:  DPO preference pairs (chosen=verified, rejected=dropped)

DPO pair format (compatible with HuggingFace trl library):
{
    "prompt": "<AST facts for the file>",
    "chosen": "<finding that passed verification>",
    "rejected": "<finding that failed verification>"
}

Usage:
    python -m rlaif.format_dpo --db rlaif/data/training_data.db --output rlaif/data/
    python -m rlaif.format_dpo --db rlaif/data/training_data.db --min-examples 100
"""

import os
import json
import sqlite3
import random
import hashlib
import argparse
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rlaif.format_dpo")


def load_training_data(db_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load verified and dropped findings from SQLite."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    verified = [dict(r) for r in conn.execute(
        "SELECT * FROM training_data WHERE verified = 1 ORDER BY category, repo"
    ).fetchall()]

    dropped = [dict(r) for r in conn.execute(
        "SELECT * FROM training_data WHERE verified = 0 ORDER BY category, repo"
    ).fetchall()]

    conn.close()

    logger.info(f"Loaded {len(verified)} verified, {len(dropped)} dropped findings")
    return verified, dropped


def format_finding_as_json(row: Dict) -> str:
    """Format a DB row as a finding JSON string."""
    finding = {
        "category": row["category"],
        "severity": row["severity"],
        "file": row["file_path"],
        "line": None,
        "description": row["description"],
        "suggestion": row["suggestion"],
        "confidence": row["confidence"],
    }
    return json.dumps(finding)


def format_finding_context(row: Dict) -> str:
    """Create a context/prompt string from the file path and category."""
    return f"FILE: {row['file_path']} | CATEGORY: {row['category']} | REPO: {row['repo']}"


def build_sft_pairs(verified: List[Dict]) -> List[Dict]:
    """
    Build SFT pairs: prompt = file context, completion = verified finding.
    These teach the model what good findings look like.
    """
    pairs = []
    for row in verified:
        pair = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a code quality analyzer. Given file facts, produce accurate findings."
                },
                {
                    "role": "user",
                    "content": f"Analyze this file for {row['category']} issues:\nFILE: {row['file_path']} (repo: {row['repo']})"
                },
                {
                    "role": "assistant",
                    "content": format_finding_as_json(row)
                }
            ]
        }
        pairs.append(pair)

    logger.info(f"Built {len(pairs)} SFT pairs")
    return pairs


def _group_by_category(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return grouped


def _make_dpo_pair(
    category: str, chosen_row: Dict, dropped_row: Dict
) -> Dict:
    prompt = (
        f"Analyze this codebase for {category} issues.\n"
        f"File: {dropped_row['file_path']} (repo: {dropped_row['repo']})"
    )
    return {
        "prompt": prompt,
        "chosen": format_finding_as_json(chosen_row),
        "rejected": format_finding_as_json(dropped_row),
        "category": category,
        "chosen_notes": "Passed structural verification",
        "rejected_notes": dropped_row.get("verification_notes", ""),
    }


def build_dpo_pairs(
    verified: List[Dict],
    dropped: List[Dict],
    max_pairs: int = 10000,
) -> List[Dict]:
    """Build DPO preference pairs by matching verified/dropped per category."""
    verified_by_cat = _group_by_category(verified)
    dropped_by_cat = _group_by_category(dropped)
    all_cats = set(verified_by_cat) | set(dropped_by_cat)

    pairs = []
    categories_used = set()

    for category in all_cats:
        v_list = verified_by_cat.get(category, [])
        d_list = dropped_by_cat.get(category, [])
        if not v_list or not d_list:
            continue
        categories_used.add(category)
        for dropped_row in d_list:
            chosen_row = random.choice(v_list)
            pairs.append(_make_dpo_pair(category, chosen_row, dropped_row))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    random.shuffle(pairs)
    logger.info(f"Built {len(pairs)} DPO pairs across {len(categories_used)} categories")
    return pairs


def build_category_stats(verified: List[Dict], dropped: List[Dict]) -> Dict:
    """Build per-category statistics for reporting."""
    stats = defaultdict(lambda: {"verified": 0, "dropped": 0, "total": 0})

    for row in verified:
        cat = row["category"]
        stats[cat]["verified"] += 1
        stats[cat]["total"] += 1

    for row in dropped:
        cat = row["category"]
        stats[cat]["dropped"] += 1
        stats[cat]["total"] += 1

    # Calculate verification rate
    for cat, s in stats.items():
        s["verification_rate"] = round(s["verified"] / s["total"] * 100, 1) if s["total"] > 0 else 0

    return dict(sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True))


def _write_jsonl(path: str, rows: List[Dict], label: str):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info(f"Wrote {len(rows)} {label} → {path}")


def _write_dataset_card(
    path: str, sft_count: int, dpo_count: int,
    total_verified: int, total_dropped: int,
    top_cats: list,
):
    rate = round(
        total_verified / max(1, total_verified + total_dropped) * 100, 1
    )
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cat_rows = "\n".join(
        f"| {cat} | {s['verified']} | {s['dropped']} "
        f"| {s['verification_rate']}% |"
        for cat, s in top_cats
    )

    card = f"""---
language: en
license: apache-2.0
task_categories:
  - text-generation
tags:
  - code-quality
  - rigour
  - rlaif
  - dpo
size_categories:
  - 1K<n<10K
---

# Rigour RLAIF Training Data

Training data for fine-tuning code quality analysis models.

## Stats

- **SFT pairs**: {sft_count}
- **DPO pairs**: {dpo_count}
- **Verified**: {total_verified}, **Dropped**: {total_dropped}
- **Rate**: {rate}%

## Top Categories

| Category | Verified | Dropped | Rate |
|----------|----------|---------|------|
{cat_rows}

## Usage with trl

```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="dpo_data.jsonl")
```

Generated: {date_str} | Pipeline: rigour-labs/driftbench RLAIF
"""
    with open(path, "w") as f:
        f.write(card)
    logger.info(f"Wrote dataset card → {path}")


def export_for_huggingface(
    sft_pairs: List[Dict],
    dpo_pairs: List[Dict],
    output_dir: str,
    stats: Dict,
):
    """Export training data in HuggingFace-compatible format."""
    os.makedirs(output_dir, exist_ok=True)

    _write_jsonl(
        os.path.join(output_dir, "sft_data.jsonl"), sft_pairs, "SFT pairs"
    )
    _write_jsonl(
        os.path.join(output_dir, "dpo_data.jsonl"), dpo_pairs, "DPO pairs"
    )

    total_v = sum(s["verified"] for s in stats.values())
    total_d = sum(s["dropped"] for s in stats.values())
    top_cats = sorted(
        stats.items(), key=lambda x: x[1]["total"], reverse=True
    )[:10]

    _write_dataset_card(
        os.path.join(output_dir, "README.md"),
        len(sft_pairs), len(dpo_pairs), total_v, total_d, top_cats,
    )

    stats_path = os.path.join(output_dir, "category_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sft_pairs": len(sft_pairs),
            "dpo_pairs": len(dpo_pairs),
            "total_verified": total_v,
            "total_dropped": total_d,
            "categories": stats,
        }, f, indent=2)
    logger.info(f"Wrote stats → {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Format RLAIF data for DPO training")
    parser.add_argument("--db", type=str, default="rlaif/data/training_data.db",
                        help="Path to training data SQLite DB")
    parser.add_argument("--output", type=str, default="rlaif/data",
                        help="Output directory for JSONL files")
    parser.add_argument("--max-dpo-pairs", type=int, default=10000,
                        help="Maximum DPO pairs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.db):
        logger.error(f"Database not found: {args.db}")
        logger.error("Run 'python -m rlaif.generate' first to create training data")
        return

    # Load data
    verified, dropped = load_training_data(args.db)

    if not verified:
        logger.error("No verified findings in database. Need more training data.")
        return

    # Build pairs
    sft_pairs = build_sft_pairs(verified)
    dpo_pairs = build_dpo_pairs(verified, dropped, max_pairs=args.max_dpo_pairs)
    stats = build_category_stats(verified, dropped)

    # Export
    export_for_huggingface(sft_pairs, dpo_pairs, args.output, stats)

    logger.info(f"\nDone! Files ready for upload to HuggingFace.")


if __name__ == "__main__":
    main()
