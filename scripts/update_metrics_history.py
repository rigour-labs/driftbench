#!/usr/bin/env python3
"""
Update metrics history on HuggingFace with this run's category stats.

Appends current run's verification rates and teacher scores to a cumulative
metrics_history.json stored in the HF dataset repo. Also computes per-teacher
per-category verification rates and saves teacher_scores.json locally.

Usage:
    python scripts/update_metrics_history.py \
        --stats rlaif/data/category_stats.json \
        --db rlaif/data/training_data.db \
        --run-number 5 \
        --hf-repo rigour-labs/rigour-rlaif-data
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone


def load_category_stats(stats_path: str) -> dict:
    """Load category_stats.json from current run."""
    if not os.path.exists(stats_path):
        return {}
    with open(stats_path) as f:
        return json.load(f)


def compute_teacher_scores(db_path: str) -> dict:
    """Compute per-teacher per-category verification rates from SQLite."""
    if not os.path.exists(db_path):
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT teacher_model, category,
               SUM(CASE WHEN verified = 1 THEN 1 ELSE 0 END) AS verified_count,
               COUNT(*) AS total_count
        FROM training_data
        GROUP BY teacher_model, category
        ORDER BY teacher_model, category
    """

    scores = {}
    for row in conn.execute(query):
        teacher = row["teacher_model"]
        category = row["category"]
        total = row["total_count"]
        verified = row["verified_count"]

        if teacher not in scores:
            scores[teacher] = {}
        scores[teacher][category] = round(verified / total, 3) if total > 0 else 0.0

    conn.close()
    return scores


def download_history(hf_repo: str, token: str) -> dict:
    """Download existing metrics_history.json from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            hf_repo, "metrics_history.json",
            repo_type="dataset", token=token,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        # First run or file doesn't exist yet
        return {"runs": []}


def upload_history(hf_repo: str, token: str, history: dict):
    """Upload updated metrics_history.json to HuggingFace."""
    from huggingface_hub import HfApi
    import tempfile

    api = HfApi(token=token)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(history, f, indent=2)
        tmp_path = f.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="metrics_history.json",
        repo_id=hf_repo,
        repo_type="dataset",
    )
    os.unlink(tmp_path)
    print(f"Uploaded metrics_history.json to {hf_repo}")


def main():
    parser = argparse.ArgumentParser(description="Update metrics history")
    parser.add_argument("--stats", required=True, help="Path to category_stats.json")
    parser.add_argument("--db", default="", help="Path to training_data.db for teacher scores")
    parser.add_argument("--run-number", required=True, help="Pipeline run number")
    parser.add_argument("--hf-repo", required=True, help="HuggingFace dataset repo ID")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")

    # Load current run stats
    category_stats = load_category_stats(args.stats)
    if not category_stats:
        print("No category_stats.json found — skipping metrics history update")
        return

    # Compute teacher scores
    teacher_scores = compute_teacher_scores(args.db) if args.db else {}

    # Save teacher_scores.json locally (for artifact upload)
    if teacher_scores:
        scores_path = os.path.join(os.path.dirname(args.stats), "teacher_scores.json")
        with open(scores_path, "w") as f:
            json.dump(teacher_scores, f, indent=2)
        print(f"Teacher scores saved: {scores_path}")
        for teacher, cats in teacher_scores.items():
            avg_rate = sum(cats.values()) / len(cats) if cats else 0
            print(f"  {teacher}: avg verification rate = {avg_rate:.1%}")

    # Build run entry
    categories = {}
    total_sft, total_dpo = 0, 0
    for cat, stats in category_stats.items():
        if isinstance(stats, dict):
            categories[cat] = {
                "verified": stats.get("verified", 0),
                "dropped": stats.get("dropped", 0),
                "rate": stats.get("verification_rate", 0),
            }
            total_sft += stats.get("verified", 0)
            total_dpo += stats.get("dropped", 0)

    run_entry = {
        "run_number": int(args.run_number),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_verified": total_sft,
        "total_dropped": total_dpo,
        "categories": categories,
        "teacher_scores": teacher_scores,
    }

    # Download existing history, append, upload
    if token:
        history = download_history(args.hf_repo, token)
        history["runs"].append(run_entry)
        # Keep last 52 weeks (1 year)
        history["runs"] = history["runs"][-52:]
        upload_history(args.hf_repo, token, history)
        print(f"Metrics history updated: {len(history['runs'])} runs tracked")
    else:
        print("No HF_TOKEN — saving metrics history locally only")
        local_path = os.path.join(os.path.dirname(args.stats), "metrics_history.json")
        history = {"runs": [run_entry]}
        with open(local_path, "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
