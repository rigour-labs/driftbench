#!/usr/bin/env python3
"""
Snapshot leaderboard data from benchmark results.

Aggregates results from results/<model>/*.json and generates
a leaderboard data file for the web dashboard.
"""
import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Optional

DRIFTBENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(DRIFTBENCH_ROOT, "results")
WEB_DATA_PATH = os.path.join(
    os.path.dirname(DRIFTBENCH_ROOT),
    "rigour-web/src/app/api/stats/data.json"
)


def format_model_name(slug: str) -> str:
    """
    Format model slug to display name.

    Examples:
        'anthropic_claude-opus-4-5' -> 'Anthropic / Claude Opus 4.5'
        'openai_gpt-5.1-codex' -> 'Openai / Gpt 5.1 Codex'
    """
    parts = slug.split('_', 1)
    if len(parts) == 2:
        provider = parts[0].capitalize()
        model_name = parts[1].replace("-", " ").replace(".", ".").title()
        return f"{provider} / {model_name}"
    return slug


def calculate_model_stats(model_dir: str) -> Optional[Dict]:
    """
    Calculate statistics for a single model from its result files.

    Args:
        model_dir: Path to model results directory

    Returns:
        Dict with model stats or None if no valid results
    """
    result_files = glob.glob(os.path.join(model_dir, "*.json"))
    if not result_files:
        return None

    total = 0
    passed = 0  # No drift detected (good for LLM)
    failed = 0  # Drift detected (bad for LLM)
    errors = 0
    correct = 0  # Matches golden baseline

    for filepath in result_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                continue

            total += 1
            
            # Extract repository name (e.g. "lodash/lodash" -> "lodash")
            repo_full = data.get("repository", "unknown")
            repo_name = repo_full.split("/")[1] if "/" in repo_full else repo_full
            
            # Initialize repo stats if needed
            if repo_name not in repo_stats:
                repo_stats[repo_name] = {"total": 0, "passed": 0, "failed": 0, "errors": 0}

            repo_stats[repo_name]["total"] += 1

            # Check for errors
            if data.get("error"):
                errors += 1
                repo_stats[repo_name]["errors"] += 1
                # Treat error as failed for simplicity or track separately
                # For now just counting
                continue

            # Check pass/fail (drift detection)
            if data.get("passed"):
                passed += 1
                repo_stats[repo_name]["passed"] += 1
            else:
                failed += 1
                repo_stats[repo_name]["failed"] += 1

            # Check correctness (matches golden)
            if data.get("correct"):
                correct += 1

        except (json.JSONDecodeError, IOError) as e:
            print(f"    ⚠️  Skipping {filepath}: {e}")
            continue

    if total == 0:
        return None

    # Calculate metrics
    pass_rate = round((passed / total) * 100, 1) if total > 0 else 0.0
    accuracy = round((correct / total) * 100, 1) if total > 0 else 0.0
    error_rate = round((errors / total) * 100, 1) if total > 0 else 0.0

    return {
        "pass_rate": pass_rate,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "tasks_run": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "correct": correct,
        "repo_stats": repo_stats
    }


def calculate_stats() -> List[Dict]:
    """
    Calculate statistics for all models.

    Returns:
        List of model stat dictionaries, sorted by pass rate
    """
    stats = []

    if not os.path.exists(RESULTS_DIR):
        print(f"⚠️  Results directory not found: {RESULTS_DIR}")
        return []

    # Skip system directories
    skip_dirs = {"patches", "studio", "__pycache__", ".git"}

    model_dirs = [
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d)) and d not in skip_dirs
    ]

    for slug in model_dirs:
        model_dir = os.path.join(RESULTS_DIR, slug)
        model_stats = calculate_model_stats(model_dir)

        if not model_stats:
            continue

        stats.append({
            "name": format_model_name(slug),
            "slug": slug,
            "ddr": 100.0 - model_stats["pass_rate"],  # DDR is inverse of pass rate
            "pass_rate": model_stats["pass_rate"],
            "accuracy": model_stats["accuracy"],
            "fpr": model_stats["error_rate"],
            "tasks_run": model_stats["tasks_run"],
            "repos": model_stats["repo_stats"],  # Add breakdown
            "status": "Verified",
            "cost": 0.0,
            "latency": "0.0s",
            "verified_at": datetime.now().strftime("%Y-%m-%d")
        })

    # Sort by pass rate (higher is better)
    stats.sort(key=lambda x: x["pass_rate"], reverse=True)
    return stats


def main():
    """Generate and save leaderboard snapshot."""
    print(f"📸 Snapshotting leaderboard from {RESULTS_DIR}...")

    data = calculate_stats()

    if not data:
        print("⚠️  No results found. Check that benchmarks have been run.")
        print(f"   Expected results in: {RESULTS_DIR}/<model>/*.json")
        # Create empty file to avoid errors
        data = []
    else:
        print(f"✅ Found {len(data)} model(s):")
        for model in data:
            print(f"   - {model['name']}: {model['pass_rate']}% pass rate ({model['tasks_run']} tasks)")

    # Ensure output directory exists
    output_dir = os.path.dirname(WEB_DATA_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write data
    with open(WEB_DATA_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n📁 Saved to: {WEB_DATA_PATH}")


if __name__ == "__main__":
    main()
