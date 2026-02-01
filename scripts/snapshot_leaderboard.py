#!/usr/bin/env python3
"""
Snapshot leaderboard data from benchmark results.

Aggregates results from results/<model>/*.json and generates
a leaderboard data file for the web dashboard with repo-wise breakdown.
"""
import os
import json
import glob
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

DRIFTBENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(DRIFTBENCH_ROOT, "results")
DATASETS_DIR = os.path.join(DRIFTBENCH_ROOT, "datasets")
MODEL_CONFIG_PATH = os.path.join(DRIFTBENCH_ROOT, "model_config.json")
WEB_DATA_PATH = os.path.join(
    os.path.dirname(DRIFTBENCH_ROOT),
    "rigour-web/src/app/api/stats/data.json"
)

# Repository metadata - maps normalized name to metadata
REPO_METADATA = {
    "lodash": {"language": "javascript", "full_name": "lodash/lodash"},
    "django": {"language": "python", "full_name": "django/django"},
    "fastapi": {"language": "python", "full_name": "tiangolo/fastapi"},
    "flask": {"language": "python", "full_name": "pallets/flask"},
    "pydantic": {"language": "python", "full_name": "pydantic/pydantic"},
    "react": {"language": "javascript", "full_name": "facebook/react"},
    "next.js": {"language": "javascript", "full_name": "vercel/next.js"},
    "nextjs": {"language": "javascript", "full_name": "vercel/next.js"},
    "ui": {"language": "javascript", "full_name": "shadcn-ui/ui"},
    "shadcn-ui": {"language": "javascript", "full_name": "shadcn-ui/ui"},
    "query": {"language": "javascript", "full_name": "TanStack/query"},
    "tanstack-query": {"language": "javascript", "full_name": "TanStack/query"},
    "openai-python": {"language": "python", "full_name": "openai/openai-python"},
}

# Category descriptions
CATEGORIES = {
    "stale_drift": "Code uses deprecated/legacy patterns",
    "staleness_drift": "Code uses deprecated/legacy patterns",
    "security_drift": "Code introduces security vulnerabilities",
    "architecture_drift": "Code violates architectural boundaries",
    "pattern_drift": "Code deviates from established patterns",
    "logic_drift": "Code has logical inconsistencies",
}


def load_model_config() -> Dict[str, Any]:
    """Load model configuration for display names."""
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH, 'r') as f:
            return json.load(f).get("model_config", {})
    return {}


def get_repo_from_result(data: Dict) -> str:
    """Extract repository name from result data."""
    # Try repository field first
    repo_full = data.get("repository", "")
    if "/" in repo_full:
        return repo_full.split("/")[1]

    # Fallback: extract from task_id
    task_id = data.get("task_id", "")
    for repo in REPO_METADATA:
        if repo in task_id.lower():
            return repo

    return "unknown"


def get_category_from_result(data: Dict) -> str:
    """Extract category from result data."""
    return data.get("category", "unknown")


def scan_datasets() -> Dict[str, int]:
    """Scan datasets directory to count tasks per repo."""
    repo_tasks = {}
    for task_file in glob.glob(os.path.join(DATASETS_DIR, "**/*.json"), recursive=True):
        try:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            repo = get_repo_from_result(task_data)
            repo_tasks[repo] = repo_tasks.get(repo, 0) + 1
        except (json.JSONDecodeError, IOError):
            continue
    return repo_tasks


def calculate_model_stats(model_dir: str) -> Optional[Dict]:
    """Calculate statistics for a single model from its result files."""
    result_files = glob.glob(os.path.join(model_dir, "*.json"))
    if not result_files:
        return None

    total = 0
    passed = 0
    failed = 0
    errors = 0
    correct = 0

    # Breakdown tracking
    by_repo: Dict[str, Dict[str, int]] = {}
    by_language: Dict[str, Dict[str, int]] = {}
    by_category: Dict[str, Dict[str, int]] = {}

    for filepath in result_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                continue

            total += 1

            # Extract dimensions
            repo = get_repo_from_result(data)
            category = get_category_from_result(data)
            language = REPO_METADATA.get(repo, {}).get("language", "unknown")

            # Initialize breakdown dicts
            if repo not in by_repo:
                by_repo[repo] = {"passed": 0, "failed": 0, "total": 0}
            if language not in by_language:
                by_language[language] = {"passed": 0, "failed": 0, "total": 0}
            if category not in by_category:
                by_category[category] = {"passed": 0, "failed": 0, "total": 0}

            by_repo[repo]["total"] += 1
            by_language[language]["total"] += 1
            by_category[category]["total"] += 1

            # Check for errors
            if data.get("error"):
                errors += 1
                continue

            # Check pass/fail
            if data.get("passed"):
                passed += 1
                by_repo[repo]["passed"] += 1
                by_language[language]["passed"] += 1
                by_category[category]["passed"] += 1
            else:
                failed += 1
                by_repo[repo]["failed"] += 1
                by_language[language]["failed"] += 1
                by_category[category]["failed"] += 1

            # Check correctness
            if data.get("correct"):
                correct += 1

        except (json.JSONDecodeError, IOError) as e:
            print(f"    ⚠️  Skipping {filepath}: {e}")
            continue

    if total == 0:
        return None

    return {
        "pass_rate": round((passed / total) * 100, 1) if total > 0 else 0.0,
        "drift_detection_rate": round((failed / total) * 100, 1) if total > 0 else 0.0,
        "accuracy": round((correct / total) * 100, 1) if total > 0 else 0.0,
        "tasks_run": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "correct": correct,
        "breakdown": {
            "by_repo": by_repo,
            "by_language": by_language,
            "by_category": by_category
        }
    }


def calculate_stats() -> Dict[str, Any]:
    """Calculate statistics for all models and generate full leaderboard data."""
    model_config = load_model_config()
    repo_tasks = scan_datasets()

    # Build repositories info
    repositories = {}
    for repo, count in repo_tasks.items():
        meta = REPO_METADATA.get(repo, {"language": "unknown"})
        repositories[repo] = {
            "language": meta.get("language", "unknown"),
            "tasks": count,
            "full_name": meta.get("full_name", f"unknown/{repo}")
        }

    leaderboard = []

    if not os.path.exists(RESULTS_DIR):
        print(f"⚠️  Results directory not found: {RESULTS_DIR}")
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "total_tasks": sum(repo_tasks.values()),
            "repositories": repositories,
            "categories": CATEGORIES,
            "leaderboard": []
        }

    # Skip system directories
    skip_dirs = {"patches", "studio", "__pycache__", ".git", "leaderboard.json"}

    model_dirs = [
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d)) and d not in skip_dirs
    ]

    for slug in model_dirs:
        model_dir = os.path.join(RESULTS_DIR, slug)
        stats = calculate_model_stats(model_dir)

        if not stats:
            continue

        # Extract model identifier from slug (e.g., "anthropic_claude-opus-4-5" -> "anthropic/claude-opus-4-5")
        model_id = slug.replace("_", "/", 1)
        config = model_config.get(model_id, {})

        # Parse provider and model name
        parts = slug.split("_", 1)
        provider = parts[0].capitalize() if parts else "Unknown"

        leaderboard.append({
            "model": model_id,
            "slug": slug,
            "display_name": config.get("display_name", slug.replace("_", " ").title()),
            "provider": provider,
            "pass_rate": stats["pass_rate"],
            "drift_detection_rate": stats["drift_detection_rate"],
            "accuracy": stats["accuracy"],
            "tasks_run": stats["tasks_run"],
            "breakdown": stats["breakdown"],
            "verified_at": datetime.now().strftime("%Y-%m-%d"),
            "status": "verified" if stats["tasks_run"] >= 10 else "partial"
        })

    # Sort by pass rate (higher is better) and assign ranks
    leaderboard.sort(key=lambda x: x["pass_rate"], reverse=True)
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0",
        "total_tasks": sum(repo_tasks.values()),
        "repositories": repositories,
        "categories": CATEGORIES,
        "leaderboard": leaderboard
    }


def main():
    """Generate and save leaderboard snapshot."""
    print(f"📸 Snapshotting leaderboard from {RESULTS_DIR}...")

    data = calculate_stats()

    if not data["leaderboard"]:
        print("⚠️  No results found. Check that benchmarks have been run.")
        print(f"   Expected results in: {RESULTS_DIR}/<model>/*.json")
    else:
        print(f"✅ Found {len(data['leaderboard'])} model(s):")
        for model in data["leaderboard"]:
            print(f"   #{model['rank']} {model['display_name']}: {model['pass_rate']}% pass rate ({model['tasks_run']} tasks)")

            # Show repo breakdown
            for repo, stats in model["breakdown"]["by_repo"].items():
                if stats["total"] > 0:
                    repo_rate = round(stats["passed"] / stats["total"] * 100, 1)
                    print(f"      └─ {repo}: {repo_rate}% ({stats['passed']}/{stats['total']})")

    # Ensure output directory exists
    output_dir = os.path.dirname(WEB_DATA_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write data
    with open(WEB_DATA_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n📁 Saved to: {WEB_DATA_PATH}")

    # Also save locally for reference
    local_path = os.path.join(DRIFTBENCH_ROOT, "results", "leaderboard.json")
    with open(local_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"📁 Local copy: {local_path}")


if __name__ == "__main__":
    main()
