#!/usr/bin/env python3
"""
Snapshot leaderboard data from benchmark results.

Two data sources (merged automatically):
1. Local results: results/<model>/*.json  (from run_full_benchmark.py)
2. Batch results: results/leaderboard.json (from run_leaderboard_batch.py)

Rich metrics: per-category accuracy, ECE calibration, overconfident errors,
per-repo and per-language breakdown.
"""
import os
import json
import glob
import math
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

# Drift category descriptions (task-level)
CATEGORIES = {
    "stale_drift": "Code uses deprecated/legacy patterns",
    "staleness_drift": "Code uses deprecated/legacy patterns",
    "security_drift": "Code introduces security vulnerabilities",
    "architecture_drift": "Code violates architectural boundaries",
    "pattern_drift": "Code deviates from established patterns",
    "logic_drift": "Code has logical inconsistencies",
    "standard_drift": "Code violates naming/formatting standards",
}
BATCH_LEADERBOARD_PATH = os.path.join(RESULTS_DIR, "leaderboard.json")

# Rigour gate names (for per-gate scoring)
GATE_NAMES = [
    "file-size", "content-check", "structure-check", "ast-analysis",
    "dependency-guardian", "safety-rail", "coverage-guard", "staleness",
    "pattern-index", "file-guard", "context-drift",
]


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


def load_task_metadata() -> Dict[str, Dict]:
    """Load metadata from all task files for fallback lookups."""
    task_metadata = {}
    for task_file in glob.glob(os.path.join(DATASETS_DIR, "**/*.json"), recursive=True):
        try:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            task_id = task_data.get("id")
            if task_id:
                task_metadata[task_id] = {
                    "category": task_data.get("category", "unknown"),
                    "repository": task_data.get("repository", ""),
                }
        except (json.JSONDecodeError, IOError):
            continue
    return task_metadata


# Global task metadata cache
_TASK_METADATA: Optional[Dict[str, Dict]] = None


def get_task_metadata() -> Dict[str, Dict]:
    """Get cached task metadata."""
    global _TASK_METADATA
    if _TASK_METADATA is None:
        _TASK_METADATA = load_task_metadata()
    return _TASK_METADATA


def get_category_from_result(data: Dict) -> str:
    """Extract category from result data, with fallback to task file."""
    # Try result file first
    category = data.get("category")
    if category:
        return category

    # Fallback: lookup from task metadata using task_id
    task_id = data.get("task_id")
    if task_id:
        metadata = get_task_metadata().get(task_id, {})
        return metadata.get("category", "unknown")

    return "unknown"


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


def is_only_structure_check_failure(data: Dict) -> bool:
    """
    Check if the failure is ONLY due to structure-check (false positive).

    Structure-check failures for missing docs files (SPEC.md, ARCH.md, etc.)
    are false positives on OSS repos that don't have Rigour's default required files.
    """
    llm_result = data.get("llm_result", {})
    report = llm_result.get("report", {})

    # If overall status is PASS, not a structure-check-only failure
    if report.get("status") == "PASS":
        return False

    summary = report.get("summary", {})
    failures = report.get("failures", [])

    # Check if structure-check is the ONLY failure
    failed_gates = [gate for gate, status in summary.items() if status == "FAIL"]

    if failed_gates == ["structure-check"]:
        # Double-check that failures list only has structure-check
        failure_ids = [f.get("id") for f in failures]
        if failure_ids == ["structure-check"]:
            return True

    return False


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
    false_positives_excluded = 0

    # Confidence calibration tracking
    confidence_bins: Dict[int, Dict[str, int]] = {}  # bin_idx -> {correct, total}
    overconfident_errors = 0  # high confidence (>=0.8) + wrong
    underconfident_correct = 0  # low confidence (<0.5) + correct
    all_confidences: List[float] = []

    # Breakdown tracking
    by_repo: Dict[str, Dict[str, int]] = {}
    by_language: Dict[str, Dict[str, int]] = {}
    by_category: Dict[str, Dict[str, int]] = {}
    by_gate: Dict[str, Dict[str, int]] = {}  # Per-gate scoring
    finding_categories: Dict[str, int] = {}  # Finding-level category counts

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

            # Check pass/fail (with false positive correction)
            actual_passed = data.get("passed")

            # Correct for structure-check false positives
            if not actual_passed and is_only_structure_check_failure(data):
                actual_passed = True
                false_positives_excluded += 1

            if actual_passed:
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
            is_correct = bool(data.get("correct"))
            if is_correct:
                correct += 1

            # ── Confidence calibration ──
            confidence = data.get("confidence")
            if confidence is not None:
                conf = float(confidence)
                all_confidences.append(conf)

                # Bin into 10 buckets (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
                bin_idx = min(int(conf * 10), 9)
                if bin_idx not in confidence_bins:
                    confidence_bins[bin_idx] = {"correct": 0, "total": 0}
                confidence_bins[bin_idx]["total"] += 1
                if is_correct:
                    confidence_bins[bin_idx]["correct"] += 1

                # Track overconfident errors and underconfident correct
                if conf >= 0.8 and not is_correct:
                    overconfident_errors += 1
                if conf < 0.5 and is_correct:
                    underconfident_correct += 1

            # ── Per-gate scoring ──
            # Extract which gates passed/failed from the LLM result report
            llm_report = data.get("llm_result", {}).get("report", {})
            summary = llm_report.get("summary", {})
            for gate_name, gate_status in summary.items():
                if gate_name not in by_gate:
                    by_gate[gate_name] = {"passed": 0, "failed": 0, "total": 0}
                by_gate[gate_name]["total"] += 1
                if gate_status == "PASS":
                    by_gate[gate_name]["passed"] += 1
                elif gate_status == "FAIL":
                    by_gate[gate_name]["failed"] += 1

            # ── Per-finding-category scoring ──
            # Count individual finding categories (e.g., STALENESS_NO_VAR, etc.)
            failures = llm_report.get("failures", [])
            for failure in failures:
                fid = failure.get("id", "unknown")
                finding_categories[fid] = finding_categories.get(fid, 0) + 1

        except (json.JSONDecodeError, IOError) as e:
            print(f"    ⚠️  Skipping {filepath}: {e}")
            continue

    if total == 0:
        return None

    # Calculate rates based on tasks that actually ran (excluding errors)
    tasks_completed = total - errors

    # ── ECE (Expected Calibration Error) ──
    # Weighted average of |accuracy_in_bin - avg_confidence_in_bin| across bins
    ece = 0.0
    n_with_conf = sum(b["total"] for b in confidence_bins.values())
    if n_with_conf > 0:
        for bin_idx, bucket in confidence_bins.items():
            if bucket["total"] == 0:
                continue
            bin_accuracy = bucket["correct"] / bucket["total"]
            bin_confidence = (bin_idx + 0.5) / 10  # midpoint of bin
            ece += (bucket["total"] / n_with_conf) * abs(bin_accuracy - bin_confidence)

    return {
        "pass_rate": round((passed / tasks_completed) * 100, 1) if tasks_completed > 0 else 0.0,
        "drift_detection_rate": round((failed / tasks_completed) * 100, 1) if tasks_completed > 0 else 0.0,
        "error_rate": round((errors / total) * 100, 1) if total > 0 else 0.0,
        "accuracy": round((correct / tasks_completed) * 100, 1) if tasks_completed > 0 else 0.0,
        "tasks_run": total,
        "tasks_completed": tasks_completed,
        "passed": passed,
        "failed": failed,  # Real drift detected by Rigour
        "errors": errors,
        "correct": correct,
        "false_positives_excluded": false_positives_excluded,
        # Confidence calibration metrics
        "calibration_ece": round(ece, 4),
        "overconfident_errors": overconfident_errors,
        "underconfident_correct": underconfident_correct,
        "avg_confidence": round(sum(all_confidences) / len(all_confidences), 3) if all_confidences else None,
        "breakdown": {
            "by_repo": by_repo,
            "by_language": by_language,
            "by_category": by_category,
            "by_gate": by_gate,
            "by_finding": dict(sorted(finding_categories.items(), key=lambda x: x[1], reverse=True)),
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
            "error_rate": stats.get("error_rate", 0.0),
            "accuracy": stats["accuracy"],
            "tasks_run": stats["tasks_run"],
            "tasks_completed": stats.get("tasks_completed", stats["tasks_run"]),
            "tasks_total": sum(repo_tasks.values()),
            "errors": stats.get("errors", 0),
            "false_positives_excluded": stats.get("false_positives_excluded", 0),
            # Rich calibration metrics
            "calibration_ece": stats.get("calibration_ece", 0.0),
            "overconfident_errors": stats.get("overconfident_errors", 0),
            "underconfident_correct": stats.get("underconfident_correct", 0),
            "avg_confidence": stats.get("avg_confidence"),
            "breakdown": stats["breakdown"],
            "verified_at": datetime.now().strftime("%Y-%m-%d"),
            "status": "verified" if stats.get("tasks_completed", stats["tasks_run"]) >= 20 else "partial",
            "source": "local",
        })

    # ── Merge batch leaderboard results ──
    # If run_leaderboard_batch.py has produced results, merge them in
    if os.path.exists(BATCH_LEADERBOARD_PATH):
        try:
            with open(BATCH_LEADERBOARD_PATH, 'r') as f:
                batch_data = json.load(f)

            local_models = {e["model"] for e in leaderboard}

            for batch_model in batch_data.get("models", []):
                model_id = batch_model.get("model", "")
                if model_id in local_models:
                    # Update existing entry with batch metrics where richer
                    for entry in leaderboard:
                        if entry["model"] == model_id:
                            entry["calibration_ece"] = batch_model.get("calibration_ece", entry.get("calibration_ece", 0))
                            entry["overconfident_errors"] = batch_model.get("overconfident_errors", entry.get("overconfident_errors", 0))
                            if "by_category" in batch_model:
                                entry["breakdown"]["by_category_batch"] = batch_model["by_category"]
                            break
                else:
                    # New model from batch — add it
                    provider = batch_model.get("provider", "unknown").capitalize()
                    by_cat = batch_model.get("by_category", {})
                    overall_acc = batch_model.get("overall_accuracy", 0)

                    leaderboard.append({
                        "model": model_id,
                        "slug": model_id.replace("/", "_"),
                        "display_name": batch_model.get("display_name", model_id),
                        "provider": provider,
                        "pass_rate": 0.0,  # N/A for batch (binary drift detection)
                        "drift_detection_rate": 0.0,
                        "error_rate": 0.0,
                        "accuracy": round(overall_acc * 100, 1),
                        "tasks_run": batch_model.get("scenarios_evaluated", 0),
                        "tasks_completed": batch_model.get("scenarios_evaluated", 0),
                        "tasks_total": sum(repo_tasks.values()),
                        "errors": 0,
                        "false_positives_excluded": 0,
                        "calibration_ece": batch_model.get("calibration_ece", 0),
                        "overconfident_errors": batch_model.get("overconfident_errors", 0),
                        "underconfident_correct": batch_model.get("underconfident_correct", 0),
                        "avg_confidence": batch_model.get("avg_confidence"),
                        "breakdown": {
                            "by_repo": batch_model.get("by_repository", {}),
                            "by_language": {},
                            "by_category": {
                                cat: {"passed": d.get("correct", 0), "failed": d.get("total", 0) - d.get("correct", 0), "total": d.get("total", 0)}
                                for cat, d in by_cat.items()
                            },
                            "by_gate": {},
                            "by_finding": {},
                        },
                        "verified_at": datetime.now().strftime("%Y-%m-%d"),
                        "status": "verified" if batch_model.get("scenarios_evaluated", 0) >= 20 else "partial",
                        "source": "batch",
                    })

            print(f"  📦 Merged {len(batch_data.get('models', []))} models from batch leaderboard")
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠️  Could not read batch leaderboard: {e}")

    # Sort by: 1) Accuracy (higher is better), 2) Tasks completed
    leaderboard.sort(key=lambda x: (x.get("accuracy", 0), x.get("tasks_completed", x["tasks_run"])), reverse=True)
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "2.0",
        "total_tasks": sum(repo_tasks.values()),
        "total_scenarios": 54,
        "repositories": repositories,
        "categories": CATEGORIES,
        "methodology": {
            "false_positive_explanation": (
                "Structure-check failures are excluded when they are the ONLY failure reason. "
                "These occur because Rigour's default configuration requires documentation files "
                "(SPEC.md, ARCH.md, DECISIONS.md, TASKS.md) that don't exist in most open-source "
                "repositories. This is a configuration issue, not a model drift detection failure."
            ),
            "scoring": (
                "Accuracy = correct predictions / total scenarios. "
                "ECE = Expected Calibration Error (lower is better). "
                "Overconfident errors = high confidence (>=0.8) but wrong answer. "
                "Ranking prioritizes accuracy across all 54 scenarios."
            ),
            "error_explanation": (
                "Errors indicate infrastructure failures (e.g., model timeout, git clone failed) "
                "where the model did not produce output to evaluate."
            )
        },
        "leaderboard": leaderboard
    }


def main():
    """Generate and save leaderboard snapshot."""
    print(f"📸 Snapshotting leaderboard from {RESULTS_DIR}...")

    data = calculate_stats()

    if not data["leaderboard"]:
        print("⚠️  No results found. Check that benchmarks have been run.")
        print(f"   Expected results in: {RESULTS_DIR}/<model>/*.json")
        print(f"   Or batch results in: {BATCH_LEADERBOARD_PATH}")
    else:
        print(f"✅ Found {len(data['leaderboard'])} model(s):")
        for model in data["leaderboard"]:
            ece = model.get("calibration_ece", 0)
            oc = model.get("overconfident_errors", 0)
            src = model.get("source", "local")
            print(f"   #{model['rank']} {model['display_name']}: "
                  f"{model['accuracy']}% accuracy "
                  f"(ECE: {ece:.3f}, overconfident: {oc}) [{src}]")

            # Show category breakdown
            by_cat = model["breakdown"].get("by_category", {})
            for cat, stats in sorted(by_cat.items()):
                if stats["total"] > 0:
                    cat_rate = round(stats["passed"] / stats["total"] * 100, 1)
                    print(f"      {cat}: {cat_rate}% ({stats['passed']}/{stats['total']})")

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
