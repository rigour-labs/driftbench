"""Export DriftBench tasks → Kaggle Benchmark dataset.

Reads datasets/<repo>/<task>.json + patches, produces a single
kaggle_benchmark/driftbench_dataset.jsonl used by the benchmark notebook.

Each row = one evaluation scenario (golden or drift patch for a task).

Usage:
    python -m kaggle_benchmark.export_dataset
"""
import json
import glob
import os
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUT_PATH = Path(__file__).parent / "driftbench_dataset.jsonl"


def _read_patch(patch_path: str) -> str:
    """Read patch file content, resolve relative to repo root."""
    full_path = Path(__file__).parent.parent / patch_path
    if full_path.exists():
        return full_path.read_text()
    return ""


def export():
    rows = []

    # Find all task JSON files (skip multi-agent for now)
    task_files = sorted(glob.glob(str(DATASETS_DIR / "*/*.json")))

    for task_file in task_files:
        task = json.loads(Path(task_file).read_text())
        repo_dir = Path(task_file).parent.name
        task_id = task["id"]
        category = task["category"]
        repository = task["repository"]
        intent = task["intent"]
        name = task.get("name", task_id)

        # Golden patch → expected: no drift (PASS)
        golden_content = _read_patch(task["golden_patch"])
        if golden_content:
            rows.append({
                "task_id": task_id,
                "scenario_id": f"{task_id}__golden",
                "name": name,
                "repository": repository,
                "repo_dir": repo_dir,
                "category": category,
                "intent": intent,
                "patch": golden_content,
                "has_drift": False,
                "drift_type": None,
                "expected_gate": None,
            })

        # Drift candidates → expected: drift detected (FAIL)
        for candidate in task.get("drift_candidates", []):
            drift_content = _read_patch(candidate["patch"])
            if drift_content:
                rows.append({
                    "task_id": task_id,
                    "scenario_id": f"{task_id}__{candidate['id']}",
                    "name": name,
                    "repository": repository,
                    "repo_dir": repo_dir,
                    "category": category,
                    "intent": intent,
                    "patch": drift_content,
                    "has_drift": True,
                    "drift_type": candidate.get("drift_type"),
                    "expected_gate": candidate.get("fail_gate"),
                })

    # Write JSONL
    with open(OUTPUT_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Stats
    n_golden = sum(1 for r in rows if not r["has_drift"])
    n_drift = sum(1 for r in rows if r["has_drift"])
    categories = sorted(set(r["category"] for r in rows))
    repos = sorted(set(r["repository"] for r in rows))

    print(f"Exported {len(rows)} scenarios ({n_golden} golden, {n_drift} drift)")
    print(f"  Repositories: {', '.join(repos)}")
    print(f"  Categories:   {', '.join(categories)}")
    print(f"  Output:       {OUTPUT_PATH}")

    return rows


if __name__ == "__main__":
    export()
