#!/usr/bin/env bash
# publish.sh — Upload DriftBench to Kaggle as a Community Benchmark.
#
# Prerequisites:
#   pip install kaggle kaggle-benchmarks
#   ~/.kaggle/kaggle.json configured
#
# Steps:
#   1. Export DriftBench tasks → JSONL dataset
#   2. Upload dataset to Kaggle Datasets
#   3. Push benchmark notebook to Kaggle Kernels
#
# Usage:
#   cd driftbench/
#   bash kaggle_benchmark/publish.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== DriftBench Kaggle Benchmark Publisher ==="
echo ""

# ── Step 1: Export dataset ──
echo "[1/3] Exporting DriftBench scenarios to JSONL..."
cd "$REPO_DIR"
python -m kaggle_benchmark.export_dataset
echo ""

DATASET_FILE="$SCRIPT_DIR/driftbench_dataset.jsonl"
SCENARIOS=$(wc -l < "$DATASET_FILE" | tr -d ' ')
echo "  → $SCENARIOS scenarios exported"
echo ""

# ── Step 2: Upload dataset to Kaggle ──
echo "[2/3] Uploading dataset to Kaggle..."

# Create a temporary upload directory with just the data + metadata
UPLOAD_DIR=$(mktemp -d)
cp "$DATASET_FILE" "$UPLOAD_DIR/"
cp "$SCRIPT_DIR/dataset-metadata.json" "$UPLOAD_DIR/"

cd "$UPLOAD_DIR"

# Create or update the dataset
if kaggle datasets status rigourlabs/driftbench-scenarios 2>/dev/null; then
    echo "  Dataset exists — creating new version..."
    kaggle datasets version -p . -m "Update: $SCENARIOS scenarios" --dir-mode zip
else
    echo "  Creating new dataset..."
    kaggle datasets create -p . --dir-mode zip
fi

# Update dataset metadata (subtitle, description, keywords)
echo "  Updating dataset metadata..."
kaggle datasets metadata -p . rigourlabs/driftbench-scenarios 2>/dev/null || true

rm -rf "$UPLOAD_DIR"
echo "  → Dataset uploaded: https://www.kaggle.com/datasets/rigourlabs/driftbench-scenarios"
echo ""

# ── Step 3: Push benchmark notebook ──
echo "[3/3] Pushing benchmark notebook to Kaggle..."
cd "$SCRIPT_DIR"
kaggle kernels push -p .
echo "  → Notebook pushed: https://www.kaggle.com/code/rigourlabs/driftbench-code-drift-detection"
echo ""

echo "=== Done! ==="
echo ""
echo "Next steps:"
echo "  1. Go to https://www.kaggle.com/code/rigourlabs/driftbench-code-drift-detection-benchmark"
echo "  2. Run the notebook to generate initial leaderboard results"
echo "  3. Visit https://www.kaggle.com/benchmarks to see the DriftBench leaderboard"
echo ""
echo "To run locally first:"
echo "  python -m kaggle_benchmark.driftbench_tasks"
