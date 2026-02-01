#!/bin/bash

# DriftBench Evaluation Runner
# Usage: ./run_eval.sh --task datasets/logic_drift/fastapi_auth.json

set -e

echo "🏎️  Initializing DriftBench..."

# Ensure python environment is ready
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r runner/requirements.txt
else
    source venv/bin/activate
fi

# Run the engine or harness based on input
if [[ "$1" == "--model" ]] || [[ "$1" == "--tool" ]]; then
    # Mapping --tool to --model for README compatibility
    SHIFTED_ARGS=("${@/--tool/--model}")
    python3 runner/harness.py "${SHIFTED_ARGS[@]}"
else
    python3 runner/engine.py "$@"
fi
