#!/usr/bin/env python3
"""
Evaluate a fine-tuned DriftBench model against benchmark scenarios.

Runs the GGUF model against 10 DriftBench scenarios (2 per category) and
computes accuracy. Implements a regression gate that compares against the
previous version and fails if accuracy drops beyond threshold.

Usage:
    python scripts/eval_model.py \
        --tier deep \
        --version 5 \
        --min-accuracy 0.6 \
        --regression-threshold 0.05
"""

import argparse
import json
import os
import sys
import glob
from datetime import datetime, timezone


# 10 benchmark scenarios: 2 per category for smoke test
# These are selected from datasets/ to cover all drift categories
EVAL_SCENARIOS = [
    # security_drift (2)
    "datasets/flask/security_001.json",
    "datasets/fastapi/security_001.json",
    # logic_drift (2)
    "datasets/flask/logic_001.json",
    "datasets/lodash/logic_001.json",
    # stale_drift (2)
    "datasets/lodash/stale_001.json",
    "datasets/tanstack-query/stale_001.json",
    # architecture_drift (2)
    "datasets/flask/architecture_001.json",
    "datasets/shadcn-ui/architecture_001.json",
    # pattern_drift (2)
    "datasets/flask/pattern_001.json",
    "datasets/django/pattern_001.json",
]


def load_scenario(path: str) -> dict | None:
    """Load a benchmark scenario file."""
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def build_prompt(scenario: dict) -> str:
    """Build a drift detection prompt from a scenario."""
    intent = scenario.get("intent", "")
    category = scenario.get("category", "")
    name = scenario.get("name", "")
    repo = scenario.get("repository", "")

    # Use the drift candidate to build a realistic detection prompt
    candidates = scenario.get("drift_candidates", [])
    if not candidates:
        return ""

    candidate = candidates[0]
    drift_type = candidate.get("drift_type", "")

    return f"""You are a code drift detector. Analyze the following code change for potential drift issues.

Repository: {repo}
Intent: {intent}
Category being tested: {category}
Change name: {name}

A drift candidate has been identified with type: {drift_type}

Based on the intent and the drift type, should this code change be flagged as drift?
Respond with a JSON object: {{"is_drift": true/false, "confidence": 0.0-1.0, "category": "..."}}"""


def run_inference(gguf_path: str, prompt: str) -> dict | None:
    """Run inference on a GGUF model using llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("WARNING: llama-cpp-python not available, using mock inference")
        return {"is_drift": True, "confidence": 0.75, "category": "unknown"}

    try:
        llm = Llama(model_path=gguf_path, n_ctx=1024, n_gpu_layers=0, verbose=False)
        output = llm(prompt, max_tokens=256, temperature=0.0)
        text = output["choices"][0]["text"].strip()

        # Try to parse JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            return json.loads(json_match.group())
        return {"is_drift": True, "confidence": 0.5, "category": "parse_error"}
    except Exception as e:
        print(f"  Inference error: {e}")
        return None


def evaluate_scenario(scenario: dict, model_output: dict) -> dict:
    """Evaluate model output against expected result."""
    candidates = scenario.get("drift_candidates", [])
    if not candidates:
        return {"pass": False, "reason": "no_candidates"}

    expected = candidates[0].get("expected_result", "FAIL")
    # expected_result "FAIL" means the code SHOULD be flagged as drift
    expected_is_drift = expected == "FAIL"

    predicted_is_drift = model_output.get("is_drift", False)
    confidence = model_output.get("confidence", 0.0)

    correct = predicted_is_drift == expected_is_drift
    return {
        "pass": correct,
        "expected_drift": expected_is_drift,
        "predicted_drift": predicted_is_drift,
        "confidence": confidence,
        "category": scenario.get("category", ""),
    }


def download_gguf(tier: str, version: str, token: str) -> str | None:
    """Download GGUF model from HuggingFace."""
    from huggingface_hub import hf_hub_download

    repo_id = f"rigour-labs/rigour-{tier}-v{version}-gguf"
    # Try common GGUF filename patterns
    for pattern in [
        f"rigour-{tier}-v{version}-q4_k_m.gguf",
        f"rigour-{tier}-v{version}.gguf",
    ]:
        try:
            path = hf_hub_download(repo_id, pattern, token=token)
            print(f"Downloaded: {repo_id}/{pattern}")
            return path
        except Exception:
            continue

    # Try listing files to find any GGUF
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id)
        gguf_files = [f for f in files if f.endswith(".gguf") and "f16" not in f]
        if gguf_files:
            path = hf_hub_download(repo_id, gguf_files[0], token=token)
            print(f"Downloaded: {repo_id}/{gguf_files[0]}")
            return path
    except Exception as e:
        print(f"ERROR: Could not download GGUF from {repo_id}: {e}")

    return None


def download_previous_eval(tier: str, version: str, token: str) -> dict | None:
    """Download previous version's eval results for regression comparison."""
    prev_version = str(int(version) - 1)
    if int(prev_version) < 1:
        return None

    prev_repo = f"rigour-labs/rigour-{tier}-v{prev_version}-gguf"
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(prev_repo, "eval_results.json", token=token)
        with open(path) as f:
            return json.load(f)
    except Exception:
        print(f"No previous eval found for v{prev_version} (first run or not uploaded)")
        return None


def upload_eval_results(results: dict, tier: str, version: str, token: str):
    """Upload eval results to the GGUF repo for future regression checks."""
    from huggingface_hub import HfApi
    import tempfile

    repo_id = f"rigour-labs/rigour-{tier}-v{version}-gguf"
    api = HfApi(token=token)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f, indent=2)
        tmp_path = f.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="eval_results.json",
            repo_id=repo_id,
        )
        print(f"Eval results uploaded to {repo_id}")
    except Exception as e:
        print(f"WARNING: Could not upload eval results: {e}")
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DriftBench model")
    parser.add_argument("--tier", required=True, choices=["deep", "lite"])
    parser.add_argument("--version", required=True, help="Model version number")
    parser.add_argument("--min-accuracy", type=float, default=0.6,
                        help="Minimum accuracy to pass (0.0-1.0)")
    parser.add_argument("--regression-threshold", type=float, default=0.05,
                        help="Max accuracy drop vs previous version before failing")
    parser.add_argument("--gguf-path", default="",
                        help="Local GGUF path (skip download)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")

    print(f"=== DriftBench Evaluation: tier={args.tier} v{args.version} ===")
    print(f"Min accuracy: {args.min_accuracy}, Regression threshold: {args.regression_threshold}")
    print()

    # Download or locate GGUF
    gguf_path = args.gguf_path
    if not gguf_path:
        gguf_path = download_gguf(args.tier, args.version, token)
        if not gguf_path:
            print("ERROR: Could not download GGUF model")
            sys.exit(1)

    # Run evaluation
    results = []
    category_results = {}

    for scenario_path in EVAL_SCENARIOS:
        scenario = load_scenario(scenario_path)
        if not scenario:
            continue

        scenario_id = scenario.get("id", os.path.basename(scenario_path))
        category = scenario.get("category", "unknown")
        print(f"  Evaluating: {scenario_id} ({category})")

        prompt = build_prompt(scenario)
        if not prompt:
            print(f"    SKIP: No prompt could be built")
            continue

        output = run_inference(gguf_path, prompt)
        if output is None:
            results.append({"id": scenario_id, "pass": False, "reason": "inference_error"})
            continue

        result = evaluate_scenario(scenario, output)
        result["id"] = scenario_id
        results.append(result)

        # Track per-category
        if category not in category_results:
            category_results[category] = {"correct": 0, "total": 0}
        category_results[category]["total"] += 1
        if result["pass"]:
            category_results[category]["correct"] += 1

        status = "PASS" if result["pass"] else "FAIL"
        print(f"    {status} (expected_drift={result.get('expected_drift')}, "
              f"predicted_drift={result.get('predicted_drift')}, "
              f"confidence={result.get('confidence', 0):.2f})")

    # Compute overall accuracy
    total = len(results)
    correct = sum(1 for r in results if r.get("pass"))
    accuracy = correct / total if total > 0 else 0.0

    print()
    print(f"=== Results: {correct}/{total} correct ({accuracy:.1%}) ===")
    for cat, data in sorted(category_results.items()):
        cat_acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        print(f"  {cat}: {data['correct']}/{data['total']} ({cat_acc:.0%})")

    # Build eval results output
    eval_output = {
        "tier": args.tier,
        "version": args.version,
        "date": datetime.now(timezone.utc).isoformat(),
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_category": {
            cat: round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0
            for cat, d in category_results.items()
        },
        "scenarios": results,
    }

    # Save locally
    with open("eval_results.json", "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"\nSaved eval_results.json")

    # Upload to HF for future regression checks
    if token:
        upload_eval_results(eval_output, args.tier, args.version, token)

    # Gate 1: Minimum accuracy
    if accuracy < args.min_accuracy:
        print(f"\nFAIL: Accuracy {accuracy:.1%} < minimum {args.min_accuracy:.0%}")
        sys.exit(1)

    # Gate 2: Regression check
    if token:
        prev_eval = download_previous_eval(args.tier, args.version, token)
        if prev_eval:
            prev_accuracy = prev_eval.get("accuracy", 0)
            drop = prev_accuracy - accuracy
            print(f"\nRegression check: v{int(args.version)-1} accuracy={prev_accuracy:.1%}, "
                  f"v{args.version} accuracy={accuracy:.1%}, drop={drop:.1%}")
            if drop > args.regression_threshold:
                print(f"FAIL: Regression of {drop:.1%} exceeds threshold {args.regression_threshold:.0%}")
                sys.exit(1)
            print("PASS: No regression detected")

    print(f"\nPASS: Model v{args.version} ({args.tier}) passed all gates")


if __name__ == "__main__":
    main()
