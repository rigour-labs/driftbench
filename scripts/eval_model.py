#!/usr/bin/env python3
"""
Evaluate a GGUF model on the DriftBench items and decide whether it may be published.

Every task contributes its golden patch (no drift) and its drift patch, with
answer-revealing comments and docstrings removed (evalgate/sanitize.py). The
model sees the intent and the patch, never the drift type. The gate fails a
model that answers the same way to everything, that has a false-positive rate
above the cap, whose balanced accuracy is too low, or whose replies cannot be
parsed. Missing inference support is an error, never a pass.

Usage:
    python scripts/eval_model.py --tier deep --version 6.0.0
    python scripts/eval_model.py --tier lite --gguf-path model.gguf --version 6.0.0 --baseline-version 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalgate.metrics import Prediction, Thresholds, gate, report  # noqa: E402
from evalgate.prompt import build_prompt, parse_answer  # noqa: E402
from evalgate.scenarios import load_items  # noqa: E402

CONTEXT_TOKENS = 8192
MAX_REPLY_TOKENS = 64


def load_model(gguf_path: str):
    try:
        from llama_cpp import Llama
    except ImportError:
        sys.exit("ERROR: llama-cpp-python is required to evaluate a model (pip install llama-cpp-python)")
    return Llama(model_path=gguf_path, n_ctx=CONTEXT_TOKENS, n_gpu_layers=0, seed=42, verbose=False)


def ask(llm, prompt: str) -> str:
    reply = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=MAX_REPLY_TOKENS,
    )
    return reply["choices"][0]["message"]["content"] or ""


def download_gguf(tier: str, version: str, token: str) -> str | None:
    from huggingface_hub import hf_hub_download
    repo_id = f"rigour-labs/rigour-{tier}-v{version}-gguf"
    try:
        return hf_hub_download(repo_id, f"rigour-{tier}-v{version}-q4_k_m.gguf", token=token or None)
    except Exception as error:  # noqa: BLE001 - reported and turned into an exit
        print(f"ERROR: could not download {repo_id}: {error}")
        return None


def baseline_report(tier: str, version: str, token: str) -> dict | None:
    """The published eval report of an earlier version, if it has one in the new format."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(f"rigour-labs/rigour-{tier}-v{version}-gguf", "eval_results.json", token=token or None)
    except Exception:  # noqa: BLE001 - no baseline is a normal first run
        return None
    return json.loads(Path(path).read_text()).get("report")


def upload_results(results: dict, tier: str, version: str, token: str) -> None:
    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj=json.dumps(results, indent=2).encode(), path_in_repo="eval_results.json",
        repo_id=f"rigour-labs/rigour-{tier}-v{version}-gguf",
    )


def evaluate(llm) -> tuple[list[Prediction], list[dict]]:
    predictions, rows = [], []
    for item in load_items():
        answer = parse_answer(ask(llm, build_prompt(item)))
        predictions.append(Prediction(item.id, item.has_drift, answer))
        rows.append({"id": item.id, "category": item.category, "expected": item.has_drift, "predicted": answer})
        print(f"  {item.id}: expected={item.has_drift} predicted={answer}")
    return predictions, rows


def regression_failures(current: dict, baseline: dict | None, max_drop: float) -> list[str]:
    if not baseline:
        return []
    drop = baseline["balanced_accuracy"] - current["balanced_accuracy"]
    return [f"balanced accuracy dropped {drop:.2f} vs baseline (max {max_drop:.2f})"] if drop > max_drop else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Rigour model and gate its publication")
    parser.add_argument("--tier", required=True, choices=["deep", "lite"])
    parser.add_argument("--version", required=True, help="Version being evaluated, as published (e.g. 5 or 6.0.0)")
    parser.add_argument("--gguf-path", default="", help="Local GGUF (skips the download)")
    parser.add_argument("--baseline-version", default="", help="Published version to compare against")
    parser.add_argument("--min-balanced-accuracy", type=float, default=Thresholds.min_balanced_accuracy)
    parser.add_argument("--max-false-positive-rate", type=float, default=Thresholds.max_false_positive_rate)
    parser.add_argument("--max-regression", type=float, default=0.05)
    parser.add_argument("--output", default="eval_results.json", help="Where to write the report")
    parser.add_argument("--upload", action="store_true", help="Upload the report to the model repo")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    gguf = args.gguf_path or download_gguf(args.tier, args.version, token)
    if not gguf:
        sys.exit(2)

    predictions, rows = evaluate(load_model(gguf))
    result = report(predictions)
    thresholds = Thresholds(args.min_balanced_accuracy, args.max_false_positive_rate)
    baseline = baseline_report(args.tier, args.baseline_version, token) if args.baseline_version else None
    failures = gate(result, thresholds) + regression_failures(result.to_dict(), baseline, args.max_regression)

    output = {
        "tier": args.tier, "version": args.version, "date": datetime.now(timezone.utc).isoformat(),
        "report": result.to_dict(), "thresholds": thresholds.__dict__, "gate_failures": failures, "items": rows,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output["report"], indent=2))
    if args.upload and token:
        upload_results(output, args.tier, args.version, token)

    if failures:
        print("FAIL: " + "; ".join(failures))
        sys.exit(1)
    print(f"PASS: {args.tier} v{args.version} may be published")


if __name__ == "__main__":
    main()
