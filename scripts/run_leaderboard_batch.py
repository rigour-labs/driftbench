#!/usr/bin/env python3
"""Batch-based drift detection leaderboard evaluator.

Submits 54 drift scenarios to LLM providers using batch APIs (50% off)
or direct calls (free tiers). Collects results, computes rich metrics
per category/repo/gate, confidence calibration, and overconfident errors.

Usage:
    # Single model (Anthropic batch — 50% off):
    python scripts/run_leaderboard_batch.py --provider anthropic --model claude-sonnet-4-6

    # Single model (Gemini free tier — $0):
    python scripts/run_leaderboard_batch.py --provider gemini --model gemini-2.5-flash

    # All models from config:
    python scripts/run_leaderboard_batch.py --all

    # Dry run (no API calls):
    python scripts/run_leaderboard_batch.py --dry-run
"""

import os
import re
import json
import time
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("leaderboard_batch")

ROOT = Path(__file__).parent.parent
DATASETS_DIR = ROOT / "datasets"
KAGGLE_BENCHMARK_DIR = ROOT / "kaggle_benchmark"
RESULTS_DIR = ROOT / "results"
MODEL_CONFIG_PATH = ROOT / "model_config.json"

DRIFT_CATEGORIES = [
    "logic_drift", "security_drift", "stale_drift",
    "architecture_drift", "pattern_drift", "standard_drift",
]

# Providers that support async batch API (50% cheaper)
BATCH_PROVIDERS = {"anthropic", "openai", "together", "together_ai"}

# Providers with free tiers (direct calls, no batch needed)
FREE_PROVIDERS = {"gemini", "deepseek", "groq"}

SYSTEM_PROMPT = """\
You are an expert code reviewer specializing in detecting code drift.

Code drift occurs when a code patch subtly violates the original intent,
project conventions, or introduces quality issues while appearing functional.

Categories of drift:
- logic_drift: Incorrect control flow, wrong variable scope, missing error handling
- security_drift: Disabled protections, exposed secrets, unsafe operations
- stale_drift: Using var instead of const/let, CommonJS instead of ESM, old APIs
- architecture_drift: Circular imports, wrong abstraction layer, coupling violations
- pattern_drift: Inconsistent with project conventions, wrong framework idioms
- standard_drift: Naming violations, incorrect exports, formatting issues

You must respond with EXACTLY this JSON structure (no other text):
{
    "has_drift": true/false,
    "confidence": 0.0-1.0,
    "drift_type": "category_name" or null,
    "explanation": "Brief explanation"
}
"""


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """Single evaluation scenario (golden or drift patch)."""
    task_id: str
    scenario_id: str
    name: str
    repository: str
    category: str
    intent: str
    patch: str
    has_drift: bool
    drift_type: Optional[str] = None
    expected_gate: Optional[str] = None


@dataclass
class Prediction:
    """Model prediction for a scenario."""
    scenario_id: str
    has_drift: bool
    confidence: float
    drift_type: Optional[str]
    explanation: str
    correct: bool
    raw_response: str = ""


@dataclass
class ModelResult:
    """Aggregated results for one model."""
    model: str
    provider: str
    display_name: str
    overall_accuracy: float = 0.0
    scenarios_evaluated: int = 0
    by_category: Dict[str, Dict] = field(default_factory=dict)
    by_repository: Dict[str, Dict] = field(default_factory=dict)
    overconfident_errors: int = 0  # high confidence + wrong
    underconfident_correct: int = 0  # low confidence + right
    calibration_ece: float = 0.0
    predictions: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SCENARIO LOADING
# ---------------------------------------------------------------------------

def load_scenarios() -> List[Scenario]:
    """Load all drift scenarios from datasets/ directory."""
    scenarios = []
    task_files = sorted(DATASETS_DIR.glob("*/*.json"))

    for task_file in task_files:
        try:
            task = json.loads(task_file.read_text())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Skipping {task_file}: {e}")
            continue

        task_id = task["id"]
        category = task["category"]
        repository = task["repository"]
        intent = task["intent"]
        name = task.get("name", task_id)

        # Golden patch → no drift
        golden_path = ROOT / task["golden_patch"]
        if golden_path.exists():
            scenarios.append(Scenario(
                task_id=task_id,
                scenario_id=f"{task_id}__golden",
                name=name,
                repository=repository,
                category=category,
                intent=intent,
                patch=golden_path.read_text(),
                has_drift=False,
            ))

        # Drift candidates → has drift
        for candidate in task.get("drift_candidates", []):
            drift_path = ROOT / candidate["patch"]
            if drift_path.exists():
                scenarios.append(Scenario(
                    task_id=task_id,
                    scenario_id=f"{task_id}__{candidate['id']}",
                    name=name,
                    repository=repository,
                    category=category,
                    intent=intent,
                    patch=drift_path.read_text(),
                    has_drift=True,
                    drift_type=candidate.get("drift_type"),
                    expected_gate=candidate.get("fail_gate"),
                ))

    logger.info(
        f"Loaded {len(scenarios)} scenarios "
        f"({sum(1 for s in scenarios if not s.has_drift)} golden, "
        f"{sum(1 for s in scenarios if s.has_drift)} drift)"
    )
    return scenarios


# ---------------------------------------------------------------------------
# PROMPT BUILDING
# ---------------------------------------------------------------------------

def build_prompt(scenario: Scenario) -> str:
    """Build drift detection prompt for a single scenario."""
    return f"""## Code Review Task

**Intent:**
{scenario.intent}

**Patch:**
```diff
{scenario.patch}
```

Does this patch introduce code drift? Respond with JSON only."""


def sanitize_custom_id(s: str) -> str:
    """Sanitize string for Anthropic batch custom_id (alphanumeric + _ -)."""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", s)[:64]


# ---------------------------------------------------------------------------
# PROVIDER BACKENDS
# ---------------------------------------------------------------------------

def call_anthropic_batch(
    scenarios: List[Scenario], model: str
) -> Dict[str, str]:
    """Submit to Anthropic Message Batches API (50% off)."""
    from rlaif.batch_provider import create_batch, poll_batch, collect_results

    # Strip provider prefix for Anthropic SDK
    model_name = model.replace("anthropic/", "")

    prompts = [
        {
            "custom_id": sanitize_custom_id(s.scenario_id),
            "content": build_prompt(s),
        }
        for s in scenarios
    ]

    logger.info(f"Submitting {len(prompts)} prompts to Anthropic Batch API ({model_name})...")
    batch_id = create_batch(prompts, model=model_name, system_prompt=SYSTEM_PROMPT)

    logger.info(f"Batch {batch_id}: polling for completion...")
    status = poll_batch(batch_id)
    if status != "ended":
        raise RuntimeError(f"Batch {batch_id} failed: {status}")

    results_map, succeeded, errored = collect_results(batch_id)
    logger.info(f"Batch complete: {succeeded} succeeded, {errored} errored")
    return results_map


def call_openai_batch(
    scenarios: List[Scenario], model: str
) -> Dict[str, str]:
    """Submit to OpenAI Batch API (50% off)."""
    import openai
    import tempfile

    client = openai.OpenAI()
    model_name = model.replace("openai/", "")

    # Build JSONL for batch
    requests = []
    for s in scenarios:
        requests.append({
            "custom_id": sanitize_custom_id(s.scenario_id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(s)},
                ],
                "max_tokens": 1024,
                "temperature": 0.1,
            },
        })

    # Write JSONL to temp file and upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
        jsonl_path = f.name

    logger.info(f"Uploading {len(requests)} requests to OpenAI Batch API...")
    with open(jsonl_path, "rb") as upload_f:
        batch_file = client.files.create(file=upload_f, purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    os.unlink(jsonl_path)

    # Poll
    logger.info(f"OpenAI batch {batch.id}: polling...")
    while True:
        batch = client.batches.retrieve(batch.id)
        if batch.status == "completed":
            break
        if batch.status in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"OpenAI batch failed: {batch.status}")
        logger.info(f"  Status: {batch.status}")
        time.sleep(30)

    # Collect results
    logger.info(
        f"OpenAI batch {batch.id} finished: status={batch.status} "
        f"output_file_id={batch.output_file_id} error_file_id={batch.error_file_id} "
        f"counts={batch.request_counts}"
    )

    # Handle batch errors — if all requests failed, output_file_id is None
    if batch.error_file_id:
        try:
            error_content = client.files.content(batch.error_file_id)
            logger.error(f"OpenAI batch errors:\n{error_content.text[:2000]}")
        except Exception as e:
            logger.error(f"Could not retrieve error file: {e}")

    if not batch.output_file_id:
        raise RuntimeError(
            f"OpenAI batch {batch.id} completed but output_file_id is None. "
            f"All {batch.request_counts.total} requests may have failed. "
            f"Check error_file_id={batch.error_file_id} for details."
        )

    output = client.files.content(batch.output_file_id)
    results_map = {}
    for line in output.text.strip().split("\n"):
        entry = json.loads(line)
        cid = entry["custom_id"]
        resp = entry.get("response", {})
        if resp.get("status_code") != 200:
            error_body = resp.get("body", {}).get("error", {})
            logger.warning(f"Batch entry {cid} failed: {error_body.get('message', 'unknown error')}")
            continue
        text = resp["body"]["choices"][0]["message"]["content"]
        results_map[cid] = text

    logger.info(f"OpenAI batch complete: {len(results_map)}/{len(requests)} results")
    return results_map


def _get_rate_limit_delay(model: str, provider: str) -> float:
    """Get rate limit delay from model_config.json, with sensible defaults."""
    config = load_model_config()
    model_info = config.get(model) or config.get(f"{provider}/{model}") or {}
    rpm = model_info.get("rate_limit_rpm")
    if rpm:
        return 60.0 / rpm + 0.5  # add 0.5s safety margin

    # Defaults for known free-tier providers
    defaults = {"gemini": 6.5, "groq": 2.0, "deepseek": 1.0}
    return defaults.get(provider, 0.5)


def call_direct(
    scenarios: List[Scenario], model: str, provider: str
) -> Dict[str, str]:
    """Call LLM directly via litellm (for free-tier providers)."""
    import litellm

    results_map = {}
    total = len(scenarios)
    delay = _get_rate_limit_delay(model, provider)
    logger.info(f"Rate limit delay: {delay:.1f}s between requests")

    for i, s in enumerate(scenarios):
        cid = sanitize_custom_id(s.scenario_id)
        prompt = build_prompt(s)

        try:
            response = litellm.completion(
                model=f"{provider}/{model}" if "/" not in model else model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            text = response.choices[0].message.content
            results_map[cid] = text
        except Exception as e:
            logger.warning(f"[{i+1}/{total}] Failed {s.scenario_id}: {e}")
            continue

        if (i + 1) % 10 == 0:
            logger.info(f"[{i+1}/{total}] Progress: {len(results_map)} succeeded")

        # Rate limiting
        if delay > 0:
            time.sleep(delay)

    logger.info(f"Direct calls complete: {len(results_map)}/{total} succeeded")
    return results_map


def call_provider(
    scenarios: List[Scenario], model: str, provider: str
) -> Dict[str, str]:
    """Route to the correct provider backend."""
    # Normalize provider name for routing
    normalized = provider.replace("_ai", "").replace("_", "")

    if normalized == "anthropic":
        return call_anthropic_batch(scenarios, model)
    elif normalized == "openai":
        return call_openai_batch(scenarios, model)
    elif normalized in ("together",) and provider in BATCH_PROVIDERS:
        # Together.ai has batch support but uses OpenAI-compatible API
        return call_openai_batch(scenarios, model)
    else:
        return call_direct(scenarios, model, provider)


# ---------------------------------------------------------------------------
# RESULT PARSING & METRICS
# ---------------------------------------------------------------------------

def parse_response(text: str) -> Dict:
    """Parse JSON response from model, handling code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()
    return json.loads(text)


def evaluate_results(
    scenarios: List[Scenario],
    results_map: Dict[str, str],
) -> Tuple[List[Prediction], Dict]:
    """Parse responses and compute all metrics."""
    # Build lookup
    scenario_map = {sanitize_custom_id(s.scenario_id): s for s in scenarios}

    predictions = []
    for cid, response_text in results_map.items():
        scenario = scenario_map.get(cid)
        if not scenario:
            logger.warning(f"No scenario found for custom_id: {cid}")
            continue

        try:
            result = parse_response(response_text)
            model_says_drift = result.get("has_drift", False)
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
            drift_type = result.get("drift_type")
            explanation = result.get("explanation", "")
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.warning(f"Parse error for {scenario.scenario_id}: {e}")
            # Count parse errors as wrong — don't silently default
            model_says_drift = not scenario.has_drift  # intentionally wrong
            confidence = 0.0
            drift_type = None
            explanation = f"(parse error: {e})"

        correct = model_says_drift == scenario.has_drift
        predictions.append(Prediction(
            scenario_id=scenario.scenario_id,
            has_drift=model_says_drift,
            confidence=confidence,
            drift_type=drift_type,
            explanation=explanation,
            correct=correct,
            raw_response=response_text[:500],
        ))

    # Compute metrics
    metrics = compute_metrics(scenarios, predictions)
    return predictions, metrics


def compute_metrics(
    scenarios: List[Scenario], predictions: List[Prediction]
) -> Dict:
    """Compute rich metrics: by_category, by_repo, ECE, overconfident errors."""
    pred_map = {p.scenario_id: p for p in predictions}
    scenario_map = {s.scenario_id: s for s in scenarios}

    by_category = {}
    by_repo = {}
    all_confidences = []  # (confidence, correct) for ECE
    overconfident = 0
    underconfident = 0

    for p in predictions:
        s = scenario_map.get(p.scenario_id)
        if not s:
            continue

        # Per-category
        cat = s.category
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0, "confidences": []}
        by_category[cat]["total"] += 1
        by_category[cat]["correct"] += int(p.correct)
        by_category[cat]["confidences"].append(p.confidence)

        # Per-repo
        repo = s.repository.split("/")[-1] if "/" in s.repository else s.repository
        if repo not in by_repo:
            by_repo[repo] = {"correct": 0, "total": 0}
        by_repo[repo]["total"] += 1
        by_repo[repo]["correct"] += int(p.correct)

        # Confidence tracking
        all_confidences.append((p.confidence, p.correct))

        # Overconfident: confidence >= 0.8 and wrong
        if p.confidence >= 0.8 and not p.correct:
            overconfident += 1
        # Underconfident: confidence < 0.5 and correct
        if p.confidence < 0.5 and p.correct:
            underconfident += 1

    # Compute accuracies
    for cat_data in by_category.values():
        cat_data["accuracy"] = (
            cat_data["correct"] / cat_data["total"]
            if cat_data["total"] > 0 else 0.0
        )
        del cat_data["confidences"]

    for repo_data in by_repo.values():
        repo_data["accuracy"] = (
            repo_data["correct"] / repo_data["total"]
            if repo_data["total"] > 0 else 0.0
        )

    # Overall
    total_correct = sum(int(p.correct) for p in predictions)
    total = len(predictions)
    overall_accuracy = total_correct / total if total > 0 else 0.0

    # ECE (Expected Calibration Error) — 10 bins
    ece = compute_ece(all_confidences, n_bins=10)

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "scenarios_evaluated": total,
        "by_category": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                            for kk, vv in v.items()}
                        for k, v in by_category.items()},
        "by_repository": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                               for kk, vv in v.items()}
                          for k, v in by_repo.items()},
        "overconfident_errors": overconfident,
        "underconfident_correct": underconfident,
        "calibration_ece": round(ece, 4),
    }


def compute_ece(confidences: List[Tuple[float, bool]], n_bins: int = 10) -> float:
    """Expected Calibration Error — measures if confidence matches accuracy."""
    if not confidences:
        return 0.0

    bins = [[] for _ in range(n_bins)]
    for conf, correct in confidences:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    ece = 0.0
    total = len(confidences)
    for bin_items in bins:
        if not bin_items:
            continue
        avg_conf = sum(c for c, _ in bin_items) / len(bin_items)
        avg_acc = sum(int(c) for _, c in bin_items) / len(bin_items)
        ece += abs(avg_acc - avg_conf) * len(bin_items) / total

    return ece


# ---------------------------------------------------------------------------
# MULTI-MODEL RUNNER
# ---------------------------------------------------------------------------

def load_model_config() -> Dict:
    """Load model_config.json."""
    if MODEL_CONFIG_PATH.exists():
        return json.loads(MODEL_CONFIG_PATH.read_text()).get("model_config", {})
    return {}


def run_all_models(scenarios: List[Scenario], dry_run: bool = False) -> List[Dict]:
    """Run leaderboard for all models in config."""
    config = load_model_config()
    all_results = []

    for model_key, model_info in config.items():
        provider = model_key.split("/")[0] if "/" in model_key else "anthropic"
        display_name = model_info.get("display_name", model_key)
        batch_enabled = model_info.get("batch_enabled", provider in BATCH_PROVIDERS)

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {display_name} ({model_key})")
        logger.info(f"Provider: {provider}, Batch: {batch_enabled}")
        logger.info(f"{'='*60}")

        if dry_run:
            logger.info("(dry run — skipping API calls)")
            all_results.append({
                "model": model_key,
                "provider": provider,
                "display_name": display_name,
                "overall_accuracy": 0.0,
                "scenarios_evaluated": 0,
                "dry_run": True,
            })
            continue

        try:
            results_map = call_provider(scenarios, model_key, provider)
            predictions, metrics = evaluate_results(scenarios, results_map)

            result = {
                "model": model_key,
                "provider": provider,
                "display_name": display_name,
                **metrics,
            }
            all_results.append(result)

            # Save per-model results
            model_slug = model_key.replace("/", "_")
            model_dir = RESULTS_DIR / "leaderboard" / model_slug
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "metrics.json").write_text(json.dumps(result, indent=2))
            (model_dir / "predictions.json").write_text(
                json.dumps([asdict(p) for p in predictions], indent=2)
            )

            # Print summary
            print(f"\n  {display_name}: {metrics['overall_accuracy']:.1%} overall")
            for cat, data in sorted(metrics["by_category"].items()):
                print(f"    {cat:22s} {data['accuracy']:.1%} ({data['correct']}/{data['total']})")
            print(f"    Overconfident errors: {metrics['overconfident_errors']}")
            print(f"    Calibration ECE:      {metrics['calibration_ece']:.4f}")

        except Exception as e:
            logger.error(f"Failed to evaluate {model_key}: {e}")
            all_results.append({
                "model": model_key,
                "provider": provider,
                "display_name": display_name,
                "error": str(e),
            })

    return all_results


def generate_leaderboard(results: List[Dict], scenario_count: int = 0) -> Dict:
    """Generate final leaderboard JSON for the website."""
    # Sort by accuracy (descending), filter out errors
    scored = [r for r in results if "error" not in r and not r.get("dry_run")]
    scored.sort(key=lambda r: r.get("overall_accuracy", 0), reverse=True)

    for i, r in enumerate(scored):
        r["rank"] = i + 1

    total = scenario_count or (scored[0]["scenarios_evaluated"] if scored else 0)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "DriftBench v1",
        "total_scenarios": total,
        "description": (
            "DriftBench evaluates LLM ability to detect code drift — "
            "subtle quality issues that slip past conventional review."
        ),
        "models": scored,
        "errors": [r for r in results if "error" in r],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DriftBench Leaderboard (Batch)")
    parser.add_argument("--provider", default=None, help="LLM provider")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--all", action="store_true", help="Run all models from config")
    parser.add_argument("--dry-run", action="store_true", help="No API calls")
    parser.add_argument("--output", default="results/leaderboard.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load scenarios
    scenarios = load_scenarios()
    if not scenarios:
        logger.error("No scenarios found. Run: python -m kaggle_benchmark.export_dataset")
        sys.exit(1)

    print(f"\nDriftBench Leaderboard Evaluator")
    print(f"================================")
    print(f"Scenarios: {len(scenarios)}")

    if args.all or (args.provider is None and args.model is None):
        # Run all models from config
        results = run_all_models(scenarios, dry_run=args.dry_run)
    else:
        # Single model
        provider = args.provider or "anthropic"
        model = args.model or "claude-sonnet-4-6"
        display_name = model.split("/")[-1] if "/" in model else model

        print(f"Model:     {model}")
        print(f"Provider:  {provider}")
        print()

        if args.dry_run:
            results = [{"model": model, "provider": provider,
                        "display_name": display_name, "dry_run": True}]
        else:
            results_map = call_provider(scenarios, model, provider)
            predictions, metrics = evaluate_results(scenarios, results_map)
            results = [{
                "model": model,
                "provider": provider,
                "display_name": display_name,
                **metrics,
            }]

    # Generate leaderboard
    leaderboard = generate_leaderboard(results, scenario_count=len(scenarios))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(leaderboard, indent=2))

    # Also save to rigour-web data path if it exists
    web_data = ROOT / "rigour-web" / "src" / "app" / "api" / "stats" / "data.json"
    if web_data.parent.exists():
        web_data.write_text(json.dumps(leaderboard, indent=2))
        logger.info(f"Published to rigour-web: {web_data}")

    print(f"\nLeaderboard saved: {output_path}")

    # Print final table
    if leaderboard["models"]:
        print(f"\n{'Rank':<6}{'Model':<30}{'Accuracy':<12}{'Security':<12}{'Logic':<12}{'Overconf':<10}")
        print("-" * 82)
        for m in leaderboard["models"]:
            sec = m.get("by_category", {}).get("security_drift", {}).get("accuracy", 0)
            log = m.get("by_category", {}).get("logic_drift", {}).get("accuracy", 0)
            print(
                f"#{m['rank']:<5}{m['display_name']:<30}"
                f"{m['overall_accuracy']:.1%}{'':>5}"
                f"{sec:.1%}{'':>5}"
                f"{log:.1%}{'':>5}"
                f"{m.get('overconfident_errors', 0)}"
            )


if __name__ == "__main__":
    main()
