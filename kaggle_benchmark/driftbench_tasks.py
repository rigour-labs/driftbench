"""DriftBench — Kaggle Community Benchmark.

Evaluates LLMs on code drift detection: given a code patch and the original
intent, can the model identify whether the patch introduces drift (security
issues, stale patterns, logic errors, architectural violations)?

This is the benchmark task file that runs on Kaggle via kaggle-benchmarks SDK.
Each scenario presents a patch and asks the model to classify it as
drift-free or drifted, with specific drift type identification.

Notebook:  https://www.kaggle.com/code/rigourlabs/driftbench-code-drift-detection-benchmark
Dataset:   https://www.kaggle.com/datasets/rigourlabs/driftbench-scenarios
"""
import json
import kaggle_benchmarks as kbench
import pandas as pd

# ── Drift categories the model must recognize ──
DRIFT_CATEGORIES = {
    "logic_drift": "Code has logical inconsistencies or incorrect control flow",
    "security_drift": "Code introduces security vulnerabilities",
    "stale_drift": "Code uses deprecated/legacy patterns instead of modern equivalents",
    "staleness_drift": "Code uses deprecated/legacy patterns instead of modern equivalents",
    "architecture_drift": "Code violates architectural boundaries or design patterns",
    "pattern_drift": "Code deviates from established project patterns/conventions",
    "standard_drift": "Code violates coding standards (naming, formatting, exports)",
}

# ── System prompt for drift detection ──
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
    "explanation": "Brief explanation of the drift found or why the code is clean"
}
"""


# ── Primary benchmark task: Drift Detection ──
@kbench.task(name="drift_detection")
def detect_drift(llm, intent: str, patch: str, has_drift: bool,
                 drift_type: str, category: str, task_id: str) -> bool:
    """Given a patch and original intent, detect whether drift is present.

    The model receives:
    - The original coding intent (what the developer was asked to do)
    - The actual patch (unified diff)

    It must determine if the patch faithfully implements the intent or
    introduces drift. This tests the core capability needed for automated
    code quality gates.
    """
    prompt = f"""\
## Code Review Task

**Original Intent:**
{intent}

**Submitted Patch:**
```diff
{patch}
```

Analyze this patch against the original intent. Does it introduce code drift?
Respond with the JSON structure specified in your instructions."""

    response = llm.prompt(prompt, system=SYSTEM_PROMPT)

    # Parse model response
    try:
        # Handle markdown-wrapped JSON
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        model_says_drift = result.get("has_drift", False)
    except (json.JSONDecodeError, AttributeError):
        # If model can't produce valid JSON, count as wrong
        model_says_drift = None

    # Assert: model's classification matches ground truth
    if model_says_drift is None:
        kbench.assertions.fail(
            expectation="Model should return valid JSON with has_drift field"
        )
        return False

    correct = model_says_drift == has_drift
    if not correct:
        if has_drift:
            kbench.assertions.fail(
                expectation=f"Model should detect {drift_type} drift in {task_id}, "
                           f"but classified patch as clean"
            )
        else:
            kbench.assertions.fail(
                expectation=f"Model should classify {task_id} golden patch as clean, "
                           f"but reported false positive drift"
            )

    return correct


# ── Secondary task: Drift Type Classification (harder) ──
@kbench.task(name="drift_classification")
def classify_drift(llm, intent: str, patch: str, has_drift: bool,
                   drift_type: str, category: str, task_id: str) -> bool:
    """Same as detect_drift but also requires correct drift TYPE identification.

    Only evaluated on scenarios where has_drift=True. The model must not only
    detect that drift exists, but correctly categorize it (security, logic,
    stale, architecture, pattern, standard).
    """
    if not has_drift:
        # Golden patches: just need correct "no drift" classification
        return detect_drift.run(
            llm=llm, intent=intent, patch=patch, has_drift=has_drift,
            drift_type=drift_type, category=category, task_id=task_id
        )

    prompt = f"""\
## Code Review Task — Detailed Classification

**Original Intent:**
{intent}

**Submitted Patch:**
```diff
{patch}
```

Analyze this patch against the original intent. Identify the specific type of
drift present (if any). Respond with the JSON structure specified."""

    response = llm.prompt(prompt, system=SYSTEM_PROMPT)

    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        model_says_drift = result.get("has_drift", False)
        model_drift_type = result.get("drift_type", "")
    except (json.JSONDecodeError, AttributeError):
        kbench.assertions.fail(
            expectation="Model should return valid JSON"
        )
        return False

    if not model_says_drift:
        kbench.assertions.fail(
            expectation=f"Model should detect drift in {task_id}"
        )
        return False

    # Normalize category names for matching
    expected = category.replace("_drift", "").replace("staleness", "stale")
    actual = (model_drift_type or "").replace("_drift", "").replace("staleness", "stale")

    correct_type = expected in actual or actual in expected
    if not correct_type:
        kbench.assertions.fail(
            expectation=f"Expected drift type '{category}' but model classified as "
                       f"'{model_drift_type}' for {task_id}"
        )

    return correct_type


# ── Code Generation task: Can the model write drift-free code? ──
@kbench.task(name="drift_free_generation")
def generate_without_drift(llm, intent: str, patch: str, has_drift: bool,
                           drift_type: str, category: str,
                           task_id: str) -> bool:
    """Given only the intent, generate code and check for common drift patterns.

    Only uses golden scenarios (has_drift=False) as ground truth. The model
    generates code from scratch, and we check for common drift anti-patterns
    using heuristic rules (no Rigour CLI needed).
    """
    if has_drift:
        # Skip drift scenarios for generation tasks
        return True

    gen_prompt = f"""\
## Code Generation Task

Implement the following change. Return ONLY a unified diff (git patch format).
Do not include any explanation or markdown formatting.

**Intent:**
{intent}

Start your response with `--- ` (the beginning of the patch)."""

    response = llm.prompt(gen_prompt)

    # Basic drift pattern checks (heuristic, no Rigour needed)
    code = response.lower()
    issues = []

    # Staleness checks
    if "var " in code and ("const " not in code and "let " not in code):
        if ".js" in intent.lower() or "es6" in intent.lower():
            issues.append("Uses 'var' instead of const/let (staleness)")

    if "module.exports" in code and "esm" in intent.lower():
        issues.append("Uses CommonJS instead of ES modules (staleness)")

    if "require(" in code and "import" in intent.lower():
        issues.append("Uses require() instead of import (staleness)")

    # Security checks
    if "csrf" in intent.lower() and "csrf_enabled = false" in code:
        issues.append("Disables CSRF protection (security)")

    if "password" in code and ("==" in code or "!=" in code):
        if "hash" not in code and "bcrypt" not in code:
            issues.append("Plain text password comparison (security)")

    # Architecture checks
    if "circular" in intent.lower() and "from src." in code:
        issues.append("Direct import creating circular dependency (architecture)")

    if "global " in code and "g object" in intent.lower():
        issues.append("Module-level globals instead of request context (logic)")

    if issues:
        kbench.assertions.fail(
            expectation=f"Generated code should be drift-free but has: "
                       + "; ".join(issues)
        )
        return False

    return True


# ── Load dataset & run evaluation ──
def load_dataset():
    """Load DriftBench scenarios from JSONL dataset."""
    import os
    dataset_path = os.path.join(os.path.dirname(__file__), "driftbench_dataset.jsonl")

    # Also check Kaggle dataset mount path
    kaggle_path = "/kaggle/input/driftbench-scenarios/driftbench_dataset.jsonl"

    path = kaggle_path if os.path.exists(kaggle_path) else dataset_path

    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            rows.append({
                "intent": row["intent"],
                "patch": row["patch"],
                "has_drift": row["has_drift"],
                "drift_type": row.get("drift_type") or "",
                "category": row["category"],
                "task_id": row["scenario_id"],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Local testing: run a single scenario
    df = load_dataset()
    print(f"Loaded {len(df)} scenarios")
    print(f"  Golden (no drift): {(~df['has_drift']).sum()}")
    print(f"  Drift:             {df['has_drift'].sum()}")
    print(f"\nCategories: {sorted(df['category'].unique())}")

    # Run drift detection benchmark
    print("\n=== Running Drift Detection Benchmark ===")
    results = detect_drift.evaluate(
        llm=[kbench.llm],
        evaluation_data=df,
    )
    print(results)
