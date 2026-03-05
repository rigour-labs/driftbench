import json
import subprocess
import os
import pandas as pd
import kaggle_benchmarks as kbench

# ---------------------------------------------------------------------------
# LOAD DATASET
# Benchmark env doesn't mount /kaggle/input — download via Kaggle API
# ---------------------------------------------------------------------------
DATA_DIR = "/tmp/driftbench"
DATASET_FILE = os.path.join(DATA_DIR, "driftbench_dataset.jsonl")

if not os.path.exists(DATASET_FILE):
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("rigourlabs/driftbench-scenarios", path=DATA_DIR, unzip=True)
    except Exception as e:
        print(f"Kaggle API failed ({e}), trying CLI...")
        subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "rigourlabs/driftbench-scenarios",
            "-p", DATA_DIR, "--unzip",
        ], check=True)

rows = []
with open(DATASET_FILE) as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)
df["drift_type"] = df["drift_type"].fillna("")

print(f"DriftBench: {len(df)} scenarios ({(~df['has_drift']).sum()} clean, {df['has_drift'].sum()} drift)")

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
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


def _parse(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# TASK: DRIFT DETECTION
# ---------------------------------------------------------------------------
@kbench.task(name="drift_detection", description="Can the LLM detect code drift in patches?")
def drift_detection(llm, intent: str, patch: str, has_drift: bool,
                    category: str, scenario_id: str) -> bool:
    """Binary: does the patch introduce drift?"""
    prompt = f"""{SYSTEM_PROMPT}

## Code Review

**Intent:**
{intent}

**Patch:**
```diff
{patch}
```

Does this patch introduce code drift? Respond with JSON."""

    response: str = llm.prompt(prompt)

    try:
        result = _parse(response)
        model_says_drift = result.get("has_drift", False)
    except (json.JSONDecodeError, AttributeError):
        kbench.assertions.assert_true(False, expectation="Model should return valid JSON")
        return False

    correct = model_says_drift == has_drift
    kbench.assertions.assert_true(
        correct,
        expectation=f"{'Should detect drift' if has_drift else 'Should classify as clean'} in {scenario_id} ({category})"
    )
    return correct


# ---------------------------------------------------------------------------
# RUN EVALUATION
# ---------------------------------------------------------------------------
print("Running drift detection on all 54 scenarios...")

results = drift_detection.evaluate(
    llm=[kbench.llm],
    evaluation_data=df[["intent", "patch", "has_drift", "category", "scenario_id"]],
)
print(results)

# ---------------------------------------------------------------------------
# SELECT MAIN TASK FOR LEADERBOARD
# ---------------------------------------------------------------------------
# %choose drift_detection
