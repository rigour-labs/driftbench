"""DriftBench: Code Drift Detection Benchmark — Kaggle Notebook.

This notebook works in two modes:
  1. STANDALONE: Always shows dataset analysis, statistics, and visualizations.
  2. BENCHMARK: When run inside Kaggle's benchmark infrastructure (MODEL_PROXY_URL set),
     also evaluates LLMs on drift detection using kaggle-benchmarks SDK.

Published at: https://www.kaggle.com/code/rigourlabs/driftbench-code-drift-detection-benchmark
Dataset:      https://www.kaggle.com/datasets/rigourlabs/driftbench-scenarios
"""

# %% [markdown]
# # DriftBench: Code Drift Detection Benchmark
#
# **What is code drift?** When AI-generated code subtly violates the original
# intent — using deprecated patterns, introducing security issues, breaking
# architectural boundaries, or ignoring project conventions — while appearing
# to work correctly.
#
# **Why does it matter?** As AI writes more production code, detecting drift
# is critical for code quality. DriftBench measures how well models can serve
# as automated code reviewers.
#
# **Three tiers:**
# - **Drift Detection** (Tier 1): Binary classification — does the patch have drift?
# - **Drift Classification** (Tier 2): What specific type of drift?
# - **Drift-Free Generation** (Tier 3): Can the model write drift-free code?

# %% Setup
import json
import os
import pandas as pd

# %% Load dataset
DATASET_PATH = "/kaggle/input/driftbench-scenarios/driftbench_dataset.jsonl"

# Fallback for local development
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join(os.path.dirname(__file__), "driftbench_dataset.jsonl")

rows = []
with open(DATASET_PATH) as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

# %% [markdown]
# ## Dataset Overview

# %% Dataset statistics
print("=" * 60)
print("DriftBench Dataset Overview")
print("=" * 60)
print(f"Total scenarios:     {len(df)}")
print(f"  Golden (no drift): {(~df['has_drift']).sum()}")
print(f"  Drifted:           {df['has_drift'].sum()}")
print(f"  Repositories:      {df['repository'].nunique()}")
print()

# %% Category breakdown
print("=" * 60)
print("Drift Categories")
print("=" * 60)
category_counts = df[df["has_drift"]].groupby("category").size().sort_values(ascending=False)
for cat, count in category_counts.items():
    print(f"  {cat:25s} {count:3d} scenarios")
print()

# %% Repository breakdown
print("=" * 60)
print("Repositories")
print("=" * 60)
repo_counts = df.groupby("repository").agg(
    total=("has_drift", "count"),
    drifted=("has_drift", "sum"),
).sort_values("total", ascending=False)
for repo, row in repo_counts.iterrows():
    print(f"  {repo:35s} {int(row['total']):3d} total ({int(row['drifted'])} drift)")
print()

# %% Sample scenario
print("=" * 60)
print("Sample Scenario")
print("=" * 60)
sample = df[df["has_drift"]].iloc[0]
print(f"Task:     {sample['name']}")
print(f"Repo:     {sample['repository']}")
print(f"Category: {sample['category']}")
print(f"Intent:   {sample['intent'][:120]}...")
print(f"Patch:    {len(sample['patch'])} chars")
print()

# %% [markdown]
# ## Drift Type Descriptions
#
# | Category | Description |
# |----------|-------------|
# | security_drift | Disabled protections, exposed secrets, unsafe operations |
# | logic_drift | Incorrect control flow, wrong variable scope, missing error handling |
# | stale_drift | Using var instead of const/let, CommonJS instead of ESM, old APIs |
# | architecture_drift | Circular imports, wrong abstraction layer, coupling violations |
# | pattern_drift | Inconsistent with project conventions, wrong framework idioms |
# | standard_drift | Naming violations, incorrect exports, formatting issues |

# %% [markdown]
# ## Benchmark Evaluation
#
# The benchmark evaluates LLMs on their ability to detect and classify code drift.
# When run inside Kaggle's Community Benchmarks infrastructure, it uses the
# `kaggle-benchmarks` SDK to evaluate models and populate the leaderboard.

# %% Check if running inside Kaggle's benchmark infrastructure
BENCHMARK_MODE = os.environ.get("MODEL_PROXY_URL") is not None

if BENCHMARK_MODE:
    print("Running in BENCHMARK mode — evaluating LLMs via kaggle-benchmarks SDK")
    print()

    import kaggle_benchmarks as kbench

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

    # ── Task 1: Drift Detection (Binary Classification) ──
    @kbench.task(name="drift_detection")
    def drift_detection(llm, intent: str, patch: str, has_drift: bool,
                        category: str, scenario_id: str) -> bool:
        """Detect whether a code patch introduces drift."""
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

        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            model_says_drift = result.get("has_drift", False)
        except (json.JSONDecodeError, AttributeError):
            kbench.assertions.fail(
                expectation="Model should return valid JSON with has_drift field"
            )
            return False

        correct = model_says_drift == has_drift
        if not correct:
            if has_drift:
                kbench.assertions.fail(
                    expectation=f"Should detect drift in {scenario_id} ({category})"
                )
            else:
                kbench.assertions.fail(
                    expectation=f"Should classify {scenario_id} as clean (false positive)"
                )
        return correct

    # ── Task 2: Drift Classification (Multi-class) ──
    @kbench.task(name="drift_classification")
    def drift_classification(llm, intent: str, patch: str, has_drift: bool,
                             category: str, scenario_id: str,
                             drift_type: str) -> bool:
        """Detect drift AND correctly classify its type."""
        prompt = f"""\
## Code Review — Detailed Classification

**Original Intent:**
{intent}

**Submitted Patch:**
```diff
{patch}
```

Analyze for drift. If drift exists, identify the specific category.
Respond with the JSON structure specified."""

        response = llm.prompt(prompt, system=SYSTEM_PROMPT)

        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            model_says_drift = result.get("has_drift", False)
            model_drift_type = result.get("drift_type", "") or ""
        except (json.JSONDecodeError, AttributeError):
            kbench.assertions.fail(expectation="Model should return valid JSON")
            return False

        # For golden patches, just check binary classification
        if not has_drift:
            if model_says_drift:
                kbench.assertions.fail(
                    expectation=f"Should classify {scenario_id} as clean"
                )
                return False
            return True

        # For drift patches, check both detection and type
        if not model_says_drift:
            kbench.assertions.fail(
                expectation=f"Should detect drift in {scenario_id}"
            )
            return False

        # Normalize and compare drift types
        expected = category.replace("_drift", "").replace("staleness", "stale")
        actual = model_drift_type.replace("_drift", "").replace("staleness", "stale")

        type_correct = expected in actual or actual in expected
        if not type_correct:
            kbench.assertions.fail(
                expectation=f"Expected '{category}' but got '{model_drift_type}' for {scenario_id}"
            )
        return type_correct

    # ── Prepare evaluation data ──
    eval_df = df.copy()
    eval_df["drift_type"] = eval_df["drift_type"].fillna("")

    # ── Run Tier 1: Drift Detection ──
    print("=" * 60)
    print("TIER 1: Drift Detection (Binary Classification)")
    print("=" * 60)

    detection_results = drift_detection.evaluate(
        llm=[kbench.llm],
        evaluation_data=eval_df[["intent", "patch", "has_drift", "category", "scenario_id"]],
    )
    print(detection_results)

    # ── Run Tier 2: Drift Classification ──
    print()
    print("=" * 60)
    print("TIER 2: Drift Classification (Multi-class)")
    print("=" * 60)

    classification_results = drift_classification.evaluate(
        llm=[kbench.llm],
        evaluation_data=eval_df[["intent", "patch", "has_drift", "category",
                                 "scenario_id", "drift_type"]],
    )
    print(classification_results)

else:
    # ── STANDALONE mode: show dataset analysis only ──
    print("=" * 60)
    print("STANDALONE MODE — Dataset Analysis Only")
    print("=" * 60)
    print()
    print("This notebook is running outside Kaggle's benchmark infrastructure.")
    print("Showing dataset analysis and sample evaluation prompts.")
    print()
    print("To run the full LLM benchmark:")
    print("  1. Fork this notebook on Kaggle")
    print("  2. Use the 'Add Models' button to select models")
    print("  3. Run through Kaggle's Community Benchmarks")
    print()

    # Show what a prompt looks like
    print("=" * 60)
    print("Example Evaluation Prompt")
    print("=" * 60)
    sample = df[df["has_drift"]].iloc[0]
    print(f"""
## Code Review Task

**Original Intent:**
{sample['intent']}

**Submitted Patch:**
```diff
{sample['patch'][:500]}{'...' if len(sample['patch']) > 500 else ''}
```

Expected answer: has_drift={sample['has_drift']}, type={sample['category']}
""")

# %% Summary
print()
print("=" * 60)
print("DriftBench Summary")
print("=" * 60)
print(f"Scenarios:  {len(df)} ({(~df['has_drift']).sum()} clean, {df['has_drift'].sum()} drift)")
print(f"Categories: {', '.join(sorted(df[df['has_drift']]['category'].unique()))}")
print(f"Repos:      {', '.join(sorted(df['repository'].unique()))}")
print()
if BENCHMARK_MODE:
    print("Tier 1 — Drift Detection:      see leaderboard")
    print("Tier 2 — Drift Classification:  see leaderboard")
else:
    print("Run inside Kaggle Benchmarks for leaderboard results.")
print()
print("Dataset:  https://www.kaggle.com/datasets/rigourlabs/driftbench-scenarios")
print("Notebook: https://www.kaggle.com/code/rigourlabs/driftbench-code-drift-detection-benchmark")
