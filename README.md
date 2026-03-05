# DriftBench

**The Full-Spectrum PR Drift & Intent Preservation Benchmark for AI Code Generation.**

DriftBench measures the ability of AI coding tools to preserve product intent and engineering invariants when making changes. While traditional benchmarks focus on "passing tests," DriftBench detects **drift** — changes that are syntactically correct and pass unit tests but violate core design patterns, security rules, or implicit business logic.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## Why DriftBench?

Traditional code benchmarks (HumanEval, MBPP, SWE-bench) measure whether AI-generated code is *correct*. DriftBench measures whether it's *appropriate*:

- Does it follow the project's established patterns?
- Does it avoid introducing security vulnerabilities?
- Does it use modern, non-deprecated APIs?
- Does it respect architectural boundaries?

An AI agent might write code that passes all tests but uses `var` instead of `const`, implements custom auth instead of using the existing `AuthService`, or accidentally logs sensitive data.

---

## Drift Categories

DriftBench evaluates AI agents across 7 major dimensions:

| Category | Description | Example Failure |
|----------|-------------|-----------------|
| **Staleness Drift** | Using deprecated/legacy patterns | Using `var` when project uses ES6+ |
| **Security Drift** | Introducing vulnerabilities | SQL injection, PII logging |
| **Architecture Drift** | Violating structural boundaries | Circular dependencies, layer violations |
| **Pattern Drift** | Deviating from established patterns | Re-implementing existing utilities |
| **Logic Drift** | Logical inconsistencies | Bypassing auth checks |
| **Standard Drift** | Quality gate violations | High cyclomatic complexity |
| **Agent Team Drift** | Multi-agent coordination issues | Cross-agent conflicts, handoff failures |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for Rigour CLI)
- API keys for the models you want to test

### Installation

```bash
# Clone the repository
git clone https://github.com/rigour-labs/driftbench.git
cd driftbench

# Install Python dependencies
pip install -r requirements.txt

# Install Rigour CLI (drift detection engine)
npm install -g @rigour-labs/cli

# Verify installation
npx @rigour-labs/cli --version
```

### Configuration

Create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=...
```

### Running Benchmarks

```bash
# Run a single task with a specific model
python -m runner.harness --model anthropic/claude-opus-4-6 --task lodash-stale-001

# Run all tasks for a model
python -m runner.harness --model anthropic/claude-opus-4-6 --all

# Run full benchmark (all models from model_config.json, all tasks)
python scripts/run_full_benchmark.py

# Run full benchmark with parallel workers (recommended: 4–6)
python scripts/run_full_benchmark.py --parallel 4

# Run a specific model only
python scripts/run_full_benchmark.py --model anthropic/claude-sonnet-4

# Dry run to see what would execute without running anything
python scripts/run_full_benchmark.py --dry-run

# Clean worker directories before a fresh run
python scripts/run_full_benchmark.py --clean --parallel 4

# Generate leaderboard from results
python scripts/snapshot_leaderboard.py
```

---

## Project Structure

```
driftbench/
├── datasets/                    # Benchmark tasks organized by repository
│   ├── lodash/                  # Tasks for lodash/lodash
│   ├── flask/                   # Tasks for pallets/flask
│   ├── django/                  # Tasks for django/django
│   ├── fastapi/                 # Tasks for tiangolo/fastapi
│   ├── shadcn/                  # Tasks for shadcn-ui/ui
│   └── tanstack/                # Tasks for TanStack/query
├── runner/
│   ├── engine.py                # Core benchmark engine (git ops, patch apply, rigour CLI)
│   ├── harness.py               # LLM harness (litellm, patch generation, retry)
│   └── log.py                   # Thread-safe logging with task-context prefixing
├── rlaif/                       # RLAIF training data pipeline (separate from benchmark)
│   ├── facts.py                 # AST fact extraction
│   ├── verifier.py              # 4-tier structural verification (14 checks)
│   ├── provider.py              # Teacher model calls with exponential backoff retry
│   ├── generate.py              # Pipeline orchestrator + CLI (live and batch modes)
│   ├── batch_provider.py        # Anthropic Batch API client (50% cheaper)
│   ├── batch_orchestrator.py    # Batch submit/collect with Pass@2 retry
│   ├── format_dpo.py            # DPO/SFT formatter with retry weighting
│   ├── finetune.py              # QLoRA fine-tune (SFT + DPO)
│   ├── export_gguf.py           # GGUF export + HuggingFace upload
│   └── repos_training.json      # 34 curated training repos (no eval overlap)
├── reporting/
│   └── aggregator.py            # Results aggregation and leaderboard generation
├── scripts/
│   ├── run_full_benchmark.py    # Full benchmark runner (parallel workers, all models)
│   └── snapshot_leaderboard.py  # Snapshot results into leaderboard JSON
├── results/                     # Benchmark results per model (gitignored)
├── model_config.json            # Model registry (name, max_tokens, features)
└── .env.example                 # Environment variables template
```

---

## Task Format

Each task is a JSON file with the following structure:

```json
{
    "id": "lodash-stale-001",
    "category": "stale_drift",
    "name": "Legacy Variable Declaration",
    "repository": "lodash/lodash",
    "intent": "Create a helper function following ES6+ standards",
    "base_sha": "main",
    "golden_patch": "datasets/lodash/patches/helper_stale_gold.patch",
    "rigour_config": "datasets/lodash/.rigour/config.yaml",
    "drift_candidates": [
        {
            "id": "stale-var-001",
            "patch": "datasets/lodash/patches/helper_stale_drift.patch",
            "drift_type": "staleness",
            "expected_result": "FAIL",
            "fail_gate": "staleness"
        }
    ]
}
```

The `golden_patch` is the correct reference implementation (should pass Rigour with no drift). Each `drift_candidates` entry is a deliberately flawed patch that should trigger drift detection.

---

## How It Works

1. **Task Loading**: Each task defines an `intent` (what the AI should implement), a target `repository`, a `base_sha` to pin the evaluation, and a `golden_patch` as the correct reference.
2. **LLM Generation**: The harness prompts the LLM to produce a unified diff patch. Up to 3 attempts with 5s/15s backoff are made on transient API errors (429, 503).
3. **Repo Setup**: The engine clones the target repo (shallow by default, full as fallback) and checks out the exact `base_sha`. All git operations have timeouts to prevent hangs.
4. **Patch Application**: The generated patch is applied using a 5-strategy cascade — `git apply`, `git apply --3way`, `git apply --reject`, `patch -p1`, then direct file creation for new-file patches.
5. **Drift Detection**: Rigour runs only against the modified files (incremental analysis) to avoid false positives from pre-existing issues.
6. **Scoring**: The LLM result is compared against the golden baseline. `passed` = no drift detected. `correct` = LLM result matches golden (both pass or both fail).

### Parallel Mode

The full benchmark runner supports parallel execution via `--parallel N`. Each task gets its own isolated workspace (`.drift_workers/<task_id>/`) so git clones and file writes never collide. Results are collected back to the main `results/` directory after each task completes.

```bash
# Recommended for full runs — 4 workers is a good balance
python scripts/run_full_benchmark.py --parallel 4
```

### Key Metrics

- **Pass Rate**: Percentage of tasks where the LLM-generated patch had no drift detected
- **DDR (Drift Detection Rate)**: How often the model introduces detectable drift
- **Accuracy**: Whether the model's pass/fail result matches the golden baseline

---

## Leaderboard

Live results at [rigour.run](https://rigour.run) (coming soon).

| Model | Display Name | Pass Rate | DDR | Tasks | Status |
|-------|-------------|-----------|-----|-------|--------|
| `anthropic/claude-opus-4-6` | Claude Opus 4.6 | --% | --% | 50 | 🆕 Pending |
| `anthropic/claude-opus-4-5` | Claude Opus 4.5 | --% | --% | 50 | Running |
| `anthropic/claude-sonnet-4` | Claude Sonnet 4 | --% | --% | 50 | Pending |
| `openai/gpt-5.2` | GPT-5.2 | --% | --% | 50 | Pending |
| `gemini/gemini-3-pro-preview` | Gemini 3 Pro | --% | --% | 50 | Pending |

*Results update automatically after each benchmark run via `scripts/snapshot_leaderboard.py`.*

Model identifiers match `model_config.json` and are passed directly to LiteLLM.

---

## DriftBench vs RLAIF — What's the Difference?

This repository contains **two separate systems** that share some infrastructure but serve different purposes:

| | DriftBench (Benchmark) | RLAIF (Training Pipeline) |
|---|---|---|
| **Purpose** | Evaluate AI models on code quality | Generate training data for Rigour's local Qwen model |
| **The LLM's role** | *Subject being tested* — generates patches | *Teacher labelling data* — reviews code quality |
| **Execution** | Sequential (must wait for each LLM response to apply patch + run Rigour) | Async-friendly (findings are independent) |
| **Batch API** | ❌ Cannot use — sequential by design | ✅ Uses Anthropic Batch API (50% cheaper) |
| **Output** | Pass/fail scores per model | DPO training pairs for QLoRA fine-tuning |
| **Entry point** | `runner/harness.py`, `scripts/run_full_benchmark.py` | `rlaif/generate.py` |

**Why DriftBench can't use the Batch API:** The benchmark must apply the LLM's patch and run Rigour before it can evaluate the next task. There's no way to submit all requests first and collect later — each step depends on the previous result.

**Why RLAIF uses the Batch API:** Each repo's AST facts are independent. All prompts can be submitted at once, and findings are collected asynchronously (usually under 1 hour). This saves ~50% on teacher model costs for weekly data generation runs.

---

## RLAIF Training Pipeline

The RLAIF pipeline generates DPO training data for fine-tuning Rigour's local Qwen model, which powers the `rigour check --deep` analysis mode.

### Model Tiers

| Tier | Base Model | Use Case | CLI Flag |
|------|-----------|----------|----------|
| **deep** (default) | Qwen2.5-Coder-1.5B-Instruct | Full power, code-specialized pretrain + QLoRA, company-hosted | *(none — default)* |
| **lite** | Qwen3.5-0.8B | Lightweight, runs on any CPU, ships as default sidecar | `--lite` |
| **legacy** | Qwen2.5-Coder-0.5B-Instruct | Previous default, kept for reproducibility | `--legacy` |

**Two product versions:** The **deep** model (Qwen2.5-Coder-1.5B) is the full-power version — code-specialized pretrain + QLoRA fine-tuning gives the best detection accuracy. Companies host this for their team. The **lite** model (Qwen3.5-0.8B) is the lightweight version that ships as the default sidecar in the Rigour CLI — runs on any laptop CPU with ~3× faster inference. Both models are trained via the same RLAIF pipeline.

### Pipeline Stages

1. **Clone & Extract** — Clone public repos and extract AST facts (classes, functions, imports, metrics) using `rlaif/facts.py`
2. **Teacher Analysis** — Send facts in batches to a strong teacher model (Claude, GPT, DeepSeek, etc.) to generate quality findings. Transient failures retry with exponential backoff (2s → 4s → 8s).
3. **Structural Verification** — Each finding passes 14 checks across 4 tiers (entity existence, metric thresholds, cross-file relationships, confidence floors) in `rlaif/verifier.py`
4. **Pass@2 Retry** — Rejected findings get a second chance: the rejection reason + original AST facts are sent back to the teacher as a refinement prompt. Corrected findings go through the same 14 checks.
5. **DPO Formatting** — Verified and rejected pairs are formatted into DPO training data. Pass@2 recoveries are weighted at 0.6× (lower than first-pass verified to account for correction bias).
6. **QLoRA Fine-tune** — `rlaif/finetune.py` runs SFT + DPO training on the formatted data
7. **GGUF Export** — `rlaif/export_gguf.py` exports the merged model to GGUF for local inference inside the Rigour CLI sidecar

### Pass@2 Retry

When the verifier rejects a finding (e.g., *"entity UserManager not found in AST"*), the rejection reason plus the original facts are sent back to the teacher model as a correction prompt. The corrected finding goes through the same 14 structural checks. Findings that pass on retry are tagged `pass2:verified_retry`. Expected gain: **15–20% more verified training data per run** with no additional repos required.

### Batch API (50% Cost Savings)

For weekly scheduled runs where latency doesn't matter, submit all prompts to the Anthropic Message Batches API upfront and collect results asynchronously (usually under 1 hour). Pass@2 retries for batch findings still use the live API since they require individual round-trips.

```bash
# Live mode — any provider, synchronous
python -m rlaif.generate --provider deepseek --model-name deepseek-chat --repo "expressjs/express"

# Live mode — Anthropic teacher, all repos in repos_training.json
python -m rlaif.generate --provider anthropic --model-name claude-sonnet-4-20250514

# Batch mode — 50% cheaper, Anthropic only, async submit
python -m rlaif.generate --batch --repo "expressjs/express"

# Collect completed batch results
python -m rlaif.generate --batch-collect <BATCH_ID> --output rlaif/data

# Collect all pending batches at once
python -m rlaif.generate --batch-collect-all --output rlaif/data

# Disable Pass@2 retry (faster, fewer API calls)
python -m rlaif.generate --no-retry --repo "expressjs/express"

# Format DPO pairs for fine-tuning
python -m rlaif.format_dpo --db rlaif/data/training_data.db

# Fine-tune Qwen2.5-Coder-1.5B via QLoRA (default — best quality)
python -m rlaif.finetune --sft rlaif/data/sft_data.jsonl --dpo rlaif/data/dpo_data.jsonl

# Fine-tune lite model (Qwen3.5-0.8B, lightweight sidecar)
python -m rlaif.finetune --lite --output rlaif/models/rigour-v1-lite

# Fine-tune legacy model (Qwen2.5-Coder-0.5B, for reproducibility)
python -m rlaif.finetune --legacy --output rlaif/models/rigour-v1-legacy

# Export to GGUF for local inference
python -m rlaif.export_gguf --model rlaif/models/rigour-v1/merged --output rlaif/models/rigour-v1

# Export + upload to HuggingFace
python -m rlaif.export_gguf --model rlaif/models/rigour-v1/merged --output rlaif/models/rigour-v1 --upload rigour-labs/rigour-deep-v1-gguf
```

Supported providers: Anthropic, OpenAI, DeepSeek, Groq, Together AI, Fireworks, Mistral, Gemini, Ollama, and any OpenAI SDK-compatible endpoint. The full pipeline runs weekly via GitHub Actions (`.github/workflows/rlaif-pipeline.yml`).

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding New Tasks

1. Create a new directory under `datasets/<repo>/`
2. Add a task JSON with `intent`, `repository`, `base_sha`, and `golden_patch`
3. Create the golden patch file (the correct reference implementation)
4. Add at least one drift candidate patch (the deliberately flawed version)
5. Configure Rigour rules in `datasets/<repo>/.rigour/config.yaml`

### Adding New Repositories

1. Add the repository to `datasets/` with an appropriate Rigour config
2. Register it in `model_config.json` if needed
3. Submit at least 3 tasks covering different drift categories

### Adding New Models

Add an entry to `model_config.json`:

```json
{
    "model_config": {
        "provider/model-name": {
            "mode": "chat",
            "max_tokens": 4096,
            "display_name": "Human-Readable Name"
        }
    }
}
```

The key must be a valid LiteLLM model string (e.g., `anthropic/claude-sonnet-4`, `openai/gpt-5.2`).

---

## Roadmap

- [ ] Expand to 100+ tasks across 20 repositories
- [ ] Add Python-specific drift detection (mypy, ruff integration)
- [ ] Support for multi-file changes
- [x] CI/CD integration for automated benchmarking (GitHub Actions weekly pipeline)
- [ ] Public API for running benchmarks
- [x] QLoRA fine-tune script for Qwen model training (Qwen2.5-Coder-1.5B default)
- [x] GGUF export + HuggingFace upload
- [x] Qwen3.5-0.8B as lite tier (lightweight sidecar for individual devs)
- [x] Anthropic Batch API for 50% cost savings on RLAIF
- [x] Pass@2 retry pipeline for 15–20% more training data
- [x] Parallel benchmark runner with isolated workspaces
- [x] Exponential backoff retry on transient LLM API errors
- [x] Subprocess timeouts on all git and Rigour CLI operations
- [ ] Auto-update local Qwen model from HuggingFace in Rigour CLI

---

## Powered By

- **[Rigour](https://github.com/rigour-labs/rigour)** — Code quality gate engine
- **[LiteLLM](https://github.com/BerriAI/litellm)** — Universal LLM API interface

---

## Rigour CLI Reference

| Command | Purpose |
|:--------|:--------|
| `rigour check` | Validate changes against configured gates |
| `rigour check --json` | Machine-readable JSON output (used by DriftBench) |
| `rigour check --ci` | CI mode with appropriate exit codes |
| `rigour check --deep` | Deep LLM-powered analysis (lite: Qwen3.5-0.8B, or `--pro` for full: Qwen2.5-Coder-1.5B) |
| `rigour init` | Set up Rigour in a project |
| `rigour explain` | Detailed explanation of last check results |
| `rigour run` | Supervisor loop for iterative refinement |
| `rigour studio` | Dashboard for monitoring |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use DriftBench in your research, please cite:

```bibtex
@software{driftbench2026,
  title = {DriftBench: A Benchmark for Measuring AI Code Drift},
  author = {Rigour Labs},
  year = {2026},
  url = {https://github.com/rigour-labs/driftbench}
}
```
