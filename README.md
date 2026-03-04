# DriftBench

**The Full-Spectrum PR Drift & Intent Preservation Benchmark for AI Code Generation.**

DriftBench measures the ability of AI coding tools to preserve product intent and engineering invariants when making changes. While traditional benchmarks focus on "passing tests," DriftBench detects **drift**—changes that are syntactically correct and pass unit tests but violate core design patterns, security rules, or implicit business logic.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## Why DriftBench?

Traditional code benchmarks (HumanEval, MBPP, SWE-bench) measure whether AI-generated code is *correct*. DriftBench measures whether it's *appropriate*:

- Does it follow the project's established patterns?
- Does it avoid introducing security vulnerabilities?
- Does it use modern, non-deprecated APIs?
- Does it respect architectural boundaries?

An AI agent might write code that passes all tests but uses `var` instead of `const`, implements custom auth instead of using the existing `AuthService`, or accidentally logs sensitive data.

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

# Verify version
npx rigour --version
```

### Configuration

Create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=...
```

### Running Benchmarks

```bash
# Run a single task
python -m runner.harness --model anthropic/claude-opus-4-5-20251101 --task lodash-stale-001

# Run all tasks for a model
python -m runner.harness --model anthropic/claude-opus-4-5-20251101 --all

# Run full benchmark (all models, all tasks)
python scripts/run_full_benchmark.py

# Dry run to see what would execute
python scripts/run_full_benchmark.py --dry-run

# Generate leaderboard from results
python scripts/snapshot_leaderboard.py
```

## Project Structure

```
driftbench/
├── datasets/                 # Benchmark tasks organized by repository
│   ├── lodash/              # Tasks for lodash/lodash
│   ├── flask/               # Tasks for pallets/flask
│   ├── django/              # Tasks for django/django
│   └── ...
├── runner/
│   ├── engine.py            # Core benchmark execution engine
│   └── harness.py           # LLM harness for generating patches
├── rlaif/                   # RLAIF training data pipeline
│   ├── facts.py             # AST fact extraction
│   ├── verifier.py          # 4-tier structural verification (14 checks)
│   ├── provider.py          # Provider-agnostic teacher + Pass@2 retry
│   ├── generate.py          # Pipeline orchestrator + CLI
│   ├── batch_provider.py    # Anthropic Batch API (50% cheaper)
│   ├── batch_orchestrator.py # Batch submit/collect + retry
│   ├── format_dpo.py        # DPO/SFT formatter with retry weighting
│   ├── finetune.py          # QLoRA fine-tune (SFT + DPO)
│   ├── export_gguf.py       # GGUF export + HuggingFace upload
│   └── repos_training.json  # 30 training repos (no eval overlap)
├── scripts/
│   ├── run_full_benchmark.py    # Run all models against all tasks
│   └── snapshot_leaderboard.py  # Generate leaderboard JSON
├── results/                 # Benchmark results (gitignored)
├── model_config.json        # Model configuration
└── .env.example             # Environment variables template
```

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
    "rigour_config": "datasets/lodash/rigour_config.yaml",
    "drift_candidates": [...]
}
```

## Leaderboard

Live results are available at [rigour.run](https://rigour.run) (coming soon).

| Model | Pass Rate | DDR | Tasks | Status |
|-------|-----------|-----|-------|--------|
| Claude Opus 4.6 | --% | --% | 50 | 🆕 Pending |
| GPT-5.3 Codex | --% | --% | 50 | 🆕 Pending |
| Claude Opus 4.5 | --% | --% | 50 | Running |
| Claude Sonnet 4 | --% | --% | 50 | Pending |
| GPT-5.2 | --% | --% | 50 | Pending |
| Gemini 3 Pro | --% | --% | 50 | Pending |

*Results are updated automatically after benchmark runs.*

## How It Works

1. **Task Selection**: Each task defines an intent (what the AI should implement) and a target repository
2. **LLM Generation**: The harness prompts the LLM to generate a unified diff patch
3. **Patch Application**: The generated patch is applied to a clean checkout of the repository
4. **Drift Detection**: Rigour analyzes the modified files for violations
5. **Scoring**: Results are compared against golden patches to determine pass/fail

### Key Metrics

- **Pass Rate**: Percentage of tasks where no drift was detected
- **DDR (Drift Detection Rate)**: How often the model introduces detectable drift
- **Accuracy**: Whether the model's result matches the golden baseline

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding New Tasks

1. Create a new directory under `datasets/<repo>/`
2. Add task JSON with intent, repository, and golden patch
3. Create the golden patch file (expected correct implementation)
4. Add a drift patch (expected incorrect implementation for validation)
5. Configure Rigour rules in `rigour_config.yaml`

### Adding New Repositories

1. Add repository metadata to `scripts/snapshot_leaderboard.py`
2. Create dataset directory with appropriate Rigour config
3. Submit at least 3 tasks covering different drift categories

## RLAIF Training Pipeline

DriftBench includes a full RLAIF (Reinforcement Learning from AI Feedback) pipeline for training Rigour's deep analysis model. The pipeline clones public repos, extracts AST facts, uses a strong teacher model to generate findings, validates them through 14 structural verification checks across 4 tiers, and outputs DPO training pairs for fine-tuning.

### Pipeline Steps

The pipeline runs in 6 stages: clone repos and extract AST facts, send facts to teacher model for analysis, run structural verification (14 checks across entity, metric, cross-file, and confidence tiers), retry rejected findings with Pass@2, format verified/rejected pairs into DPO training data, and fine-tune via QLoRA.

### Pass@2 Retry Pipeline

Rejected findings get a second chance. When the verifier rejects a finding (e.g., "entity UserManager not in AST"), the rejection reason plus the original AST facts are sent back to the teacher as a refinement prompt. The teacher produces a corrected finding, which goes through the same 14 verification checks again. Findings that pass on retry are tagged `verified_retry` and weighted at 0.6x in DPO training (lower than first-pass verified findings to account for potential correction bias). Findings that fail twice become permanent negative examples. Expected gain: 15-20% more verified training data per run, zero additional repos needed.

### Batch API (50% Cheaper)

For weekly scheduled runs where latency doesn't matter, the pipeline uses the Anthropic Message Batches API which is 50% cheaper than live calls. Batches are submitted asynchronously and polled until completion (usually under 1 hour). Pass@2 retries for batch-collected findings run via live API since they need individual round-trips.

### Usage

```bash
# Live mode — generate training data with any provider
python -m rlaif.generate --provider deepseek --model-name deepseek-chat --repo "expressjs/express"

# Batch mode — 50% cheaper via Anthropic Batch API
python -m rlaif.generate --batch --repo "expressjs/express"
python -m rlaif.generate --batch-collect-all --output rlaif/data

# Disable Pass@2 retry (faster, fewer API calls)
python -m rlaif.generate --no-retry --repo "expressjs/express"

# Format DPO pairs for fine-tuning
python -m rlaif.format_dpo --db rlaif/data/training_data.db

# Fine-tune Qwen via QLoRA
python -m rlaif.finetune --sft rlaif/data/sft_data.jsonl --dpo rlaif/data/dpo_data.jsonl

# Export to GGUF for local inference
python -m rlaif.export_gguf --model rlaif/models/rigour-v1/merged --output rlaif/models/rigour-v1
```

Supports Anthropic, OpenAI, DeepSeek, Groq, Together, Fireworks, Mistral, Gemini, Ollama, and any OpenAI SDK-compatible endpoint. The full pipeline runs weekly via GitHub Actions (see `.github/workflows/rlaif-pipeline.yml`).

## Roadmap

- [ ] Expand to 100+ tasks across 20 repositories
- [ ] Add Python-specific drift detection (mypy, ruff integration)
- [ ] Support for multi-file changes
- [x] CI/CD integration for automated benchmarking (GitHub Actions weekly pipeline)
- [ ] Public API for running benchmarks
- [x] QLoRA fine-tune script for Qwen model training
- [x] GGUF export + HuggingFace upload
- [x] Anthropic Batch API for 50% cost savings
- [x] Pass@2 retry pipeline for 15-20% more training data
- [ ] Auto-update model from HuggingFace in Rigour CLI

## Powered By

- **[Rigour](https://github.com/rigour-labs/rigour)** - Code quality gate engine
- **[LiteLLM](https://github.com/BerriAI/litellm)** - Universal LLM API interface

---

## 🛠️ Rigour CLI Commands Reference

| Command | Purpose |
| :--- | :--- |
| `rigour check` | Validates staged changes against safety rules |
| `rigour check --ci` | CI mode with appropriate output |
| `rigour init` | Setup Rigour in project |
| `rigour explain` | Detailed explanation of validation results |
| `rigour run` | Supervisor loop for iterative refinement |
| `rigour studio` | Dashboard for monitoring |

## License

MIT License - see [LICENSE](LICENSE) for details.

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
