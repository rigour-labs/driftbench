# DriftBench 🏎️

**The Full-Spectrum PR Drift & Intent Preservation Benchmark.**

DriftBench measures the ability of AI tools to preserve product intent and engineering invariants when making changes. While traditional benchmarks focus on "passing tests," DriftBench detects "drift"—changes that are syntactically correct and pass unit tests but violate core design patterns, security rules, or implicit business logic.

## 🌈 Full-Spectrum Metrics

DriftBench evaluates AI agents across 6 major dimensions:

| Category | Primary Metric | Agent Failure Example |
| :--- | :--- | :--- |
| **Logic Drift** | Intent Alignment | Bypassing an auth check in a new endpoint. |
| **Pattern Drift** | Code Reuse | Re-implementing a `fetch` wrapper instead of using `ApiClient`. |
| **Arch Drift** | Structural Integrity | Introducing a circular dependency between layers. |
| **Stale Drift** | Modernity | Using `moment.js` when the project is standardized on `date-fns`. |
| **Security Drift** | Safety Rings | Accidental PII logging or direct DB access. |
| **Standard Drift** | Quality Gates | Refactoring a simple function into a high-complexity mess. |

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- Docker (for sandboxed execution)
- Rigour CLI (`npm install -g @rigour-labs/cli`)

### Run Benchmark (LLM Evaluation)
```bash
./run_eval.sh --tool anthropic/claude-4-5-opus
# OR
./run_eval.sh --tool openai/gpt-5.2-codex
```

## 🛰️ Project Status: Infrastructure Ready
All tools for **DriftBench v0.1** are built and verified.
- [x] 50-Task Dataset
- [x] Execution Engine
- [x] LLM Evaluation Harness (LiteLLM)
- [x] Leaderboard Aggregator

### 🚀 Next Step: Baseline Execution
To generate the first official scores, run the harness against your preferred models:
```bash
export ANTHROPIC_API_KEY="your_key"
./run_eval.sh --model claude-3-5-sonnet
```

### ☁️ Railway Deployment (Optional)
Deploy this repo to Railway to:
1.  **Enable Remote PR Benchmarking**: Call the API from GitHub Actions.
2.  **Live Leaderboard**: Keep `LEADERBOARD.md` auto-updated via the `/leaderboard` endpoint.
3.  **Public API**: Let other tools trigger runs via `POST /run/{task_id}`.

## 📊 Leaderboard (Draft)
*Results pending baseline execution.*
| Model | Global DDR | FPR | Status |
| :--- | :--- | :--- | :--- |
| Claude 3.5 Sonnet | -- | -- | Awaiting Run |
| GPT-4o | -- | -- | Awaiting Run |

## 📄 License
MIT
