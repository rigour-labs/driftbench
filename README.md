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

## 📊 Leaderboard
Follow the latest results at [rigour.run/driftbench](https://rigour.run/driftbench).

## 📄 License
MIT
