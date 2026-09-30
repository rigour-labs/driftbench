# RLAIF Pipeline

Synthetic training data generator for Rigour's deep analysis model. Uses a strong teacher model (Claude, GPT-4o, DeepSeek, etc.) to label public repos, then filters findings through structural verification to produce high-quality DPO training pairs.

## How It Works

```
Public repos  →  AST fact extraction  →  Teacher model labels  →  Structural verifier  →  SQLite
                  (deterministic)         (any provider)           (14 checks, 4 tiers)     ↓
                                                                                        DPO pairs → HuggingFace
```

1. **Clone** public GitHub repos (30 training repos, separate from DriftBench eval set)
2. **Extract** AST facts — classes, functions, imports, exports, Go structs/interfaces
3. **Label** — teacher model analyzes facts and produces findings with confidence scores
4. **Verify** — 14 structural checks filter hallucinated findings (entity name matching, cross-file import graphs, language-specific naming rules, confidence floors)
5. **Store** — verified = positive examples, dropped = negative examples → SQLite
6. **Format** — SFT pairs + DPO preference pairs → JSONL for HuggingFace trl

## Module Structure

```
rlaif/
├── facts.py              # AST fact extraction (Python port of rigour-core)
├── verifier.py           # 4-tier structural verification (14 category checks)
├── provider.py           # Provider-agnostic teacher model (litellm routing)
├── generate.py           # Pipeline orchestrator + CLI
├── format_dpo.py         # DPO/SFT formatter + HuggingFace export
├── repos_training.json   # 30 public repos for training (no overlap with eval)
└── README.md
```

## Quick Start

```bash
# Install deps
pip install -r requirements.txt
pip install litellm huggingface_hub

# Run with Anthropic (default)
ANTHROPIC_API_KEY=sk-ant-... python -m rlaif.generate \
  --repo "expressjs/express" --output rlaif/data

# Run with DeepSeek
python -m rlaif.generate \
  --provider deepseek --model-name deepseek-chat \
  --api-key sk-... --repo "expressjs/express"

# Run with local Ollama
python -m rlaif.generate \
  --provider ollama --model-name qwen2.5-coder:7b \
  --api-base http://localhost:11434 --repo "expressjs/express"

# Format DPO pairs
python -m rlaif.format_dpo --db rlaif/data/training_data.db
```

## Provider Support

Works with any OpenAI SDK-compatible provider via `--provider` + `--model-name`:

| Provider | Example |
|---|---|
| Anthropic | `--provider anthropic --model-name claude-sonnet-4-6` |
| OpenAI | `--provider openai --model-name gpt-4o` |
| DeepSeek | `--provider deepseek --model-name deepseek-chat` |
| Groq | `--provider groq --model-name llama-3.1-70b-versatile` |
| Together | `--provider together --model-name meta-llama/Llama-3.1-70B-Instruct-Turbo` |
| Ollama | `--provider ollama --model-name qwen2.5-coder:7b --api-base http://localhost:11434` |
| Any proxy | `--provider openai --model-name my-model --api-base https://my-proxy.com/v1` |

Or set env vars: `MODEL_PROVIDER`, `MODEL_NAME`, `API_KEY`, `API_BASE`.

## Verification Tiers

The verifier prevents false positives by requiring structural evidence before accepting a finding:

| Tier | Categories | Check |
|---|---|---|
| **Entity name** | lazy_class, feature_envy, api_design, etc. | Entity must exist in AST facts |
| **Structural** | dead_code, naming_convention, hardcoded_config, data_clump, performance | Language-specific rules (Go snake_case, Python camelCase, import graph) |
| **Cross-file** | circular_dependency, dry_violation, shotgun_surgery, inappropriate_intimacy | Must find evidence across multiple files |
| **Confidence floor** | architecture, package_cohesion, code_smell, language_idiom | Raised from 0.3 to 0.5 |

## CI/CD

The GitHub Actions workflow (`.github/workflows/rlaif-pipeline.yml`) runs weekly and uploads to HuggingFace. See [docs/RLAIF_SETUP.md](../docs/RLAIF_SETUP.md) for secrets setup.

## Max review model: free, measured fine-tuning

The `max` tier is Qwen2.5-Coder-7B, the strongest model a 16GB laptop runs.
It is fine-tuned on review pairs labelled by later fixes, with no teacher model
and no API spend, and ships only if it beats the stock model on the Review Arena.

1. **Mine pairs** from training repositories. Evaluation repositories are refused.

   ```bash
   RIGOUR_CLI=/path/to/rigour/packages/rigour-cli/dist/cli.js \
     python -m rlaif.review.mine_pairs --repo honojs/hono --max-commits 400
   python -m rlaif.review.export_sft --out rlaif/data/review_sft.jsonl
   ```

2. **Train on a free GPU**: a Kaggle notebook with a T4, about 30 GPU-hours a
   week (Colab works too).

   ```bash
   pip install torch transformers peft trl datasets bitsandbytes accelerate huggingface_hub
   python -m rlaif.finetune --max --sft rlaif/data/review_sft.jsonl --dpo "" \
     --sft-epochs 1 --output rlaif/models/rigour-max
   python -m rlaif.export_gguf --model rlaif/models/rigour-max/merged --max --version 6.0.0
   ```

3. **Measure before publishing.** Score the GGUF on the arena against the stock max model:

   ```bash
   RIGOUR_MODEL_PATH=/path/to/rigour-max-v6.0.0-q4_k_m.gguf \
     python -m arena run --repo <eval repo> --sample 40 --tool rigour-max-candidate
   python -m arena run --repo <eval repo> --sample 40 --tool rigour-max
   python -m arena score --all
   ```

4. **Publish** only when judged recall beats `rigour-max` and judged precision
   does not drop. Upload with `export_gguf --upload`, then bump
   `latest_version.json` last. Rigour picks the model up on its next run.
