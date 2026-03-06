# Manual Training Guide — Rigour RLAIF Pipeline

Run the full fine-tuning → export → upload pipeline on a Lightning.ai GPU studio, bypassing GitHub Actions for GPU-dependent work.

## Architecture

```
GitHub Actions (automated, weekly)        Lightning.ai (manual, GPU)
┌─────────────────────────────┐          ┌──────────────────────────────┐
│ Job 1: Crawl repos          │          │                              │
│ Job 2: Generate RLAIF data  │───────►  │  ./scripts/run_training.sh   │
│         (LLM + AST verify)  │  HF hub  │    ├─ finetune (QLoRA)       │
│         Upload to HF        │          │    ├─ dequantize (fp16)       │
└─────────────────────────────┘          │    ├─ export GGUF (Q4_K_M)   │
                                         │    └─ upload to HuggingFace   │
                                         └──────────────────────────────┘
```

Data generation runs on GitHub Actions (CPU-only, uses LLM APIs). Training runs manually on Lightning.ai because it needs a GPU.

## Prerequisites

### 1. Lightning.ai Studio

Create a GPU studio (T4 minimum, A10 recommended). SSH in or use the terminal.

### 2. HuggingFace Token

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Needs **write** access to `rigour-labs/` repos.

### 3. Dependencies

```bash
# IMPORTANT: Check if torch+CUDA is already installed before running pip install torch.
# Lightning.ai ships with CUDA torch pre-installed. Running `pip install torch`
# will overwrite it with CPU-only torch and break everything.

python3 -c "import torch; print(torch.cuda.is_available())"  # Should print True

# If True, skip torch install. Only install the rest:
pip install transformers peft trl datasets bitsandbytes huggingface_hub accelerate

# If False (fresh env), install everything:
pip install torch transformers peft trl datasets bitsandbytes huggingface_hub accelerate
```

### 4. Clone the Repo

```bash
git clone https://github.com/rigour-labs/driftbench.git
cd driftbench
chmod +x scripts/run_training.sh
```

## Usage

### Full Run (Both Tiers)

```bash
./scripts/run_training.sh
```

Trains both `deep` (Qwen2.5-Coder-1.5B) and `lite` (Qwen2.5-Coder-0.5B), auto-resolves the next version number from HuggingFace.

### Single Tier

```bash
./scripts/run_training.sh --tier deep
./scripts/run_training.sh --tier lite
```

### Explicit Version

```bash
./scripts/run_training.sh --version 7
```

### Skip Fine-tuning (Export Only)

If you already fine-tuned and uploaded the merged model to HuggingFace, skip straight to GGUF export:

```bash
./scripts/run_training.sh --skip-finetune --version 5
```

The script downloads the merged model from `rigour-labs/rigour-{tier}-v{version}-merged` on HuggingFace and exports it.

### Skip Export

If you only want to fine-tune without GGUF export:

```bash
./scripts/run_training.sh --skip-export
```

### Dry Run

See what would execute without running anything:

```bash
./scripts/run_training.sh --dry-run
```

### Combine Flags

```bash
./scripts/run_training.sh --tier lite --version 6 --skip-finetune --dry-run
```

## What the Script Does

### Step 1: Preflight

- Checks `HF_TOKEN` is set
- Verifies GPU availability (torch + CUDA)
- Auto-resolves version from `latest_version.json` on HuggingFace (or uses `--version`)

### Step 2: Fine-tune (per tier)

Calls `scripts/finetune_model.py`:

1. Downloads SFT + DPO data from `rigour-labs/rigour-rlaif-data` on HuggingFace
2. Loads base model with QLoRA (4-bit quantization, LoRA rank 16)
3. Runs SFT (supervised fine-tuning) phase
4. Runs DPO (direct preference optimization) phase
5. Merges LoRA adapter back into base model (fp16, on CPU for clean weights)
6. Uploads merged model to `rigour-labs/rigour-{tier}-v{version}-merged`

After both tiers complete, updates `latest_version.json` on HuggingFace so the Rigour product auto-downloads the new model.

### Step 3: Export GGUF (per tier)

1. Builds `llama.cpp` if not already present (cmake + make)
2. Downloads merged model from HuggingFace if not local
3. Dequantizes from bitsandbytes 4-bit to fp16 if needed (`scripts/dequantize_model.py`)
4. Converts to GGUF format using `llama.cpp/convert_hf_to_gguf.py`
5. Quantizes to Q4_K_M (4-bit, good quality/size tradeoff)
6. Uploads quantized GGUF to `rigour-labs/rigour-{tier}-v{version}-gguf`

## Tier Configurations

| Parameter | Deep (1.5B) | Lite (0.5B) |
|-----------|------------|-------------|
| Base model | Qwen2.5-Coder-1.5B-Instruct | Qwen2.5-Coder-0.5B-Instruct |
| SFT epochs | 3 | 2 |
| SFT batch × grad_accum | 4 × 4 | 8 × 2 |
| DPO epochs | 1 | 1 |
| DPO batch × grad_accum | 2 × 8 | 4 × 4 |
| Max length | 1024 | 768 |
| SFT learning rate | 2e-4 | 3e-4 |
| DPO learning rate | 5e-5 | 8e-5 |

## HuggingFace Repos

After a successful run (e.g., version 6):

| Repo | Contents |
|------|----------|
| `rigour-labs/rigour-rlaif-data` | Training data (SFT + DPO) + `latest_version.json` |
| `rigour-labs/rigour-deep-v6-merged` | Deep tier merged model (fp16) |
| `rigour-labs/rigour-lite-v6-merged` | Lite tier merged model (fp16) |
| `rigour-labs/rigour-deep-v6-gguf` | Deep tier GGUF (Q4_K_M) |
| `rigour-labs/rigour-lite-v6-gguf` | Lite tier GGUF (Q4_K_M) |

## Versioning & Rollback

Each training run creates versioned repos (v5, v6, v7...). Old versions are never overwritten.

The `latest_version.json` file on `rigour-labs/rigour-rlaif-data` tells the Rigour product which version to download. To rollback:

```bash
# Point latest_version.json back to v5
python scripts/update_version.py --version 5
```

The Rigour product checks this file once per day (with a 5-second timeout) and auto-downloads the referenced version.

## Troubleshooting

### "CUDA required for bitsandbytes dequantization"

You're on a CPU-only machine. Dequantization needs a GPU. Use a Lightning.ai studio with a GPU attached.

### "pip install torch" broke CUDA

Lightning.ai pre-installs torch with CUDA. Running `pip install torch` installs the CPU-only version. Fix:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Or better: don't run `pip install torch` if CUDA torch is already working.

### llama.cpp build fails

If the llama.cpp directory is corrupted from a previous run:

```bash
rm -rf llama.cpp
./scripts/run_training.sh --skip-finetune --version 5
```

The script will re-clone and build llama.cpp.

### Tokenizer errors during dequantization

The `dequantize_model.py` script copies tokenizer files directly instead of reloading them via `AutoTokenizer.from_pretrained()`, which can crash with transformers version mismatches. If you still see tokenizer errors, ensure the merged model directory has all `tokenizer*` files.

## Individual Scripts

Each stage can be run independently for debugging:

```bash
# Fine-tune only
python scripts/finetune_model.py --tier deep --version 6 --upload

# Dequantize only
python scripts/dequantize_model.py --model-dir rlaif/models/rigour-deep-v6/merged

# Update version pointer
python scripts/update_version.py --version 6 --dry-run
```

All scripts support `--help` for full argument documentation.
