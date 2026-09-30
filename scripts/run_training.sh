#!/usr/bin/env bash
#
# run_training.sh — Full training pipeline for Lightning.ai GPU studio.
#
# Runs everything locally: finetune → dequantize → export GGUF → evaluate → publish.
# Publishing (--upload) happens only after every tier passes the eval gate, and
# latest_version.json is bumped last, so clients are never told about a model
# that failed evaluation or has not been uploaded yet.
# No GitHub Actions needed. SSH in, run this, watch it work.
#
# Usage:
#   ./scripts/run_training.sh                     # Both tiers, auto-version (v6, v7, ...)
#   ./scripts/run_training.sh --version 5         # Explicit version
#   ./scripts/run_training.sh --tier lite         # Single tier
#   ./scripts/run_training.sh --skip-finetune     # Skip to export (model already on HF)
#   ./scripts/run_training.sh --dry-run           # Print what would run
#   ./scripts/run_training.sh --status            # Check training status (after closing terminal)
#
# Cloud-safe: run with nohup so it survives terminal close:
#   nohup ./scripts/run_training.sh --tier lite --version 1.0.0 > training.log 2>&1 &
#   cat rlaif/models/training-status.json          # Check progress anytime
#
# Prerequisites:
#   - Lightning.ai studio with GPU (T4/A10/etc.)
#   - export HF_TOKEN=hf_xxx
#   - pip install torch transformers peft trl datasets bitsandbytes huggingface_hub accelerate
#   - git clone llama.cpp (script handles this)
#
set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────
TIER="both"
VERSION=""
SKIP_FINETUNE=false
SKIP_EXPORT=false
UPLOAD=false
BASELINE_VERSION=""
DRY_RUN=false
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ─── Parse args ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --tier)        TIER="$2"; shift 2 ;;
    --version)     VERSION="$2"; shift 2 ;;
    --skip-finetune) SKIP_FINETUNE=true; shift ;;
    --skip-export) SKIP_EXPORT=true; shift ;;
    --upload)      UPLOAD=true; shift ;;
    --baseline-version) BASELINE_VERSION="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=true; shift ;;
    --status)
      STATUS_FILE="rlaif/models/training-status.json"
      if [ -f "$STATUS_FILE" ]; then
        echo "═══════════════════════════════════════════"
        echo "  Training Status"
        echo "═══════════════════════════════════════════"
        python3 -c "
import json
with open('$STATUS_FILE') as f:
    s = json.load(f)
for k, v in s.items():
    print(f'  {k}: {v}')
"
      else
        echo "No training status found. Training may not have started yet."
        echo "Check: cat training.log"
      fi
      exit 0 ;;
    -h|--help)
      echo "Usage: $0 --version MAJOR.MINOR.PATCH [--tier deep|lite|both] [--upload] [--baseline-version V] [--skip-finetune] [--skip-export] [--dry-run]"
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

cd "$REPO_ROOT"

# ─── Preflight checks ─────────────────────────────────────────
echo "═══════════════════════════════════════════"
echo "  Rigour Training Pipeline (Local Runner)"
echo "═══════════════════════════════════════════"

# ─── Resolve version (SemVer: MAJOR.MINOR.PATCH) ─────────────
# MAJOR: training format change, base model change
# MINOR: new repos, dataset update, hyperparameters
# PATCH: bug fix, retrain same data
if [ -z "$VERSION" ]; then
  echo "ERROR: --version is required (SemVer format, e.g., 2.0.0)"
  echo ""
  echo "  To see current version:"
  echo "    python scripts/update_version.py --bump patch --dry-run"
  echo ""
  echo "  Version guidelines:"
  echo "    MAJOR (X.0.0): New training format, base model change, pipeline rewrite"
  echo "    MINOR (0.X.0): New training repos, dataset expansion, hyperparameter tuning"
  echo "    PATCH (0.0.X): Bug fix, retrain with same data/format"
  echo ""
  echo "  Examples:"
  echo "    ./run_training.sh --version 2.0.0  # New aligned prompt format + enterprise repos"
  echo "    ./run_training.sh --version 2.1.0  # Added 10 more repos, same format"
  echo "    ./run_training.sh --version 2.0.1  # Fixed tokenizer bug, retrained"
  exit 1
fi

# Validate SemVer format
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "ERROR: Version must be SemVer format (MAJOR.MINOR.PATCH), got: $VERSION"
  echo "  Examples: 2.0.0, 2.1.0, 2.0.1"
  exit 1
fi

echo "Version: v${VERSION}"

# ─── Build tier list ──────────────────────────────────────────
if [ "$TIER" = "both" ]; then
  TIERS=("deep" "lite")
else
  TIERS=("$TIER")
fi

echo "Tiers: ${TIERS[*]}"
echo "Version: v${VERSION}"
echo "Skip finetune: ${SKIP_FINETUNE}"
echo "Skip export: ${SKIP_EXPORT}"
echo ""

if [ "$DRY_RUN" = true ]; then
  echo "[DRY RUN] Would run the following:"
  for t in "${TIERS[@]}"; do
    if [ "$SKIP_FINETUNE" = false ]; then
      echo "  python scripts/finetune_model.py --tier $t --version $VERSION $([ "$UPLOAD" = true ] && echo --upload)"
    fi
    if [ "$SKIP_EXPORT" = false ]; then
      echo "  python scripts/dequantize_model.py --model-dir rlaif/models/rigour-${t}-v${VERSION}/merged"
      echo "  python -m rlaif.export_gguf --model rlaif/models/rigour-${t}-v${VERSION}/merged --output rlaif/models/rigour-${t}-v${VERSION} --llama-cpp-path llama.cpp --version $VERSION $([ $t = lite ] && echo --lite)"
    fi
    echo "  python scripts/eval_model.py --tier $t --version $VERSION --gguf-path <quantized gguf> --output rlaif/models/rigour-${t}-v${VERSION}/eval_results.json $([ -n "$BASELINE_VERSION" ] && echo --baseline-version $BASELINE_VERSION)"
  done
  if [ "$UPLOAD" = true ]; then
    echo "  (only if every tier passed) upload each GGUF + eval_results.json, then:"
    echo "  python scripts/update_version.py --version $VERSION --changelog 'Training run v${VERSION}'"
  fi
  exit 0
fi

# ─── Preflight: token and dependencies (after the dry-run exit, so a dry run changes nothing) ───
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=hf_xxx"
  exit 1
fi

echo "Installing dependencies..."
# Use python3 -m pip to ensure we install to the SAME Python that runs scripts
PIP="python3 -m pip"

# Detect platform: CUDA, MPS (Apple Silicon), or CPU
DEVICE=$(python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        print('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print('mps')
    else:
        print('cpu')
except ImportError:
    print('none')
" 2>/dev/null || echo "none")

CORE_DEPS="transformers peft trl datasets huggingface_hub accelerate"

if [ "$DEVICE" = "cuda" ]; then
  echo "  CUDA detected, skipping torch reinstall"
  $PIP uninstall -y torchvision torchaudio 2>/dev/null || true
  $PIP install -q $CORE_DEPS bitsandbytes 2>&1 | tail -1
elif [ "$DEVICE" = "mps" ]; then
  echo "  Apple Silicon (MPS) detected — using fp16 training (no QLoRA needed)"
  $PIP install -q $CORE_DEPS 2>&1 | tail -1
else
  echo "  Installing all packages including torch"
  $PIP install -q torch $CORE_DEPS 2>&1 | tail -1
fi
echo "  Dependencies ready"

echo "Checking GPU..."
python3 -c "
import torch
print(f'  torch={torch.__version__}')
print(f'  CUDA={torch.cuda.is_available()}')
mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
print(f'  MPS={mps}')
if torch.cuda.is_available():
    print(f'  GPU={torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'  VRAM={props.total_memory / 1e9:.1f} GB')
elif mps:
    print(f'  GPU=Apple Silicon (Metal Performance Shaders)')
    import subprocess
    try:
        mem = subprocess.check_output(['sysctl', '-n', 'hw.memsize']).decode().strip()
        print(f'  Unified Memory={int(mem) / 1e9:.0f} GB')
    except Exception:
        pass
else:
    print('  WARNING: No GPU — training will be very slow')
"


# ─── Step 1: Fine-tune ────────────────────────────────────────
if [ "$SKIP_FINETUNE" = false ]; then
  for t in "${TIERS[@]}"; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  FINETUNE: ${t} tier (v${VERSION})"
    echo "════════════════════════════════════════"
    UPLOAD_FLAG=""
    if [ "$UPLOAD" = true ]; then
      UPLOAD_FLAG="--upload"
    fi
    python scripts/finetune_model.py --tier "$t" --version "$VERSION" $UPLOAD_FLAG
  done
else
  echo "Skipping finetune (--skip-finetune)"
fi

# ─── Step 2: Export GGUF ──────────────────────────────────────
if [ "$SKIP_EXPORT" = false ]; then
  # Setup llama.cpp if needed
  if [ ! -f llama.cpp/build/bin/llama-quantize ]; then
    echo ""
    echo "Setting up llama.cpp..."
    rm -rf llama.cpp llama_cpp_fresh 2>/dev/null || true
    git clone --depth 1 https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    pip install -r requirements.txt 2>/dev/null || pip install gguf || true
    cmake -B build
    cmake --build build --config Release -j"$(nproc)"
    cd ..
    echo "llama.cpp ready"
  else
    echo "llama.cpp already built"
  fi

  for t in "${TIERS[@]}"; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  EXPORT GGUF: ${t} tier (v${VERSION})"
    echo "════════════════════════════════════════"

    MODEL_DIR="rlaif/models/rigour-${t}-v${VERSION}"
    MERGED_DIR="${MODEL_DIR}/merged"

    # Download merged model from HF if not present locally
    if [ ! -d "$MERGED_DIR" ]; then
      echo "Downloading merged model from HuggingFace..."
      python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download('rigour-labs/rigour-${t}-v${VERSION}-merged',
                  local_dir='${MERGED_DIR}',
                  token=os.environ['HF_TOKEN'])
print('Downloaded')
"
    fi

    # Dequantize if needed
    python scripts/dequantize_model.py --model-dir "$MERGED_DIR"

    # Fix tokenizer config — transformers saves extra_special_tokens as a list
    # but AutoTokenizer.from_pretrained() expects a dict. Patch it before GGUF export.
    python3 -c "
import json, os
tc_path = os.path.join('${MERGED_DIR}', 'tokenizer_config.json')
if os.path.exists(tc_path):
    with open(tc_path) as f:
        tc = json.load(f)
    changed = False
    if isinstance(tc.get('extra_special_tokens'), list):
        # Convert list to dict: ['<|tok|>'] -> {'<|tok|>': '<|tok|>'}
        tc['extra_special_tokens'] = {t: t for t in tc['extra_special_tokens']}
        changed = True
    if 'added_tokens_decoder' in tc and isinstance(tc['added_tokens_decoder'], list):
        tc.pop('added_tokens_decoder')
        changed = True
    if changed:
        with open(tc_path, 'w') as f:
            json.dump(tc, f, indent=2, ensure_ascii=False)
        print('  Fixed tokenizer_config.json (extra_special_tokens list -> dict)')
    else:
        print('  tokenizer_config.json OK')
"

    # Export GGUF
    TIER_FLAG=""
    if [ "$t" = "lite" ]; then
      TIER_FLAG="--lite"
    elif [ "$t" = "legacy" ]; then
      TIER_FLAG="--legacy"
    fi

    python -m rlaif.export_gguf \
      --model "$MERGED_DIR" \
      --output "$MODEL_DIR" \
      --llama-cpp-path llama.cpp \
      --version "$VERSION" \
      $TIER_FLAG

    echo "✓ ${t} GGUF exported"
  done
else
  echo "Skipping export (--skip-export)"
fi

# ─── Step 3: Evaluate (the gate) ──────────────────────────────
$PIP install -q llama-cpp-python 2>&1 | tail -1
for t in "${TIERS[@]}"; do
  echo ""
  echo "════════════════════════════════════════"
  echo "  EVAL GATE: ${t} tier (v${VERSION})"
  echo "════════════════════════════════════════"
  GGUF=$(ls rlaif/models/rigour-${t}-v${VERSION}/*.gguf 2>/dev/null | grep -v f16 | head -1 || true)
  if [ -z "$GGUF" ]; then
    echo "ERROR: no quantized GGUF for ${t} v${VERSION}; run the export step first"
    exit 1
  fi
  BASELINE_FLAG=""
  if [ -n "$BASELINE_VERSION" ]; then
    BASELINE_FLAG="--baseline-version $BASELINE_VERSION"
  fi
  # Exits non-zero when the gate fails, which stops the script before anything is published.
  python scripts/eval_model.py --tier "$t" --version "$VERSION" --gguf-path "$GGUF" \
    --output "rlaif/models/rigour-${t}-v${VERSION}/eval_results.json" $BASELINE_FLAG
done

# ─── Step 4: Publish (only with --upload, only after every tier passed) ───
if [ "$UPLOAD" = true ]; then
  for t in "${TIERS[@]}"; do
    python3 -c "
import glob
from huggingface_hub import HfApi
from rlaif.export_gguf import upload_to_huggingface
tier, version = '${t}', '${VERSION}'
repo = f'rigour-labs/rigour-{tier}-v{version}-gguf'
gguf = [f for f in glob.glob(f'rlaif/models/rigour-{tier}-v{version}/*.gguf') if 'f16' not in f][0]
upload_to_huggingface(gguf, repo, version=version, tier=tier)
HfApi().upload_file(path_or_fileobj=f'rlaif/models/rigour-{tier}-v{version}/eval_results.json',
                    path_in_repo='eval_results.json', repo_id=repo)
"
    echo "✓ ${t} GGUF and eval results uploaded"
  done
  # Last: clients follow latest_version.json, so it moves only when every tier is live.
  python scripts/update_version.py --version "$VERSION" --changelog "Training run v${VERSION}"
else
  echo ""
  echo "Not published: pass --upload to upload the models and move latest_version.json."
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  DONE — v${VERSION} complete"
echo "═══════════════════════════════════════════"
echo ""
echo "Models on HuggingFace:"
for t in "${TIERS[@]}"; do
  echo "  Merged: https://huggingface.co/rigour-labs/rigour-${t}-v${VERSION}-merged"
  echo "  GGUF:   https://huggingface.co/rigour-labs/rigour-${t}-v${VERSION}-gguf"
done
