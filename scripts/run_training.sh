#!/usr/bin/env bash
#
# run_training.sh — Full training pipeline for Lightning.ai GPU studio.
#
# Runs everything locally: finetune → dequantize → export GGUF → upload.
# No GitHub Actions needed. SSH in, run this, watch it work.
#
# Usage:
#   ./scripts/run_training.sh                     # Both tiers, auto-version (v6, v7, ...)
#   ./scripts/run_training.sh --version 5         # Explicit version
#   ./scripts/run_training.sh --tier lite         # Single tier
#   ./scripts/run_training.sh --skip-finetune     # Skip to export (model already on HF)
#   ./scripts/run_training.sh --dry-run           # Print what would run
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
DRY_RUN=false
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ─── Parse args ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --tier)        TIER="$2"; shift 2 ;;
    --version)     VERSION="$2"; shift 2 ;;
    --skip-finetune) SKIP_FINETUNE=true; shift ;;
    --skip-export) SKIP_EXPORT=true; shift ;;
    --dry-run)     DRY_RUN=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--tier deep|lite|both] [--version N] [--skip-finetune] [--skip-export] [--dry-run]"
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

cd "$REPO_ROOT"

# ─── Preflight checks ─────────────────────────────────────────
echo "═══════════════════════════════════════════"
echo "  Rigour Training Pipeline (Local Runner)"
echo "═══════════════════════════════════════════"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=hf_xxx"
  exit 1
fi

echo "Checking GPU..."
python3 -c "
import torch
print(f'  torch={torch.__version__}')
print(f'  CUDA={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU={torch.cuda.get_device_name(0)}')
    print(f'  VRAM={torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU — training will be very slow')
" 2>/dev/null || {
  echo "ERROR: torch not installed. Run: pip install torch"
  exit 1
}

# ─── Resolve version ──────────────────────────────────────────
if [ -z "$VERSION" ]; then
  VERSION=$(python3 -c "
import os, json
try:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download('rigour-labs/rigour-rlaif-data', 'latest_version.json',
                           repo_type='dataset', token=os.environ.get('HF_TOKEN', ''))
    with open(path) as f:
        data = json.load(f)
    current = data.get('version', 0)
    print(current + 1)
except Exception:
    print(1)
" 2>/dev/null || echo "1")
  echo "Auto-resolved version: v${VERSION}"
else
  echo "Using version: v${VERSION}"
fi

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
      echo "  python scripts/finetune_model.py --tier $t --version $VERSION --upload"
    fi
    echo "  python scripts/update_version.py --version $VERSION"
    if [ "$SKIP_EXPORT" = false ]; then
      echo "  python scripts/dequantize_model.py --model-dir rlaif/models/rigour-${t}-v${VERSION}/merged"
      echo "  python -m rlaif.export_gguf --model rlaif/models/rigour-${t}-v${VERSION}/merged --output rlaif/models/rigour-${t}-v${VERSION} --llama-cpp-path llama.cpp --version $VERSION $([ $t = lite ] && echo --lite)"
    fi
  done
  exit 0
fi

# ─── Step 1: Fine-tune ────────────────────────────────────────
if [ "$SKIP_FINETUNE" = false ]; then
  for t in "${TIERS[@]}"; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  FINETUNE: ${t} tier (v${VERSION})"
    echo "════════════════════════════════════════"
    python scripts/finetune_model.py --tier "$t" --version "$VERSION" --upload
  done

  echo ""
  echo "Updating latest_version.json on HuggingFace..."
  python scripts/update_version.py --version "$VERSION"
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

    # Upload GGUF to HuggingFace
    python3 -c "
import os, glob
from rlaif.export_gguf import upload_to_huggingface
tier = '${t}'
version = '${VERSION}'
pattern = f'rlaif/models/rigour-{tier}-v{version}/*.gguf'
gguf_files = glob.glob(pattern)
gguf_files = [f for f in gguf_files if 'f16' not in f]
if gguf_files:
    upload_to_huggingface(
        gguf_files[0],
        f'rigour-labs/rigour-{tier}-v{version}-gguf',
        version=version, tier=tier,
    )
else:
    print('ERROR: No quantized GGUF found')
    exit(1)
"
    echo "✓ ${t} GGUF uploaded"
  done
else
  echo "Skipping export (--skip-export)"
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
