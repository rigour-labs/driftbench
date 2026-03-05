"""Export fine-tuned model to GGUF format for Rigour CLI.

Converts a HuggingFace model to GGUF using llama.cpp's convert script,
then optionally uploads to HuggingFace for distribution.

Requirements:
    pip install huggingface_hub
    git clone https://github.com/ggerganov/llama.cpp (for convert script)

Usage:
    # Convert to GGUF
    python -m rlaif.export_gguf \
        --model rlaif/models/rigour-v1/merged \
        --output rlaif/models/rigour-v1

    # Convert + upload to HuggingFace
    python -m rlaif.export_gguf \
        --model rlaif/models/rigour-v1/merged \
        --output rlaif/models/rigour-v1 \
        --upload rigour-labs/rigour-deep-v1-gguf

    # Lite model (Qwen3.5-0.8B, lightweight)
    python -m rlaif.export_gguf --model rlaif/models/rigour-v1-lite/merged --lite

    # Legacy model (Qwen2.5-Coder-0.5B, for reproducibility)
    python -m rlaif.export_gguf --model rlaif/models/rigour-v1-legacy/merged --legacy
"""

import os
import json
import logging
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rlaif.export_gguf")

# Output filenames must match rigour-core types.ts convention
GGUF_FILENAMES = {
    "deep": "rigour-deep-v{version}-q4_k_m.gguf",       # Qwen2.5-Coder-1.5B (full, company-hosted)
    "lite": "rigour-lite-v{version}-q4_k_m.gguf",        # Qwen3.5-0.8B (lightweight sidecar)
    "legacy": "rigour-legacy-v{version}-q4_k_m.gguf",    # Qwen2.5-Coder-0.5B
}

QUANTIZATION = "q4_k_m"  # Same quant level as stock Qwen models


def find_llama_cpp_convert(llama_cpp_path: str = "") -> str:
    """Find the llama.cpp convert script."""
    candidates = [
        os.path.join(llama_cpp_path, "convert_hf_to_gguf.py"),
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
        shutil.which("convert_hf_to_gguf.py") or "",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def find_llama_quantize(llama_cpp_path: str = "") -> str:
    """Find the llama-quantize binary."""
    candidates = [
        os.path.join(llama_cpp_path, "build/bin/llama-quantize"),
        os.path.join(llama_cpp_path, "llama-quantize"),
        shutil.which("llama-quantize") or "",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def convert_to_gguf(
    model_path: str, output_dir: str, llama_cpp_path: str
) -> str:
    """Convert HF model to f16 GGUF."""
    convert_script = find_llama_cpp_convert(llama_cpp_path)
    if not convert_script:
        raise FileNotFoundError(
            "llama.cpp convert_hf_to_gguf.py not found. "
            "Clone llama.cpp or pass --llama-cpp-path"
        )

    f16_path = os.path.join(output_dir, "model-f16.gguf")
    logger.info(f"Converting {model_path} -> {f16_path}")

    result = subprocess.run(
        ["python", convert_script, model_path, "--outfile", f16_path],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Convert failed: {result.stderr}")

    logger.info(f"F16 GGUF created: {f16_path}")
    return f16_path


def quantize_gguf(
    f16_path: str, output_path: str, llama_cpp_path: str
) -> str:
    """Quantize f16 GGUF to q4_k_m."""
    quantize_bin = find_llama_quantize(llama_cpp_path)
    if not quantize_bin:
        raise FileNotFoundError(
            "llama-quantize not found. Build llama.cpp or pass --llama-cpp-path"
        )

    logger.info(f"Quantizing {f16_path} -> {output_path} ({QUANTIZATION})")

    result = subprocess.run(
        [quantize_bin, f16_path, output_path, QUANTIZATION],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Quantize failed: {result.stderr}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Quantized GGUF: {output_path} ({size_mb:.0f}MB)")
    return output_path


def upload_to_huggingface(
    gguf_path: str, repo_id: str, version: str, tier: str,
):
    """Upload GGUF to HuggingFace model repo."""
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        logger.error("HF_TOKEN not set. Cannot upload.")
        return

    api = HfApi(token=token)
    try:
        create_repo(repo_id, repo_type="model", token=token, exist_ok=True)
    except Exception as e:
        logger.warning(f"Repo creation note: {e}")

    filename = os.path.basename(gguf_path)
    logger.info(f"Uploading {filename} to {repo_id}...")

    api.upload_file(
        path_or_fileobj=gguf_path, path_in_repo=filename,
        repo_id=repo_id, repo_type="model",
    )

    # Upload model card
    card = _build_model_card(repo_id, tier, version, gguf_path)
    api.upload_file(
        path_or_fileobj=card.encode(), path_in_repo="README.md",
        repo_id=repo_id, repo_type="model",
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Uploaded! Model available at: {url}")
    logger.info(
        f"Direct GGUF URL: {url}/resolve/main/{filename}"
    )


def _build_model_card(
    repo_id: str, tier: str, version: str, gguf_path: str,
) -> str:
    size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base_map = {
        "deep": ("Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "qwen2"),
        "lite": ("Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B", "qwen3"),
        "legacy": ("Qwen2.5-Coder-0.5B", "Qwen/Qwen2.5-Coder-0.5B-Instruct", "qwen2"),
    }
    base_name, base_id, tag = base_map.get(tier, base_map["deep"])
    return f"""---
language: en
license: apache-2.0
tags:
  - rigour
  - code-quality
  - gguf
  - {tag}
base_model: {base_id}
---

# Rigour Deep Analysis Model ({tier} v{version})

Fine-tuned {base_name} for code quality analysis.
Trained via RLAIF: strong teacher labels + structural verification + DPO.

| Property | Value |
|---|---|
| Base | {base_name} |
| Quant | Q4_K_M |
| Size | {size_mb:.0f}MB |
| Tier | {tier} |
| Version | {version} |
| Date | {date} |

## Usage with Rigour

Update `types.ts` in rigour-core to point to this model,
or place the GGUF file in `~/.rigour/models/`.

Built by: rigour-labs/driftbench RLAIF pipeline
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export fine-tuned model to GGUF",
    )
    p.add_argument(
        "--model", type=str, required=True,
        help="Path to merged HF model directory",
    )
    p.add_argument(
        "--output", type=str, default="rlaif/models",
        help="Output directory for GGUF files",
    )
    p.add_argument(
        "--lite", action="store_true", help="Lite tier (Qwen3.5-0.8B, lightweight sidecar)",
    )
    p.add_argument(
        "--legacy", action="store_true",
        help="Legacy tier (Qwen2.5-Coder-0.5B, for reproducibility)",
    )
    p.add_argument(
        "--version", type=str, default="1",
        help="Model version number",
    )
    p.add_argument(
        "--llama-cpp-path", type=str, default="",
        help="Path to llama.cpp directory",
    )
    p.add_argument(
        "--upload", type=str, default="",
        help="HuggingFace repo to upload",
    )
    p.add_argument(
        "--skip-quantize", action="store_true",
        help="Keep f16, skip quantization",
    )
    return p


def main():
    args = _build_parser().parse_args()
    tier = "legacy" if args.legacy else ("lite" if args.lite else "deep")
    os.makedirs(args.output, exist_ok=True)

    # Step 1: Convert HF -> GGUF f16
    f16_path = convert_to_gguf(
        args.model, args.output, args.llama_cpp_path
    )

    # Step 2: Quantize
    if args.skip_quantize:
        final_path = f16_path
    else:
        filename = GGUF_FILENAMES[tier].format(version=args.version)
        final_path = os.path.join(args.output, filename)
        quantize_gguf(f16_path, final_path, args.llama_cpp_path)
        os.remove(f16_path)

    logger.info(f"GGUF ready: {final_path}")

    # Step 3: Upload
    if args.upload:
        upload_to_huggingface(
            final_path, args.upload, args.version, tier
        )


if __name__ == "__main__":
    main()
