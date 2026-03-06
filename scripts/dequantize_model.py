#!/usr/bin/env python3
"""
Dequantize a bitsandbytes 4-bit model to fp16 for GGUF export.

If the model is already in fp16 (no quantization_config in config.json),
this script is a no-op.

Usage:
    python scripts/dequantize_model.py --model-dir rlaif/models/rigour-deep-v5/merged

Requires: torch, transformers, bitsandbytes, accelerate
Requires GPU (CUDA) — bitsandbytes 4-bit cannot load on CPU.

Can be tested locally with:
    python scripts/dequantize_model.py --model-dir /path/to/model --dry-run
"""
import argparse
import glob
import json
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description="Dequantize bitsandbytes model to fp16")
    parser.add_argument("--model-dir", required=True, help="Path to merged model directory")
    parser.add_argument("--dry-run", action="store_true", help="Check if dequantization is needed without doing it")
    args = parser.parse_args()

    model_dir = args.model_dir
    config_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"ERROR: No config.json found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    if "quantization_config" not in config:
        print("Model is already in fp16 — no dequantization needed")
        sys.exit(0)

    quant_type = config["quantization_config"].get("quant_method", "unknown")
    print(f"Found {quant_type} quantization — dequantization required")

    if args.dry_run:
        print("(dry run — would dequantize to fp16)")
        sys.exit(0)

    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for bitsandbytes dequantization. Run on GPU.")
        sys.exit(1)

    print(f"Loading model from {model_dir} for dequantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Dequantizing to fp16...")
    model = model.dequantize()
    model = model.half()

    # Save dequantized model weights
    clean_dir = model_dir + "_fp16"
    print(f"Saving fp16 weights to {clean_dir}...")
    model.save_pretrained(clean_dir)

    # Copy tokenizer files from original dir (avoids reloading tokenizer
    # which can fail with transformers version mismatches)
    tokenizer_patterns = ["tokenizer*", "special_tokens*", "vocab*", "merges*"]
    for pattern in tokenizer_patterns:
        for f in glob.glob(os.path.join(model_dir, pattern)):
            shutil.copy2(f, clean_dir)
            print(f"  Copied {os.path.basename(f)}")

    # Replace original with clean version
    print("Replacing original with fp16 version...")
    shutil.rmtree(model_dir)
    shutil.move(clean_dir, model_dir)
    print("Dequantized and saved as fp16")


if __name__ == "__main__":
    main()
