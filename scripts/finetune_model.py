#!/usr/bin/env python3
"""
Fine-tune a Rigour model tier using QLoRA (SFT + DPO).

Usage:
    python scripts/finetune_model.py --tier deep --version 6
    python scripts/finetune_model.py --tier lite --version 6 --upload

Environment variables:
    HF_TOKEN: HuggingFace token for downloading data and uploading models

Can be tested locally (CPU, slow) or on GPU (Lightning.ai, fast).
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import torch


# ─── Tier-specific configuration ───────────────────────────────────────────
# MPS overrides: larger batches (unified memory fits it), lower grad_accum
# to maximize GPU utilization on Apple Silicon.
TIER_CONFIG = {
    "deep": {
        "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "sft_epochs": 3, "sft_batch": 2, "sft_grad_accum": 8,
        "dpo_epochs": 1, "dpo_batch": 2, "dpo_grad_accum": 8,
        "max_length": 2048, "lr_sft": 2e-4, "lr_dpo": 5e-5,
    },
    "lite": {
        "base_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "sft_epochs": 2, "sft_batch": 4, "sft_grad_accum": 4,
        "dpo_epochs": 1, "dpo_batch": 4, "dpo_grad_accum": 4,
        "max_length": 1536, "lr_sft": 3e-4, "lr_dpo": 8e-5,
    },
}

# MPS overrides — M4 Pro has 18-48GB unified memory shared with GPU.
# 0.5B model in fp16 ≈ 1GB, 1.5B ≈ 3GB — tons of headroom for larger batches.
MPS_OVERRIDES = {
    "deep": {"sft_batch": 8, "sft_grad_accum": 2, "dpo_batch": 4, "dpo_grad_accum": 4},
    "lite": {"sft_batch": 16, "sft_grad_accum": 1, "dpo_batch": 8, "dpo_grad_accum": 2},
}

DATASET_ID = "rigour-labs/rigour-rlaif-data"


# ─── Persistent training log ──────────────────────────────────────────────
# Writes status to a JSON file so you can check what happened after closing
# the terminal. Check: cat rlaif/models/training-status.json
def setup_logging(output_dir: str):
    """Setup file + console logging so training output survives terminal close."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("rigour-train")


STATUS_FILE = "rlaif/models/training-status.json"


def write_status(status: dict):
    """Write training status to a persistent JSON file."""
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


def read_status() -> dict:
    """Read previous training status if it exists."""
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {}


def get_device_type() -> str:
    """Detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_gpu_info():
    device = get_device_type()
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device == "mps":
        print("GPU: Apple Silicon (Metal Performance Shaders)")
        print("Mode: fp16 full fine-tune (no QLoRA — models small enough)")
    else:
        print("WARNING: No GPU — training will be slow on CPU")


def download_data(hf_token: str):
    """Download SFT and DPO datasets from HuggingFace."""
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset

    print("Downloading training data...")
    sft_path = hf_hub_download(DATASET_ID, "sft_data.jsonl", repo_type="dataset", token=hf_token)
    dpo_path = hf_hub_download(DATASET_ID, "dpo_data.jsonl", repo_type="dataset", token=hf_token)

    sft_ds = load_dataset("json", data_files=sft_path, split="train")
    dpo_ds = load_dataset("json", data_files=dpo_path, split="train")
    print(f"SFT: {len(sft_ds)} examples, DPO: {len(dpo_ds)} pairs")
    return sft_ds, dpo_ds


def load_model_and_tokenizer(base_model: str):
    """Load base model with QLoRA (CUDA), LoRA+fp16 (MPS/Apple Silicon), or fp32 (CPU)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = get_device_type()
    print(f"Loading {base_model} (device={device})...")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True}

    if device == "cuda":
        # CUDA: use 4-bit QLoRA for memory efficiency
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs["device_map"] = "auto"
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif device == "mps":
        # Apple Silicon: fp16 on MPS — models are small enough (0.5B/1.5B)
        # Don't use device_map="auto" with MPS — it doesn't support it well.
        # Load to CPU first, then move to MPS after LoRA wrapping.
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    if device == "cuda":
        model = prepare_model_for_kbit_training(model)

    # Dropout=0 on MPS — dropout forces synchronization barriers on Metal,
    # and with small models + limited data, regularization from LoRA rank is enough.
    dropout = 0.0 if device == "mps" else 0.05

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if device == "mps":
        model = model.to("mps")

    # torch.compile() — 20-40% speedup by fusing operations.
    # CUDA only. MPS inductor backend hits INT_MAX tensor dim limits with
    # attention matrices, causing "MPSGraph does not support tensor dims
    # larger than INT_MAX" crashes. Disabled on MPS until PyTorch fixes this.
    if device == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile() enabled")
        except Exception as e:
            print(f"torch.compile() skipped: {e}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    return model, tokenizer


def train_sft(model, tokenizer, sft_ds, cfg, output_dir: str):
    """Run supervised fine-tuning phase."""
    if len(sft_ds) == 0:
        print("No SFT data — skipping SFT phase")
        return

    from trl import SFTTrainer, SFTConfig

    device = get_device_type()
    dtype_args = {}
    if device == "cuda":
        dtype_args["fp16"] = True
    elif device == "mps":
        dtype_args["fp16"] = True  # MPS supports fp16 training
    steps_per_epoch = max(1, len(sft_ds) // (cfg["sft_batch"] * cfg["sft_grad_accum"]))

    # Multiprocess data loading — use CPU cores for tokenization while GPU trains
    num_workers = 4 if device in ("cuda", "mps") else 0

    total_steps = steps_per_epoch * cfg["sft_epochs"]
    # Save checkpoint every ~25% of training — crash recovery for cloud runs
    save_steps = max(50, total_steps // 4)

    sft_args = SFTConfig(
        output_dir=os.path.join(output_dir, "sft"),
        num_train_epochs=cfg["sft_epochs"],
        per_device_train_batch_size=cfg["sft_batch"],
        gradient_accumulation_steps=cfg["sft_grad_accum"],
        learning_rate=cfg["lr_sft"],
        warmup_steps=max(1, int(steps_per_epoch * cfg["sft_epochs"] * 0.1)),
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,  # Keep only last 2 checkpoints to save disk
        max_length=cfg["max_length"],
        dataset_text_field="text",  # Explicit field — avoids packing ambiguity
        dataloader_pin_memory=device != "mps",
        dataloader_num_workers=num_workers,
        **dtype_args,
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=sft_ds, args=sft_args)

    write_status({"phase": "sft", "status": "running", "total_steps": total_steps,
                  "tier": cfg.get("_tier", ""), "device": device})
    trainer.train()
    write_status({"phase": "sft", "status": "complete", "total_steps": total_steps})
    print("SFT complete")


def train_dpo(model, tokenizer, dpo_ds, cfg, output_dir: str):
    """Run direct preference optimization phase."""
    if len(dpo_ds) == 0:
        print("No DPO data — skipping DPO phase")
        return

    from trl import DPOTrainer, DPOConfig

    tokenizer.padding_side = "left"  # Required by DPOTrainer
    device = get_device_type()
    dtype_args = {}
    if device in ("cuda", "mps"):
        dtype_args["fp16"] = True
    steps_per_epoch = max(1, len(dpo_ds) // (cfg["dpo_batch"] * cfg["dpo_grad_accum"]))

    num_workers = 4 if device in ("cuda", "mps") else 0

    total_steps = steps_per_epoch * cfg["dpo_epochs"]
    save_steps = max(50, total_steps // 4)

    dpo_args = DPOConfig(
        output_dir=os.path.join(output_dir, "dpo"),
        num_train_epochs=cfg["dpo_epochs"],
        per_device_train_batch_size=cfg["dpo_batch"],
        gradient_accumulation_steps=cfg["dpo_grad_accum"],
        learning_rate=cfg["lr_dpo"],
        warmup_steps=max(1, int(steps_per_epoch * 0.1)),
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        max_length=cfg["max_length"],
        beta=0.1,
        dataloader_pin_memory=device != "mps",
        dataloader_num_workers=num_workers,
        **dtype_args,
    )
    trainer = DPOTrainer(model=model, processing_class=tokenizer, train_dataset=dpo_ds, args=dpo_args)

    write_status({"phase": "dpo", "status": "running", "total_steps": total_steps})
    trainer.train()
    write_status({"phase": "dpo", "status": "complete", "total_steps": total_steps})
    print("DPO complete")


def merge_and_save(model, tokenizer, base_model: str, output_dir: str):
    """Save LoRA adapter, reload base in fp16, merge cleanly for GGUF export."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    # Save LoRA adapter
    print("Saving LoRA adapter...")
    adapter_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Reload base in fp16 on CPU and merge — produces clean weights for GGUF
    print(f"Reloading {base_model} in fp16 on CPU for clean merge...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_path)
    merged = merged.merge_and_unload()

    merged_path = os.path.join(output_dir, "merged")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print("Merged model saved in fp16")
    return merged_path


def upload_to_hf(merged_path: str, tier: str, version: str, sft_count: int, dpo_count: int, hf_token: str):
    """Upload merged model to HuggingFace."""
    from huggingface_hub import HfApi

    repo_id = f"rigour-labs/rigour-{tier}-v{version}-merged"
    print(f"Uploading merged model to {repo_id}...")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=merged_path,
        repo_id=repo_id,
        commit_message=f"QLoRA fine-tune v{version}: tier={tier} SFT={sft_count} DPO={dpo_count}",
    )
    print(f"Model uploaded: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Rigour model with QLoRA")
    parser.add_argument("--tier", required=True, choices=list(TIER_CONFIG.keys()), help="Model tier")
    parser.add_argument("--version", required=True, help="Model version in SemVer (e.g., 2.0.0)")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after training")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="HuggingFace token")
    args = parser.parse_args()

    tier = args.tier
    version = args.version
    cfg = {**TIER_CONFIG[tier]}  # copy so we don't mutate the original
    cfg["_tier"] = tier  # tag for status tracking
    device = get_device_type()

    # Apply MPS batch size overrides for Apple Silicon
    if device == "mps" and tier in MPS_OVERRIDES:
        cfg.update(MPS_OVERRIDES[tier])
        print(f"Applied MPS overrides: batch={cfg['sft_batch']}, grad_accum={cfg['sft_grad_accum']}")

    base_model = cfg["base_model"]
    output_dir = args.output_dir or f"rlaif/models/rigour-{tier}-v{version}"
    hf_token = args.hf_token

    # Setup persistent logging — survives terminal close
    logger = setup_logging(output_dir)
    start_time = time.time()

    # Write initial status — check this file when you come back
    write_status({
        "tier": tier, "version": version, "device": device,
        "phase": "starting", "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
    })

    logger.info(f"Training tier={tier} version=v{version}")
    print_gpu_info()

    try:
        # Download data
        sft_ds, dpo_ds = download_data(hf_token)
        if len(sft_ds) == 0 and len(dpo_ds) == 0:
            write_status({"phase": "done", "status": "skipped", "reason": "no training data"})
            print("No training data — skipping")
            sys.exit(0)

        write_status({"phase": "loading_model", "status": "running",
                      "sft_examples": len(sft_ds), "dpo_pairs": len(dpo_ds)})

        # Load model
        model, tokenizer = load_model_and_tokenizer(base_model)

        # Train
        train_sft(model, tokenizer, sft_ds, cfg, output_dir)
        train_dpo(model, tokenizer, dpo_ds, cfg, output_dir)

        # Merge cleanly for GGUF export
        write_status({"phase": "merging", "status": "running"})
        merged_path = merge_and_save(model, tokenizer, base_model, output_dir)

        # Upload
        if args.upload and hf_token:
            write_status({"phase": "uploading", "status": "running"})
            upload_to_hf(merged_path, tier, version, len(sft_ds), len(dpo_ds), hf_token)

        elapsed = time.time() - start_time
        write_status({
            "tier": tier, "version": version, "device": device,
            "phase": "done", "status": "success",
            "elapsed_minutes": round(elapsed / 60, 1),
            "sft_examples": len(sft_ds), "dpo_pairs": len(dpo_ds),
            "merged_path": merged_path,
            "uploaded": args.upload,
        })
        logger.info(f"Fine-tuning complete! tier={tier} version=v{version} ({elapsed/60:.1f} min)")

    except Exception as e:
        elapsed = time.time() - start_time
        write_status({
            "tier": tier, "version": version, "device": device,
            "phase": "error", "status": "failed",
            "error": str(e), "error_type": type(e).__name__,
            "elapsed_minutes": round(elapsed / 60, 1),
        })
        logger.error(f"Training FAILED after {elapsed/60:.1f} min: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
