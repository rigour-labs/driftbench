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
import os
import sys
import torch


# ─── Tier-specific configuration ───────────────────────────────────────────
TIER_CONFIG = {
    "deep": {
        "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "sft_epochs": 3, "sft_batch": 4, "sft_grad_accum": 4,
        "dpo_epochs": 1, "dpo_batch": 2, "dpo_grad_accum": 8,
        "max_length": 1024, "lr_sft": 2e-4, "lr_dpo": 5e-5,
    },
    "lite": {
        "base_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "sft_epochs": 2, "sft_batch": 8, "sft_grad_accum": 2,
        "dpo_epochs": 1, "dpo_batch": 4, "dpo_grad_accum": 4,
        "max_length": 768, "lr_sft": 3e-4, "lr_dpo": 8e-5,
    },
}

DATASET_ID = "rigour-labs/rigour-rlaif-data"


def print_gpu_info():
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
    """Load base model with QLoRA quantization and LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"Loading {base_model}...")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if torch.cuda.is_available():
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
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

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype_args = {"bf16": True} if use_bf16 else ({"fp16": True} if torch.cuda.is_available() else {})
    steps_per_epoch = max(1, len(sft_ds) // (cfg["sft_batch"] * cfg["sft_grad_accum"]))

    sft_args = SFTConfig(
        output_dir=os.path.join(output_dir, "sft"),
        num_train_epochs=cfg["sft_epochs"],
        per_device_train_batch_size=cfg["sft_batch"],
        gradient_accumulation_steps=cfg["sft_grad_accum"],
        learning_rate=cfg["lr_sft"],
        warmup_steps=max(1, int(steps_per_epoch * cfg["sft_epochs"] * 0.1)),
        logging_steps=10,
        save_strategy="epoch",
        max_length=cfg["max_length"],
        **dtype_args,
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=sft_ds, args=sft_args)
    trainer.train()
    print("SFT complete")


def train_dpo(model, tokenizer, dpo_ds, cfg, output_dir: str):
    """Run direct preference optimization phase."""
    if len(dpo_ds) == 0:
        print("No DPO data — skipping DPO phase")
        return

    from trl import DPOTrainer, DPOConfig

    tokenizer.padding_side = "left"  # Required by DPOTrainer
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype_args = {"bf16": True} if use_bf16 else ({"fp16": True} if torch.cuda.is_available() else {})
    steps_per_epoch = max(1, len(dpo_ds) // (cfg["dpo_batch"] * cfg["dpo_grad_accum"]))

    dpo_args = DPOConfig(
        output_dir=os.path.join(output_dir, "dpo"),
        num_train_epochs=cfg["dpo_epochs"],
        per_device_train_batch_size=cfg["dpo_batch"],
        gradient_accumulation_steps=cfg["dpo_grad_accum"],
        learning_rate=cfg["lr_dpo"],
        warmup_steps=max(1, int(steps_per_epoch * 0.1)),
        logging_steps=10,
        save_strategy="epoch",
        max_length=cfg["max_length"],
        beta=0.1,
        **dtype_args,
    )
    trainer = DPOTrainer(model=model, processing_class=tokenizer, train_dataset=dpo_ds, args=dpo_args)
    trainer.train()
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
    parser.add_argument("--version", required=True, help="Model version (e.g., 6)")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after training")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="HuggingFace token")
    args = parser.parse_args()

    tier = args.tier
    version = args.version
    cfg = TIER_CONFIG[tier]
    base_model = cfg["base_model"]
    output_dir = args.output_dir or f"rlaif/models/rigour-{tier}-v{version}"
    hf_token = args.hf_token

    print(f"Training tier={tier} version=v{version}")
    print_gpu_info()

    # Download data
    sft_ds, dpo_ds = download_data(hf_token)
    if len(sft_ds) == 0 and len(dpo_ds) == 0:
        print("No training data — skipping")
        sys.exit(0)

    # Load model
    model, tokenizer = load_model_and_tokenizer(base_model)

    # Train
    train_sft(model, tokenizer, sft_ds, cfg, output_dir)
    train_dpo(model, tokenizer, dpo_ds, cfg, output_dir)

    # Merge cleanly for GGUF export
    merged_path = merge_and_save(model, tokenizer, base_model, output_dir)

    # Upload
    if args.upload and hf_token:
        upload_to_hf(merged_path, tier, version, len(sft_ds), len(dpo_ds), hf_token)

    print(f"Fine-tuning complete! tier={tier} version=v{version}")


if __name__ == "__main__":
    main()
