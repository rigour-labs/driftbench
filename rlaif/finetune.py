"""QLoRA fine-tune script for Rigour deep analysis model.

Takes SFT + DPO training data from HuggingFace (or local JSONL)
and fine-tunes Qwen using QLoRA + trl DPOTrainer.

Supports three tiers:
  - deep:    Qwen3.5-0.8B          (default, hybrid GDN+MoE, ~3x faster CPU)
  - pro:     Qwen2.5-Coder-1.5B    (higher capacity, code-specialized pretrain)
  - legacy:  Qwen2.5-Coder-0.5B    (previous default, kept for reproducibility)

Requirements:
    pip install torch transformers peft trl datasets bitsandbytes

Usage:
    # Fine-tune with DPO (recommended — uses Qwen3.5-0.8B by default)
    python -m rlaif.finetune \
        --sft rlaif/data/sft_data.jsonl \
        --dpo rlaif/data/dpo_data.jsonl \
        --output rlaif/models/rigour-v1

    # Fine-tune from HuggingFace dataset
    python -m rlaif.finetune \
        --hf-dataset rigour-labs/rigour-rlaif-data \
        --output rlaif/models/rigour-v1

    # Pro model (1.5B, code-specialized)
    python -m rlaif.finetune --pro --output rlaif/models/rigour-v1-pro

    # Legacy model (previous Qwen2.5-Coder-0.5B for reproducibility)
    python -m rlaif.finetune --legacy --output rlaif/models/rigour-v1-legacy
"""

import os
import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rlaif.finetune")

# Base models (same as rigour-core/src/inference/types.ts)
# Qwen 3.5 is the new default — hybrid GDN+MoE gives ~3x faster CPU inference
# No Qwen3.5-Coder variant exists yet; RLAIF fine-tuning handles code specialization
BASE_MODELS = {
    "deep": "Qwen/Qwen3.5-0.8B",                     # NEW default: faster CPU, March 2026
    "pro": "Qwen/Qwen2.5-Coder-1.5B-Instruct",       # Higher capacity, code pretrain
    "legacy": "Qwen/Qwen2.5-Coder-0.5B-Instruct",    # Previous default, reproducibility
}

# QLoRA defaults tuned for code quality analysis
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _load_local_data(sft_path: str, dpo_path: str):
    """Load training data from local JSONL files."""
    from datasets import Dataset

    sft_data, dpo_data = [], []

    if sft_path and os.path.exists(sft_path):
        with open(sft_path) as f:
            sft_data = [json.loads(line) for line in f if line.strip()]
        logger.info(f"Loaded {len(sft_data)} SFT examples")

    if dpo_path and os.path.exists(dpo_path):
        with open(dpo_path) as f:
            dpo_data = [json.loads(line) for line in f if line.strip()]
        logger.info(f"Loaded {len(dpo_data)} DPO pairs")

    sft_dataset = Dataset.from_list(sft_data) if sft_data else None
    dpo_dataset = Dataset.from_list(dpo_data) if dpo_data else None
    return sft_dataset, dpo_dataset


def _load_hf_data(dataset_id: str):
    """Load training data from HuggingFace dataset."""
    from datasets import load_dataset

    logger.info(f"Loading dataset from HuggingFace: {dataset_id}")
    sft_ds = load_dataset(
        "json", data_files=f"hf://datasets/{dataset_id}/sft_data.jsonl",
        split="train",
    )
    dpo_ds = load_dataset(
        "json", data_files=f"hf://datasets/{dataset_id}/dpo_data.jsonl",
        split="train",
    )
    logger.info(f"Loaded {len(sft_ds)} SFT, {len(dpo_ds)} DPO from HF")
    return sft_ds, dpo_ds


def _setup_model_and_tokenizer(base_model: str, use_4bit: bool):
    """Load base model with QLoRA config."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logger.info(f"Loading base model: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        import torch
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.1f}%)"
    )
    return model, tokenizer


def run_sft(model, tokenizer, sft_dataset, output_dir: str, epochs: int):
    """Run supervised fine-tuning phase."""
    from trl import SFTTrainer, SFTConfig

    logger.info(f"Starting SFT training ({len(sft_dataset)} examples)")

    training_args = SFTConfig(
        output_dir=os.path.join(output_dir, "sft"),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=1024,  # trl >= 0.15 renamed max_seq_length → max_length
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=sft_dataset,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "sft-final"))
    logger.info("SFT training complete")
    return model


def run_dpo(model, tokenizer, dpo_dataset, output_dir: str, epochs: int):
    """Run DPO preference optimization phase."""
    from trl import DPOTrainer, DPOConfig

    logger.info(f"Starting DPO training ({len(dpo_dataset)} pairs)")

    training_args = DPOConfig(
        output_dir=os.path.join(output_dir, "dpo"),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=1024,
        max_prompt_length=512,
        beta=0.1,
    )

    trainer = DPOTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dpo_dataset,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "dpo-final"))
    logger.info("DPO training complete")
    return model


def merge_and_save(model, tokenizer, output_dir: str):
    """Merge LoRA adapter back into base model and save."""
    logger.info("Merging LoRA adapter into base model...")
    merged = model.merge_and_unload()
    merged_path = os.path.join(output_dir, "merged")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    logger.info(f"Merged model saved to {merged_path}")
    return merged_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QLoRA fine-tune for Rigour deep analysis",
    )
    p.add_argument(
        "--sft", type=str, default="rlaif/data/sft_data.jsonl",
        help="Path to SFT JSONL",
    )
    p.add_argument(
        "--dpo", type=str, default="rlaif/data/dpo_data.jsonl",
        help="Path to DPO JSONL",
    )
    p.add_argument(
        "--hf-dataset", type=str, default="",
        help="HuggingFace dataset ID (e.g., rigour-labs/rigour-rlaif-data)",
    )
    p.add_argument(
        "--output", type=str, default="rlaif/models/rigour-v1",
        help="Output directory for fine-tuned model",
    )
    p.add_argument(
        "--pro", action="store_true",
        help="Use Qwen2.5-Coder-1.5B (higher capacity, code pretrain)",
    )
    p.add_argument(
        "--legacy", action="store_true",
        help="Use previous Qwen2.5-Coder-0.5B (for reproducibility)",
    )
    p.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit quantization (needs more VRAM)",
    )
    p.add_argument(
        "--sft-epochs", type=int, default=3, help="SFT training epochs",
    )
    p.add_argument(
        "--dpo-epochs", type=int, default=1, help="DPO training epochs",
    )
    p.add_argument(
        "--skip-sft", action="store_true",
        help="Skip SFT, only run DPO",
    )
    return p


def main():
    args = _build_parser().parse_args()

    tier = "legacy" if args.legacy else ("pro" if args.pro else "deep")
    base_model = BASE_MODELS[tier]
    logger.info(f"Tier: {tier} | Base model: {base_model}")

    # Load data
    if args.hf_dataset:
        sft_dataset, dpo_dataset = _load_hf_data(args.hf_dataset)
    else:
        sft_dataset, dpo_dataset = _load_local_data(args.sft, args.dpo)

    if not sft_dataset and not dpo_dataset:
        logger.error("No training data found. Run rlaif.generate first.")
        return

    # Setup model
    model, tokenizer = _setup_model_and_tokenizer(
        base_model, use_4bit=not args.no_4bit
    )
    os.makedirs(args.output, exist_ok=True)

    # Phase 1: SFT
    if sft_dataset and not args.skip_sft:
        model = run_sft(
            model, tokenizer, sft_dataset, args.output, args.sft_epochs
        )

    # Phase 2: DPO
    if dpo_dataset:
        model = run_dpo(
            model, tokenizer, dpo_dataset, args.output, args.dpo_epochs
        )

    # Phase 3: Merge
    merged_path = merge_and_save(model, tokenizer, args.output)

    logger.info(f"\nFine-tuning complete!")
    logger.info(f"Merged model: {merged_path}")
    logger.info(
        f"Next: python -m rlaif.export_gguf "
        f"--model {merged_path} --output {args.output}"
    )


if __name__ == "__main__":
    main()
