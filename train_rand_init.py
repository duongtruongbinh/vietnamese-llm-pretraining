#!/usr/bin/env python3
"""
Pre-training Script for Vietnamese GPT-2 — Random Weight Initialization
========================================================================
Train GPT-2 Small from scratch (random init) on Vietnamese data.
Same data pipeline and Chinchilla budget as continual pre-training version,
but all weights are randomly initialized (no English GPT-2 transfer).
"""

import glob
import math
import os
import unicodedata
from typing import Dict, List, Any

# Environment variables to prevent multiprocessing issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if 'WANDB_PROJECT' not in os.environ:
    os.environ['WANDB_PROJECT'] = 'vietnamese-gpt2'

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    GPT2Config,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ========================== Configuration ==========================
# Model settings
BASE_MODEL = "gpt2"  # GPT-2 Small architecture (124M parameters)
TOKENIZER_DIR = "./vietnamese_tokenizer"
OUTPUT_DIR = "./vietnamese_gpt2_rand_init_mixed"
LOGGING_DIR = "./logs"

# Dataset settings
# Mix: 70% BKAINewsCorpus (news) + 30% Wikipedia VI — Chinchilla-scaled
# Chinchilla token budget: 20 × N_params for GPT-2 Small (124M)
TOKEN_BUDGET = 20 * 124_000_000  # 2.48B tokens
DATASET_CONFIGS = [
    {"name": "bkai-foundation-models/BKAINewsCorpus", "split": "train", "text_col": "text", "weight": 0.7},
    {"name": "vietgpt/wikipedia_vi",                  "split": "train", "text_col": "text", "weight": 0.3},
]
MAX_LENGTH = 1024  # Block size for grouped texts

# Training hyperparameters
LEARNING_RATE = 5e-4   # Higher LR suitable for random init (vs 5e-5 for continual)
WEIGHT_DECAY = 0.01
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 64  # 2 x 64 x 1 GPU = 128 effective batch
WARMUP_RATIO = 0.1
FP16 = False
BF16 = True
GRADIENT_CHECKPOINTING = True

# Data processing
PREPROCESSING_NUM_WORKERS = 4
DATALOADER_NUM_WORKERS = 0
EVAL_SPLIT_RATIO = 0.01  # 1% for evaluation

# Wandb configuration
WANDB_RUN_NAME = "gpt2-small-vietnamese-rand-init"


def is_main_process() -> bool:
    """Check if this is the main process (rank 0) in DDP training."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)


def load_and_prepare_tokenizer() -> GPT2TokenizerFast:
    print(f"Loading tokenizer from: {TOKENIZER_DIR}")
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process():
        print(f"Tokenizer vocab size: {len(tokenizer):,}")
        test_text = "Việt Nam là một đất nước"
        test_tokens = tokenizer.tokenize(test_text)
        print(f"Sample tokenization: '{test_text}'")
        print(f"  → Tokens: {test_tokens[:8]}...")
        print(f"  → Token count: {len(test_tokens)} tokens")

    return tokenizer


def load_and_prepare_model(tokenizer: GPT2TokenizerFast) -> GPT2LMHeadModel:
    """
    Initialize GPT-2 Small from RANDOM weights using Vietnamese vocab size.
    No pretrained English weights are loaded.
    """
    _main = is_main_process()

    if _main:
        print(f"\nInitializing model from RANDOM weights (architecture: {BASE_MODEL})")

    # Load GPT-2 Small config only (architecture), then build from scratch
    config = GPT2Config.from_pretrained(BASE_MODEL)
    config.vocab_size = len(tokenizer)  # Set vocab size upfront — no resize needed
    config.attn_implementation = "flash_attention_2"

    if _main:
        print(f"Model architecture:")
        print(f"  - n_layer: {config.n_layer}")
        print(f"  - n_head: {config.n_head}")
        print(f"  - n_embd: {config.n_embd}")
        print(f"  - vocab_size: {config.vocab_size}")
        print(f"  - Attention: flash_attention_2")

    # Random init — GPT2LMHeadModel() with no pretrained weights
    model = GPT2LMHeadModel(config)

    # Cast to BF16 for Flash Attention 2 compatibility
    model = model.to(torch.bfloat16)

    # Tie lm_head ↔ wte weights
    config.tie_word_embeddings = True
    model.tie_weights()
    if _main:
        print(f"Weight tying (lm_head ↔ wte): ENABLED")

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        if _main:
            print("Gradient checkpointing: ENABLED")

    if _main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters:")
        print(f"  - Total: {total_params:,}")
        print(f"  - Trainable: {trainable_params:,}")

    return model


def load_and_prepare_dataset(tokenizer: GPT2TokenizerFast):
    """
    Load and process dataset using Group Texts technique.
    Subsample/upsample each source to hit Chinchilla budget at target weights.
    """
    _main = is_main_process()

    if _main:
        print(f"\n{'='*60}")
        print("DATASET PREPARATION")
        print(f"{'='*60}")

    # 1. Load and mix datasets — subsample/upsample per Chinchilla budget & weight
    all_datasets = []
    for cfg in DATASET_CONFIGS:
        weight = cfg["weight"]
        if _main:
            print(f"\nLoading dataset: {cfg['name']} (weight={weight})")
        ds = load_dataset(cfg["name"], split=cfg["split"])
        if cfg["text_col"] != "text":
            ds = ds.rename_column(cfg["text_col"], "text")
        ds = ds.select_columns(["text"])

        # Estimate average tokens/sample via a small probe
        probe_size = min(500, len(ds))
        probe_texts = [normalize_text(ds[i]["text"]) for i in range(probe_size)]
        probe_ids = tokenizer(
            probe_texts, truncation=False, return_attention_mask=False
        )["input_ids"]
        avg_tokens = sum(len(ids) for ids in probe_ids) / probe_size

        # Target samples for this dataset's share of the Chinchilla token budget
        target_tokens = TOKEN_BUDGET * weight
        target_samples = int(target_tokens / avg_tokens)

        if _main:
            print(f"  → {len(ds):,} samples | avg {avg_tokens:.0f} tokens/sample")
            print(f"  → Token target: {target_tokens/1e9:.2f}B → {target_samples:,} samples needed")

        if target_samples > len(ds):
            repeat = math.ceil(target_samples / len(ds))
            ds = concatenate_datasets([ds] * repeat)
            if _main:
                print(f"  → Upsampled {repeat}× → {len(ds):,} samples (before trim)")

        ds = ds.shuffle(seed=42).select(range(min(target_samples, len(ds))))
        if _main:
            print(f"  → Final: {len(ds):,} samples")
        all_datasets.append(ds)

    # Concatenate and shuffle all domains together
    dataset = concatenate_datasets(all_datasets)
    dataset = dataset.shuffle(seed=42)
    if _main:
        print(f"\nCombined dataset: {len(dataset):,} total samples")

    # 2. Tokenize function
    eos_token_id = tokenizer.eos_token_id

    def tokenize_function(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        normalized_texts = [normalize_text(text) for text in examples["text"]]
        tokenized = tokenizer(
            normalized_texts,
            truncation=False,
            return_attention_mask=False,
        )
        for i in range(len(tokenized["input_ids"])):
            tokenized["input_ids"][i].append(eos_token_id)
        return tokenized

    # 3. Group texts function
    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        if total_length >= MAX_LENGTH:
            total_length = (total_length // MAX_LENGTH) * MAX_LENGTH
        result = {
            k: [t[i : i + MAX_LENGTH] for i in range(0, total_length, MAX_LENGTH)]
            for k, t in concatenated.items()
        }
        result["labels"] = [block[:] for block in result["input_ids"]]
        return result

    # 4. Apply tokenization
    if _main:
        print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=PREPROCESSING_NUM_WORKERS,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # 5. Apply group texts
    if _main:
        print(f"\nGrouping texts into blocks of {MAX_LENGTH} tokens...")
    grouped_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=PREPROCESSING_NUM_WORKERS,
        desc="Grouping texts",
    )

    if _main:
        print(f"Total training blocks: {len(grouped_dataset):,}")
        print(f"Tokens per block: {MAX_LENGTH}")
        print(f"Total tokens: {len(grouped_dataset) * MAX_LENGTH:,}")

    # 6. Train/eval split
    if _main:
        print(f"\nSplitting dataset (eval ratio: {EVAL_SPLIT_RATIO})")
    split_dataset = grouped_dataset.train_test_split(test_size=EVAL_SPLIT_RATIO, seed=42)

    if _main:
        print(f"Train samples: {len(split_dataset['train']):,}")
        print(f"Eval samples: {len(split_dataset['test']):,}")

    return split_dataset


def create_trainer(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    train_dataset,
    eval_dataset,
) -> Trainer:
    _main = is_main_process()

    # Compute max_steps from Chinchilla token budget
    num_gpus = max(1, torch.cuda.device_count())
    tokens_per_step = (
        PER_DEVICE_TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
        * num_gpus
        * MAX_LENGTH
    )
    max_steps = TOKEN_BUDGET // tokens_per_step

    if _main:
        print(f"\n{'='*60}")
        print("TRAINER CONFIGURATION")
        print(f"{'='*60}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Training params — max_steps enforces Chinchilla token budget
        max_steps=max_steps,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Optimizer params
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",

        # Mixed precision
        fp16=False,
        bf16=BF16 and torch.cuda.is_available(),

        # Evaluation and Saving
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_dir=LOGGING_DIR,
        logging_steps=100,
        report_to=["tensorboard", "wandb"],
        run_name=WANDB_RUN_NAME,

        # Performance
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=True,

        # Reproducibility
        seed=42,
        data_seed=42,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * max(1, torch.cuda.device_count())
    )

    if _main:
        print(f"Batch size per device: {training_args.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Learning rate: {training_args.learning_rate}")
        print(f"Weight decay: {training_args.weight_decay}")
        print(f"BF16: {training_args.bf16}")
        print(f"Token budget (Chinchilla 20×): {TOKEN_BUDGET/1e9:.2f}B tokens")
        print(f"Tokens per step: {tokens_per_step:,}")
        print(f"Max steps: {max_steps:,}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    return trainer


def main():
    _main = is_main_process()

    if _main:
        print("=" * 60)
        print("VIETNAMESE GPT-2 PRE-TRAINING (RANDOM INIT)")
        print("=" * 60)

    if torch.cuda.is_available():
        if _main:
            print(f"\nGPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        if _main:
            print("\nWARNING: No GPU detected. Training will be slow.")

    # 1. Load tokenizer
    tokenizer = load_and_prepare_tokenizer()

    # 2. Initialize model from random weights
    model = load_and_prepare_model(tokenizer)

    # 3. Prepare dataset
    dataset = load_and_prepare_dataset(tokenizer)

    # 4. Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    # 5. Train (with resume support)
    if _main:
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}\n")

    checkpoint_dirs = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    resume_from_checkpoint = None

    if checkpoint_dirs:
        def get_step(path):
            try:
                return int(os.path.basename(path).split('-')[-1])
            except (ValueError, IndexError):
                return 0
        checkpoint_dirs_sorted = sorted(checkpoint_dirs, key=get_step, reverse=True)

        for ckpt in checkpoint_dirs_sorted:
            model_file = os.path.join(ckpt, 'model.safetensors')
            if os.path.exists(model_file) and os.path.getsize(model_file) > 1_000_000:
                resume_from_checkpoint = ckpt
                break

        if resume_from_checkpoint:
            if _main:
                print(f"Found checkpoint: {resume_from_checkpoint}")
                print("Resuming training from checkpoint...")
        else:
            if _main:
                print("Checkpoints found but appear invalid. Starting fresh...")
    else:
        if _main:
            print("No checkpoint found. Starting fresh training...")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 6. Save final model
    if _main:
        print(f"\n{'='*60}")
        print("SAVING FINAL MODEL")
        print(f"{'='*60}")

    final_output_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_output_dir)

    if _main:
        print(f"\n✓ Model saved to: {final_output_dir}")

    # 7. Final evaluation
    if _main:
        print(f"\n{'='*60}")
        print("FINAL EVALUATION")
        print(f"{'='*60}")

    eval_results = trainer.evaluate()
    if _main:
        print(f"Eval Loss: {eval_results['eval_loss']:.4f}")
        print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)


if __name__ == "__main__":
    main()
