#!/usr/bin/env python3
"""
Continual Pre-training Script for Vietnamese GPT-2
===================================================
Continue pre-training GPT-2 Small model from English to Vietnamese.
"""

import glob
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
BASE_MODEL = "gpt2"  # GPT-2 Small (124M parameters)
TOKENIZER_DIR = "./vietnamese_tokenizer"
OUTPUT_DIR = "./vietnamese_gpt2_pre_end_mixed_data"
LOGGING_DIR = "./logs"

# Dataset settings
# Mix: BKAINewsCorpus (news) + Wikipedia VI
DATASET_CONFIGS = [
    {"name": "bkai-foundation-models/BKAINewsCorpus", "split": "train", "text_col": "text"},
    {"name": "vietgpt/wikipedia_vi",                  "split": "train", "text_col": "text"},
]
MAX_LENGTH = 1024  # Block size for grouped texts

# Training hyperparameters
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 2  # Reduced from 4 to avoid OOM
PER_DEVICE_EVAL_BATCH_SIZE = 2  # Reduced from 4 to avoid OOM
GRADIENT_ACCUMULATION_STEPS = 64  # 2 (batch) x 64 (accum) x 1 (GPU) = 128 effective
WARMUP_RATIO = 0.1
FP16 = True
GRADIENT_CHECKPOINTING = True

# Data processing
PREPROCESSING_NUM_WORKERS = 4
DATALOADER_NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues with CUDA
EVAL_SPLIT_RATIO = 0.01  # 1% for evaluation

# Wandb configuration
WANDB_RUN_NAME = "gpt2-small-vietnamese-continual"


def is_main_process() -> bool:
    """Check if this is the main process (rank 0) in DDP training."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def normalize_text(text: str) -> str:
    """
    Normalize text to Unicode NFC form.
    """
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)


def load_and_prepare_tokenizer() -> GPT2TokenizerFast:
    """
    Load the trained Vietnamese tokenizer.
    """
    print(f"Loading tokenizer from: {TOKENIZER_DIR}")
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR)

    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process():
        print(f"Tokenizer vocab size: {len(tokenizer):,}")

        # Validate Vietnamese tokenization
        test_text = "Việt Nam là một đất nước"
        test_tokens = tokenizer.tokenize(test_text)
        print(f"Sample tokenization: '{test_text}'")
        print(f"  → Tokens: {test_tokens[:8]}...")
        print(f"  → Token count: {len(test_tokens)} tokens")

    return tokenizer


def load_and_prepare_model(tokenizer: GPT2TokenizerFast) -> GPT2LMHeadModel:
    """
    Load GPT-2 model and resize embeddings for new vocabulary.
    """
    _main = is_main_process()

    if _main:
        print(f"\nLoading base model: {BASE_MODEL}")

    # Load config to verify GPT-2 Small architecture
    config = GPT2Config.from_pretrained(BASE_MODEL)
    if _main:
        print(f"Model architecture:")
        print(f"  - n_layer: {config.n_layer}")
        print(f"  - n_head: {config.n_head}")
        print(f"  - n_embd: {config.n_embd}")
        print(f"  - Original vocab_size: {config.vocab_size}")

    # Use Flash Attention 2 for faster training.
    # Requires model to be loaded in fp16/bf16.
    attn_implementation = "flash_attention_2"
    if _main:
        print(f"Attention implementation: {attn_implementation}")

    # Load model in FP16 for Flash Attention 2 compatibility.
    model = GPT2LMHeadModel.from_pretrained(
        BASE_MODEL,
        config=config,
        attn_implementation=attn_implementation,
        torch_dtype=torch.float16,
    )

    # Resize token embeddings to match new tokenizer
    original_vocab_size = model.config.vocab_size
    new_vocab_size = len(tokenizer)

    if _main:
        print(f"\nResizing embeddings: {original_vocab_size} -> {new_vocab_size}")

        if original_vocab_size == new_vocab_size:
            print("⚠️  WARNING: Vocab sizes are identical!")
            print("   This suggests you may be using the original GPT-2 tokenizer,")
            print("   not a Vietnamese-optimized tokenizer.")
            print("   Consider training a Vietnamese-specific tokenizer with different vocab size.")

    model.resize_token_embeddings(new_vocab_size)

    # Re-tie weights after resize to fix lm_head.weight missing in DDP
    model.config.tie_word_embeddings = True
    model.tie_weights()
    if _main:
        print(f"Weight tying (lm_head ↔ wte): ENABLED")

    # Enable gradient checkpointing to save VRAM
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        if _main:
            print("Gradient checkpointing: ENABLED")

    # Count parameters
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
    """
    _main = is_main_process()

    if _main:
        print(f"\n{'='*60}")
        print("DATASET PREPARATION")
        print(f"{'='*60}")

    # 1. Load and mix datasets
    all_datasets = []
    for cfg in DATASET_CONFIGS:
        if _main:
            print(f"\nLoading dataset: {cfg['name']}")
        ds = load_dataset(cfg["name"], split=cfg["split"])
        # Keep only the text column
        if cfg["text_col"] != "text":
            ds = ds.rename_column(cfg["text_col"], "text")
        ds = ds.select_columns(["text"])
        if _main:
            print(f"  → {len(ds):,} samples")
        all_datasets.append(ds)

    # Concatenate and shuffle
    dataset = concatenate_datasets(all_datasets)
    dataset = dataset.shuffle(seed=42)
    if _main:
        print(f"\nCombined dataset: {len(dataset):,} total samples")
    
    # 2. Tokenize function
    eos_token_id = tokenizer.eos_token_id
    
    def tokenize_function(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        """Tokenize, normalize NFC, and add EOS between documents."""
        # Normalize Unicode NFC
        normalized_texts = [normalize_text(text) for text in examples["text"]]
        
        # Tokenize
        tokenized = tokenizer(
            normalized_texts,
            truncation=False,  # Don't truncate, will group later
            return_attention_mask=False,
        )
        
        # Add EOS token at the end of each document for separation
        for i in range(len(tokenized["input_ids"])):
            tokenized["input_ids"][i].append(eos_token_id)
        
        return tokenized
    
    # 3. Group texts function
    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        """
        Concatenate all texts and split into blocks of MAX_LENGTH.
        This technique ensures no padding/waste in the data.
        """
        # Concatenate all input_ids
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        
        # Calculate total length
        total_length = len(concatenated["input_ids"])
        
        # Split into complete blocks (discard remainder)
        if total_length >= MAX_LENGTH:
            total_length = (total_length // MAX_LENGTH) * MAX_LENGTH
        
        # Split into chunks
        result = {
            k: [t[i : i + MAX_LENGTH] for i in range(0, total_length, MAX_LENGTH)]
            for k, t in concatenated.items()
        }
        
        # Add labels (same as input_ids for CLM)
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
    """
    Create Trainer with optimized configuration.
    """
    _main = is_main_process()

    if _main:
        print(f"\n{'='*60}")
        print("TRAINER CONFIGURATION")
        print(f"{'='*60}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # Training params
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # Optimizer params
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        
        # Mixed precision
        fp16=FP16 and torch.cuda.is_available(),
        
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
    
    # Data collator for Causal Language Modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not Masked LM
    )
    
    # Print configuration
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
        print(f"FP16: {training_args.fp16}")
        print(f"Epochs: {training_args.num_train_epochs}")
    
    # Create trainer
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
    """
    Main training pipeline.
    """
    _main = is_main_process()

    if _main:
        print("=" * 60)
        print("VIETNAMESE GPT-2 CONTINUAL PRE-TRAINING")
        print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        if _main:
            print(f"\nGPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        if _main:
            print("\nWARNING: No GPU detected. Training will be slow.")
    
    # 1. Load tokenizer
    tokenizer = load_and_prepare_tokenizer()
    
    # 2. Load model
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
    
    # Check for existing checkpoints to resume from
    checkpoint_dirs = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    resume_from_checkpoint = None
    
    if checkpoint_dirs:
        # Sort by step number, try from latest to oldest
        def get_step(path):
            try:
                return int(os.path.basename(path).split('-')[-1])
            except (ValueError, IndexError):
                return 0
        checkpoint_dirs_sorted = sorted(checkpoint_dirs, key=get_step, reverse=True)
        
        # Use the latest valid checkpoint, skip corrupted ones
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
