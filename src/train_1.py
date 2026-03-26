#!/usr/bin/env python3
from loguru import logger
import os
from itertools import chain
import pandas as pd

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
from transformers.trainer_utils import get_last_checkpoint

from src.config import (
    BASE_MODEL, TOKENIZER_DIR, CHECKPOINT_DIR,
    TOKEN_BUDGET, MAX_LENGTH, EVAL_SPLIT_RATIO,
    PREPROCESSING_NUM_WORKERS, DATASETS,
    LEARNING_RATE, WEIGHT_DECAY,
    PER_DEVICE_TRAIN_BATCH_SIZE, PER_DEVICE_EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS, WARMUP_RATIO, BF16,
    GRADIENT_CHECKPOINTING, DATALOADER_NUM_WORKERS,
    WANDB_RUN_NAME,
)
from src.utils import normalize_text, PerplexityCallback

def load_and_prepare_dataset(tokenizer):
    all_datasets = []
    for src in DATASETS:
        ds = load_dataset("parquet", data_files=src["path"], split="train")
        ds = ds.select_columns(["text"])
        ds = ds.shuffle(seed=42)

        weight = src["weight"]
        if weight > 1:
            ds = concatenate_datasets([ds] * weight)

        logger.info("  {}: {} samples (weight={})", src["path"], f"{len(ds):,}", weight)
        all_datasets.append(ds)

    dataset = concatenate_datasets(all_datasets).shuffle(seed=42)
    eos_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        texts = [normalize_text(t) for t in examples["text"]]
        tokenized = tokenizer(texts, truncation=False, return_attention_mask=False)
        for ids in tokenized["input_ids"]:
            ids.append(eos_token_id)
        return tokenized

    def group_texts(examples):
        concatenated = {k: list(chain.from_iterable(examples[k])) for k in examples}
        total_length = len(concatenated["input_ids"])
        if total_length >= MAX_LENGTH:
            total_length = (total_length // MAX_LENGTH) * MAX_LENGTH
        result = {
            k: [t[i : i + MAX_LENGTH] for i in range(0, total_length, MAX_LENGTH)]
            for k, t in concatenated.items()
        }
        result["labels"] = [block[:] for block in result["input_ids"]]
        return result

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=PREPROCESSING_NUM_WORKERS,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    grouped = tokenized.map(
        group_texts,
        batched=True,
        num_proc=PREPROCESSING_NUM_WORKERS,
        desc="Grouping texts",
    )

    split = grouped.train_test_split(test_size=EVAL_SPLIT_RATIO, seed=42)

    total_tokens = len(grouped) * MAX_LENGTH
    logger.info(
        "Dataset: {} train / {} eval blocks ({:.2f}B tokens)",
        f"{len(split['train']):,}",
        f"{len(split['test']):,}",
        total_tokens / 1e9,
    )

    return split

def main():

    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU: {} ({:.1f} GB)", torch.cuda.get_device_name(0), mem_gb)
    else:
        logger.warning("No GPU detected. Training will be slow.")

    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer: {} (vocab_size={})", TOKENIZER_DIR, f"{len(tokenizer):,}")

    config = GPT2Config.from_pretrained(BASE_MODEL)
    config.vocab_size = len(tokenizer)
    config.use_cache = False
    config.attn_implementation = "flash_attention_2"
    # config.scale_attn_by_inverse_layer_idx = True
    # config.reorder_and_upcast_attn = True
    model = GPT2LMHeadModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: random init, {:.1f}M params, flash_attention_2, bf16", n_params / 1e6)

    dataset = load_and_prepare_dataset(tokenizer)

    num_gpus = max(1, torch.cuda.device_count())
    tokens_per_step = (
        PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * num_gpus * MAX_LENGTH
    )
    max_steps = TOKEN_BUDGET // tokens_per_step

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        max_steps=max_steps,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=BF16,
        eval_strategy="steps", save_strategy="steps",
        eval_steps=500, save_steps=500,
        save_total_limit=3, 
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        report_to=["wandb"],
        run_name=WANDB_RUN_NAME,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        seed=42, data_seed=42,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    effective_batch = (
        PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * num_gpus
    )
    logger.info(
        "Training: lr={}, batch={}, max_steps={} ({:.2f}B tokens)",
        LEARNING_RATE, effective_batch, f"{max_steps:,}", TOKEN_BUDGET / 1e9,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[PerplexityCallback()],
    )

    resume_from = get_last_checkpoint(CHECKPOINT_DIR) if os.path.isdir(CHECKPOINT_DIR) else None
    if resume_from:
        logger.info("Resuming from: {}", resume_from)
    else:
        logger.info("Starting fresh training...")

    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = os.path.join(CHECKPOINT_DIR, "final")
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)

    result = trainer.evaluate()
    loss = result["eval_loss"]
    ppl = torch.exp(torch.tensor(loss)).item()
    logger.info("Done. Eval loss={:.4f}, perplexity={:.2f}", loss, ppl)
    logger.info("Model saved to: {}", final_dir)
    
    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(os.path.join(CHECKPOINT_DIR, "log_history.csv"), index=False)

if __name__ == "__main__":
    main()
