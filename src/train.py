#!/usr/bin/env python3
import logging
import os

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
    PREPROCESSING_NUM_WORKERS, DATASET_CONFIGS,
    LEARNING_RATE, WEIGHT_DECAY,
    PER_DEVICE_TRAIN_BATCH_SIZE, PER_DEVICE_EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS, WARMUP_RATIO, BF16,
    GRADIENT_CHECKPOINTING, DATALOADER_NUM_WORKERS,
    WANDB_RUN_NAME,
)
from src.utils import configure_root_logging, normalize_text

logger = logging.getLogger(__name__)


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def load_and_prepare_tokenizer() -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_main_process():
        logger.info("Tokenizer: %s (vocab_size=%s)", TOKENIZER_DIR, f"{len(tokenizer):,}")
    return tokenizer


def load_and_prepare_model(tokenizer: GPT2TokenizerFast) -> GPT2LMHeadModel:
    config = GPT2Config.from_pretrained(BASE_MODEL)
    config.vocab_size = len(tokenizer)
    config.attn_implementation = "flash_attention_2"

    model = GPT2LMHeadModel(config)
    model = model.to(torch.bfloat16)

    config.tie_word_embeddings = True
    model.tie_weights()

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "Model: random init, %.1fM params, flash_attention_2, bf16",
            total_params / 1e6,
        )

    return model


def load_and_prepare_dataset(tokenizer: GPT2TokenizerFast):
    _main = is_main_process()

    all_datasets = []
    for cfg in DATASET_CONFIGS:
        if "path" in cfg:
            ds = load_dataset("parquet", data_files=cfg["path"], split="train")
            src = cfg["path"]
        else:
            ds = load_dataset(cfg["name"], split=cfg["split"])
            src = cfg["name"]

        if cfg["text_col"] != "text":
            ds = ds.rename_column(cfg["text_col"], "text")
        ds = ds.select_columns(["text"])

        weight = cfg.get("weight", 1)
        if weight > 1:
            ds = concatenate_datasets([ds] * weight)

        ds = ds.shuffle(seed=42)
        if _main:
            logger.info("  %s: %s samples", src, f"{len(ds):,}")
        all_datasets.append(ds)

    dataset = concatenate_datasets(all_datasets).shuffle(seed=42)

    eos_token_id = tokenizer.eos_token_id

    def tokenize_function(examples: dict[str, list]) -> dict[str, list[list[int]]]:
        normalized_texts = [normalize_text(text) for text in examples["text"]]
        tokenized = tokenizer(
            normalized_texts,
            truncation=False,
            return_attention_mask=False,
        )
        for i in range(len(tokenized["input_ids"])):
            tokenized["input_ids"][i].append(eos_token_id)
        return tokenized

    def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
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

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=PREPROCESSING_NUM_WORKERS,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    grouped_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=PREPROCESSING_NUM_WORKERS,
        desc="Grouping texts",
    )

    split_dataset = grouped_dataset.train_test_split(test_size=EVAL_SPLIT_RATIO, seed=42)

    if _main:
        total_tokens = len(grouped_dataset) * MAX_LENGTH
        logger.info(
            "Dataset: %s train / %s eval blocks (%.2fB tokens)",
            f"{len(split_dataset['train']):,}",
            f"{len(split_dataset['test']):,}",
            total_tokens / 1e9,
        )

    return split_dataset


def create_trainer(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    train_dataset,
    eval_dataset,
) -> Trainer:
    _main = is_main_process()

    num_gpus = max(1, torch.cuda.device_count())
    tokens_per_step = (
        PER_DEVICE_TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
        * num_gpus
        * MAX_LENGTH
    )
    max_steps = TOKEN_BUDGET // tokens_per_step

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        max_steps=max_steps,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=BF16,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        report_to=["wandb"],
        run_name=WANDB_RUN_NAME,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=True,
        seed=42,
        data_seed=42,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * max(1, torch.cuda.device_count())
    )

    if _main:
        logger.info(
            "Training: lr=%s, batch=%s, max_steps=%s (%.2fB tokens)",
            LEARNING_RATE,
            effective_batch_size,
            f"{max_steps:,}",
            TOKEN_BUDGET / 1e9,
        )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )


def main():
    configure_root_logging()
    _main = is_main_process()

    if _main:
        if torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0), mem_gb)
        else:
            logger.warning("No GPU detected. Training will be slow.")

    tokenizer = load_and_prepare_tokenizer()
    model = load_and_prepare_model(tokenizer)
    dataset = load_and_prepare_dataset(tokenizer)
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    resume_from_checkpoint = None
    if os.path.isdir(CHECKPOINT_DIR):
        resume_from_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)

    if _main:
        if resume_from_checkpoint:
            logger.info("Resuming from: %s", resume_from_checkpoint)
        else:
            logger.info("Starting fresh training...")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_output_dir = os.path.join(CHECKPOINT_DIR, "final")
    trainer.save_model(final_output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_output_dir)

    eval_results = trainer.evaluate()
    if _main:
        loss = eval_results['eval_loss']
        ppl = torch.exp(torch.tensor(loss)).item()
        logger.info("Done. Eval loss=%.4f, perplexity=%.2f", loss, ppl)
        logger.info("Model saved to: %s", final_output_dir)


if __name__ == "__main__":
    main()
