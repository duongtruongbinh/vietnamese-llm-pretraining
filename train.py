#!/usr/bin/env python3
import glob
import os
import unicodedata
from typing import Dict, List, Any

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

BASE_MODEL    = "gpt2"
TOKENIZER_DIR = "./artifacts/tokenizer"
OUTPUT_DIR    = "./artifacts/checkpoints/scratch_init"

TOKEN_BUDGET              = 2_480_000_000
MAX_LENGTH                = 1024
EVAL_SPLIT_RATIO          = 0.01
PREPROCESSING_NUM_WORKERS = 4
DATASET_CONFIGS           = [
    {
        "path": "data/train/bkai_train.parquet",
        "text_col": "text",
    },
    {
        "path": "data/train/vi_wiki_articles_clean.parquet",
        "text_col": "text",
    },
]

LEARNING_RATE               = 5e-4
WEIGHT_DECAY                = 0.01
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE  = 2
GRADIENT_ACCUMULATION_STEPS = 64
WARMUP_RATIO                = 0.1
BF16                        = True
GRADIENT_CHECKPOINTING      = True
DATALOADER_NUM_WORKERS      = 0

WANDB_RUN_NAME = "gpt2-small-vietnamese-rand-init"


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)


def load_and_prepare_tokenizer() -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_main_process():
        print(f"Tokenizer: {TOKENIZER_DIR} (vocab_size={len(tokenizer):,})")
    return tokenizer


def load_and_prepare_model(tokenizer: GPT2TokenizerFast) -> GPT2LMHeadModel:
    _main = is_main_process()

    # Load GPT-2 Small config only (architecture), then build from scratch
    config = GPT2Config.from_pretrained(BASE_MODEL)
    config.vocab_size = len(tokenizer)  # Set vocab size upfront — no resize needed
    config.attn_implementation = "flash_attention_2"

    model = GPT2LMHeadModel(config)
    model = model.to(torch.bfloat16)  # BF16 for Flash Attention 2 compatibility

    # Tie lm_head ↔ wte weights
    config.tie_word_embeddings = True
    model.tie_weights()

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    if _main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: random init, {total_params/1e6:.1f}M params, flash_attention_2, bf16")

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

        if "vi_wiki" in src:
            ds = concatenate_datasets([ds] * 3)

        ds = ds.shuffle(seed=42)
        if _main:
            print(f"  {src}: {len(ds):,} samples")
        all_datasets.append(ds)

    dataset = concatenate_datasets(all_datasets).shuffle(seed=42)

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
        print(f"Dataset: {len(split_dataset['train']):,} train / {len(split_dataset['test']):,} eval blocks ({total_tokens/1e9:.2f}B tokens)")

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
        output_dir=OUTPUT_DIR,
        max_steps=max_steps,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=BF16 and torch.cuda.is_available(),
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
        print(f"Training: lr={LEARNING_RATE}, batch={effective_batch_size}, max_steps={max_steps:,} ({TOKEN_BUDGET/1e9:.2f}B tokens)")

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
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        else:
            print("WARNING: No GPU detected. Training will be slow.")

    tokenizer = load_and_prepare_tokenizer()
    model = load_and_prepare_model(tokenizer)
    dataset = load_and_prepare_dataset(tokenizer)
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    checkpoint_dirs = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    resume_from_checkpoint = None

    if checkpoint_dirs:
        def get_step(path):
            try:
                return int(os.path.basename(path).split('-')[-1])
            except (ValueError, IndexError):
                return 0
        for ckpt in sorted(checkpoint_dirs, key=get_step, reverse=True):
            model_file = os.path.join(ckpt, 'model.safetensors')
            if os.path.exists(model_file) and os.path.getsize(model_file) > 1_000_000:
                resume_from_checkpoint = ckpt
                break

    if _main:
        if resume_from_checkpoint:
            print(f"Resuming from: {resume_from_checkpoint}")
        else:
            print("Starting fresh training...")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_output_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_output_dir)

    eval_results = trainer.evaluate()
    if _main:
        loss = eval_results['eval_loss']
        print(f"Done. Eval loss={loss:.4f}, perplexity={torch.exp(torch.tensor(loss)):.2f}")
        print(f"Model saved to: {final_output_dir}")


if __name__ == "__main__":
    main()
