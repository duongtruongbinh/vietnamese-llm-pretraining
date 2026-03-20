#!/usr/bin/env python3
"""SFT GPT-2 for 5-word quatrain generation (4 lines x 5 words)."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_PROJECT", "vietnamese-gpt2")

import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)
from config import (
    MODEL_DIR, BF16, WARMUP_RATIO,
    POEM_DATA_PATH, POEM_CHECKPOINT_DIR, POEM_PREFIX,
    POEM_EPOCHS, POEM_BATCH_SIZE, POEM_LEARNING_RATE,
    POEM_WEIGHT_DECAY, POEM_MAX_LENGTH,
)
from utils import normalize_text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model from {MODEL_DIR}")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16 if BF16 else torch.float32,
    )
    model.tie_weights()
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    print(f"Loading data from {POEM_DATA_PATH}")
    ds = load_dataset("json", data_files=POEM_DATA_PATH, split="train")
    print(f"  Samples: {len(ds)}")

    eos = tokenizer.eos_token
    prefix_len = len(tokenizer(POEM_PREFIX, add_special_tokens=False)["input_ids"])

    def tokenize(batch):
        texts = [POEM_PREFIX + normalize_text(t) + eos for t in batch["text"]]
        enc = tokenizer(texts, truncation=True, max_length=POEM_MAX_LENGTH, padding="max_length")

        # Mask prefix and padding in labels
        enc["labels"] = [
            [-100 if i < prefix_len or enc["attention_mask"][j][i] == 0 else tok
             for i, tok in enumerate(ids)]
            for j, ids in enumerate(enc["input_ids"])
        ]
        return enc

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(ds['train'])} | Eval: {len(ds['test'])}")

    args = TrainingArguments(
        output_dir=POEM_CHECKPOINT_DIR,
        num_train_epochs=POEM_EPOCHS,
        per_device_train_batch_size=POEM_BATCH_SIZE,
        per_device_eval_batch_size=POEM_BATCH_SIZE,
        learning_rate=POEM_LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=POEM_WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        bf16=BF16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to=["wandb"],
        run_name="gpt2-sft-poem",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
    )

    print(f"\nTraining: epochs={POEM_EPOCHS}, batch={POEM_BATCH_SIZE}, lr={POEM_LEARNING_RATE}")
    trainer.train()

    final_dir = os.path.join(POEM_CHECKPOINT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    result = trainer.evaluate()
    loss = result["eval_loss"]
    ppl = torch.exp(torch.tensor(loss)).item()
    print(f"\nDone. Loss={loss:.4f}, PPL={ppl:.1f}")
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()
