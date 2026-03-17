#!/usr/bin/env python3
import os
import unicodedata
from typing import Iterator

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast


# Mix: BKAINewsCorpus (news) + Wikipedia VI (local parquet files)
DATASET_CONFIGS = [
    {"path": "data/train/bkai_train.parquet",              "text_col": "text"},
    {"path": "data/train/vi_wiki_articles_clean.parquet",  "text_col": "text"},
]
VOCAB_SIZE = 50257  # Keep the same vocab size as original GPT-2
MIN_FREQUENCY = 2
SPECIAL_TOKEN = "<|endoftext|>"
OUTPUT_DIR = "./artifacts/tokenizer"


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)


def get_training_corpus(datasets_list, batch_size: int = 1000) -> Iterator[list]:
    for dataset in datasets_list:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            yield [normalize_text(text) for text in batch["text"]]


def train_tokenizer():
    all_datasets = []
    for cfg in DATASET_CONFIGS:
        if "path" in cfg:
            print(f"Loading: {cfg['path']}")
            ds = load_dataset("parquet", data_files=cfg["path"], split="train")
        else:
            print(f"Loading: {cfg['name']}")
            ds = load_dataset(cfg["name"], split=cfg["split"])

        if cfg["text_col"] != "text":
            ds = ds.rename_column(cfg["text_col"], "text")
        ds = ds.select_columns(["text"])
        print(f"  {len(ds):,} samples")
        all_datasets.append(ds)

    tokenizer = ByteLevelBPETokenizer()
    print(f"\nTraining tokenizer: vocab_size={VOCAB_SIZE:,}, min_freq={MIN_FREQUENCY}")
    tokenizer.train_from_iterator(
        get_training_corpus(all_datasets),
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=[SPECIAL_TOKEN],
        show_progress=True,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))

    gpt2_tokenizer = GPT2TokenizerFast(
        tokenizer_file=os.path.join(OUTPUT_DIR, "tokenizer.json"),
        bos_token=SPECIAL_TOKEN,
        eos_token=SPECIAL_TOKEN,
        unk_token=SPECIAL_TOKEN,
        pad_token=SPECIAL_TOKEN,
        model_max_length=1024,
    )
    gpt2_tokenizer.save_pretrained(OUTPUT_DIR)

    test_text = "Xin chào Việt Nam! Đây là bài kiểm tra tokenizer."
    encoded = gpt2_tokenizer.encode(normalize_text(test_text))
    decoded = gpt2_tokenizer.decode(encoded)
    print(f"\nVocab size: {len(gpt2_tokenizer):,}")
    print(f"Test: '{test_text}' → {len(encoded)} tokens → '{decoded}'")
    print(f"Saved to: {OUTPUT_DIR}")

    return gpt2_tokenizer


if __name__ == "__main__":
    train_tokenizer()
