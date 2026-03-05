#!/usr/bin/env python3
"""
Tokenizer Training Script for Vietnamese GPT-2
================================================
Train a ByteLevelBPETokenizer from BKAINewsCorpus dataset for Vietnamese.
"""

import os
import unicodedata
from typing import Iterator

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast


# =============================== Configuration ================================
# Mix: BKAINewsCorpus (news) + Wikipedia VI
DATASET_CONFIGS = [
    {"name": "bkai-foundation-models/BKAINewsCorpus", "split": "train", "text_col": "text"},
    {"name": "vietgpt/wikipedia_vi",                  "split": "train", "text_col": "text"},
]
VOCAB_SIZE = 50257  # Keep the same vocab size as original GPT-2
MIN_FREQUENCY = 2
SPECIAL_TOKEN = "<|endoftext|>"
OUTPUT_DIR = "./vietnamese_tokenizer"


def normalize_text(text: str) -> str:
    """
    Normalize text to Unicode NFC form.
    
    Args:
        text: Input text
        
    Returns:
        NFC-normalized text
    """
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)


def get_training_corpus(datasets_list, batch_size: int = 1000) -> Iterator[list]:
    """
    Generator to yield batches of text from multiple datasets.

    Args:
        datasets_list: List of Hugging Face datasets
        batch_size: Number of samples per batch

    Yields:
        List of NFC-normalized texts
    """
    for dataset in datasets_list:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            yield [normalize_text(text) for text in batch["text"]]


def train_tokenizer():
    """
    Train a ByteLevelBPETokenizer from BKAINewsCorpus.
    """
    print("=" * 60)
    print("VIETNAMESE GPT-2 TOKENIZER TRAINING")
    print("=" * 60)
    
    # 1. Load datasets
    all_datasets = []
    total_samples = 0
    for cfg in DATASET_CONFIGS:
        print(f"\n[1/{len(DATASET_CONFIGS)}+] Loading dataset: {cfg['name']}")
        ds = load_dataset(cfg["name"], split=cfg["split"])
        if cfg["text_col"] != "text":
            ds = ds.rename_column(cfg["text_col"], "text")
        ds = ds.select_columns(["text"])
        print(f"      Samples: {len(ds):,}")
        total_samples += len(ds)
        all_datasets.append(ds)
    print(f"\n      Total samples (combined): {total_samples:,}")
    
    # 2. Initialize ByteLevelBPETokenizer
    print("\n[2/5] Initializing ByteLevelBPETokenizer...")
    tokenizer = ByteLevelBPETokenizer()

    # 3. Train tokenizer
    print(f"\n[3/5] Training tokenizer with vocab_size={VOCAB_SIZE:,}")
    print(f"      Special token: {SPECIAL_TOKEN}")
    print(f"      Min frequency: {MIN_FREQUENCY}")

    # Train from iterator (streams all datasets)
    tokenizer.train_from_iterator(
        get_training_corpus(all_datasets),
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=[SPECIAL_TOKEN],
        show_progress=True,
    )
    
    # 4. Save tokenizer
    print(f"\n[4/5] Saving tokenizer to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save tokenizer in tokenizer.json format (standard tokenizers library format)
    tokenizer.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))

    # 5. Convert to GPT2TokenizerFast format
    print("\n[5/5] Converting to GPT2TokenizerFast format...")
    
    # Reload with GPT2TokenizerFast from tokenizer.json
    gpt2_tokenizer = GPT2TokenizerFast(
        tokenizer_file=os.path.join(OUTPUT_DIR, "tokenizer.json"),
        bos_token=SPECIAL_TOKEN,
        eos_token=SPECIAL_TOKEN,
        unk_token=SPECIAL_TOKEN,
        pad_token=SPECIAL_TOKEN,
        model_max_length=1024,
    )
    
    # Save in standard transformers format
    gpt2_tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Vocab size: {len(gpt2_tokenizer):,}")
    print(f"BOS token: {gpt2_tokenizer.bos_token} (id={gpt2_tokenizer.bos_token_id})")
    print(f"EOS token: {gpt2_tokenizer.eos_token} (id={gpt2_tokenizer.eos_token_id})")
    print(f"UNK token: {gpt2_tokenizer.unk_token} (id={gpt2_tokenizer.unk_token_id})")
    print(f"PAD token: {gpt2_tokenizer.pad_token} (id={gpt2_tokenizer.pad_token_id})")
    
    # Test tokenization
    test_text = "Xin chào Việt Nam! Đây là bài kiểm tra tokenizer."
    test_text_normalized = normalize_text(test_text)
    encoded = gpt2_tokenizer.encode(test_text_normalized)
    decoded = gpt2_tokenizer.decode(encoded)
    
    print(f"\nTest tokenization:")
    print(f"  Input:   '{test_text}'")
    print(f"  Tokens:  {len(encoded)} tokens")
    print(f"  IDs:     {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
    print(f"  Decoded: '{decoded}'")
    
    print("\n" + "=" * 60)
    print(f"✓ Tokenizer saved successfully to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return gpt2_tokenizer


if __name__ == "__main__":
    train_tokenizer()
