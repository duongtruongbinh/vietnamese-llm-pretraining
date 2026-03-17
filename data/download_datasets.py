#!/usr/bin/env python3
import os
from pathlib import Path

from datasets import load_dataset


OUTPUT_DIR = Path(__file__).parent / "train"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def download_and_save_dataset(
    name: str,
    split: str,
    output_filename: str,
    max_rows: int = None,
):
    print(f"Downloading: {name}")
    ds = load_dataset(name, split=split)

    if max_rows and len(ds) > max_rows:
        ds = ds.select(range(max_rows))

    output_path = OUTPUT_DIR / output_filename
    ds.to_parquet(output_path)

    print(f"  Saved {len(ds):,} rows → {output_path} ({format_size(os.path.getsize(output_path))})")


def main():
    download_and_save_dataset(
        name="bkai-foundation-models/BKAINewsCorpus",
        split="train",
        output_filename="bkai_train.parquet",
        max_rows=2_740_000,  # Limit to 2.74M rows (target_samples from config)
    )


if __name__ == "__main__":
    main()
