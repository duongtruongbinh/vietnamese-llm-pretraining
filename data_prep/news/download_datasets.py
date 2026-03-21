#!/usr/bin/env python3
"""Download news corpus (BKAI) and save as Parquet under data/train/."""

import logging
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from datasets import load_dataset
from src.utils import configure_root_logging, format_size

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "data" / "train"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_and_save_dataset(
    name: str,
    split: str,
    output_filename: str,
    max_rows: int | None = None,
) -> None:
    logger.info("Downloading: %s", name)
    ds = load_dataset(name, split=split)

    if max_rows and len(ds) > max_rows:
        ds = ds.select(range(max_rows))

    output_path = OUTPUT_DIR / output_filename
    ds.to_parquet(output_path)

    logger.info(
        "Saved %s rows → %s (%s)",
        f"{len(ds):,}",
        output_path,
        format_size(os.path.getsize(output_path)),
    )


def main() -> None:
    configure_root_logging()
    download_and_save_dataset(
        name="bkai-foundation-models/BKAINewsCorpus",
        split="train",
        output_filename="bkai_train.parquet",
        max_rows=2_740_000,
    )


if __name__ == "__main__":
    main()
