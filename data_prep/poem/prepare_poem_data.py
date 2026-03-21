#!/usr/bin/env python3
"""Preprocess poem data: extract valid 5-word quatrains (4 lines x 5 words)."""

import html
import json
import logging
import os
import re

import pandas as pd

from src.config import (
    POEM_RAW_CSV, POEM_DATA_PATH,
    POEM_LINES_PER_STANZA, POEM_WORDS_PER_LINE,
)
from src.utils import configure_root_logging, normalize_text

logger = logging.getLogger(__name__)

_TRAILING_PUNCT = re.compile(r'[\"\u201c\u201d\u2018\u2019":;,\-–—]+$')
_LEADING_PUNCT = re.compile(r'^[\"\u201c\u201d\u2018\u2019]+')


def clean_html_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</img>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<i>.*?</i>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?b>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?div[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("\u00a0", " ")
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_source_column(source: str) -> str:
    if pd.isna(source) or not source:
        return ""
    source = re.sub(r"Bình luận nhanh.*", "", source, flags=re.IGNORECASE)
    return source.strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["content"] = df["content"].apply(
        lambda x: clean_html_tags(str(x)) if pd.notna(x) else ""
    )
    df["source"] = df["source"].apply(
        lambda x: clean_source_column(str(x)) if pd.notna(x) else ""
    )
    df["title"] = df["title"].apply(
        lambda x: html.unescape(str(x)).strip() if pd.notna(x) else ""
    )
    df = df[df["content"].str.strip() != ""].copy()
    return df


def clean_line(line: str) -> str:
    """Normalize a poem line."""
    line = re.sub(r"\s+", " ", line).strip()
    line = _LEADING_PUNCT.sub("", line).strip()
    line = _TRAILING_PUNCT.sub("", line).strip()
    line = normalize_text(line)
    return line


def count_words(line: str) -> int:
    """Count words in a line (excluding punctuation)."""
    cleaned = re.sub(r"[.,!?;:\"'\u201c\u201d\u2018\u2019\-–—()]", " ", line)
    return len(cleaned.split())


def extract_valid_stanzas(content: str) -> list[str]:
    """Extract stanzas with exactly 4 lines x 5 words each."""
    raw_stanzas = re.split(r"\n\s*\n", content.strip())
    valid = []

    for raw in raw_stanzas:
        lines = [clean_line(l) for l in raw.strip().split("\n") if l.strip()]
        lines = [l for l in lines if l]

        if len(lines) != POEM_LINES_PER_STANZA:
            continue

        if all(count_words(l) == POEM_WORDS_PER_LINE for l in lines):
            valid.append("\n".join(lines))

    return valid


def main() -> None:
    configure_root_logging()
    df = pd.read_csv(POEM_RAW_CSV, encoding="utf-8-sig")
    df = clean_dataframe(df)

    output_dir = os.path.dirname(POEM_DATA_PATH)
    os.makedirs(output_dir, exist_ok=True)

    all_stanzas: list[str] = []
    for _, row in df.iterrows():
        content = str(row["content"])
        stanzas = extract_valid_stanzas(content)
        all_stanzas.extend(stanzas)

    unique_stanzas = list(dict.fromkeys(all_stanzas))

    with open(POEM_DATA_PATH, "w", encoding="utf-8") as f:
        for stanza in unique_stanzas:
            json.dump({"text": stanza}, f, ensure_ascii=False)
            f.write("\n")

    logger.info("Saved %d stanzas to %s", len(unique_stanzas), POEM_DATA_PATH)


if __name__ == "__main__":
    main()
