#!/usr/bin/env python3
"""Deduplicate pretraining corpus and produce an audit report.

Strategy:
  1. Global exact document dedup via normalized-text SHA-256
  2. Exact paragraph dedup for selected sources
  3. Final exact document dedup after paragraph dedup

Usage:
    python data_prep/deduplicate.py
"""

import argparse
import hashlib
import json
from loguru import logger
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from transformers import GPT2TokenizerFast

from src import config as cfg
from src.utils import format_size, normalize_text

DEDUP_DIR = Path(cfg.DEDUP_DIR)

# Paragraph dedup applies to these sources only (matched by filename stem).
PARA_STEMS = {"bkai_train", "vi_wiki_articles_clean"}
MIN_PARA_CHARS = 50
MIN_DOC_CHARS = 20
BATCH_SIZE = 50_000
TOKEN_BATCH_SIZE = 8192

def sha_bytes(text):
    return hashlib.sha256(normalize_text(text).encode("utf-8")).digest()

def dedup_paragraphs(text, seen_paras):
    kept = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) < MIN_PARA_CHARS:
            kept.append(para)
            continue
        h = sha_bytes(para)
        if h in seen_paras:
            continue
        seen_paras.add(h)
        kept.append(para)
    return "\n\n".join(kept).strip()

def flush_rows(writer, out_path, rows):
    if not rows:
        return writer
    table = pa.table({"text": rows})
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
    writer.write_table(table)
    rows.clear()
    return writer

def dedup_all():
    os.makedirs(DEDUP_DIR, exist_ok=True)

    seen_docs_raw = set()
    seen_docs_final = set()

    outputs = []
    report = {"sources": {}}
    total_original = 0
    total_kept = 0
    
    for path in cfg.RAW_DATASETS:
        if not os.path.exists(path):
            logger.warning("Skipping (not found): {}", path)
            continue

        name = Path(path).stem
        filename = Path(path).name
        parquet = pq.ParquetFile(path)
        original_docs = parquet.metadata.num_rows

        out_path = DEDUP_DIR / filename
        if out_path.exists():
            out_path.unlink()

        para_seen = set()
        use_para = name in PARA_STEMS
        writer = None
        out_rows = []

        stats = {
            "original_docs": original_docs,
            "removed_by_doc_dedup": 0,
            "removed_by_paragraph_dedup": 0,
            "removed_as_too_short": 0,
            "deduped_docs": 0,
        }

        pbar = tqdm(
            parquet.iter_batches(batch_size=BATCH_SIZE, columns=["text"]),
            total=(original_docs + BATCH_SIZE - 1) // BATCH_SIZE,
            desc=f"Deduplicating {name}",
            unit="batch",
            dynamic_ncols=True,
        )

        for batch in pbar:
            for text in batch.column(0).to_pylist():
                total_original += 1
                text = text or ""

                if not text:
                    stats["removed_as_too_short"] += 1
                    continue

                raw_hash = sha_bytes(text)
                if raw_hash in seen_docs_raw:
                    stats["removed_by_doc_dedup"] += 1
                    continue
                seen_docs_raw.add(raw_hash)

                if use_para:
                    new_text = dedup_paragraphs(text, para_seen)
                    if new_text != text.strip():
                        stats["removed_by_paragraph_dedup"] += 1
                else:
                    new_text = text.strip()

                if len(new_text) < MIN_DOC_CHARS:
                    stats["removed_as_too_short"] += 1
                    continue

                final_hash = sha_bytes(new_text)
                if final_hash in seen_docs_final:
                    stats["removed_by_doc_dedup"] += 1
                    continue
                seen_docs_final.add(final_hash)

                out_rows.append(new_text)
                stats["deduped_docs"] += 1
                total_kept += 1

                if len(out_rows) >= BATCH_SIZE:
                    writer = flush_rows(writer, out_path, out_rows)

        writer = flush_rows(writer, out_path, out_rows)
        if writer is not None:
            writer.close()
        else:
            pq.write_table(pa.table({"text": []}), out_path, compression="snappy")

        output_size = format_size(out_path.stat().st_size)
        stats["removed_docs"] = stats["original_docs"] - stats["deduped_docs"]
        stats["duplicate_rate"] = (
            round(stats["removed_docs"] / stats["original_docs"], 4)
            if stats["original_docs"] else 0.0
        )
        stats["output_path"] = str(out_path)
        stats["output_size"] = output_size

        report["sources"][name] = stats
        outputs.append({"name": name, "path": str(out_path), "rows": stats["deduped_docs"]})

        logger.info(
            "{} {} -> {} (-{})",
            name,
            f"{stats['original_docs']:,}",
            f"{stats['deduped_docs']:,}",
            f"{stats['removed_docs']:,}",
        )

    report["total_original_docs"] = total_original
    report["total_deduped_docs"] = total_kept
    report["total_removed_docs"] = total_original - total_kept
    report["duplicate_rate"] = (
        round((total_original - total_kept) / total_original, 4)
        if total_original else 0.0
    )

    return outputs, report

# ── Token audit ──────────────────────────────────────────────────────────────

def count_tokens(outputs):
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.TOKENIZER_DIR)
    tokenizer.model_max_length = 10**9
    counts = {}

    for item in outputs:
        parquet = pq.ParquetFile(item["path"])
        total_tokens = 0
        total_batches = (item["rows"] + TOKEN_BATCH_SIZE - 1) // TOKEN_BATCH_SIZE

        pbar = tqdm(
            parquet.iter_batches(batch_size=TOKEN_BATCH_SIZE, columns=["text"]),
            total=total_batches,
            desc=f"Tokenizing {item['name']}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in pbar:
            texts = [normalize_text(t or "") for t in batch.column(0).to_pylist()]
            enc = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_length=True,
            )
            total_tokens += sum(enc["length"])

        counts[item["name"]] = total_tokens
        logger.info("{} {} tokens", item["name"], f"{total_tokens:,}")

    return counts

def build_token_audit(token_counts):
    """Build token audit dict with unique counts and weighted effective counts."""
    weights = {Path(d["path"]).stem: d["weight"] for d in cfg.DATASETS}

    unique_tokens = sum(token_counts.values())
    sources = {}
    effective_total = 0

    for name, tokens in token_counts.items():
        w = weights.get(name, 1)
        weighted = tokens * w
        effective_total += weighted
        sources[name] = {"tokens": tokens, "weight": w, "weighted_tokens": weighted}

    budget = cfg.TOKEN_BUDGET
    gap = effective_total - budget

    return {
        "sources": sources,
        "unique_tokens": unique_tokens,
        "effective_tokens_with_weights": effective_total,
        "token_budget": budget,
        "repeat_factor": round(budget / unique_tokens, 2) if unique_tokens else None,
        "enough_for_budget": effective_total >= budget,
        "budget_gap_or_surplus": gap,
    }

# ── Reporting ────────────────────────────────────────────────────────────────

def print_summary(report):
    logger.info("")
    logger.info("=" * 72)
    logger.info("DEDUPLICATION SUMMARY")
    logger.info("=" * 72)

    for name, s in report["sources"].items():
        logger.info(
            "{} {} -> {}  (-{}, {:.1f}%)",
            name,
            f"{s['original_docs']:,}",
            f"{s['deduped_docs']:,}",
            f"{s['removed_docs']:,}",
            s["duplicate_rate"] * 100,
        )

    logger.info("-" * 72)
    logger.info(
        "{} {} -> {}  (-{}, {:.1f}%)",
        "TOTAL",
        f"{report['total_original_docs']:,}",
        f"{report['total_deduped_docs']:,}",
        f"{report['total_removed_docs']:,}",
        report["duplicate_rate"] * 100,
    )

    ta = report.get("token_audit")
    if not ta:
        logger.info("=" * 72)
        return

    logger.info("")
    logger.info("TOKEN AUDIT (unique)")
    logger.info("  Unique tokens after dedup : {}", f"{ta['unique_tokens']:,}")
    logger.info("  Training token budget     : {}", f"{ta['token_budget']:,}")
    logger.info("  Repeat factor (unweighted): {:.2f}x", ta["repeat_factor"])

    logger.info("")
    logger.info("TOKEN AUDIT (weighted training mixture)")
    for name, s in ta["sources"].items():
        logger.info(
            "  {} {}  x{}  = {}",
            name, f"{s['tokens']:,}", s["weight"], f"{s['weighted_tokens']:,}",
        )
    logger.info("  {} {}", "Effective tokens", f"{ta['effective_tokens_with_weights']:,}")
    logger.info("  {} {}", "Token budget", f"{ta['token_budget']:,}")

    gap = ta["budget_gap_or_surplus"]
    if ta["enough_for_budget"]:
        logger.info("  → Surplus: {} tokens above budget", f"{gap:,}")
    else:
        logger.info("  → Gap: {} tokens below budget", f"{-gap:,}")

    logger.info("=" * 72)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    outputs, report = dedup_all()
    logger.info("Counting tokens...")
    token_counts = count_tokens(outputs)
    report["token_audit"] = build_token_audit(token_counts)

    report_path = DEDUP_DIR / "dedup_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report: {}", report_path)

    print_summary(report)

if __name__ == "__main__":
    main()
