#!/usr/bin/env python3
"""Shared utility functions."""

import logging
import unicodedata
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def normalize_text(text: str) -> str:
    """Apply Unicode NFC normalization, returning empty string for None."""
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)


def format_size(num_bytes: int) -> str:
    """Format byte count as human-readable string (B/KB/MB/GB/TB)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def configure_root_logging(level: int = logging.INFO) -> None:
    """Initialize root logging once."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_gpt2_lm_head(
    model_dir: str | Path,
    *,
    torch_dtype: torch.dtype | None = None,
    tie_weights: bool = False,
    pad_token_to_eos: bool = False,
    eval_mode: bool | None = True,
) -> tuple[GPT2LMHeadModel, GPT2TokenizerFast, str]:
    """Load GPT2LMHeadModel and tokenizer from a saved directory.

    Returns (model, tokenizer, device) where device is cuda or cpu.
    """
    model_dir = str(Path(model_dir).resolve())
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    if pad_token_to_eos and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw: dict = {}
    if torch_dtype is not None:
        load_kw["torch_dtype"] = torch_dtype
    model = GPT2LMHeadModel.from_pretrained(model_dir, **load_kw)
    if tie_weights:
        model.tie_weights()
    if pad_token_to_eos:
        model.config.pad_token_id = tokenizer.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if eval_mode is True:
        model.eval()
    elif eval_mode is False:
        model.train()

    return model, tokenizer, device


def gpt2_generate_texts(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    device: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
    num_return_sequences: int,
) -> list[str]:
    """Run `model.generate` and return decoded strings (NFC-normalized prompt)."""
    prompt = normalize_text(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
