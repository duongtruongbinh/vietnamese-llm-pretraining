#!/usr/bin/env python3
"""Generate 5-word quatrains from stage-2 model."""

from loguru import logger

import torch

from src.config import (
    POEM_MODEL_DIR, POEM_PREFIX, POEM_MAX_LENGTH,
    TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY,
)
from src.utils import generate_texts, load_gpt2

DEFAULT_POEM_GEN_CONFIG = {
    "max_new_tokens": POEM_MAX_LENGTH,
    "num_return_sequences": 2,
    "do_sample": True,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "repetition_penalty": REPETITION_PENALTY,
}


def _parse_user_value(raw: str, current):
    """Parse interactive config input into current value's type."""
    if isinstance(current, bool):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid bool value: {raw}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return type(current)(raw)


def generate_poems(model, tokenizer, device, prompt: str = "", **kwargs) -> list[str]:
    """Generate poems from optional first-line prompt; strip stage-2 prefix from output."""
    full_prompt = POEM_PREFIX + prompt
    texts = generate_texts(
        model,
        tokenizer,
        device,
        full_prompt,
        **kwargs,
    )
    return [
        t[len(POEM_PREFIX):].strip() if t.startswith(POEM_PREFIX) else t
        for t in texts
    ]


def interactive_mode(model, tokenizer, device) -> None:
    """Run interactive generation loop for stage-2 poem model."""
    logger.info("Interactive mode.")
    logger.info("Commands: 'config' to edit generation params, 'quit' to exit.")
    gen_config = dict(DEFAULT_POEM_GEN_CONFIG)

    while True:
        try:
            prompt = input("\nFirst line: ").strip()
            if prompt.lower() in {"q", "quit", "exit"}:
                break
            if prompt.lower() == "config":
                for key, value in gen_config.items():
                    new_value = input(f"  {key} [{value}]: ").strip()
                    if new_value:
                        try:
                            gen_config[key] = _parse_user_value(new_value, value)
                        except ValueError as exc:
                            logger.warning("Skip invalid value for {}: {}", key, exc)
                continue
            if not prompt:
                continue

            for poem in generate_poems(model, tokenizer, device, prompt, **gen_config):
                logger.info("{}\n", poem)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception("Generation error: {}", e)


def main():
    model, tokenizer, device = load_gpt2(
        POEM_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        tie_weights=True,
        pad_token_to_eos=True,
        eval_mode=True,
    )

    prompts = [
        "",
        "Trăng sáng trên đầu núi",
        "Mùa thu lá vàng rơi",
        "Hoa sen nở trắng hồ",
    ]

    for prompt in prompts:
        logger.info("\n[Prompt: {}]", prompt or "(empty)")
        for poem in generate_poems(model, tokenizer, device, prompt, **DEFAULT_POEM_GEN_CONFIG):
            logger.info("{}\n", poem)

    user_input = input("Enter interactive mode? [Y/n]: ").strip().lower()
    if user_input != "n":
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
