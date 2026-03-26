#!/usr/bin/env python3
"""Generate text from stage-1 base model."""

from loguru import logger

from src.config import (
    MODEL_DIR, MAX_NEW_TOKENS, TEMPERATURE,
    TOP_K, TOP_P, REPETITION_PENALTY,
)
from src.utils import generate_texts, load_gpt2

DEFAULT_GEN_CONFIG = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": TEMPERATURE,
    "top_k": TOP_K,
    "top_p": TOP_P,
    "repetition_penalty": REPETITION_PENALTY,
    "do_sample": True,
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


def interactive_mode(model, tokenizer, device) -> None:
    """Run interactive generation loop for stage-1 model."""
    logger.info("Interactive mode.")
    logger.info("Commands: 'config' to edit generation params, 'quit' to exit.")
    gen_config = dict(DEFAULT_GEN_CONFIG)

    while True:
        try:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ("quit", "exit", "q"):
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

            for text in generate_texts(model, tokenizer, device, prompt, **gen_config):
                logger.info("\n{}\n", text)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception("Generation error: {}", e)


def main():
    model, tokenizer, device = load_gpt2(MODEL_DIR, eval_mode=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        "Loaded model from {} ({}) — {:.1f}M params, vocab={}",
        MODEL_DIR, device, n_params, f"{len(tokenizer):,}",
    )

    test_prompts = [
        "Việt Nam là một đất nước",
        "Hôm nay thời tiết rất đẹp,",
        "Trong lịch sử Việt Nam,",
        "Công nghệ trí tuệ nhân tạo",
        "Hà Nội là thủ đô của",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        logger.info("[{}/{}] {}", i, len(test_prompts), prompt)
        for text in generate_texts(model, tokenizer, device, prompt, **DEFAULT_GEN_CONFIG):
            logger.info("{}", text)
        logger.info("")

    user_input = input("Enter interactive mode? [Y/n]: ").strip().lower()
    if user_input != "n":
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
