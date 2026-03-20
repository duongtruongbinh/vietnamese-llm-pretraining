#!/usr/bin/env python3
"""Generate 5-word quatrains from SFT model."""

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from config import (
    POEM_MODEL_DIR, POEM_PREFIX,
    TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY,
)


def load_model():
    tokenizer = GPT2TokenizerFast.from_pretrained(POEM_MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(POEM_MODEL_DIR, torch_dtype=torch.bfloat16)
    model.tie_weights()
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return tokenizer, model, device


def generate(tokenizer, model, device, prompt="", num_samples=2):
    """Generate poem from prompt (can be empty or first line)."""
    text = POEM_PREFIX + prompt
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            num_return_sequences=num_samples,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        if text.startswith("thơ:"):
            text = text[4:].strip()
        results.append(text)
    return results


def main():
    tokenizer, model, device = load_model()

    prompts = [
        "",
        "Trăng sáng trên đầu núi",
        "Mùa thu lá vàng rơi",
        "Hoa sen nở trắng hồ",
    ]

    for prompt in prompts:
        print(f"\n[Prompt: {prompt or '(empty)'}]")
        poems = generate(tokenizer, model, device, prompt)
        for poem in poems:
            print(poem)
            print()

    # Interactive mode
    print("Interactive mode (type 'q' to quit)")
    while True:
        prompt = input("\nFirst line: ").strip()
        if prompt.lower() == "q":
            break

        poems = generate(tokenizer, model, device, prompt)
        for poem in poems:
            print(poem)
            print()


if __name__ == "__main__":
    main()
