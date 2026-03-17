#!/usr/bin/env python3
import unicodedata
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


MODEL_DIR = "./artifacts/checkpoints/scratch_init/final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.2
DO_SAMPLE = True
NUM_RETURN_SEQUENCES = 1


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)


def load_model_and_tokenizer():
    print(f"Loading model from: {MODEL_DIR} ({DEVICE})")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, vocab={len(tokenizer):,}")
    return model, tokenizer


def generate_text(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
    do_sample: bool = DO_SAMPLE,
    num_return_sequences: int = NUM_RETURN_SEQUENCES,
) -> list:
    prompt = normalize_text(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

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


def interactive_mode(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast):
    print("\nInteractive mode. Type 'quit' to exit, 'config' to change params.\n")

    config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
    }

    while True:
        try:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                break

            if prompt.lower() == "config":
                for key, value in config.items():
                    new_value = input(f"  {key} [{value}]: ").strip()
                    if new_value:
                        config[key] = type(value)(new_value)
                continue

            if not prompt:
                continue

            for text in generate_text(model, tokenizer, prompt, **config):
                print(f"\n{text}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def run_test_examples(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast):
    test_prompts = [
        "Việt Nam là một đất nước",
        "Hôm nay thời tiết rất đẹp,",
        "Trong lịch sử Việt Nam,",
        "Công nghệ trí tuệ nhân tạo",
        "Hà Nội là thủ đô của",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"[{i}/{len(test_prompts)}] {prompt}")
        for text in generate_text(model, tokenizer, prompt):
            print(text)
        print()


def main():
    model, tokenizer = load_model_and_tokenizer()
    run_test_examples(model, tokenizer)

    user_input = input("Enter interactive mode? [Y/n]: ").strip().lower()
    if user_input != "n":
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
