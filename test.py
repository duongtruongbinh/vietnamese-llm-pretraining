#!/usr/bin/env python3
"""
Text Generation Test Script for Vietnamese GPT-2
=================================================
Test the trained model by generating text from Vietnamese prompts.
"""

import unicodedata
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ========================== Configuration ==========================
MODEL_DIR = "./vietnamese_gpt2_rand_init_mixed/final"  # Path to trained model
# MODEL_DIR = "./vietnamese_gpt2_pre_en_mixed_data/checkpoint-3000"  # Path to trained model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generation parameters
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7      # Lower = more focused, Higher = more creative
TOP_K = 50              # Top-K sampling: consider top 50 tokens
TOP_P = 0.95            # Nucleus sampling: cumulative probability
REPETITION_PENALTY = 1.2  # Penalize repetition
DO_SAMPLE = True        # Use sampling instead of greedy
NUM_RETURN_SEQUENCES = 1

# Consider adjusting these if output quality is poor:
# - Lower TEMPERATURE (0.6-0.7) for more coherent but less creative text
# - Higher REPETITION_PENALTY (1.3-1.5) if text repeats too much
# - Lower TOP_P (0.85-0.9) for more focused generations


def normalize_text(text: str) -> str:
    """
    Normalize text to Unicode NFC form.
    """
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)


def load_model_and_tokenizer():
    """
    Load the trained model and tokenizer.
    """
    print(f"Loading model from: {MODEL_DIR}")
    print(f"Device: {DEVICE}")
    
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Vocab size: {len(tokenizer):,}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test tokenizer with Vietnamese text
    test_text = "Việt Nam là một đất nước"
    test_tokens = tokenizer.tokenize(test_text)
    print(f"\nTokenizer test: '{test_text}'")
    print(f"  → {len(test_tokens)} tokens: {test_tokens[:8]}...")
    
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
    """
    Generate text from a prompt.
    
    Args:
        model: Trained GPT2LMHeadModel
        tokenizer: GPT2TokenizerFast
        prompt: Vietnamese prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling
        top_p: Top-p (nucleus) sampling
        repetition_penalty: Penalty for word repetition
        do_sample: True for sampling, False for greedy
        num_return_sequences: Number of sequences to generate
        
    Returns:
        List of generated texts
    """
    # Normalize to NFC
    prompt = normalize_text(prompt)
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Generate
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
    
    # Decode
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts


def interactive_mode(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast):
    """
    Interactive mode for continuous testing.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter a Vietnamese prompt to generate text.")
    print("Type 'quit' or 'exit' to exit.")
    print("Type 'config' to change generation parameters.")
    print("=" * 60 + "\n")
    
    # Default config
    config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
    }
    
    while True:
        try:
            prompt = input("\n📝 Prompt: ").strip()
            
            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if prompt.lower() == "config":
                print("\nCurrent configuration:")
                for key, value in config.items():
                    new_value = input(f"  {key} [{value}]: ").strip()
                    if new_value:
                        config[key] = type(value)(new_value)
                print("Configuration updated!")
                continue
            
            if not prompt:
                print("Please enter a prompt.")
                continue
            
            print("\n🤖 Generating...")
            generated = generate_text(
                model, tokenizer, prompt, **config
            )
            
            print("\n" + "-" * 40)
            print("📄 Generated text:")
            print("-" * 40)
            for i, text in enumerate(generated):
                if len(generated) > 1:
                    print(f"\n[{i+1}]")
                print(text)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_test_examples(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast):
    """
    Run test examples.
    """
    test_prompts = [
        "Việt Nam là một đất nước",
        "Hôm nay thời tiết rất đẹp,",
        "Trong lịch sử Việt Nam,",
        "Công nghệ trí tuệ nhân tạo",
        "Hà Nội là thủ đô của",
    ]
    
    print("\n" + "=" * 60)
    print("TEST EXAMPLES")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Prompt: \"{prompt}\"")
        print("-" * 40)
        
        generated = generate_text(model, tokenizer, prompt)
        for text in generated:
            print(text)
        
        print("-" * 40)


def main():
    """
    Main function.
    """
    print("=" * 60)
    print("VIETNAMESE GPT-2 TEXT GENERATION")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Run test examples
    run_test_examples(model, tokenizer)
    
    # Enter interactive mode
    print("\n" + "=" * 60)
    user_input = input("Enter interactive mode? [Y/n]: ").strip().lower()
    if user_input != "n":
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
