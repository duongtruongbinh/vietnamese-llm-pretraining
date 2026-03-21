#!/usr/bin/env python3
"""Centralized configuration for the Vietnamese GPT-2 pretraining project."""

# ── Paths ────────────────────────────────────────────────────────────────────
TOKENIZER_DIR = "./artifacts/tokenizer"
CHECKPOINT_DIR = "./artifacts/checkpoints/scratch_init"
MODEL_DIR = "./artifacts/checkpoints/scratch_init/final"

# ── Dataset ──────────────────────────────────────────────────────────────────
# weight: how many times to duplicate a dataset during pretraining (default 1).
DATASET_CONFIGS = [
    {"path": "data/train/bkai_train.parquet", "text_col": "text"},
    {"path": "data/train/vi_wiki_articles_clean.parquet", "text_col": "text", "weight": 3},
]

# ── Tokenizer training ──────────────────────────────────────────────────────
VOCAB_SIZE = 50257
MIN_FREQUENCY = 2
SPECIAL_TOKEN = "<|endoftext|>"

# ── Model ────────────────────────────────────────────────────────────────────
BASE_MODEL = "gpt2"
MAX_LENGTH = 1024

# ── Training hyperparameters ─────────────────────────────────────────────────
TOKEN_BUDGET = 2_480_000_000
EVAL_SPLIT_RATIO = 0.01
PREPROCESSING_NUM_WORKERS = 4
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 64
WARMUP_RATIO = 0.1
BF16 = True
GRADIENT_CHECKPOINTING = True
DATALOADER_NUM_WORKERS = 0
WANDB_RUN_NAME = "gpt2-small-vietnamese-rand-init"

# ── Inference defaults ───────────────────────────────────────────────────────
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.2
DO_SAMPLE = True
NUM_RETURN_SEQUENCES = 1

# ── Poem SFT ─────────────────────────────────────────────────────────────────
POEM_DATA_PATH = "data/sft/poem_stanzas.jsonl"
POEM_RAW_CSV = "data/raws/poem_dataset.csv"
POEM_CHECKPOINT_DIR = "./artifacts/checkpoints/sft_poem"
POEM_MODEL_DIR = "./artifacts/checkpoints/sft_poem/final"
POEM_PREFIX = "thơ:\n"
POEM_LINES_PER_STANZA = 4
POEM_WORDS_PER_LINE = 5
POEM_EPOCHS = 30
POEM_BATCH_SIZE = 32
POEM_LEARNING_RATE = 5e-5
POEM_WEIGHT_DECAY = 0.1
POEM_MAX_LENGTH = 64
