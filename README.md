# Vietnamese GPT-2 Pre-training

Pre-train GPT-2 from scratch on Vietnamese text data (BKAI News + Wikipedia).

## Requirements

- Python >= 3.11
- CUDA GPU

## Setup

```bash
uv sync
# or: pip install -r requirements.txt
```

## Project Structure

```
config.py              # All hyperparameters and paths
train.py               # Pre-training script (DDP)
inference.py           # Text generation
sft_poem.py            # SFT for poem generation
generate_poem.py       # Generate poems
tokenizer.py           # Train BPE tokenizer
data/
  download_datasets.py # Download BKAI corpus
  crawl_vi_wiki.py     # Crawl Vietnamese Wikipedia
  process_vi_wiki.py   # Clean wikitext to plaintext
  crawl_poem.py        # Crawl poem metadata
  scrape_poem_content.py # Scrape poem content
  prepare_poem_data.py # Extract valid stanzas
scripts/
  train.sh             # Multi-GPU pre-training
  train_sft_poem.sh    # SFT poem training
artifacts/
  tokenizer/           # Trained tokenizer
  checkpoints/         # Model checkpoints
```

## Pre-training Pipeline

### 1. Prepare Data

```bash
python data/download_datasets.py
python data/crawl_vi_wiki.py 
python data/process_vi_wiki.py
```

### 2. Train Tokenizer

```bash
python tokenizer.py
```

### 3. Train Model

```bash
# Single GPU
python train.py

# Multi-GPU
bash scripts/train.sh
```

### 4. Inference

```bash
python inference.py
```

## Poem SFT Pipeline

### 1. Prepare Poem Data

```bash
python data/crawl_poem.py
python data/scrape_poem_content.py
python data/prepare_poem_data.py
```

### 2. Train SFT

```bash
bash scripts/train_sft_poem.sh
```

### 3. Generate Poems

```bash
python generate_poem.py
```

## Configuration

All settings in `config.py`. Training auto-resumes from latest checkpoint.
