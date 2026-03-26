# Vietnamese GPT-2: Multi-Stage Pretraining

A clean and reproducible GPT-2 pretraining pipeline for Vietnamese, trained in two stages:

- **Stage 1**: Pretrain GPT-2 from random initialization on mixed Vietnamese corpora.
- **Stage 2**: Continue pretraining on a curated **5-word quatrain poem** corpus for style adaptation.

This repository includes data preparation, tokenizer training, multi-stage pretraining, and text generation scripts.

---

## Repository Structure

```text
vietnamese-gpt2/
├── src/
│   ├── config.py               # Central configuration: paths, datasets, hyperparameters
│   ├── utils.py                # Shared helpers for normalization, callbacks, generation
│   ├── train_tokenizer.py      # Train tokenizer from Vietnamese corpora
│   ├── train_1.py              # Stage 1 pretraining from scratch
│   ├── train_2.py              # Stage 2 continued pretraining on poem corpus
│   ├── generate_base.py        # Generate text with the stage-1 model
│   └── generate_poem.py        # Generate poem-style text with the stage-2 model
├── data_prep/
│   ├── news/download_datasets.py
│   ├── wiki/crawl_vi_wiki.py
│   ├── wiki/process_vi_wiki.py
│   ├── poem/crawl_poem.py
│   ├── poem/scrape_poem_content.py
│   ├── poem/prepare_poem_data.py
│   └── deduplicate.py
├── scripts/
│   ├── train_1.sh
│   └── train_2.sh
├── artifacts/                  # Tokenizer, checkpoints, logs, final models
└── data/                       # Raw and processed datasets
````


## Training Overview

### Stage 1: Base Language Pretraining

Train GPT-2 from random initialization on mixed Vietnamese corpora such as news and Wikipedia.

### Stage 2: Domain Adaptation for Poetry

Continue pretraining the stage-1 model on a Vietnamese poem corpus to adapt the model toward 5-word quatrain generation.

## Requirements

* Python **3.11+**
* CUDA-compatible GPU
* [uv](https://github.com/astral-sh/uv) for environment and package management

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/duongtruongbinh/vietnamese-gpt2
uv sync
uv pip install -e .
```

Run all commands from the repository root.


## Pipeline

### 1. Prepare raw corpora

```bash
uv run python data_prep/news/download_datasets.py
uv run python data_prep/wiki/crawl_vi_wiki.py
uv run python data_prep/wiki/process_vi_wiki.py
```

### 2. Train the tokenizer

```bash
uv run python src/train_tokenizer.py
```

### 3. Deduplicate the pretraining data

```bash
uv run python data_prep/deduplicate.py
```

### 4. Run stage 1 pretraining

```bash
bash scripts/train_1.sh
```

### 5. Prepare the poem corpus

```bash
uv run python data_prep/poem/crawl_poem.py
uv run python data_prep/poem/scrape_poem_content.py
uv run python data_prep/poem/prepare_poem_data.py
```

### 6. Run stage 2 continued pretraining

```bash
bash scripts/train_2.sh
```

## Text Generation

Generate text with the base model:

```bash
uv run python src/generate_base.py
```

Generate poem-style text with the stage-2 model:

```bash
uv run python src/generate_poem.py
```


## Configuration

All important paths and hyperparameters are managed in:

```text
src/config.py
```

This includes:

* Dataset paths
* Tokenizer directory
* Checkpoint directory
* Sequence length
* Batch size
* Learning rate
* Training budget
* Logging and runtime settings


## Outputs

Training artifacts are stored under:

```text
artifacts/
```

Typical outputs include:

* Trained tokenizer
* Intermediate checkpoints
* Final stage-1 model
* Final stage-2 model
* Training logs

---

## Notes

* Stage 1 is intended for **general Vietnamese language modeling**.
* Stage 2 is intended for **style adaptation**, not full instruction tuning.
* For best results, ensure corpus quality and deduplication are completed before training.
* A GPU is strongly recommended for both tokenizer experimentation and model training.

---

## Project Goal

This project aims to provide a simple, practical, and extensible foundation for training Vietnamese GPT-2 models from scratch and adapting them to specific text styles such as poetry