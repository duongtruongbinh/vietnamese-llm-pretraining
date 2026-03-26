#!/bin/bash
# Vietnamese GPT-2 Training Script (2 GPUs, DDP)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export WANDB_PROJECT="vietnamese-gpt2"
export WANDB_MODE="online"

mkdir -p artifacts/logs

echo "=========================================="
echo "Vietnamese GPT-2 Pre-training (Random Init)"
echo "GPU: $CUDA_VISIBLE_DEVICES (${NUM_GPUS} GPUs)"
echo "=========================================="

uv run torchrun --nproc_per_node=$NUM_GPUS src/train_1.py 2>&1 | tee artifacts/logs/training_log.txt