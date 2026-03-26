#!/bin/bash
# Vietnamese GPT-2 Continued Pretraining Script (Stage 2)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export WANDB_PROJECT="vietnamese-gpt2"
export WANDB_MODE="online"

mkdir -p artifacts/logs

echo "=========================================="
echo "Vietnamese GPT-2 Continued Pretraining (Poem Corpus)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

uv run python src/train_2.py 2>&1 | tee artifacts/logs/pretrain_stage2_log.txt

echo "Training completed!"
