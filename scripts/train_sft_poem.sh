#!/bin/bash
# ==============================================
# Vietnamese GPT-2 SFT Poem Training Script
# ==============================================

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export WANDB_PROJECT="vietnamese-gpt2"
export WANDB_MODE="online"

python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

mkdir -p artifacts/logs

echo "=========================================="
echo "Vietnamese GPT-2 SFT Poem"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

python3 sft_poem.py 2>&1 | tee artifacts/logs/sft_poem_log.txt

echo ""
echo "Training completed!"
