#!/bin/bash
# ==============================================
# Vietnamese GPT-2 Training Script (2 GPUs, DDP)
# ==============================================

export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

# Set environment variables to prevent multiprocessing issues
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Wandb configuration
export WANDB_PROJECT="vietnamese-gpt2"
export WANDB_MODE="online"  # change to "offline" if no internet

# Clear GPU cache before starting
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "=========================================="
echo "Vietnamese GPT-2 Pre-training (Random Init)"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES (${NUM_GPUS} GPUs)"
echo ""

torchrun --nproc_per_node=$NUM_GPUS train.py 2>&1 | tee artifacts/logs/training_log.txt

echo ""
echo "Training completed!"
