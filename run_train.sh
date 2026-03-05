#!/bin/bash
# ==============================================
# Vietnamese GPT-2 Training Script (Multi-GPU)
# ==============================================

# Select GPU 1
export CUDA_VISIBLE_DEVICES=1
NUM_GPUS=1

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
echo "Vietnamese GPT-2 Continual Pre-training"
echo "=========================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES (${NUM_GPUS} devices)"
echo ""

# Run training with DDP via torchrun
torchrun --nproc_per_node=$NUM_GPUS train.py 2>&1 | tee training_log.txt

echo ""
echo "Training completed!"
