#!/bin/bash
# ==============================================
# Vietnamese GPT-2 Training Script (Random Init, Single GPU 1)
# ==============================================

# Select GPU 1 only
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
echo "Vietnamese GPT-2 Pre-training (Random Init)"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES (${NUM_GPUS} device)"
echo ""

# Run training on single GPU (no torchrun needed for 1 GPU)
python3 train_rand_init.py 2>&1 | tee training_log_rand_init.txt

echo ""
echo "Training completed!"
