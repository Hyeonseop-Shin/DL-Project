#!/bin/bash

# Single-GPU Training Script (Fallback Mode)
# Usage: ./scripts/train_single_gpu.sh <model_name> [gpu_id]
#
# Examples:
#   ./scripts/train_single_gpu.sh itransformer 0
#   ./scripts/train_single_gpu.sh timexer 1
#   ./scripts/train_single_gpu.sh wavenet 2
#   ./scripts/train_single_gpu.sh timesnet 3
#   ./scripts/train_single_gpu.sh waxer 4
#   ./scripts/train_single_gpu.sh taxer 5

MODEL=${1:-"itransformer"}
GPU_ID=${2:-0}

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate allganize

# Training configuration
DATASET="weather"
CITY="korea"
SEQ_LEN=512
PRED_LEN=16
BATCH_SIZE=32
EPOCHS=10
LR=3e-4

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "Starting ${MODEL} single-GPU training on GPU ${GPU_ID}..."

# Standard python execution (no torchrun)
python main.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --city ${CITY} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --gpu_num 0 \
    --mode train \
    --val True \
    --num_workers 4

echo "Single-GPU training completed for ${MODEL}"
