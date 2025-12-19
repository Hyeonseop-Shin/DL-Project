#!/bin/bash
#
# High-Throughput Training Script with Gradient Accumulation
#
# This script increases effective batch size through gradient accumulation,
# achieving similar throughput benefits without running multiple processes per GPU.
#
# IMPORTANT: NCCL backend does NOT support multiple processes per GPU!
# This is a fundamental limitation of NCCL's direct GPU-to-GPU communication.
#
# Instead, we use gradient accumulation:
#   - accum_iter=4: Accumulate gradients over 4 batches before updating
#   - Effective batch size = batch_size * accum_iter * num_gpus
#   - With batch_size=32, accum_iter=4, 8 GPUs: effective batch = 32*4*8 = 1024
#
# Usage:
#   ./scripts/train_multi_process.sh [num_gpus] [accum_iter] [model]
#
# Examples:
#   # 8 GPUs, 4x gradient accumulation (effective batch = 32*4*8 = 1024)
#   ./scripts/train_multi_process.sh 8 4
#
#   # 8 GPUs, 2x gradient accumulation, timexer model
#   ./scripts/train_multi_process.sh 8 2 timexer
#

set -e

# Configuration
NUM_GPUS=${1:-8}
ACCUM_ITER=${2:-4}
MODEL=${3:-waxer}

# Dataset configuration
DATASET="weather"
CITY="tokyo"
SEQ_LEN=512
PRED_LEN=16
PATCH_LEN=16
INPUT_DIM=6

# Model configuration
D_MODEL=64
D_FF=128
E_LAYERS=2
N_HEADS=4

# Training configuration
BATCH_SIZE=32
EPOCHS=10
LR=1e-4

# Calculate effective batch size
EFFECTIVE_BATCH=$((BATCH_SIZE * ACCUM_ITER * NUM_GPUS))

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate allganize

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

echo "=============================================="
echo "High-Throughput Training with Gradient Accumulation"
echo "=============================================="
echo "Model:             ${MODEL}"
echo "Number of GPUs:    ${NUM_GPUS}"
echo "Batch size/GPU:    ${BATCH_SIZE}"
echo "Accumulation iter: ${ACCUM_ITER}"
echo "Effective batch:   ${EFFECTIVE_BATCH}"
echo "=============================================="
echo ""
echo "Note: Using gradient accumulation instead of multiple"
echo "      processes per GPU (NCCL doesn't support the latter)"
echo ""

# Build command based on model
COMMON_ARGS="
    --model ${MODEL}
    --dataset ${DATASET}
    --city ${CITY}
    --input_dim ${INPUT_DIM}
    --seq_len ${SEQ_LEN}
    --pred_len ${PRED_LEN}
    --patch_len ${PATCH_LEN}
    --d_model ${D_MODEL}
    --d_ff ${D_FF}
    --e_layers ${E_LAYERS}
    --n_heads ${N_HEADS}
    --batch_size ${BATCH_SIZE}
    --accum_iter ${ACCUM_ITER}
    --epochs ${EPOCHS}
    --lr ${LR}
    --mode train
    --val True
    --num_workers 4
"

# Model-specific arguments
if [ "${MODEL}" = "waxer" ] || [ "${MODEL}" = "wavenet" ]; then
    MODEL_ARGS="
        --wavenet_d_model 64
        --wavenet_layers 3
        --find_unused_parameters True
    "
elif [ "${MODEL}" = "taxer" ]; then
    MODEL_ARGS="
        --times_d_model 64
        --times_d_ff 64
        --times_top_k 3
        --times_num_kernels 4
        --times_layers 2
    "
else
    MODEL_ARGS=""
fi

# Run training
echo "Starting training..."
echo ""

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    main.py \
    ${COMMON_ARGS} \
    ${MODEL_ARGS}

echo ""
echo "Training completed!"
