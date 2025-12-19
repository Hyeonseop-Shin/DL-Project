#!/bin/bash

# iTransformer Multi-GPU Training Script
# Usage: ./scripts/train_itransformer.sh [num_gpus]

NUM_GPUS=${1:-8}  # Default to 8 GPUs

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate allganize

# Training configuration
MODEL="itransformer"
DATASET="weather"
CITY="korea"
SEQ_LEN=512
PRED_LEN=16
BATCH_SIZE=32
EPOCHS=10
LR=3e-4
D_MODEL=32
D_FF=32
E_LAYERS=1
N_HEADS=1

# DDP Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

echo "Starting ${MODEL} training with ${NUM_GPUS} GPUs..."

# Run with torchrun
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    main.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --city ${CITY} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --d_model ${D_MODEL} \
    --d_ff ${D_FF} \
    --e_layers ${E_LAYERS} \
    --n_heads ${N_HEADS} \
    --mode train \
    --val True \
    --num_workers 4

echo "Training completed for ${MODEL}"
