#!/bin/bash

# WaveNet Multi-GPU Training Script
# Usage: ./scripts/train_wavenet.sh [num_gpus]

NUM_GPUS=${1:-8}

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate allganize

# Training configuration
MODEL="wavenet"
DATASET="weather"
CITY="korea"
SEQ_LEN=512
PRED_LEN=16
INPUT_DIM=30
BATCH_SIZE=32
EPOCHS=10
LR=3e-4
D_MODEL=32
E_LAYERS=3
WAVE_KERNEL_SIZE=3

# DDP Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

echo "Starting ${MODEL} training with ${NUM_GPUS} GPUs..."

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29502 \
    main.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --city ${CITY} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --input_dim ${INPUT_DIM} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --d_model ${D_MODEL} \
    --e_layers ${E_LAYERS} \
    --wave_kernel_size ${WAVE_KERNEL_SIZE} \
    --mode train \
    --val True \
    --num_workers 4 \
    --find_unused_parameters True

echo "Training completed for ${MODEL}"
