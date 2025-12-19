#!/bin/bash

# TimesNet Multi-GPU Training Script
# Usage: ./scripts/train_timesnet.sh [num_gpus]

NUM_GPUS=${1:-8}

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate allganize

# Training configuration
MODEL="timesnet"
DATASET="weather"
CITY="korea"
SEQ_LEN=512
PRED_LEN=16
INPUT_DIM=30
TOP_K=3
BATCH_SIZE=32
EPOCHS=10
LR=3e-4
D_MODEL=64
D_FF=64
E_LAYERS=2
TIME_INCEPTION=4

# DDP Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

echo "Starting ${MODEL} training with ${NUM_GPUS} GPUs..."

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29503 \
    main.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --city ${CITY} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --input_dim ${INPUT_DIM} \
    --top_k ${TOP_K} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --d_model ${D_MODEL} \
    --d_ff ${D_FF} \
    --e_layers ${E_LAYERS} \
    --time_inception ${TIME_INCEPTION} \
    --mode train \
    --val True \
    --num_workers 4

echo "Training completed for ${MODEL}"
