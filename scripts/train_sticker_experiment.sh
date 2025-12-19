#!/bin/bash
#
# Sticker Dataset Experiment - Train and Evaluate All Models
#
# This script trains all 6 models on the sticker dataset:
# - Round 1: iTransformer + TimeXer (parallel, 4 GPUs each)
# - Round 2: WaXer + TaXer (parallel, 4 GPUs each)
# - Round 3: TimesNet + WaveNet (parallel, 4 GPUs each)
#
# After training, it tests all models and generates comparison results.
#

set -e

# Configuration
EPOCHS=128
DATASET="sticker"
INPUT_DIM=15
SEQ_LEN=512
PRED_LEN=16
NUM_WORKERS=0

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate allganize

cd /home/hsshin/waxer

echo "=============================================="
echo "Sticker Dataset Experiment"
echo "=============================================="
echo "Training 6 models for ${EPOCHS} epochs"
echo "Dataset: ${DATASET}"
echo "Input dim: ${INPUT_DIM}"
echo "=============================================="

#######################
# ROUND 1: iTransformer + TimeXer
#######################
echo ""
echo "======== ROUND 1: iTransformer + TimeXer ========"
echo ""

# iTransformer on GPUs 0,1,2,3
echo "Starting iTransformer training..."
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 main.py \
    --model itransformer \
    --mode train --val True \
    --dataset ${DATASET} \
    --input_dim ${INPUT_DIM} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model 128 --d_ff 128 --e_layers 3 --n_heads 4 --top_k 2 \
    --epochs ${EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --ckpt_path checkpoints/itransformer_v1 \
    > logs/itransformer_train.log 2>&1 &
PID_ITRANS=$!

# TimeXer on GPUs 4,5,6,7
echo "Starting TimeXer training..."
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29502 main.py \
    --model timexer \
    --mode train --val True \
    --dataset ${DATASET} \
    --input_dim ${INPUT_DIM} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model 90 --d_ff 128 --e_layers 3 --n_heads 4 \
    --epochs ${EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --ckpt_path checkpoints/timexer_v1 \
    > logs/timexer_train.log 2>&1 &
PID_TIMEXER=$!

echo "Waiting for Round 1 to complete..."
wait $PID_ITRANS
echo "iTransformer training completed!"
wait $PID_TIMEXER
echo "TimeXer training completed!"

#######################
# ROUND 2: WaXer + TaXer
#######################
echo ""
echo "======== ROUND 2: WaXer + TaXer ========"
echo ""

# WaXer on GPUs 0,1,2,3
echo "Starting WaXer training..."
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 main.py \
    --model waxer \
    --mode train --val True \
    --dataset ${DATASET} \
    --input_dim ${INPUT_DIM} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model 90 --d_ff 128 --e_layers 3 --n_heads 4 \
    --wavenet_d_model 64 --wavenet_layers 3 \
    --find_unused_parameters True \
    --epochs ${EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --ckpt_path checkpoints/waxer_v1 \
    > logs/waxer_train.log 2>&1 &
PID_WAXER=$!

# TaXer on GPUs 4,5,6,7
echo "Starting TaXer training..."
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29502 main.py \
    --model taxer \
    --mode train --val True \
    --dataset ${DATASET} \
    --input_dim ${INPUT_DIM} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model 90 --d_ff 128 --e_layers 3 --n_heads 4 \
    --times_d_model 64 --times_d_ff 64 --times_top_k 3 --times_num_kernels 4 --times_layers 2 \
    --epochs ${EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --ckpt_path checkpoints/taxer_v1 \
    > logs/taxer_train.log 2>&1 &
PID_TAXER=$!

echo "Waiting for Round 2 to complete..."
wait $PID_WAXER
echo "WaXer training completed!"
wait $PID_TAXER
echo "TaXer training completed!"

#######################
# ROUND 3: TimesNet + WaveNet
#######################
echo ""
echo "======== ROUND 3: TimesNet + WaveNet ========"
echo ""

# TimesNet on GPUs 0,1,2,3
echo "Starting TimesNet training..."
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 main.py \
    --model timesnet \
    --mode train --val True \
    --dataset ${DATASET} \
    --input_dim ${INPUT_DIM} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model 128 --d_ff 128 --e_layers 3 --top_k 2 \
    --epochs ${EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --ckpt_path checkpoints/timesnet_v1 \
    > logs/timesnet_train.log 2>&1 &
PID_TIMESNET=$!

# WaveNet on GPUs 4,5,6,7
echo "Starting WaveNet training..."
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29502 main.py \
    --model wavenet \
    --mode train --val True \
    --dataset ${DATASET} \
    --input_dim ${INPUT_DIM} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model 128 --d_ff 128 --e_layers 3 \
    --find_unused_parameters True \
    --epochs ${EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --ckpt_path checkpoints/wavenet_v1 \
    > logs/wavenet_train.log 2>&1 &
PID_WAVENET=$!

echo "Waiting for Round 3 to complete..."
wait $PID_TIMESNET
echo "TimesNet training completed!"
wait $PID_WAVENET
echo "WaveNet training completed!"

echo ""
echo "=============================================="
echo "All training completed!"
echo "=============================================="
echo ""
echo "Training logs saved to logs/ directory"
echo "Checkpoints saved to checkpoints/<model>_v1/ directories"
echo ""
echo "To test models, run:"
echo "  ./scripts/test_sticker_experiment.sh"
