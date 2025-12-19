#!/bin/bash

# Master training script for sticker dataset
# Usage: ./scripts/train_sticker.sh <model_name> [num_gpus]
# Example: ./scripts/train_sticker.sh taxer 8

set -e

# Arguments
MODEL=${1:-"itransformer"}
NUM_GPUS=${2:-8}

# Validate model name
MODEL_LOWER=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')

# Environment setup
source ~/anaconda3/etc/profile.d/conda.sh
conda activate allganize
export OMP_NUM_THREADS=4

# Common parameters
SEQ_LEN=512
PRED_LEN=16
BATCH_SIZE=32
EPOCHS=35
LR=1e-4
SAVE_INTERVAL=16
VIZ_INTERVAL=32
DROPOUT=0.2

# Dataset parameters (sticker)
DATASET="sticker"
COUNTRY="canada"
STORE="all"
INPUT_DIM=15
TRAIN_RATIO=0.8

# Model-specific parameters
case $MODEL_LOWER in
    itransformer)
        D_MODEL=32
        D_FF=32
        E_LAYERS=1
        N_HEADS=1
        TOP_K=2
        MASTER_PORT=29510
        EXTRA_ARGS="--top_k $TOP_K"
        ;;
    timexer)
        D_MODEL=32
        D_FF=32
        E_LAYERS=1
        N_HEADS=1
        PATCH_LEN=16
        MASTER_PORT=29511
        EXTRA_ARGS="--patch_len $PATCH_LEN"
        ;;
    wavenet)
        D_MODEL=32
        D_FF=32
        E_LAYERS=3
        N_HEADS=1
        WAVE_KERNEL=3
        MASTER_PORT=29512
        EXTRA_ARGS="--wave_kernel_size $WAVE_KERNEL --find_unused_parameters True"
        ;;
    timesnet)
        D_MODEL=64
        D_FF=64
        E_LAYERS=2
        N_HEADS=1
        TOP_K=3
        TIME_INCEPTION=4
        MASTER_PORT=29513
        EXTRA_ARGS="--top_k $TOP_K --time_inception $TIME_INCEPTION --find_unused_parameters True"
        ;;
    waxer)
        D_MODEL=32
        D_FF=32
        E_LAYERS=1
        N_HEADS=1
        PATCH_LEN=16
        WAVENET_D_MODEL=64
        WAVENET_LAYERS=3
        MASTER_PORT=29514
        EXTRA_ARGS="--patch_len $PATCH_LEN --wavenet_d_model $WAVENET_D_MODEL --wavenet_layers $WAVENET_LAYERS --find_unused_parameters True"
        ;;
    taxer)
        D_MODEL=32
        D_FF=32
        E_LAYERS=1
        N_HEADS=1
        PATCH_LEN=16
        TIMES_D_MODEL=64
        TIMES_D_FF=64
        TIMES_TOP_K=3
        TIMES_NUM_KERNELS=4
        TIMES_LAYERS=2
        MASTER_PORT=29515
        EXTRA_ARGS="--patch_len $PATCH_LEN --times_d_model $TIMES_D_MODEL --times_d_ff $TIMES_D_FF --times_top_k $TIMES_TOP_K --times_num_kernels $TIMES_NUM_KERNELS --times_layers $TIMES_LAYERS --find_unused_parameters True"
        ;;
    *)
        echo "Error: Unknown model '$MODEL'"
        echo "Available models: itransformer, timexer, wavenet, timesnet, waxer, taxer"
        exit 1
        ;;
esac

# Find next available version number
find_next_version() {
    local base_path=$1
    local version=1
    while [ -d "${base_path}_v${version}" ]; do
        version=$((version + 1))
    done
    echo $version
}

# Create checkpoint directory with auto-version
DATASET_NAME="sticker"
CKPT_BASE="checkpoints/${DATASET_NAME}/${MODEL_LOWER}"
VERSION=$(find_next_version $CKPT_BASE)
CKPT_PATH="${CKPT_BASE}_v${VERSION}"
mkdir -p $CKPT_PATH

# Save training arguments to checkpoint directory as JSON
cat > $CKPT_PATH/args.json << EOF
{
    "date": "$(date -Iseconds)",
    "model": "$MODEL_LOWER",
    "d_model": $D_MODEL,
    "d_ff": $D_FF,
    "e_layers": $E_LAYERS,
    "n_heads": $N_HEADS,
    "dropout": $DROPOUT,
    "dataset": "$DATASET",
    "country": "$COUNTRY",
    "store": "$STORE",
    "input_dim": $INPUT_DIM,
    "train_ratio": $TRAIN_RATIO,
    "seq_len": $SEQ_LEN,
    "pred_len": $PRED_LEN,
    "batch_size": $BATCH_SIZE,
    "epochs": $EPOCHS,
    "lr": "$LR",
    "save_interval": $SAVE_INTERVAL,
    "viz_interval": $VIZ_INTERVAL,
    "num_gpus": $NUM_GPUS,
    "extra_args": "$EXTRA_ARGS"
}
EOF

echo "=============================================="
echo "Training $MODEL on Sticker Dataset"
echo "=============================================="
echo "GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE (per GPU)"
echo "Epochs: $EPOCHS"
echo "Sequence length: $SEQ_LEN"
echo "Prediction length: $PRED_LEN"
echo "Country: $COUNTRY"
echo "Store: $STORE"
echo "Checkpoint path: $CKPT_PATH"
echo "=============================================="

# Run training with torchrun (log saved to checkpoint directory)
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    main.py \
    --mode train \
    --model $MODEL_LOWER \
    --dataset $DATASET \
    --country $COUNTRY \
    --store $STORE \
    --input_dim $INPUT_DIM \
    --train_ratio $TRAIN_RATIO \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --dropout $DROPOUT \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --e_layers $E_LAYERS \
    --n_heads $N_HEADS \
    --save_interval $SAVE_INTERVAL \
    --viz_interval $VIZ_INTERVAL \
    --ckpt_path $CKPT_PATH \
    $EXTRA_ARGS \
    2>&1 | tee $CKPT_PATH/train.log

echo "Training complete. Checkpoint saved to $CKPT_PATH"
