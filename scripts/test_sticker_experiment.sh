#!/bin/bash
#
# Sticker Dataset Experiment - Test All Models
#
# This script tests all 6 trained models and saves metrics to results/
#

set -e

DATASET="sticker"
INPUT_DIM=15
SEQ_LEN=512
PRED_LEN=16

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate allganize

cd /home/hsshin/waxer

echo "=============================================="
echo "Testing All Models on Sticker Dataset"
echo "=============================================="

# Function to find best checkpoint in a directory
find_best_ckpt() {
    local ckpt_dir=$1
    local model=$2
    # Find the latest checkpoint by epoch number
    ls ${ckpt_dir}/${model}_e*_s${SEQ_LEN}_p${PRED_LEN}.pth 2>/dev/null | \
        sed 's/.*_e\([0-9]*\)_.*/\1 &/' | \
        sort -rn | \
        head -1 | \
        awk '{print $2}' | \
        xargs -I {} basename {} .pth
}

# Test iTransformer
echo ""
echo "Testing iTransformer..."
CKPT_NAME=$(find_best_ckpt "checkpoints/itransformer_v1" "itransformer")
echo "Using checkpoint: ${CKPT_NAME}"
python main.py \
    --model itransformer \
    --mode test \
    --dataset ${DATASET} \
    --input_dim ${INPUT_DIM} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model 128 --d_ff 128 --e_layers 3 --n_heads 4 --top_k 2 \
    --ckpt_path checkpoints/itransformer_v1 \
    --ckpt_name ${CKPT_NAME}

# Test TimeXer
echo ""
echo "Testing TimeXer..."
CKPT_NAME=$(find_best_ckpt "checkpoints/timexer_v1" "timexer")
echo "Using checkpoint: ${CKPT_NAME}"
python main.py \
    --model timexer \
    --mode test \
    --dataset ${DATASET} \
    --input_dim ${INPUT_DIM} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model 90 --d_ff 128 --e_layers 3 --n_heads 4 \
    --ckpt_path checkpoints/timexer_v1 \
    --ckpt_name ${CKPT_NAME}

# Test WaXer
echo ""
echo "Testing WaXer..."
CKPT_NAME=$(find_best_ckpt "checkpoints/waxer_v1" "waxer")
echo "Using checkpoint: ${CKPT_NAME}"
python main.py \
    --model waxer \
    --mode test \
    --dataset ${DATASET} \
    --input_dim ${INPUT_DIM} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model 90 --d_ff 128 --e_layers 3 --n_heads 4 \
    --wavenet_d_model 64 --wavenet_layers 3 \
    --ckpt_path checkpoints/waxer_v1 \
    --ckpt_name ${CKPT_NAME}

# Test TaXer
echo ""
echo "Testing TaXer..."
CKPT_NAME=$(find_best_ckpt "checkpoints/taxer_v1" "taxer")
echo "Using checkpoint: ${CKPT_NAME}"
python main.py \
    --model taxer \
    --mode test \
    --dataset ${DATASET} \
    --input_dim ${INPUT_DIM} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model 90 --d_ff 128 --e_layers 3 --n_heads 4 \
    --times_d_model 64 --times_d_ff 64 --times_top_k 3 --times_num_kernels 4 --times_layers 2 \
    --ckpt_path checkpoints/taxer_v1 \
    --ckpt_name ${CKPT_NAME}

# Test TimesNet
echo ""
echo "Testing TimesNet..."
CKPT_NAME=$(find_best_ckpt "checkpoints/timesnet_v1" "timesnet")
echo "Using checkpoint: ${CKPT_NAME}"
python main.py \
    --model timesnet \
    --mode test \
    --dataset ${DATASET} \
    --input_dim ${INPUT_DIM} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model 128 --d_ff 128 --e_layers 3 --top_k 2 \
    --ckpt_path checkpoints/timesnet_v1 \
    --ckpt_name ${CKPT_NAME}

# Test WaveNet
echo ""
echo "Testing WaveNet..."
CKPT_NAME=$(find_best_ckpt "checkpoints/wavenet_v1" "wavenet")
echo "Using checkpoint: ${CKPT_NAME}"
python main.py \
    --model wavenet \
    --mode test \
    --dataset ${DATASET} \
    --input_dim ${INPUT_DIM} \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --d_model 128 --d_ff 128 --e_layers 3 \
    --ckpt_path checkpoints/wavenet_v1 \
    --ckpt_name ${CKPT_NAME}

echo ""
echo "=============================================="
echo "All tests completed!"
echo "=============================================="
echo ""
echo "Results saved to results/ directory"
echo "Run 'python scripts/compare_models.py' to generate comparison"
