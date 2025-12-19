#!/bin/bash

# Evaluation script for all model checkpoints
# Usage: ./scripts/eval_models.sh [dataset] [gpu_id]
# Example: ./scripts/eval_models.sh korean 0
# Example: ./scripts/eval_models.sh all 0

set -e

# Arguments
DATASET=${1:-"all"}
GPU_ID=${2:-0}

# Environment setup
source ~/anaconda3/etc/profile.d/conda.sh
conda activate allganize
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONUNBUFFERED=1

# Output directory
OUTPUT_BASE="results/eval_results"

echo "=============================================="
echo "Model Evaluation Script"
echo "=============================================="
echo "Dataset: $DATASET"
echo "GPU: $GPU_ID"
echo "Output: $OUTPUT_BASE"
echo "=============================================="

# Available models
MODELS="itransformer timexer wavenet timesnet waxer taxer"

# Function to evaluate a single dataset
evaluate_dataset() {
    local ds=$1
    local output_dir="$OUTPUT_BASE/$ds"

    echo ""
    echo "=============================================="
    echo "Evaluating dataset: $ds"
    echo "=============================================="

    mkdir -p "$output_dir"

    python evaluation/eval_all_checkpoints.py \
        --dataset "$ds" \
        --output_dir "$output_dir" \
        2>&1 | tee "$output_dir/eval.log"

    echo "Results saved to $output_dir"
}

# Run evaluation based on dataset argument
case $DATASET in
    all)
        for ds in sticker korean global; do
            evaluate_dataset "$ds"
        done
        ;;
    sticker|korean|global)
        evaluate_dataset "$DATASET"
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo "Available: sticker, korean, global, all"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Evaluation Complete"
echo "=============================================="
echo "Results saved to: $OUTPUT_BASE"

# Print summary of all results
if [ -f "$OUTPUT_BASE/sticker/summary.json" ] || [ -f "$OUTPUT_BASE/korean/summary.json" ] || [ -f "$OUTPUT_BASE/global/summary.json" ]; then
    echo ""
    echo "Summary files generated:"
    find "$OUTPUT_BASE" -name "summary.json" 2>/dev/null | while read f; do
        echo "  - $f"
    done
fi
