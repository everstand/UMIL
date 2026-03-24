#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: bash scripts/run_ablation.sh <dataset> <split_id>"
    exit 1
fi

DATASET=$1
SPLIT_ID=$2

CONFIG_FILE="configs/${DATASET}/32_5.yaml"
TEST_KEYS="splits/${DATASET}.yml"
CKPT_PATH="outputs/${DATASET}/best_model_split${SPLIT_ID}.pth"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found at $CKPT_PATH"
    exit 1
fi

declare -a ABLATIONS=(
    "E0 raw 0.5"
    "E1 raw 0.5"
    "E2 raw 0.5"
)

for ablation in "${ABLATIONS[@]}"; do
    read -r MODE SPACE ALPHA <<< "$ablation"
    echo "Running: Mode=${MODE}, Space=${SPACE}, Alpha=${ALPHA}"
    python scripts/evaluate.py \
        --config "$CONFIG_FILE" \
        --dataset "$DATASET" \
        --test_keys "$TEST_KEYS" \
        --ckpt "$CKPT_PATH" \
        --opts EVAL.ABLATION_MODE "$MODE" EVAL.REP_SPACE "$SPACE" EVAL.ALPHA "$ALPHA"
done