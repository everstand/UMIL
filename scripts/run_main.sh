#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: bash scripts/run_main.sh <dataset> <eval_mode>"
    exit 1
fi

DATASET=$1
EVAL_MODE=$2

CONFIG_FILE="configs/${DATASET}/32_5.yaml"
SPLITS_FILE="splits/${DATASET}.yml"
OUTPUT_DIR="outputs/${DATASET}"

if [ ! -f "$CONFIG_FILE" ] || [ ! -f "$SPLITS_FILE" ]; then
    echo "Error: Missing $CONFIG_FILE or $SPLITS_FILE"
    exit 1
fi

LOG_FILE="${OUTPUT_DIR}/main_results_5fold_${EVAL_MODE}.txt"
> "$LOG_FILE"

echo "Starting 5-Fold Pipeline | Dataset: $DATASET | Mode: $EVAL_MODE"

for SPLIT_ID in 0 1 2 3 4; do
    echo ">> Processing Split $SPLIT_ID ..."
    
    # 1. Train
    python scripts/train.py \
        --config "$CONFIG_FILE" \
        --dataset "$DATASET" \
        --split_file "$SPLITS_FILE" \
        --split_id "$SPLIT_ID" \
        --output "$OUTPUT_DIR"

    CKPT_PATH="${OUTPUT_DIR}/best_model_split${SPLIT_ID}.pth"
    TMP_EVAL_FILE="${OUTPUT_DIR}/temp_eval_split${SPLIT_ID}.log"
    
    # 2. Evaluate
    python scripts/evaluate.py \
        --config "$CONFIG_FILE" \
        --dataset "$DATASET" \
        --test_keys "$SPLITS_FILE" \
        --ckpt "$CKPT_PATH" \
        --opts EVAL.ABLATION_MODE "$EVAL_MODE" 2>&1 | tee "$TMP_EVAL_FILE"
        
    # 3. Parse
    F1=$(grep "Final F-score" "$TMP_EVAL_FILE" | sed -E 's/.*Final F-score: ([0-9.]+).*/\1/' || true)
    DIV=$(grep "Diversity" "$TMP_EVAL_FILE" | sed -E 's/.*Diversity: ([0-9.]+).*/\1/' || true)
    
    if [ -z "$F1" ] || [ -z "$DIV" ]; then
        echo "Error: Failed to parse F1 or Diversity for Split $SPLIT_ID"
        exit 1
    fi
    
    echo "$SPLIT_ID $F1 $DIV" >> "$LOG_FILE"
    rm -f "$TMP_EVAL_FILE"
done

# 4. Statistical Summary
python -c "
import sys
import numpy as np

try:
    with open(sys.argv[1], 'r') as f:
        data = [line.strip().split() for line in f if len(line.strip().split()) == 3]
        f1s = [float(p[1]) for p in data]
        divs = [float(p[2]) for p in data]
except Exception as e:
    print(f'Error reading logs: {e}')
    sys.exit(1)

if len(f1s) != 5:
    print(f'Error: Expected 5 splits, got {len(f1s)}. Check for interruptions.')
    sys.exit(1)

print(f'\n[ {sys.argv[2].upper()} - 5-Fold Result ({sys.argv[3]}) ]')
print(f'F-score   : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
print(f'Diversity : {np.mean(divs):.4f} ± {np.std(divs):.4f}')
" "$LOG_FILE" "$DATASET" "$EVAL_MODE" | tee -a "$LOG_FILE"