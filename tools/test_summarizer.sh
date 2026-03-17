#!/usr/bin/env bash
# ====================================================================
# Video Summarization Evaluation Script (SumMe / TVSum)
# Usage: 
#   bash tools/test_summarizer.sh <DATASET> <CONFIG_FILE>
# Example: 
#   bash tools/test_summarizer.sh summe configs/summe/32_5.yaml
# ====================================================================

DATASET=$1
CONFIG=$2

if [ -z "$DATASET" ] || [ -z "$CONFIG" ]; then
    echo "🚨 Error: Missing arguments!"
    echo "💡 Usage: bash tools/test_summarizer.sh [summe|tvsum|all] [config_path]"
    exit 1
fi

echo "================================================="
echo "🚀 Starting Video Summarization Evaluation..."
echo "📦 Dataset : ${DATASET^^}"
echo "📄 Config  : ${CONFIG}"
echo "================================================="

# 评测 SumMe
if [ "$DATASET" == "summe" ]; then
    python test_summe.py --config ${CONFIG}

# 评测 TVSum
elif [ "$DATASET" == "tvsum" ]; then
    python test_tvsum.py --config ${CONFIG}

# 一键连跑两个数据集！
elif [ "$DATASET" == "all" ]; then
    echo "⏳ [1/2] Evaluating SumMe..."
    python test_summe.py --config ${CONFIG}
    echo "================================================="
    echo "⏳ [2/2] Evaluating TVSum..."
    python test_tvsum.py --config ${CONFIG}

else
    echo "🚨 Error: Invalid dataset! Please choose 'summe', 'tvsum', or 'all'."
fi

echo "🎉 All evaluation tasks finished!"