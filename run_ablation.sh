#!/bin/bash

# 遇到错误立即终止
set -e

# ==========================================
# 实验全局参数配置 (请根据你的物理路径修改此处)
# ==========================================
CONFIG_FILE="configs/tvsum/32_5.yaml"  # 🌟 替换为真实存在的 YAML
DATASET="tvsum"
CKPT_PATH="outputs/best_model_split0.pth"   # 替换为真实的权重路径
TEST_KEYS="splits/tvsum.yml"   # 替换为真实的测试集划分文件

echo "================================================================="
echo "🚀 开始执行视频摘要代表性先验的严格消融实验"
echo "Dataset: $DATASET | Checkpoint: $CKPT_PATH"
echo "================================================================="

# 定义消融矩阵：格式为 "ABLATION_MODE REP_SPACE ALPHA"
declare -a ABLATIONS=(
    "E0 raw 0.5"
    "E1 raw 0.5"
    "E2 raw 0.5"
    "E3 raw 0.5"
    "E3 proj 0.5"
)

for ablation in "${ABLATIONS[@]}"; do
    read -r MODE SPACE ALPHA <<< "$ablation"
    
    echo ""
    echo "-----------------------------------------------------------------"
    echo "▶ 正在评估 -> 档位: [ $MODE ] | 空间: [ $SPACE ] | Alpha: [ $ALPHA ]"
    echo "-----------------------------------------------------------------"
    
    # 🌟 使用 --opts 传递消融参数，不再产生未定义参数报错
    python scripts/evaluate.py \
        --config "$CONFIG_FILE" \
        --dataset "$DATASET" \
        --ckpt "$CKPT_PATH" \
        --test_keys "$TEST_KEYS" \
        --opts EVAL.ABLATION_MODE "$MODE" EVAL.REP_SPACE "$SPACE" EVAL.ALPHA "$ALPHA"
done

echo ""
echo "✅ 第一轮物理消融执行完毕。请从上方日志中提取各档位的 F-score 进行严谨对比。"