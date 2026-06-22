#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/run_training.sh
# One-click training script for the RAG Hallucination Detector.
# Assumes the RAGTruth-main/ folder is at the repo root.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RESPONSE_PATH="./RAGTruth-main/dataset/response.jsonl"
OUTPUT_DIR="./Hallucination_Model"
MODEL_NAME="microsoft/deberta-v3-base"
MAX_LENGTH=256
EPOCHS=5
LR=2e-5
BATCH_SIZE=16
WEIGHT_DECAY=0.01
TEST_SIZE=0.20
SEED=42

echo "============================================================"
echo "  RAG Hallucination Detector – Training"
echo "============================================================"
echo "Model     : $MODEL_NAME"
echo "Epochs    : $EPOCHS"
echo "LR        : $LR"
echo "Batch     : $BATCH_SIZE"
echo "Output    : $OUTPUT_DIR"
echo "============================================================"

python src/train.py \
    --response_path "$RESPONSE_PATH" \
    --output_dir    "$OUTPUT_DIR"    \
    --model_name    "$MODEL_NAME"    \
    --max_length    "$MAX_LENGTH"    \
    --epochs        "$EPOCHS"        \
    --lr            "$LR"            \
    --batch_size    "$BATCH_SIZE"    \
    --weight_decay  "$WEIGHT_DECAY"  \
    --test_size     "$TEST_SIZE"     \
    --seed          "$SEED"

echo "============================================================"
echo "  Training complete. Model saved to: $OUTPUT_DIR"
echo "============================================================"
