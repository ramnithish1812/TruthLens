#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/run_evaluation.sh
# Evaluate a trained checkpoint on the full RAGTruth dataset.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_DIR="./Hallucination_Model"
RAGTRUTH_PATH="./RAGTruth-main/dataset/response.jsonl"
OUTPUT_DIR="./results"

echo "============================================================"
echo "  RAG Hallucination Detector – Evaluation"
echo "============================================================"
echo "Checkpoint : $MODEL_DIR"
echo "Dataset    : $RAGTRUTH_PATH"
echo "Results    : $OUTPUT_DIR"
echo "============================================================"

python src/predict.py \
    --model_dir     "$MODEL_DIR"     \
    --eval_ragtruth "$RAGTRUTH_PATH" \
    --output_dir    "$OUTPUT_DIR"

echo "============================================================"
echo "  Evaluation complete. Results in: $OUTPUT_DIR"
echo "============================================================"
