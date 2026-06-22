"""
RAG Hallucination Detector - Inference
=======================================
Load a trained DeBERTa-v3 checkpoint and run inference on new text,
a JSONL file, or the full RAGTruth validation split.

Usage examples
--------------
# Single string
python src/predict.py \
    --model_dir ./Hallucination_Model \
    --text "The Eiffel Tower is located in Berlin."

# Batch JSONL (must contain a "response" field)
python src/predict.py \
    --model_dir ./Hallucination_Model \
    --input_jsonl ./data/my_responses.jsonl \
    --output_jsonl ./results/predictions.jsonl

# Full benchmark evaluation on RAGTruth response.jsonl
python src/predict.py \
    --model_dir ./Hallucination_Model \
    --eval_ragtruth ./RAGTruth-main/dataset/response.jsonl \
    --output_dir ./results
"""

import os
import json
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = 256
BATCH_SIZE = 32
LABEL_MAP  = {0: "faithful", 1: "hallucinated"}


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    logger.info(f"Model loaded from {model_dir} → {device}")
    return tokenizer, model, device


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_texts(texts: list[str], tokenizer, model, device) -> dict:
    """
    Returns dict with keys:
      labels     – list[int]   (0 = faithful, 1 = hallucinated)
      probs      – list[float] (probability of class 1)
      label_names – list[str]
    """
    all_preds, all_probs = [], []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    return {
        "labels"     : all_preds,
        "probs"      : all_probs,
        "label_names": [LABEL_MAP[p] for p in all_preds],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(y_true, y_pred, probs) -> dict:
    return {
        "Accuracy" : round(accuracy_score( y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall"   : round(recall_score(   y_true, y_pred), 4),
        "F1 Score" : round(f1_score(       y_true, y_pred), 4),
        "ROC-AUC"  : round(roc_auc_score(  y_true, probs ), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",      required=True)
    p.add_argument("--text",           default=None,  help="Single string to classify")
    p.add_argument("--input_jsonl",    default=None,  help="Batch JSONL with 'response' field")
    p.add_argument("--output_jsonl",   default=None,  help="Where to write predictions")
    p.add_argument("--eval_ragtruth",  default=None,  help="Path to RAGTruth response.jsonl")
    p.add_argument("--output_dir",     default="./results")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer, model, device = load_model(args.model_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Single text ───────────────────────────────────────────────────────────
    if args.text:
        result = predict_texts([args.text], tokenizer, model, device)
        label  = result["label_names"][0]
        prob   = result["probs"][0]
        print(f"\nInput  : {args.text[:120]}")
        print(f"Label  : {label}")
        print(f"P(hall): {prob:.4f}\n")

    # ── Batch JSONL ───────────────────────────────────────────────────────────
    elif args.input_jsonl:
        df = pd.read_json(args.input_jsonl, lines=True)
        result = predict_texts(df["response"].tolist(), tokenizer, model, device)
        df["pred_label"] = result["label_names"]
        df["hall_prob"]  = result["probs"]
        out = args.output_jsonl or os.path.join(args.output_dir, "predictions.jsonl")
        df.to_json(out, orient="records", lines=True)
        logger.info(f"Predictions written → {out}")

    # ── RAGTruth benchmark ────────────────────────────────────────────────────
    elif args.eval_ragtruth:
        df = pd.read_json(args.eval_ragtruth, lines=True)
        df["labels"] = df["labels"].apply(lambda x: 1 if len(x) > 0 else 0)
        result = predict_texts(df["response"].tolist(), tokenizer, model, device)

        y_true = df["labels"].tolist()
        y_pred = result["labels"]
        probs  = result["probs"]

        metrics = benchmark(y_true, y_pred, probs)

        print("\n" + "="*50)
        print("  RAGTruth Benchmark Results")
        print("="*50)
        for k, v in metrics.items():
            print(f"  {k:<12}: {v:.4f}")
        print("="*50)
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["faithful","hallucinated"], digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        out_path = os.path.join(args.output_dir, "benchmark_metrics.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved → {out_path}")

    else:
        print("Provide --text, --input_jsonl, or --eval_ragtruth")


if __name__ == "__main__":
    main()
