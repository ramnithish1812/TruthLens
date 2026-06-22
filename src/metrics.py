"""
metrics.py
==========
Evaluation utilities for the RAG hallucination detection project.

Provides:
  - compute_metrics()    – used as Trainer callback
  - full_benchmark()     – post-training detailed report
  - save_results()       – persist benchmark to JSON + CSV
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import evaluate as hf_evaluate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

# ── HuggingFace metric objects (loaded once) ──────────────────────────────────
_accuracy  = hf_evaluate.load("accuracy")
_precision = hf_evaluate.load("precision")
_recall    = hf_evaluate.load("recall")
_f1        = hf_evaluate.load("f1")


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Callback for HuggingFace Trainer.
    Returns accuracy, precision, recall, f1 on the validation set.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy" : _accuracy.compute( predictions=preds, references=labels)["accuracy"],
        "precision": _precision.compute(predictions=preds, references=labels)["precision"],
        "recall"   : _recall.compute(   predictions=preds, references=labels)["recall"],
        "f1"       : _f1.compute(       predictions=preds, references=labels)["f1"],
    }


def full_benchmark(
    y_true: list[int],
    y_pred: list[int],
    probs:  list[float],
) -> Dict[str, Any]:
    """
    Compute a comprehensive set of evaluation metrics.

    Parameters
    ----------
    y_true : ground-truth binary labels (0 = faithful, 1 = hallucinated)
    y_pred : predicted binary labels
    probs  : predicted probability of class 1 (hallucinated)

    Returns
    -------
    dict with scalar metrics + 'report' (classification_report str)
          + 'confusion_matrix' (2×2 list)
    """
    metrics = {
        "Accuracy" : round(float(accuracy_score( y_true, y_pred)),               4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall"   : round(float(recall_score(   y_true, y_pred, zero_division=0)), 4),
        "F1 Score" : round(float(f1_score(       y_true, y_pred, zero_division=0)), 4),
        "ROC-AUC"  : round(float(roc_auc_score(  y_true, probs)),                4),
    }
    metrics["report"]           = classification_report(
        y_true, y_pred,
        target_names=["faithful", "hallucinated"],
        digits=4,
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    # Pretty-print
    print("\n" + "=" * 55)
    print("  Hallucination Detection – Benchmark Results")
    print("=" * 55)
    for k in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        print(f"  {k:<12}: {metrics[k]:.4f}")
    print("=" * 55)
    print("\nClassification Report:\n")
    print(metrics["report"])
    print("Confusion Matrix (faithful | hallucinated):")
    cm = metrics["confusion_matrix"]
    print(f"  TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
    print(f"  FN={cm[1][0]:>5}  TP={cm[1][1]:>5}\n")

    return metrics


def save_results(metrics: Dict[str, Any], output_dir: str) -> None:
    """Persist metrics to JSON and a one-row CSV for easy diffing across runs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out / "benchmark_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics → {json_path}")

    # CSV (scalar rows only)
    import pandas as pd
    scalar_keys = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    row = {k: metrics[k] for k in scalar_keys if k in metrics}
    csv_path = out / "benchmark_metrics.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    logger.info(f"CSV     → {csv_path}")
