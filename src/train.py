"""
RAG Hallucination Detector - DeBERTa-v3 Fine-tuning
=====================================================
Trains microsoft/deberta-v3-base on the RAGTruth dataset
for binary hallucination classification.

Dataset  : RAGTruth (Wu et al., 2023)  https://arxiv.org/abs/2401.00396
Backbone : microsoft/deberta-v3-base
Task     : Binary sequence classification (hallucinated / faithful)
"""

import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME  = "microsoft/deberta-v3-base"
MAX_LENGTH  = 256
LABEL_NAMES = ["faithful", "hallucinated"]


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_ragtruth(response_path: str) -> pd.DataFrame:
    """Load RAGTruth response.jsonl and convert span-level labels to binary."""
    df = pd.read_json(response_path, lines=True)
    # Convert list of annotation dicts → binary label
    df["labels"] = df["labels"].apply(lambda x: 1 if len(x) > 0 else 0)
    logger.info(f"Loaded {len(df)} rows | label dist:\n{df['labels'].value_counts()}")
    return df


def build_hf_datasets(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified train/val split → HuggingFace Dataset objects."""
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["labels"]
    )
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")
    return (
        Dataset.from_pandas(train_df, preserve_index=False),
        Dataset.from_pandas(val_df, preserve_index=False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def make_tokenize_fn(tokenizer, max_length: int):
    """Returns a batched tokenisation function for HF .map()."""
    def tokenize(batch):
        return tokenizer(
            batch["response"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    return tokenize


COLUMNS_TO_DROP = ["id", "source_id", "model", "temperature", "split", "quality", "response"]


def preprocess(dataset: Dataset, tokenize_fn) -> Dataset:
    dataset = dataset.map(tokenize_fn, batched=True)
    drop_cols = [c for c in COLUMNS_TO_DROP if c in dataset.column_names]
    dataset = dataset.remove_columns(drop_cols)
    dataset.set_format("torch")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

accuracy_metric  = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric    = evaluate.load("recall")
f1_metric        = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy" : accuracy_metric.compute( predictions=preds, references=labels)["accuracy"],
        "precision": precision_metric.compute(predictions=preds, references=labels)["precision"],
        "recall"   : recall_metric.compute(   predictions=preds, references=labels)["recall"],
        "f1"       : f1_metric.compute(       predictions=preds, references=labels)["f1"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train DeBERTa-v3 hallucination detector")
    parser.add_argument("--response_path",  type=str,   default="./RAGTruth-main/dataset/response.jsonl")
    parser.add_argument("--output_dir",     type=str,   default="./Hallucination_Model")
    parser.add_argument("--model_name",     type=str,   default=MODEL_NAME)
    parser.add_argument("--max_length",     type=int,   default=MAX_LENGTH)
    parser.add_argument("--epochs",         type=int,   default=5)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--test_size",      type=float, default=0.20)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--fp16",           action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Load & split data ─────────────────────────────────────────────────
    df = load_ragtruth(args.response_path)
    train_ds, val_ds = build_hf_datasets(df, args.test_size, args.seed)

    # ── 2. Tokeniser ─────────────────────────────────────────────────────────
    tokenizer   = AutoTokenizer.from_pretrained(args.model_name)
    tokenize_fn = make_tokenize_fn(tokenizer, args.max_length)
    train_ds    = preprocess(train_ds, tokenize_fn)
    val_ds      = preprocess(val_ds,   tokenize_fn)

    # ── 3. Model ─────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "faithful", 1: "hallucinated"},
        label2id={"faithful": 0, "hallucinated": 1},
        torch_dtype=torch.float32,
    ).float()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Model on {device} | dtype {next(model.parameters()).dtype}")

    # ── 4. Training arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir              = args.output_dir,
        overwrite_output_dir    = True,
        num_train_epochs        = args.epochs,
        learning_rate           = args.lr,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        weight_decay            = args.weight_decay,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        logging_strategy        = "steps",
        logging_steps           = 100,
        load_best_model_at_end  = True,
        metric_for_best_model   = "f1",
        greater_is_better       = True,
        fp16                    = args.fp16,
        bf16                    = False,
        report_to               = "none",
        seed                    = args.seed,
    )

    # ── 5. Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
    )

    # ── 6. Train ─────────────────────────────────────────────────────────────
    train_result = trainer.train()
    logger.info(f"Training finished: {train_result}")

    # ── 7. Evaluate ──────────────────────────────────────────────────────────
    metrics = trainer.evaluate()
    logger.info("Validation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── 8. Save ──────────────────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Persist metrics JSON
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model + tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
