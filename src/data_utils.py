"""
data_utils.py
=============
Shared data-loading and preprocessing utilities for the RAG hallucination
detection pipeline. Used by both train.py and predict.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Columns not needed after tokenisation
COLUMNS_TO_DROP = [
    "id", "source_id", "model",
    "temperature", "split", "quality", "response",
]


def load_ragtruth(response_path: str | Path) -> pd.DataFrame:
    """
    Read RAGTruth response.jsonl and convert span-level annotation lists
    into a binary label:
        0 → faithful   (empty labels list)
        1 → hallucinated (one or more annotation spans)
    """
    df = pd.read_json(str(response_path), lines=True)
    df["labels"] = df["labels"].apply(lambda x: 1 if len(x) > 0 else 0)
    n_total = len(df)
    n_hall  = df["labels"].sum()
    logger.info(
        f"Loaded {n_total} rows  |  "
        f"hallucinated={n_hall} ({100*n_hall/n_total:.1f}%)  "
        f"faithful={n_total-n_hall} ({100*(n_total-n_hall)/n_total:.1f}%)"
    )
    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df) with stratification on the 'labels' column."""
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["labels"]
    )
    logger.info(f"Train: {len(train_df)}  |  Val: {len(val_df)}")
    return train_df, val_df


def to_hf_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[Dataset, Dataset]:
    """Convert DataFrames to HuggingFace Dataset objects."""
    return (
        Dataset.from_pandas(train_df, preserve_index=False),
        Dataset.from_pandas(val_df,   preserve_index=False),
    )


def make_tokenize_fn(tokenizer, max_length: int = 256):
    """
    Return a batched tokenisation function compatible with Dataset.map().

    Parameters
    ----------
    tokenizer  : HuggingFace tokenizer
    max_length : int  –  truncation / padding length
    """
    def tokenize(batch):
        return tokenizer(
            batch["response"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    return tokenize


def preprocess_dataset(dataset: Dataset, tokenize_fn) -> Dataset:
    """
    Apply tokenisation, drop non-model columns, and set tensor format.
    """
    dataset = dataset.map(tokenize_fn, batched=True)
    drop_cols = [c for c in COLUMNS_TO_DROP if c in dataset.column_names]
    dataset = dataset.remove_columns(drop_cols)
    dataset.set_format("torch")
    return dataset
