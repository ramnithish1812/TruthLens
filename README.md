# TruthLens

A DeBERTa-v3-based hallucination detector for Retrieval-Augmented Generation (RAG) responses, fine-tuned on [RAGTruth](https://arxiv.org/abs/2401.00396).

TruthLens performs binary sequence classification on a single response string — no retrieved source passages required at inference time — making it usable as a lightweight, low-latency post-generation guard rail in production RAG pipelines.

```
f(response) → {0: faithful, 1: hallucinated}
```

> Full write-up with methodology, related work, and discussion: see `TruthLens_IEEE.docx` in this repo.

---

## Results

All numbers below are taken directly from `hallucination_model.ipynb` — nothing here is estimated or extrapolated.

The model was trained for 5 epochs with `load_best_model_at_end=True` and `metric_for_best_model="f1"`. **Epoch 3 was selected as the best checkpoint.** Epochs 4–5 show validation loss climbing while training loss keeps falling — classic overfitting — so they are reported for transparency but are not the deployed model.

### Reported model (Epoch 3, validation split)

| Metric | Score |
|---|---|
| Accuracy | 76.11% |
| Precision (hallucinated) | 73.09% |
| Recall (hallucinated) | 70.52% |
| F1 (hallucinated) | 71.78% |
| AUC-ROC | 0.8356 |

### Per-class report (epoch 3)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Faithful (0) | 78.26% | 80.35% | 79.29% | 2,025 |
| Hallucinated (1) | 73.09% | 70.52% | 71.78% | 1,533 |
| **Weighted avg** | 76.03% | 76.11% | 76.05% | 3,558 |

Note: recall < precision on the hallucinated class. For a safety-critical guard rail you generally want the opposite (catch every hallucination, tolerate some false alarms) — see [Limitations](#limitations).

### Epoch-wise training dynamics

| Epoch | Train Loss | Val Loss | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 1 | 0.5282 | 0.5379 | 75.52% | 80.59% | 56.88% | 66.69% |
| 2 | 0.4823 | 0.5400 | 72.63% | 65.59% | 76.71% | 70.72% |
| **3*** | **0.4556** | **0.5025** | **76.11%** | **73.09%** | **70.52%** | **71.78%** |
| 4 | 0.3498 | 0.6772 | 68.02% | 58.70% | 86.95% | 70.08% |
| 5 | 0.2933 | 0.8234 | 68.16% | 58.72% | 87.87% | 70.39% |

\* best checkpoint, selected by validation F1

---

## Dataset

[RAGTruth](https://github.com/ParticleMedia/RAGTruth) — 17,790 manually annotated RAG responses from six LLMs (GPT-3.5-turbo, GPT-4, Llama-2 7B/13B/70B-chat, Mistral-7B-Instruct) across summarisation, question-answering, and data-to-text tasks.

A response is labelled hallucinated if it contains one or more annotated hallucination spans, else faithful.

| Split | Total | Faithful | Hallucinated | Hall. rate |
|---|---|---|---|---|
| Full dataset | 17,790 | 10,126 | 7,664 | 43.10% |
| Train (80%) | 14,232 | 8,101 | 6,131 | 43.09% |
| Validation (20%) | 3,558 | 2,025 | 1,533 | 43.09% |

Split via `sklearn.model_selection.train_test_split(test_size=0.2, random_state=42, stratify=labels)`.

---

## Model

| | |
|---|---|
| Backbone | `microsoft/deberta-v3-base` |
| Parameters | ~184M |
| Classification head | Linear, `[CLS]` (768-dim) → 2 logits, softmax |
| Max sequence length | 256 tokens |
| Precision | float32 |

## Training configuration

```python
TrainingArguments(
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=False,
    bf16=False,
)
```

Optimizer: AdamW. Trained on a single **NVIDIA Tesla T4** (Google Colab) — full 5-epoch run took ~90 minutes (5,396s).

### Environment

```
transformers==4.53.0
datasets==3.6.0
accelerate==1.8.1
evaluate==0.4.5
scikit-learn==1.7.0
torch==2.11.0+cu128
```

---

## Usage

### Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("path/to/Hallucination_Model")
model = AutoModelForSequenceClassification.from_pretrained("path/to/Hallucination_Model")
model.eval()

def predict(response: str) -> dict:
    inputs = tokenizer(response, truncation=True, padding="max_length",
                        max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {"faithful": probs[0].item(), "hallucinated": probs[1].item()}

predict("The Eiffel Tower was completed in 1889 in Paris.")
```

### Reproducing training

Open `hallucination_model.ipynb` in Google Colab (GPU runtime), upload the RAGTruth dataset ZIP when prompted in the "Upload dataset ZIP" cell, and run all cells top to bottom. The notebook saves the final model + tokenizer to `Hallucination_Model/` and zips it for download.

---

## Limitations

- **Overfits past epoch 3.** Training loss falls monotonically across all 5 epochs while validation loss rises after epoch 3 — the model memorizes training data rather than generalizing further. No early stopping or extra regularization beyond weight decay (0.01) is currently applied.
- **Recall < precision on the hallucinated class** (70.52% vs. 73.09%). This means more hallucinations are missed than falsely flagged — the opposite of what you'd want from a guard rail in a safety-critical pipeline.
- **No access to retrieved source passages at inference time.** The classifier sees only the generated response, so it can't cross-reference claims against the grounding context the way the original human annotators did.
- **256-token truncation** may hide hallucinations that occur later in long responses.
- **Distribution shift risk.** RAGTruth's six source LLMs (GPT-3.5/4, Llama-2, Mistral) and three task types may not represent newer models (e.g., GPT-4o, Claude) or production retrieval corpora.

## Future work

- Early stopping / stronger regularization to address epoch-3+ overfitting
- Threshold recalibration or class-weighted loss to raise recall on the hallucinated class
- Two-encoder setup incorporating retrieved passages for cross-referencing
- Span-level (not just response-level) hallucination prediction
- Controlled benchmarking against published baselines (SummaC, fine-tuned BERT/RoBERTa, RAGTruth's Llama-2-7B detector)

---

## Citation

```bibtex
@misc{ragtruth2024,
  title={RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models},
  author={Wu, Yuanhao and others},
  year={2024},
  eprint={2401.00396},
  archivePrefix={arXiv}
}

@inproceedings{deberta-v3,
  title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing},
  author={He, Pengcheng and Gao, Jianfeng and Chen, Weizhu},
  booktitle={ICLR},
  year={2023}
}
```

