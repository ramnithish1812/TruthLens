"""
RAG Hallucination Detector – src package
=========================================
DeBERTa-v3-base fine-tuned on RAGTruth for binary hallucination detection.

Modules
-------
train      – full fine-tuning pipeline (data → model → checkpoint)
predict    – inference on single strings, JSONL files, or RAGTruth benchmark
data_utils – shared data-loading and preprocessing helpers
metrics    – evaluation metrics (Trainer callback + detailed benchmark report)
"""
