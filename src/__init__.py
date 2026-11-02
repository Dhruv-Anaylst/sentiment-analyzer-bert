"""
Sentiment Analyzer — Fine-tuning BERT for multi-class sentiment classification.

Modules:
    dataset:  PyTorch Dataset with HuggingFace tokenizer integration.
    model:    BERT-based classification architecture.
    train:    Training loop with validation, scheduling, and checkpointing.
    evaluate: Metrics computation, classification reports, confusion matrices.
    predict:  Single-text inference with confidence scores.
    utils:    Reproducibility, device management, plotting, and I/O helpers.
"""

__version__ = "1.0.0"
__author__ = "Dhruv"
