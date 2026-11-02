"""PyTorch Dataset for sentiment classification with HuggingFace tokenizer."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SentimentDataset(Dataset):
    """A PyTorch Dataset that tokenizes text reviews for BERT-based classification.

    Each sample is tokenized with a HuggingFace tokenizer and returned as a
    dictionary of tensors ready for the model's forward pass.

    Args:
        texts: List of raw review strings.
        labels: List of integer sentiment labels (0=negative, 1=neutral, 2=positive).
        tokenizer: A HuggingFace ``PreTrainedTokenizer`` instance.
        max_len: Maximum sequence length after tokenization (truncation/padding).

    Example::

        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = SentimentDataset(
            texts=["Great product!", "It was okay"],
            labels=[2, 1],
            tokenizer=tokenizer,
            max_len=256,
        )
        sample = dataset[0]
        # sample keys: input_ids, attention_mask, label
    """

    LABEL_MAP: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizer,
        max_len: int = 256,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                f"Length mismatch: {len(texts)} texts vs {len(labels)} labels."
            )
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Tokenize and return a single sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary with ``input_ids``, ``attention_mask``, and ``label`` tensors.
        """
        text = str(self.texts[index])
        label = self.labels[index]

        encoding: dict[str, Any] = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),          # (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0), # (max_len,)
            "label": torch.tensor(label, dtype=torch.long),
        }

    @classmethod
    def label_to_name(cls, label: int) -> str:
        """Convert an integer label to its human-readable sentiment name."""
        return cls.LABEL_MAP.get(label, "unknown")

    @classmethod
    def name_to_label(cls, name: str) -> int:
        """Convert a sentiment name to its integer label."""
        inverse = {v: k for k, v in cls.LABEL_MAP.items()}
        if name.lower() not in inverse:
            raise ValueError(
                f"Unknown sentiment name '{name}'. "
                f"Valid names: {list(inverse.keys())}"
            )
        return inverse[name.lower()]
