"""Inference module: load a trained model and predict sentiment with confidence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from src.dataset import SentimentDataset
from src.model import SentimentClassifier

logger = logging.getLogger(__name__)


class SentimentPredictor:
    """High-level inference wrapper for the trained sentiment model.

    Handles model loading, tokenization, and prediction in a single class.

    Args:
        model_path: Path to the saved model checkpoint (``.pt`` file).
        model_name: HuggingFace model identifier used during training.
        num_classes: Number of sentiment classes.
        max_len: Maximum token length (must match training configuration).
        device: Target device; defaults to CUDA if available.

    Example::

        predictor = SentimentPredictor("models/best_model.pt")
        result = predictor.predict("This product is amazing!")
        print(result)
        # {
        #     "text": "This product is amazing!",
        #     "sentiment": "positive",
        #     "confidence": 0.9621,
        #     "probabilities": {"negative": 0.0123, "neutral": 0.0256, "positive": 0.9621},
        # }
    """

    def __init__(
        self,
        model_path: str | Path,
        model_name: str = "bert-base-uncased",
        num_classes: int = 3,
        max_len: int = 256,
        device: torch.device | None = None,
    ) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_len = max_len
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load tokenizer
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)

        # Load model
        self.model = SentimentClassifier(
            model_name=model_name,
            num_classes=num_classes,
        )
        self._load_checkpoint(Path(model_path))
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "SentimentPredictor initialized on %s from %s",
            self.device,
            model_path,
        )

    def _load_checkpoint(self, path: Path) -> None:
        """Load model weights from a training checkpoint.

        Args:
            path: Path to the ``.pt`` checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                "Loaded checkpoint from epoch %d (val_loss=%.4f)",
                checkpoint.get("epoch", -1),
                checkpoint.get("val_loss", float("nan")),
            )
        else:
            # Support raw state_dict saves
            self.model.load_state_dict(checkpoint)
            logger.info("Loaded raw state_dict checkpoint")

    @torch.no_grad()
    def predict(self, text: str) -> dict[str, Any]:
        """Predict sentiment for a single text input.

        Args:
            text: Raw review text string.

        Returns:
            Dictionary with keys:
            - ``text``: The input text.
            - ``sentiment``: Predicted class name (e.g. ``"positive"``).
            - ``confidence``: Probability of the predicted class.
            - ``probabilities``: Dict mapping each class name to its probability.
        """
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

        prob_dict = {
            SentimentDataset.label_to_name(i): round(probabilities[i].item(), 4)
            for i in range(self.num_classes)
        }

        return {
            "text": text,
            "sentiment": SentimentDataset.label_to_name(predicted_idx),
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
        }

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict sentiment for a batch of texts.

        Args:
            texts: List of raw review text strings.

        Returns:
            List of prediction dictionaries (same format as :meth:`predict`).
        """
        return [self.predict(text) for text in texts]
