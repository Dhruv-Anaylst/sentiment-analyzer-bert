"""BERT-based sentiment classification model."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel


class SentimentClassifier(nn.Module):
    """BERT encoder with a dropout + linear classification head.

    The model extracts the ``[CLS]`` token representation from the final
    hidden state of BERT and passes it through a dropout layer and a linear
    projection to produce class logits.

    Args:
        model_name: HuggingFace model identifier (e.g. ``"bert-base-uncased"``).
        num_classes: Number of output classes.
        dropout_rate: Dropout probability applied before the classification head.
        freeze_encoder_layers: Number of bottom encoder layers to freeze
            (0 = fine-tune all, 6 = freeze bottom half). Freezing reduces
            memory usage and accelerates training with minimal accuracy loss.

    Example::

        model = SentimentClassifier(
            model_name="bert-base-uncased",
            num_classes=3,
            dropout_rate=0.3,
        )
        logits = model(input_ids, attention_mask)  # (batch, 3)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        freeze_encoder_layers: int = 0,
    ) -> None:
        super().__init__()

        self.bert: BertModel = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        # Optionally freeze the bottom N encoder layers
        if freeze_encoder_layers > 0:
            self._freeze_layers(freeze_encoder_layers)

    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze the embedding layer and the first ``num_layers`` encoder layers.

        Frozen parameters do not accumulate gradients and are excluded from
        the optimizer, significantly reducing memory and compute.
        """
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze the specified number of encoder layers
        for layer_idx in range(min(num_layers, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through BERT encoder and classification head.

        Args:
            input_ids: Token IDs of shape ``(batch_size, seq_len)``.
            attention_mask: Binary mask of shape ``(batch_size, seq_len)``
                indicating real tokens (1) vs padding (0).

        Returns:
            Logits tensor of shape ``(batch_size, num_classes)``.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Extract [CLS] token representation (first token of last hidden state)
        cls_output: torch.Tensor = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        # Apply dropout and linear projection
        dropped = self.dropout(cls_output)
        logits: torch.Tensor = self.classifier(dropped)  # (batch, num_classes)

        return logits

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of (trainable) parameters in the model.

        Args:
            trainable_only: If True, count only parameters with
                ``requires_grad=True``.

        Returns:
            Total number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
