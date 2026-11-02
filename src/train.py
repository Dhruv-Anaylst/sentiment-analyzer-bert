"""Training loop with validation, learning rate scheduling, and early stopping."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.evaluate import compute_metrics

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Monitors a metric and stops training when it stops improving.

    Args:
        patience: Number of epochs to wait after the last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: One of ``"min"`` (loss) or ``"max"`` (accuracy/f1).
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter: int = 0
        self.best_score: float | None = None
        self.should_stop: bool = False

    def __call__(self, score: float) -> bool:
        """Check whether training should stop.

        Args:
            score: Current epoch's metric value.

        Returns:
            ``True`` if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        model: The classification model.
        dataloader: Training data loader.
        optimizer: Optimizer instance (e.g. AdamW).
        scheduler: Learning rate scheduler.
        loss_fn: Loss function (e.g. CrossEntropyLoss).
        device: Target device (cpu/cuda).

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss: torch.Tensor = loss_fn(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    """Run one validation epoch.

    Args:
        model: The classification model.
        dataloader: Validation data loader.
        loss_fn: Loss function.
        device: Target device.

    Returns:
        Tuple of (average_loss, accuracy, metrics_dict).
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    accuracy = metrics["accuracy"]
    return avg_loss, accuracy, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 5,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    patience: int = 3,
    checkpoint_dir: str | Path = "models",
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Full training pipeline with validation, scheduling, and early stopping.

    Args:
        model: The ``SentimentClassifier`` model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Maximum number of training epochs.
        learning_rate: Peak learning rate for AdamW.
        weight_decay: L2 regularization coefficient.
        warmup_ratio: Fraction of total steps used for linear warmup.
        patience: Early stopping patience (epochs without improvement).
        checkpoint_dir: Directory to save the best model checkpoint.
        device: Target device; defaults to CUDA if available.

    Returns:
        Dictionary of training history with keys:
        ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``, ``val_f1``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Optimizer ---
    # Separate weight-decay groups: no decay for bias and LayerNorm
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(param_groups, lr=learning_rate)

    # --- Scheduler: linear warmup → linear decay ---
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Loss function ---
    loss_fn = nn.CrossEntropyLoss()

    # --- Early stopping ---
    early_stopping = EarlyStopping(patience=patience, mode="min")

    # --- Training history ---
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    logger.info("Starting training on %s for %d epochs", device, epochs)
    logger.info(
        "Trainable parameters: %s",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device
        )

        # Validate
        val_loss, val_acc, val_metrics = validate_epoch(
            model, val_loader, loss_fn, device
        )

        epoch_time = time.time() - epoch_start

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_metrics["f1_weighted"])

        logger.info(
            "Epoch %d/%d [%.1fs] — "
            "train_loss: %.4f, train_acc: %.4f | "
            "val_loss: %.4f, val_acc: %.4f, val_f1: %.4f",
            epoch,
            epochs,
            epoch_time,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_metrics["f1_weighted"],
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
            logger.info("  ✓ Saved best model checkpoint (val_loss=%.4f)", val_loss)

        # Early stopping check
        if early_stopping(val_loss):
            logger.info(
                "Early stopping triggered after %d epochs (patience=%d)",
                epoch,
                patience,
            )
            break

    logger.info("Training complete. Best val_loss: %.4f", best_val_loss)
    return history
