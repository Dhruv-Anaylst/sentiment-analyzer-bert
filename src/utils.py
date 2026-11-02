"""Utility functions: reproducibility, device management, plotting, and I/O."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


def get_device() -> torch.device:
    """Detect and return the best available compute device.

    Returns:
        ``torch.device`` for CUDA, MPS (Apple Silicon), or CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | Path = "reports/figures/training_curves.png",
) -> None:
    """Plot training and validation loss/accuracy curves.

    Creates a two-panel figure: loss on the left, accuracy on the right.

    Args:
        history: Training history dictionary with keys ``train_loss``,
            ``train_acc``, ``val_loss``, ``val_acc``.
        save_path: File path to save the figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    axes[1].set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved to %s", save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | Path = "reports/figures/confusion_matrix.png",
    normalize: bool = True,
) -> None:
    """Plot a heatmap of the confusion matrix.

    Args:
        cm: Confusion matrix array of shape ``(n_classes, n_classes)``.
        class_names: List of class display names.
        save_path: File path to save the figure.
        normalize: If ``True``, normalize rows to show percentages.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
        title = "Confusion Matrix (Normalized)"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def save_metrics(
    metrics: dict[str, Any],
    save_path: str | Path = "reports/metrics.json",
) -> None:
    """Serialize evaluation metrics to a JSON file.

    Args:
        metrics: Dictionary of metric names to values.
        save_path: File path for the JSON output.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python builtins for JSON serialization
    serializable: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer,)):
            serializable[key] = int(value)
        elif isinstance(value, (np.floating,)):
            serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    logger.info("Metrics saved to %s", save_path)


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load metrics from a JSON file.

    Args:
        path: Path to the JSON metrics file.

    Returns:
        Dictionary of metric names to values.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)
