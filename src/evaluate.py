"""Evaluation functions: metrics, classification reports, and confusion matrices."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = ["negative", "neutral", "positive"]


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute classification metrics from true and predicted labels.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Optional list of class names for the report.

    Returns:
        Dictionary of scalar metrics: accuracy, precision, recall, f1
        (macro and weighted), and Cohen's kappa.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }

    return metrics


def generate_classification_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> str:
    """Generate a formatted sklearn classification report string.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Display names for each class.

    Returns:
        Formatted classification report as a string.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    report: str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    return report


def generate_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> np.ndarray:
    """Compute the confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Class names (used only for logging).

    Returns:
        Confusion matrix as a NumPy array of shape ``(n_classes, n_classes)``.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    cm: np.ndarray = confusion_matrix(y_true, y_pred)
    logger.info("Confusion matrix:\n%s", cm)
    return cm


def per_class_analysis(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute per-class precision, recall, F1, and support.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Display names for each class.

    Returns:
        List of dictionaries, one per class, each containing:
        ``class_name``, ``precision``, ``recall``, ``f1``, ``support``.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
    results: list[dict[str, Any]] = []

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results.append(
            {
                "class_name": class_names[i],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int(support),
            }
        )

        logger.info(
            "  %s — P: %.4f  R: %.4f  F1: %.4f  Support: %d",
            class_names[i],
            precision,
            recall,
            f1,
            support,
        )

    return results


def full_evaluation(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run a complete evaluation and return all results.

    Combines scalar metrics, the text classification report, the confusion
    matrix, and per-class analysis into a single result dictionary.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Display names for each class.

    Returns:
        Dictionary with keys: ``metrics``, ``report``, ``confusion_matrix``,
        ``per_class``.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    logger.info("=" * 60)
    logger.info("FULL EVALUATION REPORT")
    logger.info("=" * 60)

    metrics = compute_metrics(y_true, y_pred, class_names)
    report = generate_classification_report(y_true, y_pred, class_names)
    cm = generate_confusion_matrix(y_true, y_pred, class_names)
    per_class = per_class_analysis(y_true, y_pred, class_names)

    logger.info("\n%s", report)
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Weighted F1: %.4f", metrics["f1_weighted"])
    logger.info("Cohen's Kappa: %.4f", metrics["cohen_kappa"])

    return {
        "metrics": metrics,
        "report": report,
        "confusion_matrix": cm,
        "per_class": per_class,
    }
