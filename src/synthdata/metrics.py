"""Classification metrics used by every report."""

import numpy as np

from synthdata.config import CLASSES, TARGET_INDEX


def confusion(labels, predictions) -> np.ndarray:
    """Rows are true breeds, columns are predicted breeds."""
    count = len(CLASSES)
    flat = np.asarray(labels) * count + np.asarray(predictions)
    return np.bincount(flat, minlength=count * count).reshape(count, count)


def target_recall(labels, predictions) -> float:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    return float((predictions[labels == TARGET_INDEX] == TARGET_INDEX).mean())


def macro_f1(labels, predictions) -> float:
    matrix = confusion(labels, predictions)
    true_positive = np.diag(matrix).astype(float)
    denominator = (matrix.sum(axis=0) + matrix.sum(axis=1)).astype(float)
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return float(f1.mean())


def summary(labels, predictions) -> dict:
    matrix = confusion(labels, predictions)
    actual = matrix.sum(axis=1)
    recall = np.divide(
        np.diag(matrix),
        actual,
        out=np.zeros(len(CLASSES)),
        where=actual > 0,
    )
    return {
        "egyptian_mau_recall": target_recall(labels, predictions),
        "macro_f1": macro_f1(labels, predictions),
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "recall": {
            breed: float(value) for breed, value in zip(CLASSES, recall, strict=True)
        },
        "confusion": matrix.tolist(),
    }
