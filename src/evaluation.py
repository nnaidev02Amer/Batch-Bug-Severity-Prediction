"""Metric computation. All numbers shown anywhere in the project come from here."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from .utils import CLASSES


def evaluate_predictions(y_true, y_pred) -> dict:
    """Return accuracy, macro/weighted P/R/F1, per-class metrics and confusion matrix."""
    labels = CLASSES
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)[:3]
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)[:3]
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": dict(zip(["precision", "recall", "f1"], map(float, macro))),
        "weighted": dict(zip(["precision", "recall", "f1"], map(float, weighted))),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "report_text": classification_report(
            y_true, y_pred, labels=labels, zero_division=0),
    }


def majority_baseline(y_train, y_test) -> dict:
    """Accuracy/macro-F1 of always predicting the most common training class."""
    values, counts = np.unique(y_train, return_counts=True)
    majority = values[np.argmax(counts)]
    pred = [majority] * len(y_test)
    m = evaluate_predictions(list(y_test), pred)
    return {"class": str(majority), "accuracy": m["accuracy"],
            "macro_f1": m["macro"]["f1"]}
