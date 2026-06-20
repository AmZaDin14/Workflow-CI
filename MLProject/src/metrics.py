"""Metric helpers used by modelling_tuning.py.

Pure functions so they can be unit-tested without MLflow or sklearn objects.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Return a flat dict of classification metrics for MLflow logging."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-7, 1 - 1e-7))),
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }


def baseline_stats(
    X: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Per-feature mean and std of the training matrix (no NaNs)."""
    X = np.asarray(X, dtype=float)
    out: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(feature_names):
        col = X[:, j]
        col = col[~np.isnan(col)]
        out[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col, ddof=1)) if len(col) > 1 else 0.0,
        }
    return out


def drift_score(
    X_now: np.ndarray,
    baseline: Dict[str, Dict[str, float]],
) -> float:
    """Average absolute standardised mean shift across features.

    A value of 0 means identical distribution; >1 typically means noticeable
    drift. Used by the K4 exporter for the `ml_model_drift_score` gauge.
    """
    X_now = np.asarray(X_now, dtype=float)
    if X_now.ndim == 1:
        X_now = X_now.reshape(1, -1)
    if X_now.shape[0] == 0:
        return 0.0
    drifts: List[float] = []
    for j, name in enumerate(baseline.keys()):
        if j >= X_now.shape[1]:
            break
        col = X_now[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        b = baseline[name]
        std = b["std"] if b["std"] > 1e-9 else 1e-9
        drifts.append(abs(float(np.mean(col)) - b["mean"]) / std)
    if not drifts:
        return 0.0
    return float(np.mean(drifts))
