from __future__ import annotations

from collections import Counter

import numpy as np


def count_metrics(k_true: np.ndarray, k_pred: np.ndarray) -> dict:
    k_true = np.asarray(k_true, dtype=np.int64)
    k_pred = np.asarray(k_pred, dtype=np.int64)
    if len(k_true) == 0:
        return {"n": 0}
    error = np.abs(k_pred - k_true)
    return {
        "n": int(len(k_true)),
        "exact_match": float((error == 0).mean()),
        "within_1": float((error <= 1).mean()),
        "mae": float(error.mean()),
        "bias": float((k_pred - k_true).mean()),
    }


def relative_count_metrics(k_true: np.ndarray, k_pred: np.ndarray, tol: float) -> dict:
    """Speaker-count accuracy scored on error normalised by the true count.

    rel_err = |K̂ − K| / K; a recording counts as correct when rel_err <= tol.
    """
    k_true = np.asarray(k_true, dtype=np.float64)
    k_pred = np.asarray(k_pred, dtype=np.float64)
    if len(k_true) == 0:
        return {"n": 0, "tol": float(tol)}
    rel_err = np.abs(k_pred - k_true) / k_true
    return {
        "n": int(len(k_true)),
        "tol": float(tol),
        "rel_accuracy": float((rel_err <= tol).mean()),
        "rel_mae": float(rel_err.mean()),
        "rel_median": float(np.median(rel_err)),
    }


def confusion(k_true: np.ndarray, k_pred: np.ndarray) -> tuple[list[int], list[int], np.ndarray]:
    pairs = Counter(zip(map(int, k_true), map(int, k_pred)))
    rows = sorted({t for t, _ in pairs})
    cols = sorted({p for _, p in pairs})
    matrix = np.zeros((len(rows), len(cols)), dtype=np.int64)
    for (t, p), count in pairs.items():
        matrix[rows.index(t), cols.index(p)] = count
    return rows, cols, matrix
