from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def best_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(1 - sorted_labels)
    false_negative = int(sorted_labels.sum()) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1))
    return float(f1[best_index]), float(sorted_scores[best_index])


def binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    f1, threshold = best_f1(scores, labels)
    result = {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "f1": f1,
        "threshold": threshold,
    }
    return result


def pixel_metrics(
    maps: list[np.ndarray],
    masks: list[np.ndarray],
    max_pixels: int = 2_000_000,
    seed: int = 0,
) -> dict[str, float]:
    flat_scores = np.concatenate(
        [np.asarray(m, dtype=np.float32).ravel() for m in maps]
    )
    flat_masks = np.concatenate([np.asarray(m, dtype=np.uint8).ravel() for m in masks])
    if flat_scores.size > max_pixels:
        rng = np.random.default_rng(seed)
        index = rng.choice(flat_scores.size, size=max_pixels, replace=False)
        flat_scores = flat_scores[index]
        flat_masks = flat_masks[index]
    f1, threshold = best_f1(flat_scores, flat_masks)
    return {
        "pixel_auroc": float(roc_auc_score(flat_masks, flat_scores)),
        "pixel_aupr": float(average_precision_score(flat_masks, flat_scores)),
        "pixel_f1": f1,
        "pixel_threshold": threshold,
    }


def roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(np.asarray(labels), np.asarray(scores))
    return fpr, tpr
