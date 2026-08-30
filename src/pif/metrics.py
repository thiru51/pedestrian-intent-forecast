"""Metrics, implemented here rather than pulled from sklearn.

Not out of principle — sklearn is fine — but the package already depends on
numpy and torch and nothing else numeric, and average precision plus AUC plus
displacement error is about sixty lines. One fewer dependency to pin in two
install paths.

Average precision is the headline number for crossing intent. The positive
class is a minority (most window/lead pairs are not followed by a crossing),
and accuracy on an imbalanced binary problem mostly reports the class prior.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def average_precision(scores, labels) -> float:
    """Area under precision-recall, computed the way sklearn's
    average_precision_score does: sum of precision at each threshold weighted
    by the change in recall. No interpolation."""
    scores, labels = _as_float_array(scores), _as_float_array(labels)
    n_pos = labels.sum()
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / n_pos
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum(precision * (recall - prev_recall)))


def roc_auc(scores, labels) -> float:
    """Rank-based (Mann-Whitney U) rather than trapezoid over a threshold
    sweep, so ties are handled exactly."""
    scores, labels = _as_float_array(scores), _as_float_array(labels)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_stats(scores, labels, threshold: float = 0.5) -> dict[str, float]:
    scores, labels = _as_float_array(scores), _as_float_array(labels)
    pred = (scores >= threshold).astype(np.float64)
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        # A planner cares specifically about missed crossings, so the miss rate
        # is worth having in front of you and not buried inside recall.
        "miss_rate": fn / max(tp + fn, 1e-12),
        "false_alarm_rate": fp / max(fp + tn, 1e-12),
    }


def displacement_error(pred, target, mask=None) -> tuple[float, float]:
    """(ADE, FDE) in normalised image units.

    ADE averages the per-step L2 error over the horizon; FDE is the error at
    the last valid step. Both ignore masked-out steps, which is how a track
    that ends before the horizon does gets handled.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mask is None:
        mask = np.ones(pred.shape[:2], dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)

    err = np.linalg.norm(pred - target, axis=-1) * mask
    total = mask.sum()
    ade = float(err.sum() / total) if total else float("nan")

    fde_vals = []
    for i in range(pred.shape[0]):
        valid = np.nonzero(mask[i])[0]
        if len(valid):
            fde_vals.append(err[i, valid[-1]])
    fde = float(np.mean(fde_vals)) if fde_vals else float("nan")
    return ade, fde


def gesture_accuracy(logits, targets, ignore_index: int = -1) -> float:
    """Mean over groups of per-frame top-1 accuracy, skipping ignored labels."""
    correct = 0.0
    total = 0.0
    for group_logits, group_targets in zip(logits, targets):
        valid = group_targets != ignore_index
        if not valid.any():
            continue
        pred = np.asarray(group_logits).argmax(-1)
        correct += float((pred[valid] == group_targets[valid]).sum())
        total += float(valid.sum())
    return correct / total if total else float("nan")


@dataclass
class IntentMetrics:
    n: int
    positive_rate: float
    ap: float
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    miss_rate: float
    false_alarm_rate: float
    ade: float = float("nan")
    fde: float = float("nan")
    gesture_acc: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        return (
            f"n={self.n} pos={self.positive_rate:.3f} "
            f"AP={self.ap:.4f} AUC={self.auc:.4f} F1={self.f1:.4f} "
            f"acc={self.accuracy:.4f} miss={self.miss_rate:.4f} "
            f"ADE={self.ade:.5f} FDE={self.fde:.5f}"
        )


def intent_metrics(
    scores,
    labels,
    traj_pred=None,
    traj_target=None,
    traj_mask=None,
    gesture_acc: float = float("nan"),
    threshold: float = 0.5,
) -> IntentMetrics:
    scores, labels = _as_float_array(scores), _as_float_array(labels)
    stats = binary_stats(scores, labels, threshold)
    ade = fde = float("nan")
    if traj_pred is not None and traj_target is not None:
        ade, fde = displacement_error(traj_pred, traj_target, traj_mask)
    return IntentMetrics(
        n=len(labels),
        positive_rate=float(labels.mean()) if len(labels) else float("nan"),
        ap=average_precision(scores, labels),
        auc=roc_auc(scores, labels),
        ade=ade,
        fde=fde,
        gesture_acc=gesture_acc,
        **stats,
    )
