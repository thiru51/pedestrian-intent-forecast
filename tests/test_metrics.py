"""Metrics are hand-implemented, so they get checked against cases with known
answers rather than against another library."""

from __future__ import annotations

import numpy as np
import pytest

from pif.metrics import (
    average_precision,
    binary_stats,
    displacement_error,
    gesture_accuracy,
    intent_metrics,
    roc_auc,
)


def test_perfect_and_inverted_ranking():
    labels = np.array([0, 0, 1, 1])
    assert average_precision(np.array([0.1, 0.2, 0.8, 0.9]), labels) == pytest.approx(1.0)
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == pytest.approx(1.0)
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == pytest.approx(0.0)


def test_auc_of_a_constant_score_is_a_coin_flip():
    labels = np.array([0, 1, 0, 1])
    assert roc_auc(np.full(4, 0.5), labels) == pytest.approx(0.5)


def test_average_precision_worked_example():
    # Ranked: 1, 0, 1, 0. Precision at the two hits is 1/1 and 2/3, each
    # contributing a recall step of 1/2.
    ap = average_precision(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 0]))
    assert ap == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))


def test_degenerate_splits_are_nan_not_a_lie():
    # All-negative or all-positive: AP and AUC are undefined. Returning 0 or 1
    # here would look like a real score in a log.
    assert np.isnan(average_precision(np.array([0.2, 0.7]), np.array([0, 0])))
    assert np.isnan(roc_auc(np.array([0.2, 0.7]), np.array([1, 1])))


def test_binary_stats_and_miss_rate():
    scores = np.array([0.9, 0.8, 0.4, 0.1])
    labels = np.array([1, 0, 1, 0])
    s = binary_stats(scores, labels, threshold=0.5)
    assert s["accuracy"] == pytest.approx(0.5)
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(0.5)
    assert s["miss_rate"] == pytest.approx(0.5)          # one crossing missed
    assert s["false_alarm_rate"] == pytest.approx(0.5)   # one phantom
    assert s["miss_rate"] == pytest.approx(1.0 - s["recall"])


def test_displacement_error_respects_the_mask():
    pred = np.zeros((1, 3, 2))
    target = np.array([[[3.0, 4.0], [6.0, 8.0], [100.0, 100.0]]])
    mask = np.array([[1.0, 1.0, 0.0]])
    ade, fde = displacement_error(pred, target, mask)
    assert ade == pytest.approx((5.0 + 10.0) / 2)
    assert fde == pytest.approx(10.0)  # last *valid* step, not the masked one


def test_gesture_accuracy_skips_ignored_labels():
    logits = [np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])]
    targets = [np.array([0, 1, -1])]
    assert gesture_accuracy(logits, targets, ignore_index=-1) == pytest.approx(1.0)


def test_intent_metrics_bundle():
    rng = np.random.default_rng(0)
    labels = (rng.random(200) < 0.3).astype(np.float64)
    scores = np.clip(labels * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
    m = intent_metrics(scores, labels)
    assert m.n == 200
    assert 0.0 <= m.ap <= 1.0
    assert m.auc > 0.5  # the scores do carry signal
    assert m.positive_rate == pytest.approx(labels.mean())
    assert "AP=" in m.line()
    assert set(m.as_dict()) >= {"ap", "auc", "f1", "miss_rate", "ade", "fde"}
