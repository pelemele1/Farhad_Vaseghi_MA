import math

import numpy as np

from scripts.evaluate_stage_a import compute_metrics


def test_compute_metrics_perfect_predictions():
    labels = np.array([[1, 0], [0, 1], [1, 0], [0, 0]])
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.1, 0.1]])
    rows = compute_metrics(labels, probs, class_names=("a", "b"))

    row_a, row_b = rows[0], rows[1]
    assert row_a["precision"] == 1.0 and row_a["recall"] == 1.0 and row_a["f1"] == 1.0
    assert row_b["precision"] == 1.0 and row_b["recall"] == 1.0 and row_b["f1"] == 1.0
    assert row_a["ap"] == 1.0 and row_b["ap"] == 1.0
    assert row_a["support"] == 2 and row_b["support"] == 1


def test_compute_metrics_class_with_zero_positives_gets_nan_ap():
    labels = np.array([[0], [0], [0]])
    probs = np.array([[0.4], [0.6], [0.2]])
    rows = compute_metrics(labels, probs, class_names=("only_class",))

    assert rows[0]["support"] == 0
    assert math.isnan(rows[0]["ap"])
    assert rows[0]["precision"] == 0.0  # no predicted positives at threshold=0.5


def test_compute_metrics_threshold_changes_precision_recall_not_ap():
    labels = np.array([[1], [0], [1], [0]])
    probs = np.array([[0.6], [0.55], [0.4], [0.1]])

    low = compute_metrics(labels, probs, class_names=("c",), threshold=0.3)
    high = compute_metrics(labels, probs, class_names=("c",), threshold=0.5)

    # lower threshold predicts more positives -> recall can only go up or stay the same
    assert low[0]["recall"] >= high[0]["recall"]
    # AP is threshold-independent, ranks all four the same way regardless
    assert low[0]["ap"] == high[0]["ap"]
