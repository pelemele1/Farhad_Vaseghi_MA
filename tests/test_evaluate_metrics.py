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
    assert row_a["roc_auc"] == 1.0 and row_b["roc_auc"] == 1.0
    assert row_a["support"] == 2 and row_b["support"] == 1


def test_compute_metrics_class_with_zero_positives_gets_nan_ap():
    labels = np.array([[0], [0], [0]])
    probs = np.array([[0.4], [0.6], [0.2]])
    rows = compute_metrics(labels, probs, class_names=("only_class",))

    assert rows[0]["support"] == 0
    assert math.isnan(rows[0]["ap"])
    assert math.isnan(rows[0]["roc_auc"])
    assert rows[0]["precision"] == 0.0  # no predicted positives at threshold=0.5


def test_compute_metrics_class_with_all_positives_gets_nan_roc_auc():
    # AP is still defined (support > 0), but ROC-AUC needs >=1 negative too.
    labels = np.array([[1], [1], [1]])
    probs = np.array([[0.4], [0.6], [0.2]])
    rows = compute_metrics(labels, probs, class_names=("only_class",))

    assert rows[0]["support"] == 3
    assert not math.isnan(rows[0]["ap"])
    assert math.isnan(rows[0]["roc_auc"])


def test_compute_metrics_threshold_changes_precision_recall_not_ap_or_roc_auc():
    labels = np.array([[1], [0], [1], [0]])
    probs = np.array([[0.6], [0.55], [0.4], [0.1]])

    low = compute_metrics(labels, probs, class_names=("c",), threshold=0.3)
    high = compute_metrics(labels, probs, class_names=("c",), threshold=0.5)

    # lower threshold predicts more positives -> recall can only go up or stay the same
    assert low[0]["recall"] >= high[0]["recall"]
    # AP and ROC-AUC are threshold-independent, rank all four the same way regardless
    assert low[0]["ap"] == high[0]["ap"]
    assert low[0]["roc_auc"] == high[0]["roc_auc"]


def test_compute_metrics_accepts_per_class_threshold_dict():
    # class "a" only predicts positive at a low threshold, class "b" only at a high one --
    # a single shared scalar threshold couldn't get both right at once.
    labels = np.array([[1, 1], [0, 0]])
    probs = np.array([[0.3, 0.9], [0.2, 0.8]])

    rows = compute_metrics(labels, probs, class_names=("a", "b"), threshold={"a": 0.25, "b": 0.85})
    row_a, row_b = rows[0], rows[1]

    assert row_a["threshold"] == 0.25 and row_b["threshold"] == 0.85
    # a: 0.3 >= 0.25 -> predicted positive (correct), 0.2 >= 0.25 -> False (correct)
    assert row_a["precision"] == 1.0 and row_a["recall"] == 1.0
    # b: 0.9 >= 0.85 -> predicted positive (correct), 0.8 >= 0.85 -> False (correct)
    assert row_b["precision"] == 1.0 and row_b["recall"] == 1.0

    # unaffected regression check: the plain scalar-threshold path still works the same as before
    scalar_rows = compute_metrics(labels, probs, class_names=("a", "b"), threshold=0.5)
    assert scalar_rows[0]["threshold"] == 0.5 and scalar_rows[1]["threshold"] == 0.5


def test_threshold_for_recall_keeps_target_share_of_positives():
    from src.eval.thresholds import threshold_for_recall

    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.6, 0.2, 0.7, 0.1])
    t, recall, neg_pass = threshold_for_recall(y, p, 0.75)
    assert t == 0.6
    assert recall == 0.75
    assert neg_pass == 0.5
