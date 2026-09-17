import numpy as np
import pytest

from src.eval.thresholds import best_f1_threshold, tune_per_class_thresholds


def test_best_f1_threshold_finds_the_perfect_separator():
    # a clean gap between positives (0.6, 0.7) and negatives (0.1, 0.2) --
    # any threshold in (0.2, 0.6] gets precision=recall=f1=1.0.
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.6, 0.2, 0.7, 0.1])

    threshold, precision, recall, f1 = best_f1_threshold(y_true, y_prob)

    assert 0.2 < threshold <= 0.6
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == pytest.approx(1.0)


def test_tune_per_class_thresholds_returns_one_value_per_class():
    # class "a" needs a low threshold to catch its one positive, class "b" a high one --
    # confirms tuning is independent per class, not one shared sweep.
    labels = np.array([[1, 0], [0, 0], [0, 1], [0, 0]])
    probs = np.array([[0.3, 0.1], [0.25, 0.2], [0.1, 0.9], [0.2, 0.3]])

    tuned = tune_per_class_thresholds(labels, probs, class_names=("a", "b"))

    assert set(tuned.keys()) == {"a", "b"}
    # a's only positive has the highest prob (0.3) among its column -> best threshold sits at/below it
    assert tuned["a"] <= 0.3
    # b's only positive (0.9) is far above every negative in its column -> a high threshold is best
    assert tuned["b"] > 0.3
