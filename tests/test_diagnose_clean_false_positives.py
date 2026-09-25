import numpy as np

from scripts.diagnose_clean_false_positives import clean_false_positive_rates, clean_row_mask


def test_clean_row_mask_identifies_all_zero_rows():
    rows = [
        {"dirt": "0", "water": "0", "scratch": "0"},
        {"dirt": "1", "water": "0", "scratch": "0"},
        {"dirt": "0", "water": "0", "scratch": "1"},
    ]
    mask = clean_row_mask(rows, ("dirt", "water", "scratch"))
    assert mask.tolist() == [True, False, False]


def test_clean_false_positive_rates_counts_max_prob_crossings():
    class_names = ("dirt", "water", "scratch")
    rows = [
        {"dirt": "0", "water": "0", "scratch": "0"},  # clean, dirt max prob 0.9 -> FP
        {"dirt": "0", "water": "0", "scratch": "0"},  # clean, dirt max prob 0.1 -> not FP
        {"dirt": "1", "water": "0", "scratch": "0"},  # not clean, excluded regardless of prob
    ]
    # (N=3, C=3, H=2, W=2)
    probs = np.zeros((3, 3, 2, 2), dtype=np.float32)
    probs[0, 0, 0, 0] = 0.9  # sample 0, dirt channel
    probs[1, 0, 0, 0] = 0.1  # sample 1, dirt channel
    probs[2, 0, 0, 0] = 0.95  # sample 2 is not clean, must be excluded
    labels = np.zeros_like(probs)

    rates = clean_false_positive_rates(probs, labels, rows, class_names, threshold=0.5)

    fp_rate, n_clean = rates["dirt"]
    assert n_clean == 2
    assert fp_rate == 0.5  # 1 of the 2 clean images crossed the threshold
    assert rates["water"][0] == 0.0
    assert rates["scratch"][0] == 0.0


def test_clean_false_positive_rates_supports_per_class_threshold_dict():
    class_names = ("dirt", "water")
    rows = [{"dirt": "0", "water": "0"}]
    probs = np.zeros((1, 2, 2, 2), dtype=np.float32)
    probs[0, 0] = 0.4  # dirt max prob 0.4
    probs[0, 1] = 0.6  # water max prob 0.6
    labels = np.zeros_like(probs)

    # dirt threshold 0.3 -> 0.4 crosses -> FP; water threshold 0.7 -> 0.6 doesn't cross -> not FP
    rates = clean_false_positive_rates(probs, labels, rows, class_names, threshold={"dirt": 0.3, "water": 0.7})

    assert rates["dirt"][0] == 1.0
    assert rates["water"][0] == 0.0


def test_clean_false_positive_rates_nan_when_no_clean_rows():
    class_names = ("dirt",)
    rows = [{"dirt": "1"}]
    probs = np.zeros((1, 1, 2, 2), dtype=np.float32)
    labels = np.zeros_like(probs)

    rates = clean_false_positive_rates(probs, labels, rows, class_names, threshold=0.5)

    fp_rate, n_clean = rates["dirt"]
    assert n_clean == 0
    assert np.isnan(fp_rate)
