import numpy as np

from scripts.visualize_stage_b_results import class_index_for_sample


def test_class_index_for_sample_active_class():
    row = {"dirt": "1", "water": "0", "scratch": "0"}
    kind, idx = class_index_for_sample(row, ("dirt", "water", "scratch"))
    assert kind == "dirt"
    assert idx == 0


def test_class_index_for_sample_picks_first_active_when_multiple_flagged():
    # Shouldn't happen for real Stage B data (never-combine-distortions
    # rule), but the function should still behave predictably if it did.
    row = {"dirt": "0", "water": "1", "scratch": "1"}
    kind, idx = class_index_for_sample(row, ("dirt", "water", "scratch"))
    assert kind == "water"
    assert idx == 1


def test_class_index_for_sample_clean_uses_most_confident_prediction():
    row = {"dirt": "0", "water": "0", "scratch": "0"}
    # probs_chw: (C, H, W) -- class 2 (scratch) has the highest max prob
    probs_chw = np.zeros((3, 2, 2), dtype=np.float32)
    probs_chw[2, 0, 1] = 0.87
    kind, idx = class_index_for_sample(row, ("dirt", "water", "scratch"), probs_chw)
    assert kind == "clean"
    assert idx == 2


def test_class_index_for_sample_clean_without_probs_defaults_to_zero():
    row = {"dirt": "0", "water": "0", "scratch": "0"}
    kind, idx = class_index_for_sample(row, ("dirt", "water", "scratch"))
    assert kind == "clean"
    assert idx == 0
