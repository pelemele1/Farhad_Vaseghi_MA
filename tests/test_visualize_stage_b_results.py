import numpy as np

from scripts.visualize_stage_b_results import class_index_for_sample, find_combo_sample_index


def test_class_index_for_sample_active_class():
    row = {"dirt": "1", "water": "0", "scratch": "0"}
    kind, idx = class_index_for_sample(row, ("dirt", "water", "scratch"))
    assert kind == "dirt"
    assert idx == 0


def test_class_index_for_sample_picks_first_active_when_multiple_flagged():
    # Combo variants (Session 19+) do have multiple classes flagged at once --
    # class_index_for_sample only ever shows one (the overlay is single-class);
    # find_combo_sample_index (below) is what finds a genuine combo row to
    # show both classes for.
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


def test_find_combo_sample_index_finds_first_multi_active_row():
    rows = [
        {"dirt": "1", "water": "0", "scratch": "0"},
        {"dirt": "0", "water": "0", "scratch": "0"},
        {"dirt": "1", "water": "1", "scratch": "0"},  # first combo row
        {"dirt": "1", "water": "1", "scratch": "1"},
    ]
    idx, active = find_combo_sample_index(rows, ("dirt", "water", "scratch"))
    assert idx == 2
    assert active == ["dirt", "water"]


def test_find_combo_sample_index_returns_none_when_no_combos_present():
    rows = [
        {"dirt": "1", "water": "0", "scratch": "0"},
        {"dirt": "0", "water": "0", "scratch": "0"},
        {"dirt": "0", "water": "1", "scratch": "0"},
    ]
    idx, active = find_combo_sample_index(rows, ("dirt", "water", "scratch"))
    assert idx is None
    assert active == []
