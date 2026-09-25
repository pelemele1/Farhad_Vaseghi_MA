import numpy as np
import pytest

from src.eval.general_channel import (
    general_channel_consistency,
    general_tile_labels,
    general_tile_probs,
)


def test_general_tile_labels_is_logical_or_across_classes():
    # sample 0: only class 0's top-left tile is active. sample 1: all-zero.
    tile_labels = np.zeros((2, 3, 2, 2), dtype=np.uint8)
    tile_labels[0, 0, 0, 0] = 1

    general = general_tile_labels(tile_labels)

    assert general.shape == (2, 1, 2, 2)
    assert general[0, 0, 0, 0] == 1
    assert general[0, 0, 0, 1] == 0
    assert general[1].sum() == 0


def test_general_tile_probs_noisy_or_matches_hand_calculation():
    # two classes each 0.5 at the same tile -> 1 - (0.5*0.5) = 0.75
    tile_probs = np.array([[[[0.5]], [[0.5]]]])  # shape (1, 2, 1, 1)

    general = general_tile_probs(tile_probs)

    assert general.shape == (1, 1, 1, 1)
    assert general[0, 0, 0, 0] == pytest.approx(0.75)


def test_general_tile_probs_zero_when_all_classes_zero():
    tile_probs = np.zeros((1, 3, 2, 2))

    general = general_tile_probs(tile_probs)

    assert np.all(general == 0.0)


def test_general_channel_consistency_is_one_on_agreeing_array():
    # every tile has exactly one class at prob 1.0, rest 0.0 -- thresholded
    # per-class OR and thresholded noisy-OR general channel always agree.
    tile_probs = np.zeros((4, 3, 2, 2))
    tile_probs[:, 0, 0, 0] = 1.0

    consistency = general_channel_consistency(tile_probs, threshold=0.5)

    assert consistency == pytest.approx(1.0)
