import numpy as np

from scripts.evaluate_stage_b import flatten_tiles
from src.eval.metrics import compute_metrics


def test_flatten_tiles_shape_and_order():
    # N=2, C=2, H=1, W=3 -- small enough to hand-check every value's position.
    labels = np.array([
        [[[1, 0, 1]], [[0, 0, 1]]],  # sample 0: class0 tiles, class1 tiles
        [[[0, 1, 0]], [[1, 1, 0]]],  # sample 1
    ])
    probs = np.zeros_like(labels, dtype=np.float32)

    labels_flat, probs_flat = flatten_tiles(labels, probs)
    assert labels_flat.shape == (2 * 1 * 3, 2)  # N*H*W rows, C columns
    assert probs_flat.shape == labels_flat.shape

    # row order is sample-major, then H, then W (matches transpose(0,2,3,1).reshape)
    expected_rows = [
        (1, 0), (0, 0), (1, 1),  # sample 0's 3 tiles: (class0, class1)
        (0, 1), (1, 1), (0, 0),  # sample 1's 3 tiles
    ]
    assert [tuple(row) for row in labels_flat] == expected_rows


def test_flattened_tiles_feed_compute_metrics_correctly():
    # A grid where class "a" is positive in exactly the tiles predicted
    # positive, and class "b" is always negative -- checks the whole
    # flatten -> compute_metrics chain end to end, not just flatten_tiles
    # in isolation.
    labels = np.zeros((3, 2, 2, 2), dtype=np.uint8)
    labels[:, 0, 0, 0] = 1  # class "a" positive in the top-left tile of every sample
    probs = np.zeros((3, 2, 2, 2), dtype=np.float32)
    probs[:, 0, 0, 0] = 0.9  # confident correct prediction there
    probs[:, 0, 1, 1] = 0.9  # and one confident false positive elsewhere

    labels_flat, probs_flat = flatten_tiles(labels, probs)
    rows = compute_metrics(labels_flat, probs_flat, class_names=("a", "b"))

    row_a, row_b = rows[0], rows[1]
    assert row_a["support"] == 3  # one positive tile per sample, 3 samples
    assert row_a["recall"] == 1.0  # every true positive tile was caught
    assert row_a["precision"] < 1.0  # the extra false-positive tile hurts precision
    assert row_b["support"] == 0
