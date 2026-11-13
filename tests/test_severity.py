import numpy as np

from src.eval.severity import broadcast_rows_to_tiles, compute_metrics_by_severity


def _row(dirt_severity="none"):
    return {"dirt_severity": dirt_severity}


def test_compute_metrics_by_severity_hand_computed():
    # 6 samples, 1 class ("dirt"): 2 negatives, 2 low positives, 2 high
    # positives. Model scores everything perfectly.
    rows = [
        _row("none"), _row("none"),
        _row("low"), _row("low"),
        _row("high"), _row("high"),
    ]
    labels = np.array([[0], [0], [1], [1], [1], [1]])
    probs = np.array([[0.1], [0.2], [0.6], [0.7], [0.9], [0.95]])

    results = compute_metrics_by_severity(rows, labels, probs, ("dirt",), threshold=0.5)

    by_level = results["dirt"]
    assert set(by_level) == {"low", "high"}  # no "medium" samples in this split
    assert by_level["low"]["support"] == 2
    assert by_level["low"]["precision"] == 1.0 and by_level["low"]["recall"] == 1.0
    assert by_level["high"]["support"] == 2
    assert by_level["high"]["precision"] == 1.0 and by_level["high"]["recall"] == 1.0


def test_compute_metrics_by_severity_low_severity_is_harder_to_detect():
    # Same setup, but low-severity positives get LOW predicted probability
    # (the model struggles on faint distortions) -- low severity's recall
    # at threshold=0.5 should come out worse than high severity's.
    rows = [_row("none"), _row("none"), _row("low"), _row("low"), _row("high"), _row("high")]
    labels = np.array([[0], [0], [1], [1], [1], [1]])
    probs = np.array([[0.1], [0.2], [0.3], [0.4], [0.9], [0.95]])  # low positives score below 0.5

    results = compute_metrics_by_severity(rows, labels, probs, ("dirt",), threshold=0.5)

    assert results["dirt"]["low"]["recall"] == 0.0
    assert results["dirt"]["high"]["recall"] == 1.0


def test_compute_metrics_by_severity_supports_per_class_threshold_dict():
    rows = [_row("none"), _row("high")]
    labels = np.array([[0], [1]])
    probs = np.array([[0.4], [0.6]])

    results = compute_metrics_by_severity(rows, labels, probs, ("dirt",), threshold={"dirt": 0.5})

    assert results["dirt"]["high"]["threshold"] == 0.5


def test_broadcast_rows_to_tiles_repeats_each_row_grid_size_times():
    rows = [{"id": "a"}, {"id": "b"}]
    broadcast = broadcast_rows_to_tiles(rows, grid_h=2, grid_w=2)

    assert len(broadcast) == 2 * 4
    assert [r["id"] for r in broadcast] == ["a"] * 4 + ["b"] * 4


def test_broadcast_rows_to_tiles_matches_flatten_tiles_order():
    # Confirms the broadcast order lines up with
    # scripts.evaluate_stage_b.flatten_tiles's own (N,C,H,W) -> (N*H*W,C)
    # transpose+reshape order.
    from scripts.evaluate_stage_b import flatten_tiles

    rows = [{"id": "a"}, {"id": "b"}]
    n, c, h, w = 2, 1, 2, 2
    labels = np.zeros((n, c, h, w))
    labels[0] = 0.0
    labels[1] = 1.0  # every tile of image 1 is labeled 1, image 0 all 0

    labels_flat, _ = flatten_tiles(labels, labels.copy())
    broadcast = broadcast_rows_to_tiles(rows, grid_h=h, grid_w=w)

    for i, row in enumerate(broadcast):
        expected_label = 0.0 if row["id"] == "a" else 1.0
        assert labels_flat[i, 0] == expected_label
