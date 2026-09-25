from unittest import mock

import cv2 as cv
import numpy as np

from scripts.evaluate_stage_b import flatten_tiles
from scripts.visualize_stage_a_results import select_diverse_sample_indices
from scripts.visualize_stage_b_results import (
    _compute_per_class_pr_curves,
    build_report_rows,
    class_index_for_sample,
    find_combo_sample_index,
    max_prob_per_image,
    plot_stage_b_eight_column_report,
    select_report_rows,
)
from src.data.stage_b_dataset import StageBDataset
from src.soiling.dataset_builder import build_stage_b_dataset


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


# --- Per-sample report figure helpers (Session 20) ------------------------


class _FakeDataset:
    """Minimal stand-in for StageBDataset -- build_report_rows/
    select_report_rows only ever touch `.rows`, matching how
    class_index_for_sample and select_diverse_sample_indices are already
    unit-tested against plain row lists elsewhere in this project."""

    def __init__(self, rows):
        self.rows = rows


def test_build_report_rows_matches_class_index_for_sample():
    rows = [
        {"dirt": "1", "water": "0", "scratch": "0"},
        {"dirt": "0", "water": "1", "scratch": "0"},
    ]
    dataset = _FakeDataset(rows)
    class_names = ("dirt", "water", "scratch")
    probs = np.zeros((2, 3, 2, 2), dtype=np.float32)

    report_rows = build_report_rows(dataset, [0, 1], probs, class_names)

    assert report_rows == [(0, "dirt", 0), (1, "water", 1)]


def test_select_report_rows_covers_every_kind():
    rows = [
        {"dirt": "1", "water": "0", "scratch": "0"},
        {"dirt": "0", "water": "1", "scratch": "0"},
        {"dirt": "0", "water": "0", "scratch": "1"},
        {"dirt": "0", "water": "0", "scratch": "0"},
    ]
    dataset = _FakeDataset(rows)
    class_names = ("dirt", "water", "scratch")
    probs = np.zeros((4, 3, 2, 2), dtype=np.float32)

    report_rows = select_report_rows(dataset, probs, class_names, per_class=1, seed=0)

    kinds = {kind for _, kind, _ in report_rows}
    assert kinds == {"dirt", "water", "scratch", "clean"}


def test_compute_per_class_pr_curves_skips_degenerate_class():
    # class 1 (index 1) is all-negative -- no positives, curve undefined.
    labels_flat = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    probs_flat = np.array([[0.8, 0.1], [0.2, 0.3], [0.7, 0.4], [0.1, 0.2]])

    curves = _compute_per_class_pr_curves(labels_flat, probs_flat, ("a", "b"))

    assert curves["a"] is not None
    assert curves["b"] is None


def test_compute_per_class_pr_curves_returns_recall_precision_ap():
    # perfect separation: class 0 positives (index 1,2) both score higher
    # than the negatives -- AP should be 1.0.
    labels_flat = np.array([[1], [0], [1], [0]])
    probs_flat = np.array([[0.9], [0.1], [0.8], [0.2]])

    curves = _compute_per_class_pr_curves(labels_flat, probs_flat, ("a",))

    recall, precision, ap = curves["a"]
    assert ap == 1.0
    assert len(recall) == len(precision)


def test_plot_stage_b_eight_column_report_writes_expected_pages(tmp_path):
    # Matplotlib smoke test (no model/weights needed -- probs are random) --
    # confirms the figure runs end to end and paginates correctly, not that
    # the pixels are correct.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        img = rng.integers(0, 255, size=(96, 128, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    data_dir = tmp_path / "stage_b"
    build_stage_b_dataset(
        source_dir, data_dir, variants_per_image=4, seed=0,
        ratios=(0.25, 0.25, 0.5), img_size=64,
    )

    dataset = StageBDataset(data_dir, split="test")
    class_names = dataset.class_names
    labels = dataset.tile_labels.astype(np.float32)
    probs = rng.random(labels.shape).astype(np.float32)

    indices = select_diverse_sample_indices(dataset.rows, class_names, per_kind=1, seed=0)
    report_rows = build_report_rows(dataset, indices, probs, class_names)
    labels_flat, probs_flat = flatten_tiles(labels, probs)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    paths = plot_stage_b_eight_column_report(
        dataset, report_rows, probs, labels, class_names,
        labels_flat, probs_flat, out_dir, rows_per_page=2,
    )

    assert len(paths) == -(-len(report_rows) // 2)  # ceil(n / rows_per_page)
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_plot_stage_b_eight_column_report_shows_all_classes_probability(tmp_path):
    # Confirms the figure actually renders 8 columns (3 classes' raw
    # probability tiles plus a dedicated gated column, not just the
    # dominant one) -- checks the axes grid shape directly rather than
    # pixel content.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img = rng.integers(0, 255, size=(96, 128, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    data_dir = tmp_path / "stage_b"
    build_stage_b_dataset(
        source_dir, data_dir, variants_per_image=4, seed=0,
        ratios=(0.5, 0.25, 0.25), img_size=64,
    )

    dataset = StageBDataset(data_dir, split="train")
    class_names = dataset.class_names
    assert len(class_names) == 3
    labels = dataset.tile_labels.astype(np.float32)
    probs = rng.random(labels.shape).astype(np.float32)

    report_rows = [(0, "dirt", 0)]
    labels_flat, probs_flat = flatten_tiles(labels, probs)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    import matplotlib.pyplot as plt
    with mock.patch.object(plt, "subplots", wraps=plt.subplots) as spy:
        plot_stage_b_eight_column_report(
            dataset, report_rows, probs, labels, class_names,
            labels_flat, probs_flat, out_dir, rows_per_page=6,
        )
    n_rows, n_cols = spy.call_args[0][:2]
    assert (n_rows, n_cols) == (1, 8)  # 1 row, 8 columns: orig/distorted/GT/3xprob/PR/gated


def test_plot_stage_b_eight_column_report_column_8_reflects_gating(tmp_path):
    # Column 8 should show the GATED grid (all-zero for a gated-out image)
    # when gated_tile_probs is given, distinct from column 4-6's raw view.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img = rng.integers(0, 255, size=(96, 128, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    data_dir = tmp_path / "stage_b"
    build_stage_b_dataset(
        source_dir, data_dir, variants_per_image=4, seed=0,
        ratios=(0.5, 0.25, 0.25), img_size=64,
    )

    dataset = StageBDataset(data_dir, split="train")
    class_names = dataset.class_names
    labels = dataset.tile_labels.astype(np.float32)
    raw_probs = np.ones_like(labels) * 0.7  # nonzero everywhere
    gated_tile_probs = np.zeros_like(labels)  # this image was gated out

    report_rows = [(0, "dirt", 0)]
    labels_flat, probs_flat = flatten_tiles(labels, raw_probs)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    import matplotlib.pyplot as plt
    captured = {}
    original_imshow = plt.Axes.imshow

    def _spy_imshow(self, data, *args, **kwargs):
        captured.setdefault("calls", []).append(np.array(data))
        return original_imshow(self, data, *args, **kwargs)

    with mock.patch.object(plt.Axes, "imshow", _spy_imshow):
        plot_stage_b_eight_column_report(
            dataset, report_rows, raw_probs, labels, class_names,
            labels_flat, probs_flat, out_dir, rows_per_page=6,
            gate_probs={0: 0.1}, gated_tile_probs=gated_tile_probs,
        )

    # imshow call order per row: 0=clean image, 1=distorted image, 2=GT,
    # 3=dirt raw (dominant), 4=water raw, 5=scratch raw, [PR curve uses
    # .plot(), not .imshow()], 6=gated column -- the last call.
    last_grid = captured["calls"][-1]
    assert np.all(last_grid == 0.0)
    dominant_raw_grid = captured["calls"][3]
    assert np.all(dominant_raw_grid == 0.7)


def test_max_prob_per_image_reduces_over_the_tile_grid():
    probs = np.zeros((2, 3, 2, 2), dtype=np.float32)
    probs[0, 0] = [[0.1, 0.9], [0.2, 0.3]]  # image 0, class 0 -> max 0.9
    probs[1, 2] = [[0.4, 0.4], [0.4, 0.5]]  # image 1, class 2 -> max 0.5

    result = max_prob_per_image(probs)

    assert result.shape == (2, 3)
    assert result[0, 0] == 0.9
    assert result[1, 2] == 0.5
    assert result[0, 1] == 0.0  # untouched channel stays 0
