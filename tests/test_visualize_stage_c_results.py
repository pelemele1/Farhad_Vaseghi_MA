from unittest import mock

import cv2 as cv
import numpy as np

from scripts.visualize_stage_c_results import plot_stage_c_report, select_report_rows
from src.data.stage_c_dataset import StageCDataset
from src.soiling.dataset_builder import build_stage_b_dataset


class _FakeDataset:
    """Minimal stand-in for StageCDataset -- select_report_rows only ever
    touches `.rows`, matching how select_diverse_sample_indices and
    class_index_for_sample are already unit-tested against plain row lists
    elsewhere in this project."""

    def __init__(self, rows):
        self.rows = rows


def test_select_report_rows_covers_every_kind():
    rows = [
        {"dirt": "1", "water": "0", "scratch": "0"},
        {"dirt": "0", "water": "1", "scratch": "0"},
        {"dirt": "0", "water": "0", "scratch": "1"},
        {"dirt": "0", "water": "0", "scratch": "0"},
    ]
    dataset = _FakeDataset(rows)
    class_names = ("dirt", "water", "scratch")
    probs = np.zeros((4, 3, 8, 8), dtype=np.float32)

    report_rows = select_report_rows(dataset, probs, class_names, per_class=1, seed=0)

    kinds = {kind for _, kind, _ in report_rows}
    assert kinds == {"dirt", "water", "scratch", "clean"}


def test_plot_stage_c_report_writes_expected_pages(tmp_path):
    # Matplotlib smoke test (no model/weights needed -- probs are random) --
    # confirms the figure runs end to end and paginates correctly, not that
    # the pixels are correct.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        img = rng.integers(0, 255, size=(96, 128, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    data_dir = tmp_path / "stage_c"
    build_stage_b_dataset(
        source_dir, data_dir, variants_per_image=4, seed=0,
        ratios=(0.25, 0.25, 0.5), img_size=64, save_pixel_masks=True,
    )

    dataset = StageCDataset(data_dir, split="test")
    class_names = dataset.class_names
    n = len(dataset)
    probs = rng.random((n, 3, 64, 64)).astype(np.float32)
    masks = rng.random((n, 3, 64, 64)).astype(np.float32)

    report_rows = select_report_rows(dataset, probs, class_names, per_class=1, seed=0)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    paths = plot_stage_c_report(
        dataset, report_rows, probs, masks, class_names, out_dir, rows_per_page=2,
    )

    assert len(paths) == -(-len(report_rows) // 2)  # ceil(n / rows_per_page)
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_plot_stage_c_report_column_count_with_and_without_gate(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img = rng.integers(0, 255, size=(96, 128, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    data_dir = tmp_path / "stage_c"
    build_stage_b_dataset(
        source_dir, data_dir, variants_per_image=4, seed=0,
        ratios=(0.5, 0.25, 0.25), img_size=64, save_pixel_masks=True,
    )

    dataset = StageCDataset(data_dir, split="train")
    class_names = dataset.class_names
    n = len(dataset)
    probs = rng.random((n, 3, 64, 64)).astype(np.float32)
    masks = rng.random((n, 3, 64, 64)).astype(np.float32)
    report_rows = [(0, "dirt", 0)]

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    import matplotlib.pyplot as plt
    with mock.patch.object(plt, "subplots", wraps=plt.subplots) as spy:
        plot_stage_c_report(dataset, report_rows, probs, masks, class_names, out_dir, rows_per_page=6)
    n_rows, n_cols = spy.call_args[0][:2]
    assert (n_rows, n_cols) == (1, 4)  # no gate: original/distorted/GT/predicted

    with mock.patch.object(plt, "subplots", wraps=plt.subplots) as spy:
        plot_stage_c_report(
            dataset, report_rows, probs, masks, class_names, out_dir, rows_per_page=6,
            gate_probs={0: 0.9}, gated_probs=probs,
        )
    n_rows, n_cols = spy.call_args[0][:2]
    assert (n_rows, n_cols) == (1, 5)  # +1 gated column
