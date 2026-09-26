import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
import torch

from scripts.evaluate_stage_c import pool_to_grid
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import ImpairedGateHead, StageCDistortionHead
from src.soiling.dataset_builder import build_stage_b_dataset

_WEIGHTS = Path("weights/yolo11m.pt")


def test_pool_to_grid_area_averages_each_class_independently():
    # 1 sample, 2 classes, 4x4 -> pool to 2x2: each output cell is the mean
    # of a 2x2 block, same area-average semantics as
    # tile_labels.py::rasterize_tile_label's INTER_AREA downsampling.
    array = np.zeros((1, 2, 4, 4), dtype=np.float32)
    array[0, 0] = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # top-left quadrant all 1
    array[0, 1] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]  # bottom-right quadrant all 1

    pooled = pool_to_grid(array, grid_size=2)

    assert pooled.shape == (1, 2, 2, 2)
    assert np.allclose(pooled[0, 0], [[1, 0], [0, 0]])
    assert np.allclose(pooled[0, 1], [[0, 0], [0, 1]])


def test_pool_to_grid_handles_multiple_samples_independently():
    array = np.zeros((2, 1, 4, 4), dtype=np.float32)
    array[0, 0, :, :] = 1.0  # sample 0: all-positive
    array[1, 0, :, :] = 0.0  # sample 1: all-negative

    pooled = pool_to_grid(array, grid_size=2)
    assert np.allclose(pooled[0], 1.0)
    assert np.allclose(pooled[1], 0.0)


pytestmark = pytest.mark.skipif(
    not _WEIGHTS.exists(),
    reason="requires weights/yolo11m.pt (COCO-pretrained YOLOv11m) downloaded locally",
)


def test_evaluate_stage_c_runs_end_to_end(tmp_path):
    # A fresh (untrained) head is enough to exercise the eval pipeline --
    # this checks the script runs and reports every class, not that the
    # numbers are good.
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

    backbone = FrozenYOLOBackbone(str(_WEIGHTS))
    head = StageCDistortionHead(in_channels=backbone.out_channels)
    ckpt_path = tmp_path / "stage_c_head.pt"
    torch.save({"head_state_dict": head.state_dict(), "class_names": head.class_names}, ckpt_path)

    result = subprocess.run(
        [sys.executable, "scripts/evaluate_stage_c.py",
         "--checkpoint", str(ckpt_path), "--data", str(data_dir),
         "--weights", str(_WEIGHTS), "--split", "test", "--img-size", "64", "--eval-grid", "16"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("dirt", "water", "scratch"):
        assert name in result.stdout


def test_evaluate_stage_c_with_gate_checkpoint_prints_both_tables(tmp_path):
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

    backbone = FrozenYOLOBackbone(str(_WEIGHTS))
    head = StageCDistortionHead(in_channels=backbone.out_channels)
    ckpt_path = tmp_path / "stage_c_head.pt"
    torch.save({"head_state_dict": head.state_dict(), "class_names": head.class_names}, ckpt_path)

    gate_head = ImpairedGateHead(in_channels=backbone.out_channels)
    gate_ckpt_path = tmp_path / "impaired_gate_head.pt"
    torch.save({"head_state_dict": gate_head.state_dict(), "class_names": gate_head.class_names}, gate_ckpt_path)

    result = subprocess.run(
        [sys.executable, "scripts/evaluate_stage_c.py",
         "--checkpoint", str(ckpt_path), "--data", str(data_dir),
         "--weights", str(_WEIGHTS), "--split", "test", "--img-size", "64", "--eval-grid", "16",
         "--gate-checkpoint", str(gate_ckpt_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[ungated]" in result.stdout
    assert "[gated]" in result.stdout


def test_evaluate_stage_c_by_severity_prints_breakdown(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        img = rng.integers(0, 255, size=(96, 128, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    data_dir = tmp_path / "stage_c_severity"
    build_stage_b_dataset(
        source_dir, data_dir, variants_per_image=10, seed=0,
        ratios=(0.25, 0.25, 0.5), img_size=64, include_severity=True, save_pixel_masks=True,
    )

    backbone = FrozenYOLOBackbone(str(_WEIGHTS))
    head = StageCDistortionHead(in_channels=backbone.out_channels)
    ckpt_path = tmp_path / "stage_c_head.pt"
    torch.save({"head_state_dict": head.state_dict(), "class_names": head.class_names}, ckpt_path)

    result = subprocess.run(
        [sys.executable, "scripts/evaluate_stage_c.py",
         "--checkpoint", str(ckpt_path), "--data", str(data_dir),
         "--weights", str(_WEIGHTS), "--split", "test", "--img-size", "64", "--eval-grid", "16",
         "--by-severity"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Per-severity breakdown" in result.stdout
    for level in ("low", "medium", "high"):
        assert level in result.stdout
