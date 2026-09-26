import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from src.soiling.dataset_builder import build_stage_a_dataset

_WEIGHTS = Path("weights/yolo11m.pt")

pytestmark = pytest.mark.skipif(
    not _WEIGHTS.exists(),
    reason="requires weights/yolo11m.pt (COCO-pretrained YOLOv11m) downloaded locally",
)


def test_smoke_test_runs_end_to_end(tmp_path):
    # --smoke-test is the only training this project runs from the local/
    # dev side (standing rule: Claude prepares code, doesn't train it) --
    # this just verifies the backbone -> head -> loss -> optimizer.step()
    # pipeline actually wires together and runs, not that it learns anything.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        img = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    data_dir = tmp_path / "stage_a"
    build_stage_a_dataset(
        source_dir, data_dir, variants_per_image=4, seed=0, ratios=(0.5, 0.25, 0.25)
    )

    result = subprocess.run(
        [sys.executable, "scripts/train_stage_a.py", "--smoke-test",
         "--data", str(data_dir), "--weights", str(_WEIGHTS)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "epoch 1/1" in result.stdout


def test_low_severity_sample_weights_upweights_rows_with_any_low_effect():
    from scripts.train_stage_a import low_severity_sample_weights

    rows = [
        {"dirt_severity": "low", "water_severity": "none", "scratch_severity": "none"},
        {"dirt_severity": "high", "water_severity": "low", "scratch_severity": "none"},
        {"dirt_severity": "none", "water_severity": "none", "scratch_severity": "none"},
        {"dirt_severity": "medium", "water_severity": "none", "scratch_severity": "high"},
    ]
    assert low_severity_sample_weights(rows, 3.0) == [3.0, 3.0, 1.0, 1.0]
