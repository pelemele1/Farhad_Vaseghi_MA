import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
import torch

from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import StageBDistortionHead
from src.soiling.dataset_builder import build_stage_b_dataset

_WEIGHTS = Path("weights/yolo11m.pt")

pytestmark = pytest.mark.skipif(
    not _WEIGHTS.exists(),
    reason="requires weights/yolo11m.pt (COCO-pretrained YOLOv11m) downloaded locally",
)


def test_evaluate_stage_b_runs_end_to_end(tmp_path):
    # A fresh (untrained) head is enough to exercise the eval pipeline --
    # this checks the script runs and reports every class over real tile
    # counts, not that the numbers are good.
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

    backbone = FrozenYOLOBackbone(str(_WEIGHTS))
    head = StageBDistortionHead(in_channels=backbone.out_channels)
    ckpt_path = tmp_path / "stage_b_head.pt"
    torch.save({"head_state_dict": head.state_dict(), "class_names": head.class_names}, ckpt_path)

    result = subprocess.run(
        [sys.executable, "scripts/evaluate_stage_b.py",
         "--checkpoint", str(ckpt_path), "--data", str(data_dir),
         "--weights", str(_WEIGHTS), "--split", "test"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tiles" in result.stdout
    for name in ("dirt", "water", "scratch"):
        assert name in result.stdout
