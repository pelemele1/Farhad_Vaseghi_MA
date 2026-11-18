import subprocess
import sys

import cv2 as cv
import numpy as np

from src.soiling.dataset_builder import EFFECT_NAMES

# Seconds a small test build may take. The 6 classes (Session 31) draw slower than
# the old 3, and on the HPC every test file runs at once on a shared node.
_BUILD_TIMEOUT = 600


def test_build_stage_a_dataset_cli_runs_end_to_end(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        img = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    out_dir = tmp_path / "stage_a"
    result = subprocess.run(
        [sys.executable, "scripts/build_stage_a_dataset.py",
         "--source", str(source_dir), "--out", str(out_dir),
         "--variants", "4", "--seed", "0"],
        capture_output=True, text=True, timeout=_BUILD_TIMEOUT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Wrote 16 images" in result.stdout
    for name in EFFECT_NAMES:
        assert name in result.stdout

    assert len(list((out_dir / "images").glob("*.jpg"))) == 16
    assert (out_dir / "metadata.csv").exists()


def test_build_stage_a_dataset_cli_include_combos(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    out_dir = tmp_path / "stage_a_combo"
    result = subprocess.run(
        [sys.executable, "scripts/build_stage_a_dataset.py",
         "--source", str(source_dir), "--out", str(out_dir),
         "--variants", "8", "--seed", "0", "--include-combos"],
        capture_output=True, text=True, timeout=_BUILD_TIMEOUT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Wrote 16 images" in result.stdout  # 2 sources * 8 variants

    assert len(list((out_dir / "images").glob("*.jpg"))) == 16
    assert (out_dir / "metadata.csv").exists()


def test_build_stage_a_dataset_cli_include_severity(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    out_dir = tmp_path / "stage_a_severity"
    result = subprocess.run(
        [sys.executable, "scripts/build_stage_a_dataset.py",
         "--source", str(source_dir), "--out", str(out_dir),
         "--variants", "10", "--seed", "0", "--include-severity"],
        capture_output=True, text=True, timeout=_BUILD_TIMEOUT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Wrote 20 images" in result.stdout  # 2 sources * 10 variants
    assert "severity:" in result.stdout
    for level in ("low", "medium", "high"):
        assert level in result.stdout

    assert len(list((out_dir / "images").glob("*.jpg"))) == 20
    assert (out_dir / "metadata.csv").exists()
