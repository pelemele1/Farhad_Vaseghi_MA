import subprocess
import sys

import cv2 as cv
import numpy as np


def test_build_stage_b_dataset_cli_runs_end_to_end(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        img = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    out_dir = tmp_path / "stage_b"
    result = subprocess.run(
        [sys.executable, "scripts/build_stage_b_dataset.py",
         "--source", str(source_dir), "--out", str(out_dir),
         "--variants", "4", "--seed", "0", "--img-size", "64"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Wrote 16 images" in result.stdout
    assert "grid 2x2" in result.stdout  # img-size 64 // 32 = 2
    for name in ("dirt", "water", "scratch"):
        assert name in result.stdout

    assert len(list((out_dir / "images").glob("*.jpg"))) == 16
    assert (out_dir / "metadata.csv").exists()
    assert (out_dir / "tile_labels.npy").exists()
    assert (out_dir / "stage_b_meta.json").exists()


def test_build_stage_b_dataset_cli_include_combos(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    out_dir = tmp_path / "stage_b_combo"
    result = subprocess.run(
        [sys.executable, "scripts/build_stage_b_dataset.py",
         "--source", str(source_dir), "--out", str(out_dir),
         "--variants", "8", "--seed", "0", "--img-size", "64", "--include-combos"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Wrote 16 images" in result.stdout  # 2 sources * 8 variants

    assert len(list((out_dir / "images").glob("*.jpg"))) == 16
    assert (out_dir / "tile_labels.npy").exists()


def test_build_stage_b_dataset_cli_per_class_threshold_overrides(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    out_dir = tmp_path / "stage_b_thresh"
    result = subprocess.run(
        [sys.executable, "scripts/build_stage_b_dataset.py",
         "--source", str(source_dir), "--out", str(out_dir),
         "--variants", "4", "--seed", "0", "--img-size", "64",
         "--dirt-threshold", "0.20", "--water-threshold", "0.25", "--scratch-threshold", "0.02"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = {line.strip().split(":", 1)[0]: line for line in result.stdout.splitlines() if line.strip().startswith(("dirt:", "water:", "scratch:"))}
    assert "threshold=0.2" in lines["dirt"] and "threshold=0.25" not in lines["dirt"]
    assert "threshold=0.25" in lines["water"]
    assert "threshold=0.02" in lines["scratch"]


def test_build_stage_b_dataset_cli_include_severity(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), img)

    out_dir = tmp_path / "stage_b_severity"
    result = subprocess.run(
        [sys.executable, "scripts/build_stage_b_dataset.py",
         "--source", str(source_dir), "--out", str(out_dir),
         "--variants", "10", "--seed", "0", "--img-size", "64", "--include-severity"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Wrote 20 images" in result.stdout  # 2 sources * 10 variants
    assert "severity:" in result.stdout

    assert len(list((out_dir / "images").glob("*.jpg"))) == 20
    assert (out_dir / "tile_labels.npy").exists()


def test_build_stage_b_dataset_cli_rejects_img_size_not_multiple_of_32(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    cv.imwrite(str(source_dir / "00000000.jpg"),
               np.random.default_rng(0).integers(0, 255, size=(64, 96, 3), dtype=np.uint8))

    result = subprocess.run(
        [sys.executable, "scripts/build_stage_b_dataset.py",
         "--source", str(source_dir), "--out", str(tmp_path / "out"),
         "--img-size", "100"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
