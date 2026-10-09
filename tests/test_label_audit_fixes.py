"""Session 26 label audit: thin-water mask floor, invisible-effect labels, and
the one-time in-place fix for existing datasets."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from src.soiling import dataset_builder
from src.soiling.dataset_builder import EFFECT_NAMES, apply_effect_combo_with_masks, build_stage_b_dataset
from src.soiling.effects import (
    MIN_VISIBLE_PIXELS,
    THIN_WATER_MASK_FLOOR,
    add_water,
    is_visible,
    remove_thin_water_floor,
    visible_pixel_count,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _photo(seed=0, size=96):
    return np.random.default_rng(seed).integers(40, 200, size=(size, size, 3), dtype=np.uint8)


def test_thin_water_mask_has_no_floor():
    _, mask = add_water(_photo(size=256), seed=0, mechanism="thin")
    assert mask.min() < 0.01  # was >= THIN_WATER_MASK_FLOOR everywhere before the fix
    assert mask.max() > 0.5


def test_remove_thin_water_floor_maps_floor_to_zero_and_keeps_one():
    out = remove_thin_water_floor(np.array([THIN_WATER_MASK_FLOOR, 0.6, 1.0]))
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_visible_pixel_count_ignores_changes_outside_mask_and_tiny_changes():
    before = np.full((10, 10, 3), 100, dtype=np.uint8)
    after = before.copy()
    after[:5] += 50   # visible change, top half
    after[5:] += 5    # below MIN_PIXEL_CHANGE, bottom half
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[:, :5] = 1.0  # left half
    assert visible_pixel_count(before, after, mask) == 25
    assert not is_visible(before, after, np.zeros((10, 10)))


def test_invisible_effect_is_redrawn_and_never_labeled_when_always_invisible(monkeypatch):
    image = _photo()
    calls = []

    def invisible_scratch(img, seed=None):
        calls.append(seed)
        return img.copy(), np.ones(img.shape[:2], dtype=np.float32)

    monkeypatch.setattr(dataset_builder, "add_scratch", invisible_scratch)
    out, labels, masks = apply_effect_combo_with_masks(image, (False, False, True), {"scratch": 7}, {"scratch": "high"})
    assert len(calls) == dataset_builder.MAX_EFFECT_ATTEMPTS
    assert len(set(calls)) == len(calls)  # a fresh seed every attempt
    assert labels["scratch"] == 0 and labels["scratch_severity"] == "none"
    assert masks["scratch"].max() == 0
    assert np.array_equal(out, image)


def test_visible_effect_is_labeled_on_first_attempt(monkeypatch):
    image = _photo()

    def bright_scratch(img, seed=None):
        out = img.copy()
        out[:20, :20] = 255
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        mask[:20, :20] = 1.0
        return out, mask

    monkeypatch.setattr(dataset_builder, "add_scratch", bright_scratch)
    _, labels, masks = apply_effect_combo_with_masks(image, (False, False, True), {"scratch": 7}, {"scratch": "high"})
    assert labels["scratch"] == 1 and masks["scratch"].sum() == 400
    assert 400 >= MIN_VISIBLE_PIXELS


def test_fix_script_removes_floor_and_invisible_labels_then_refuses_rerun(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for i in range(2):
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), _photo(i, 64))
    data = tmp_path / "stage_b"
    rows, _ = build_stage_b_dataset(source_dir, data, variants_per_image=4, seed=0, img_size=64,
                                    save_pixel_masks=True)

    # Recreate the two pre-audit defects: a floored water mask and a label
    # for a scratch that left the image unchanged.
    water_row = next(r for r in rows if int(r["water"]))
    scratch_row = next(r for r in rows if int(r["scratch"]))
    w_path = data / "masks" / (Path(water_row["path"]).stem + ".png")
    original_w = cv.imread(str(w_path), cv.IMREAD_UNCHANGED)[..., 1].astype(int)
    w = cv.imread(str(w_path), cv.IMREAD_UNCHANGED).astype(np.float32) / 255
    w[..., 1] = THIN_WATER_MASK_FLOOR + (1 - THIN_WATER_MASK_FLOOR) * w[..., 1]
    cv.imwrite(str(w_path), np.round(w * 255).astype(np.uint8))
    clean = next(r for r in rows if r["source_id"] == scratch_row["source_id"] and not any(int(r[c]) for c in EFFECT_NAMES))
    cv.imwrite(str(data / scratch_row["path"]), cv.imread(str(data / clean["path"])))
    # ... and a dirt label whose mask came back all-zero (vendored-effect failure).
    dirt_row = next(r for r in rows if int(r["dirt"]))
    d_path = data / "masks" / (Path(dirt_row["path"]).stem + ".png")
    d = cv.imread(str(d_path), cv.IMREAD_UNCHANGED)
    d[..., 0] = 0
    cv.imwrite(str(d_path), d)

    run = [sys.executable, "scripts/fix_dataset_labels.py", "--stage-b", str(data), "--source", str(source_dir)]
    result = subprocess.run(run, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    fixed_w = cv.imread(str(w_path), cv.IMREAD_UNCHANGED)[..., 1].astype(int)
    assert np.abs(fixed_w - original_w).max() <= 2  # floor removed, original mask back
    with open(data / "metadata.csv", newline="") as f:
        fixed = {r["path"]: r for r in csv.DictReader(f)}
    assert fixed[scratch_row["path"]]["scratch"] == "0"
    assert fixed[scratch_row["path"]]["scratch_severity"] == "none"
    s_mask = cv.imread(str(data / "masks" / (Path(scratch_row["path"]).stem + ".png")), cv.IMREAD_UNCHANGED)
    assert s_mask[..., 2].max() == 0
    idx = [r["path"] for r in rows].index(scratch_row["path"])
    assert np.load(data / "tile_labels.npy")[idx, 2].sum() == 0
    assert fixed[dirt_row["path"]]["dirt"] == "1"  # re-rendered, not unlabeled
    assert cv.imread(str(d_path), cv.IMREAD_UNCHANGED)[..., 0].max() > 0
    assert (data / "images_before_label_fix" / Path(dirt_row["path"]).name).exists()
    assert json.loads((data / "stage_b_meta.json").read_text())["label_audit_fix"] is True
    assert (data / "metadata.csv.before_label_fix").exists()

    again = subprocess.run(run, cwd=REPO_ROOT, capture_output=True, text=True)
    assert again.returncode != 0
