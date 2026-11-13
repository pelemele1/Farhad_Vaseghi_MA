import json
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from scripts.make_gt_severity_independent import unscale_mask
from src.soiling.dataset_builder import EFFECT_NAMES, build_stage_b_dataset
from src.soiling.effects import SEVERITY_ALPHA

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_unscale_mask_divides_each_channel_by_its_own_alpha():
    full = np.array([[[200, 100, 0]]], dtype=np.uint8)
    saved = np.round(full * np.array([SEVERITY_ALPHA["low"], SEVERITY_ALPHA["medium"], 1.0])).astype(np.uint8)
    out = unscale_mask(saved, ["low", "medium", "none"])
    assert np.abs(out.astype(int) - full.astype(int)).max() <= 2
    assert out[0, 0, 2] == 0


def test_migration_restores_full_strength_masks_and_relabels_tiles(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        cv.imwrite(str(source_dir / f"{i:08d}.jpg"), rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
    out_dir = tmp_path / "out"
    rows, tiles = build_stage_b_dataset(
        source_dir, out_dir, variants_per_image=10, seed=0, img_size=64,
        include_severity=True, save_pixel_masks=True,
    )

    # Recreate the pre-Session-22 on-disk state: severity-scaled masks.
    originals = {}
    for row in rows:
        path = out_dir / "masks" / (Path(row["path"]).stem + ".png")
        full = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        originals[path] = full
        alphas = [SEVERITY_ALPHA.get(row[f"{n}_severity"], 1.0) for n in EFFECT_NAMES]
        cv.imwrite(str(path), np.round(full * np.array(alphas)).astype(np.uint8))
    meta = json.loads((out_dir / "stage_b_meta.json").read_text())
    meta["severity_independent_gt"] = False
    (out_dir / "stage_b_meta.json").write_text(json.dumps(meta))

    result = subprocess.run(
        [sys.executable, "scripts/make_gt_severity_independent.py", "--data", str(out_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    for path, full in originals.items():
        restored = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        assert np.abs(restored.astype(int) - full.astype(int)).max() <= 2
    assert json.loads((out_dir / "stage_b_meta.json").read_text())["severity_independent_gt"] is True
    # Near-threshold tiles may flip from uint8 rounding; the bulk must match a direct build.
    assert (np.load(out_dir / "tile_labels.npy") == tiles).mean() > 0.97

    again = subprocess.run(
        [sys.executable, "scripts/make_gt_severity_independent.py", "--data", str(out_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert again.returncode != 0
