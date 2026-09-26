"""
Converts an existing Stage B/C dataset (built with --include-severity
--save-pixel-masks before Session 22) to severity-independent ground truth,
in place, without re-running the synthesis: each saved `masks/*.png`
channel was written as round(mask * SEVERITY_ALPHA[severity] * 255), so
dividing by the row's own per-class alpha (from metadata.csv's
`{class}_severity` column) recovers the full-strength mask up to uint8
rounding (worst case 0.5/0.3 = ~1.7/255 for "low"). `tile_labels.npy` is
then re-rasterized from the recovered masks with the dataset's own
thresholds -- exactly what the current build_stage_b_dataset writes
directly. Refuses to run twice (stage_b_meta.json's
`severity_independent_gt` flag).

Usage:
    python scripts/make_gt_severity_independent.py --data data/processed/stage_b_scratch15
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np

from src.soiling.dataset_builder import EFFECT_NAMES
from src.soiling.effects import SEVERITY_ALPHA
from src.soiling.tile_labels import rasterize_tile_label


def unscale_mask(saved, severities):
    """saved: (H, W, C) uint8 mask as written by build_stage_b_dataset.
    severities: per-class severity strings in EFFECT_NAMES order ("none"
    for an inactive class, whose channel is all-zero anyway). Returns the
    full-strength (H, W, C) uint8 mask."""
    out = saved.astype(np.float32)
    for i, severity in enumerate(severities):
        if severity in SEVERITY_ALPHA:
            out[..., i] /= SEVERITY_ALPHA[severity]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    data = Path(args.data)
    meta_path = data / "stage_b_meta.json"
    meta = json.loads(meta_path.read_text())
    if meta.get("severity_independent_gt"):
        sys.exit(f"{data} already has severity-independent ground truth -- nothing to do")
    if not meta.get("has_pixel_masks"):
        sys.exit(f"{data} has no masks/ folder -- rebuild it with the current builder instead")

    with open(data / "metadata.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    old_tiles = np.load(data / "tile_labels.npy")
    new_tiles = np.empty_like(old_tiles)
    grid_h, grid_w = meta["grid_h"], meta["grid_w"]

    for idx, row in enumerate(rows):
        mask_path = data / "masks" / (Path(row["path"]).stem + ".png")
        saved = cv.imread(str(mask_path), cv.IMREAD_UNCHANGED)
        fixed = unscale_mask(saved, [row[f"{name}_severity"] for name in EFFECT_NAMES])
        cv.imwrite(str(mask_path), fixed)
        for i, name in enumerate(EFFECT_NAMES):
            new_tiles[idx, i] = rasterize_tile_label(
                fixed[..., i].astype(np.float32) / 255.0, grid_h, grid_w, meta["thresholds"][name]
            )
        if (idx + 1) % 1000 == 0:
            print(f"{idx + 1}/{len(rows)}", flush=True)

    np.save(data / "tile_labels.npy", new_tiles)
    meta["severity_independent_gt"] = True
    meta_path.write_text(json.dumps(meta, indent=2))

    per_class = old_tiles.sum(axis=(0, 2, 3)), new_tiles.sum(axis=(0, 2, 3))
    for i, name in enumerate(EFFECT_NAMES):
        print(f"{name:<8} positive tiles: {per_class[0][i]} -> {per_class[1][i]}")


if __name__ == "__main__":
    main()
