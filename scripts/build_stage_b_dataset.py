"""
Precompute the Stage B tile-grid dataset (architecture.md §2) from a source
image folder (the MIO-TCD pilot subset by default).

Usage:
    python scripts/build_stage_b_dataset.py --source data/raw/mio_tcd/images \
        --out data/processed/stage_b --variants 4 --seed 0 --img-size 512
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.soiling.dataset_builder import DEFAULT_TILE_THRESHOLDS, EFFECT_NAMES, build_stage_b_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Folder of clean source images")
    parser.add_argument("--out", default="data/processed/stage_b", help="Output directory")
    parser.add_argument("--variants", type=int, default=4,
                         help="Variants per source image (cycles clean/dirt/water/scratch, balanced)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=512,
                         help="Must be a multiple of 32 (P5 stride) -- sets the tile grid size (img-size // 32)")
    parser.add_argument("--dirt-threshold", type=float, default=None,
                         help="Override the dirt tile-coverage threshold (default "
                         f"{DEFAULT_TILE_THRESHOLDS['dirt']}, see scripts/diagnose_tile_thresholds.py "
                         "--effect dirt).")
    parser.add_argument("--water-threshold", type=float, default=None,
                         help="Override the water tile-coverage threshold (default "
                         f"{DEFAULT_TILE_THRESHOLDS['water']}, see scripts/diagnose_tile_thresholds.py "
                         "--effect water).")
    parser.add_argument("--scratch-threshold", type=float, default=None,
                         help="Override the scratch tile-coverage threshold (default "
                         f"{DEFAULT_TILE_THRESHOLDS['scratch']}, see scripts/diagnose_tile_thresholds.py "
                         "--effect scratch).")
    parser.add_argument("--include-combos", action="store_true",
                         help="Also generate multi-distortion variants (the 3 pairs + the full triple, "
                         "see COMBO_KINDS in src/soiling/dataset_builder.py) alongside the original "
                         "clean/single-effect kinds -- 8 kinds total. Default: off, original behavior.")
    parser.add_argument("--include-severity", action="store_true",
                         help="Split each single-effect kind into 3 balanced severity levels "
                         "(low/medium/high, see SEVERITY_VARIANT_KINDS in "
                         "src/soiling/dataset_builder.py) -- 10 kinds total (or 14 combined with "
                         "--include-combos). Default: off, every active class is full-strength.")
    parser.add_argument("--save-pixel-masks", action="store_true",
                         help="Also write each variant's full-resolution per-class mask to "
                         "masks/ (3-channel PNG, one channel per class, continuous [0,1] scaled "
                         "to uint8) -- this is Stage C's (pixel-level segmentation) ground "
                         "truth, computed for free in the same call that already builds this "
                         "Stage B dataset. Default: off, no masks/ folder written.")
    args = parser.parse_args()

    thresholds = None
    overrides = {"dirt": args.dirt_threshold, "water": args.water_threshold, "scratch": args.scratch_threshold}
    if any(v is not None for v in overrides.values()):
        thresholds = dict(DEFAULT_TILE_THRESHOLDS)
        for name, value in overrides.items():
            if value is not None:
                thresholds[name] = value

    rows, tile_labels = build_stage_b_dataset(
        args.source, args.out,
        variants_per_image=args.variants, seed=args.seed, img_size=args.img_size,
        thresholds=thresholds, include_combos=args.include_combos, include_severity=args.include_severity,
        save_pixel_masks=args.save_pixel_masks,
    )

    used_thresholds = thresholds if thresholds is not None else DEFAULT_TILE_THRESHOLDS
    grid_h, grid_w = tile_labels.shape[2], tile_labels.shape[3]
    by_split = Counter(r["split"] for r in rows)
    print(f"Wrote {len(rows)} images to {Path(args.out) / 'images'}")
    print(f"Metadata: {Path(args.out) / 'metadata.csv'}")
    print(f"Tile labels: {Path(args.out) / 'tile_labels.npy'} (shape {tile_labels.shape}, grid {grid_h}x{grid_w})")
    if args.save_pixel_masks:
        print(f"Pixel masks: {Path(args.out) / 'masks'} (one 3-channel PNG per image)")
    print(f"Split sizes: {dict(by_split)}")
    for i, name in enumerate(EFFECT_NAMES):
        positives = sum(r[name] for r in rows)
        print(f"  {name}: {positives}/{len(rows)} images positive ({positives / len(rows):.1%})"
              f" | threshold={used_thresholds[name]}"
              f" | tile-positive rate={tile_labels[:, i].mean():.1%}")
        if args.include_severity:
            by_level = Counter(r[f"{name}_severity"] for r in rows if r[name] == 1)
            print(f"    severity: {dict(by_level)}")


if __name__ == "__main__":
    main()
