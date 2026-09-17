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
    parser.add_argument("--scratch-threshold", type=float, default=None,
                         help="Override the scratch tile-coverage threshold (default "
                         f"{DEFAULT_TILE_THRESHOLDS['scratch']}, see scripts/diagnose_scratch_threshold.py). "
                         "dirt/water thresholds are left at their default.")
    parser.add_argument("--include-combos", action="store_true",
                         help="Also generate multi-distortion variants (the 3 pairs + the full triple, "
                         "see COMBO_KINDS in src/soiling/dataset_builder.py) alongside the original "
                         "clean/single-effect kinds -- 8 kinds total. Default: off, original behavior.")
    args = parser.parse_args()

    thresholds = None
    if args.scratch_threshold is not None:
        thresholds = dict(DEFAULT_TILE_THRESHOLDS)
        thresholds["scratch"] = args.scratch_threshold

    rows, tile_labels = build_stage_b_dataset(
        args.source, args.out,
        variants_per_image=args.variants, seed=args.seed, img_size=args.img_size,
        thresholds=thresholds, include_combos=args.include_combos,
    )

    used_thresholds = thresholds if thresholds is not None else DEFAULT_TILE_THRESHOLDS
    grid_h, grid_w = tile_labels.shape[2], tile_labels.shape[3]
    by_split = Counter(r["split"] for r in rows)
    print(f"Wrote {len(rows)} images to {Path(args.out) / 'images'}")
    print(f"Metadata: {Path(args.out) / 'metadata.csv'}")
    print(f"Tile labels: {Path(args.out) / 'tile_labels.npy'} (shape {tile_labels.shape}, grid {grid_h}x{grid_w})")
    print(f"Split sizes: {dict(by_split)}")
    for i, name in enumerate(EFFECT_NAMES):
        positives = sum(r[name] for r in rows)
        print(f"  {name}: {positives}/{len(rows)} images positive ({positives / len(rows):.1%})"
              f" | threshold={used_thresholds[name]}"
              f" | tile-positive rate={tile_labels[:, i].mean():.1%}")


if __name__ == "__main__":
    main()
