"""
Precompute the Stage A multi-label dataset from a source image folder (the
MIO-TCD pilot subset by default).

Usage:
    python scripts/build_stage_a_dataset.py --source data/raw/mio_tcd/images \
        --out data/processed/stage_a --variants 4 --seed 0
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.soiling.dataset_builder import build_stage_a_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Folder of clean source images")
    parser.add_argument("--out", default="data/processed/stage_a", help="Output directory")
    parser.add_argument("--variants", type=int, default=4,
                         help="Variants per source image (cycles clean/dirt/water/scratch, balanced; "
                         "use 8 with --include-combos, 10 with --include-severity, or 14 with both "
                         "for exact balance across the full kind pool)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-combos", action="store_true",
                         help="Also generate multi-distortion variants (the 3 pairs + the full triple, "
                         "see COMBO_KINDS in src/soiling/dataset_builder.py) alongside the original "
                         "clean/single-effect kinds -- 8 kinds total. Default: off, original behavior.")
    parser.add_argument("--include-severity", action="store_true",
                         help="Split each single-effect kind into 3 balanced severity levels "
                         "(low/medium/high, see SEVERITY_VARIANT_KINDS in "
                         "src/soiling/dataset_builder.py) -- 10 kinds total (or 14 combined with "
                         "--include-combos). Default: off, every active class is full-strength.")
    args = parser.parse_args()

    rows = build_stage_a_dataset(
        args.source, args.out,
        variants_per_image=args.variants, seed=args.seed,
        include_combos=args.include_combos, include_severity=args.include_severity,
    )

    by_split = Counter(r["split"] for r in rows)
    print(f"Wrote {len(rows)} images to {Path(args.out) / 'images'}")
    print(f"Metadata: {Path(args.out) / 'metadata.csv'}")
    print(f"Split sizes: {dict(by_split)}")
    for name in ("dirt", "water", "scratch"):
        positives = sum(r[name] for r in rows)
        print(f"  {name}: {positives}/{len(rows)} positive ({positives / len(rows):.1%})")
        if args.include_severity:
            by_level = Counter(r[f"{name}_severity"] for r in rows if r[name] == 1)
            print(f"    severity: {dict(by_level)}")


if __name__ == "__main__":
    main()
