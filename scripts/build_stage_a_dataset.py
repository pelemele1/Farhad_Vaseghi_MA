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
                         help="Variants per source image (cycles clean/dirt/water/scratch, balanced)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = build_stage_a_dataset(
        args.source, args.out,
        variants_per_image=args.variants, seed=args.seed,
    )

    by_split = Counter(r["split"] for r in rows)
    print(f"Wrote {len(rows)} images to {Path(args.out) / 'images'}")
    print(f"Metadata: {Path(args.out) / 'metadata.csv'}")
    print(f"Split sizes: {dict(by_split)}")
    for name in ("dirt", "water", "scratch"):
        positives = sum(r[name] for r in rows)
        print(f"  {name}: {positives}/{len(rows)} positive ({positives / len(rows):.1%})")


if __name__ == "__main__":
    main()
