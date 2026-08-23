"""
Sample a pilot subset of MIO-TCD-Localization's train split out of the
official tar archive (downloaded separately -- see docs/development_log.md).

Usage:
    python scripts/sample_mio_tcd.py --tar D:/MasterThesis_FAU/MIO-TCD-Localization.tar \
        --out data/raw/mio_tcd --n 1000 --seed 0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mio_tcd import build_pilot_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", required=True, help="Path to MIO-TCD-Localization.tar")
    parser.add_argument("--out", default="data/raw/mio_tcd", help="Output directory")
    parser.add_argument("--n", type=int, default=1000, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    image_paths, gt_rows, selected_ids = build_pilot_subset(
        args.tar, args.out, n=args.n, seed=args.seed
    )
    print(f"Extracted {len(image_paths)} images to {Path(args.out) / 'images'}")
    print(f"Wrote {len(gt_rows)} ground-truth rows to {Path(args.out) / 'gt_subset.csv'}")
    print(f"Manifest of sampled ids: {Path(args.out) / 'manifest.txt'}")


if __name__ == "__main__":
    main()
