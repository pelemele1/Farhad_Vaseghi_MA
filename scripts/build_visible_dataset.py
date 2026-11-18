"""
Builds the visible-change dataset used by Stages A, B and C (Session 27; see
src/soiling/visible_dataset.py).

Usage:
    python scripts/build_visible_dataset.py --source data/raw/mio_tcd/images \
        --out data/processed/visible --workers 16
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.soiling.visible_dataset import DEFAULT_TILE_THRESHOLDS, build_visible_dataset
from src.soiling.visible_labels import EFFECT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Folder of clean source photos")
    parser.add_argument("--out", default="data/processed/visible")
    parser.add_argument("--variants", type=int, default=None,
                        help="Fixed number of variants per source photo. Default: the planned "
                             "amounts (clean + one per type and severity per photo, plus each "
                             "photo's share of the two-type variants) -- see "
                             "src/soiling/dataset_builder.py::plan_variant_kinds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-sources", type=int, default=None, help="Only the first N photos (testing)")
    for name, value in DEFAULT_TILE_THRESHOLDS.items():  # e.g. --water-drops-threshold
        parser.add_argument(f"--{name.replace('_', '-')}-threshold", type=float, default=value,
                            dest=f"{name}_threshold",
                            help=f"Tile cutoff for {name}: share of a tile it must cover (default {value})")
    args = parser.parse_args()

    np.seterr(all="ignore")  # the vendored texture code divides by zero on purpose
    start = time.time()
    thresholds = {n: getattr(args, f"{n}_threshold") for n in DEFAULT_TILE_THRESHOLDS}
    rows, tiles = build_visible_dataset(args.source, args.out, variants_per_image=args.variants, seed=args.seed,
                                        img_size=args.img_size, thresholds=thresholds, workers=args.workers,
                                        max_sources=args.max_sources)
    print(f"wrote {len(rows)} images to {args.out} in {time.time() - start:.0f}s")
    two_type = sum(1 for r in rows if len(r["kind"].split("+")) == 2)
    print(f"two-type images: {two_type} ({two_type / len(rows):.1%} of {len(rows)})")
    for split in ("train", "val", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        clean = sum(1 for r in split_rows if r["dominant"] == "clean")
        print(f"{split}: {len(split_rows)} images, {clean} with nothing visible")
        for n in EFFECT_NAMES:
            by_level = Counter(r[f"{n}_severity"] for r in split_rows if r[n])
            print(f"    {n:17s} {sum(r[n] for r in split_rows):5d} images visible"
                  f"  (light {by_level['low']}, medium {by_level['medium']}, heavy {by_level['high']})")


if __name__ == "__main__":
    main()
