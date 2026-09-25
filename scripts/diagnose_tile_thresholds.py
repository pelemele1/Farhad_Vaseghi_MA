"""
Is a tile-coverage threshold (DEFAULT_TILE_THRESHOLDS in
src/soiling/dataset_builder.py) actually a reasonable cutoff, or just a
plausible-looking guess? Originally written for scratch alone (Session 18,
improvement #3 -- found the 3% guess discarded over half of real
scratch-affected tiles); generalized here to cover dirt and water too, which
have never been through this same check and are still at their untouched
original 0.15 default.

Generates a batch of real masks for the chosen effect (add_dirt / add_water /
add_scratch -- the same functions the actual dataset build calls),
rasterizes each at several candidate thresholds, and reports:
  - how many tiles (16x16 grid, img-size 512 by default) end up positive at
    each threshold -- the tile positive-rate this threshold choice actually
    produces, the number that directly drives pos_weight/imbalance
  - the coverage-fraction distribution of tiles the mask actually touches at
    all (>0 coverage) -- shows whether the current default is cutting
    through the middle of that distribution (arbitrary) or sitting cleanly
    below/above the bulk of it (a real boundary between "just grazed" and
    "actually affected" tiles)

Usage:
    python scripts/diagnose_tile_thresholds.py --effect scratch --n 30
    python scripts/diagnose_tile_thresholds.py --effect dirt --n 30
    python scripts/diagnose_tile_thresholds.py --effect water --n 30
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np

from src.soiling.dataset_builder import DEFAULT_TILE_THRESHOLDS
from src.soiling.effects import add_dirt, add_scratch, add_water
from src.soiling.tile_labels import rasterize_tile_label

_EFFECT_FUNCS = {"dirt": add_dirt, "water": add_water, "scratch": add_scratch}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--effect", choices=sorted(_EFFECT_FUNCS), default="scratch")
    parser.add_argument("--n", type=int, default=30, help="Number of independent masks to sample")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidates", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
        help="Candidate coverage thresholds to compare",
    )
    args = parser.parse_args()

    effect_fn = _EFFECT_FUNCS[args.effect]
    rng = np.random.default_rng(args.seed)
    blank = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)

    all_touched_coverages = []  # coverage fraction of every tile the mask touches at all (>0), pooled across samples
    positive_counts = {t: [] for t in args.candidates}
    total_touched = []

    for i in range(args.n):
        seed = int(rng.integers(0, 2**31 - 1))
        _, mask = effect_fn(blank, seed=seed)
        # rasterize_tile_label thresholds; replicate its resize step here to
        # also inspect the raw coverage distribution before thresholding
        coverage = cv.resize(mask.astype(np.float32), (args.grid_size, args.grid_size), interpolation=cv.INTER_AREA)
        touched = coverage[coverage > 0]
        all_touched_coverages.extend(touched.tolist())
        total_touched.append(int((coverage > 0).sum()))

        for t in args.candidates:
            label = rasterize_tile_label(mask, args.grid_size, args.grid_size, t)
            positive_counts[t].append(int(label.sum()))

    touched_arr = np.array(all_touched_coverages)
    print(f"=== {args.n} independent add_{args.effect} masks, {args.grid_size}x{args.grid_size} grid ===\n")
    print(f"tiles touched at all (coverage>0) per mask: mean={np.mean(total_touched):.1f}  "
          f"min={np.min(total_touched)}  max={np.max(total_touched)}  (out of {args.grid_size**2} tiles)\n")
    print("coverage-fraction distribution over all touched tiles (pooled across samples):")
    for p in (10, 25, 50, 75, 90, 95, 99):
        print(f"  p{p:<3d} = {np.percentile(touched_arr, p):.4f}")
    print()

    print(f"{'threshold':>10}  {'mean positive tiles/mask':>26}  {'mean positive rate':>20}")
    for t in args.candidates:
        counts = np.array(positive_counts[t])
        rate = counts.mean() / (args.grid_size ** 2)
        print(f"{t:>10.2f}  {counts.mean():>26.2f}  {rate:>19.4%}")

    print(f"\ncurrent default threshold in dataset_builder.py for {args.effect!r}: "
          f"{DEFAULT_TILE_THRESHOLDS[args.effect]} (mean positive rate at that threshold shown above)")


if __name__ == "__main__":
    main()
