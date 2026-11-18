"""
Checks a built visible-change dataset (scripts/build_visible_dataset.py)
against itself, image by image, and against the planned amounts:

  1. amounts  -- clean, every (type, severity) and every two-type pair as
                 planned (src/soiling/dataset_builder.py::plan_variant_kinds),
                 overall and per split; at most two types per image; all
                 variants of a photo in one split.
  2. labels   -- every type in the label map was applied to that image; a type
                 is flagged present exactly when it has at least
                 MIN_VISIBLE_PIXELS pixels; the stored pixel counts, severities
                 and the dominant type agree with the map; a clean variant has
                 an empty map and the label map has the image's size.
  3. tiles    -- tile_labels.npy is exactly the grid re-computed from each label
                 map with the dataset's own cutoffs; no tile names a type that
                 is not in the pixel map.

It also reports, per type and severity, how many applied types stayed visible
(an effect that never shows -- e.g. faint dust on a black night photo -- is left
unlabeled by design, so the image is not labeled with something it does not show).

Exit status 1 if any check fails. Usage:
    python scripts/check_visible_dataset.py --data data/processed/visible
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np

from src.data.visible_dataset import read_label_map
from src.soiling.dataset_builder import COMBO_KINDS, PLANNED_SINGLE_KINDS, _apportion
from src.soiling.effects import MIN_VISIBLE_PIXELS, SEVERITY_LEVELS
from src.soiling.visible_labels import CLASS_NAMES, EFFECT_NAMES, tile_class_grid

SPLITS = ("train", "val", "test")


def applied_types(kind):
    """{type: severity} applied to an image, from its kind string."""
    if kind == "clean":
        return {}
    return dict(token.split(":") for token in kind.split("+"))


def pair_of(kind):
    """The unordered pair of types of a two-type kind, else None."""
    types = tuple(applied_types(kind))
    return "+".join(sorted(types, key=EFFECT_NAMES.index)) if len(types) == 2 else None


def check_amounts(rows, errors, combos_per_kind):
    n_photos = len({r["source_id"] for r in rows})
    singles = Counter(r["kind"] for r in rows if len(applied_types(r["kind"])) <= 1)
    pairs = Counter(pair_of(r["kind"]) for r in rows if pair_of(r["kind"]))

    for kind in PLANNED_SINGLE_KINDS:
        if singles[kind] != n_photos:
            errors.append(f"amounts: {kind} has {singles[kind]} images, planned {n_photos} (one per photo)")
    for kind in COMBO_KINDS:
        if pairs[kind] != combos_per_kind:
            errors.append(f"amounts: two-type {kind} has {pairs[kind]} images, planned {combos_per_kind}")
    for r in rows:
        if len(applied_types(r["kind"])) > 2:
            errors.append(f"amounts: {r['path']} has more than two types ({r['kind']})")

    split_of = defaultdict(set)
    for r in rows:
        split_of[r["source_id"]].add(r["split"])
    mixed = [s for s, splits in split_of.items() if len(splits) > 1]
    if mixed:
        errors.append(f"amounts: {len(mixed)} photos have variants in more than one split, e.g. {mixed[0]}")

    photos_per_split = Counter(next(iter(s)) for s in split_of.values())
    ratios = tuple(photos_per_split[s] / n_photos for s in SPLITS)
    for split, per_kind in zip(SPLITS, _apportion(combos_per_kind, ratios)):
        split_pairs = Counter(pair_of(r["kind"]) for r in rows if r["split"] == split and pair_of(r["kind"]))
        uneven = {k: split_pairs[k] for k in COMBO_KINDS if split_pairs[k] != per_kind}
        if uneven:
            errors.append(f"amounts: in {split}, two-type pairs are not {per_kind} each: {uneven}")
    return n_photos, singles, pairs


def check_image(data, row, tile_grid, meta, errors):
    """Checks one image's label map, flags and tiles; returns the types the map holds."""
    labels = read_label_map(data / "labels" / (Path(row["path"]).stem + ".png"))
    image = cv.imread(str(data / row["path"]))
    where = row["path"]
    if image is None or labels is None:
        errors.append(f"labels: {where}: image or label map missing")
        return set()
    if labels.shape != image.shape[:2]:
        errors.append(f"labels: {where}: label map {labels.shape} vs image {image.shape[:2]}")
    if labels.max() >= len(CLASS_NAMES):
        errors.append(f"labels: {where}: label value {labels.max()} is not a class")

    applied = applied_types(row["kind"])
    counts = {name: int((labels == value).sum()) for value, name in enumerate(CLASS_NAMES) if value}
    in_map = {name for name, count in counts.items() if count}
    if in_map - set(applied):
        errors.append(f"labels: {where}: map has {sorted(in_map - set(applied))}, never applied ({row['kind']})")
    for name in EFFECT_NAMES:
        present = counts[name] >= MIN_VISIBLE_PIXELS
        if int(row[name]) != int(present):
            errors.append(f"labels: {where}: {name} flag {row[name]} but {counts[name]} pixels")
        if int(row[f"{name}_pixels"]) != counts[name]:
            errors.append(f"labels: {where}: {name}_pixels {row[f'{name}_pixels']} but map has {counts[name]}")
        expected_severity = applied.get(name, "none") if present else "none"
        if row[f"{name}_severity"] != expected_severity:
            errors.append(f"labels: {where}: {name}_severity {row[f'{name}_severity']}, expected {expected_severity}")
    present = [n for n in EFFECT_NAMES if int(row[n])]
    dominant = max(present, key=lambda n: counts[n]) if present else "clean"
    if row["dominant"] != dominant:
        errors.append(f"labels: {where}: dominant {row['dominant']}, map says {dominant}")
    if row["kind"] == "clean" and labels.any():
        errors.append(f"labels: {where}: clean variant has a non-empty label map")

    if tile_grid is None:
        return set(n for n in EFFECT_NAMES if int(row[n]))
    grid = tile_class_grid(labels, meta["grid_h"], meta["grid_w"], meta["tile_thresholds"])
    if not np.array_equal(grid, tile_grid):
        errors.append(f"tiles: {where}: stored tile grid differs from the one computed from the label map")
    tile_types = {CLASS_NAMES[v] for v in np.unique(tile_grid) if v}
    if tile_types - in_map:
        errors.append(f"tiles: {where}: tiles name {sorted(tile_types - in_map)}, not in the pixel map")
    return set(present)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed/visible")
    parser.add_argument("--combos-per-kind", type=int, default=None,
                        help="Planned images per two-type pair (default: the dataset's own count when "
                             "it was built with the planned amounts, else the amounts check is skipped)")
    parser.add_argument("--skip-amounts", action="store_true",
                        help="Skip the amounts check (a partial build, e.g. --max-sources)")
    parser.add_argument("--max-errors", type=int, default=30, help="Errors printed in full")
    args = parser.parse_args()

    data = Path(args.data)
    with open(data / "metadata.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    meta = json.loads((data / "visible_meta.json").read_text())
    tiles = np.load(data / "tile_labels.npy")
    errors = []

    if list(meta["class_names"]) != list(CLASS_NAMES):
        errors.append(f"meta: class_names {meta['class_names']}, the code has {list(CLASS_NAMES)}")
    if len(tiles) != len(rows):
        errors.append(f"tiles: {len(tiles)} tile grids for {len(rows)} images")

    planned = meta.get("variant_plan") == "planned amounts"
    if (planned or args.combos_per_kind is not None) and not args.skip_amounts:
        from src.soiling.dataset_builder import COMBOS_PER_KIND
        n_photos, _, _ = check_amounts(rows, errors, args.combos_per_kind or COMBOS_PER_KIND)
    else:
        n_photos = len({r["source_id"] for r in rows})
        print("amounts: skipped")

    applied_count = Counter()
    visible_count = Counter()
    for i, row in enumerate(rows):
        present = check_image(data, row, tiles[i] if i < len(tiles) else None, meta, errors)
        for name, severity in applied_types(row["kind"]).items():
            applied_count[(name, severity)] += 1
            visible_count[(name, severity)] += int(name in present)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(rows)} images checked", flush=True)

    two_type = sum(1 for r in rows if pair_of(r["kind"]))
    print(f"{len(rows)} images from {n_photos} photos; two types on {two_type} ({two_type / len(rows):.1%})")
    for split in SPLITS:
        print(f"  {split}: {sum(1 for r in rows if r['split'] == split)} images")
    print("applied -> labeled (visible) per type and severity:")
    for name in EFFECT_NAMES:
        parts = []
        for level in SEVERITY_LEVELS:
            a, v = applied_count[(name, level)], visible_count[(name, level)]
            parts.append(f"{level} {v}/{a}" + (f" ({100 * v / a:.1f}%)" if a else ""))
        print(f"  {name:17s} " + ", ".join(parts))

    if errors:
        print(f"\nFAILED: {len(errors)} problems")
        for e in errors[:args.max_errors]:
            print("  " + e)
        if len(errors) > args.max_errors:
            print(f"  ... and {len(errors) - args.max_errors} more")
        sys.exit(1)
    print("\nOK: every label agrees with its image, its label map, its tiles and the plan")


if __name__ == "__main__":
    main()
