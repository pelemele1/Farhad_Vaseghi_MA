"""
Audits every generated image's labels against the data itself, for the
Stage A dataset (metadata only) and the Stage B/C dataset (metadata, tile
labels and pixel masks):

  1. metadata  -- class flag <-> severity agree; each source photo has its 14
                  expected variants (1 clean, 9 class x severity, 4 combos);
                  all variants of a source share one split.
  2. masks     -- a labeled class has a non-empty, full-strength mask; an
                  unlabeled class's mask is empty.
  3. tiles     -- tile_labels.npy equals the grid re-rasterized from the masks;
                  no labeled class without a single positive tile.
  4. images    -- every labeled class visibly changes the image inside its
                  mask, by the generator's own rule (visible_pixel_count >=
                  MIN_VISIBLE_PIXELS against the clean variant of the same
                  photo); a clean variant matches its source photo. Stage A has
                  no stored masks: scratch masks are regenerated from their
                  seeds, dirt/water are checked over the whole frame when alone.
  5. floor     -- no water mask carries the thin-water 0.2 floor.

Variants whose labels scripts/fix_dataset_labels.py removed (label_fixes.csv)
are expected to break the 14-kinds pattern and are not flagged for it.

Prints a summary and, with --out, writes every flagged image to a CSV.

Usage:
    python scripts/audit_dataset_labels.py --stage-a data/processed/stage_a \
        --stage-b data/processed/stage_b --source data/raw/mio_tcd/images --out audit.csv
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

from scripts.fix_dataset_labels import FLOOR_DETECT, regenerated_scratch_mask
from src.soiling.dataset_builder import EFFECT_NAMES
from src.soiling.effects import MIN_VISIBLE_PIXELS, visible_pixel_count
from src.soiling.tile_labels import rasterize_tile_label

SEVERITIES = ("low", "medium", "high")
EXPECTED_KINDS = Counter(
    ["clean"]
    + [f"{c}:{s}" for c in EFFECT_NAMES for s in SEVERITIES]
    + ["dirt+water", "dirt+scratch", "water+scratch", "dirt+water+scratch"]
)


def read_rows(data):
    with open(data / "metadata.csv", newline="") as f:
        return list(csv.DictReader(f))


def kind_of(row):
    active = [c for c in EFFECT_NAMES if int(row[c])]
    if not active:
        return "clean"
    if len(active) == 1:
        return f"{active[0]}:{row[f'{active[0]}_severity']}"
    return "+".join(active)


def fixed_sources(data):
    path = data / "label_fixes.csv"
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {Path(r["image"]).stem.split("_v")[0] for r in csv.DictReader(f)}


def check_metadata(rows, name, flags, expected_breaks=frozenset()):
    by_source = defaultdict(list)
    for i, row in enumerate(rows):
        for c in EFFECT_NAMES:
            on, sev = int(row[c]), row[f"{c}_severity"]
            if (on and sev not in SEVERITIES) or (not on and sev != "none"):
                flags.append((name, row["path"], "metadata", f"{c}={on} but severity={sev}"))
        by_source[row["source_id"]].append(row)
    for source, group in by_source.items():
        kinds = Counter(kind_of(r) for r in group)
        if kinds != EXPECTED_KINDS and source not in expected_breaks:
            flags.append((name, source, "variants", f"unexpected variant kinds {dict(kinds)}"))
        if len({r["split"] for r in group}) != 1:
            flags.append((name, source, "split", "variants of one photo in several splits"))
    return Counter(r["split"] for r in rows), len(by_source)


def mean_abs_diff(a, b):
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)


def audit_stage_b(data, source_dir, flags, stats):
    rows = read_rows(data)
    meta = json.loads((data / "stage_b_meta.json").read_text())
    tiles = np.load(data / "tile_labels.npy")
    splits, n_sources = check_metadata(rows, "stage_b", flags, fixed_sources(data))
    stats["stage_b"] = {"images": len(rows), "sources": n_sources, "splits": dict(splits)}

    clean_of = {r["source_id"]: r for r in rows if kind_of(r) == "clean"}
    clean_cache = {}
    visible = defaultdict(list)  # (class, severity) -> visible pixels per image
    mask_peak = defaultdict(list)
    tile_mismatch = 0
    for i, row in enumerate(rows):
        image = cv.imread(str(data / row["path"]))
        mask = cv.imread(str(data / "masks" / Path(row["path"]).with_suffix(".png").name), cv.IMREAD_UNCHANGED)
        if image is None or mask is None or mask.shape[:2] != image.shape[:2]:
            flags.append(("stage_b", row["path"], "files", "image or mask missing / size mismatch"))
            continue
        m = mask.astype(np.float32) / 255.0

        # 2. masks
        for c, name in enumerate(EFFECT_NAMES):
            if int(row[name]):
                peak = m[..., c].max()
                mask_peak[(name, row[f"{name}_severity"])].append(peak)
                if (m[..., c] > 0.5).sum() == 0:
                    flags.append(("stage_b", row["path"], "mask", f"{name} labeled but mask empty"))
                if name == "water" and m[..., c].min() >= FLOOR_DETECT:
                    flags.append(("stage_b", row["path"], "floor", f"water mask floor {m[..., c].min():.2f}"))
            elif m[..., c].max() > 0:
                flags.append(("stage_b", row["path"], "mask", f"{name} not labeled but mask non-empty"))

        # 3. tiles
        grid = np.stack([rasterize_tile_label(m[..., c], meta["grid_h"], meta["grid_w"], meta["thresholds"][n])
                         for c, n in enumerate(EFFECT_NAMES)])
        if not np.array_equal(grid, tiles[i]):
            tile_mismatch += 1
            flags.append(("stage_b", row["path"], "tiles", f"{int((grid != tiles[i]).sum())} tiles differ from masks"))
        for c, name in enumerate(EFFECT_NAMES):
            if int(row[name]) and tiles[i, c].sum() == 0:
                flags.append(("stage_b", row["path"], "tiles", f"{name} labeled but no positive tile"))

        # 4. image evidence against the clean variant of the same photo
        src = row["source_id"]
        if src not in clean_cache:
            clean_cache.clear()
            clean_cache[src] = cv.imread(str(data / clean_of[src]["path"]))
        clean = clean_cache[src]
        if kind_of(row) == "clean":
            raw = cv.imread(str(source_dir / f"{src}.jpg"))
            if raw is not None and mean_abs_diff(clean, raw).mean() > 3.0:
                flags.append(("stage_b", row["path"], "clean", "clean variant differs from its source photo"))
            continue
        for c, name in enumerate(EFFECT_NAMES):
            if not int(row[name]):
                continue
            count = visible_pixel_count(clean, image, m[..., c])
            visible[(name, row[f"{name}_severity"])].append(count)
            if count < MIN_VISIBLE_PIXELS:
                flags.append(("stage_b", row["path"], "invisible",
                              f"{name} ({row[f'{name}_severity']}): only {count} visibly changed pixels"))
        if i % 2000 == 0:
            print(f"  stage_b {i}/{len(rows)}", flush=True)

    stats["stage_b"]["tile_mismatches"] = tile_mismatch
    stats["stage_b"]["mask_peak_min"] = {f"{k[0]}:{k[1]}": round(float(min(v)), 3) for k, v in sorted(mask_peak.items())}
    stats["stage_b"]["visible_pixels_min"] = {f"{k[0]}:{k[1]}": int(min(v)) for k, v in sorted(visible.items())}


def audit_stage_a(data, source_dir, flags, stats):
    rows = read_rows(data)
    splits, n_sources = check_metadata(rows, "stage_a", flags, fixed_sources(data))
    stats["stage_a"] = {"images": len(rows), "sources": n_sources, "splits": dict(splits)}
    clean_of = {r["source_id"]: r for r in rows if kind_of(r) == "clean"}
    changes = defaultdict(list)
    clean_cache = {}
    raw_cache = {}
    for i, row in enumerate(rows):
        src = row["source_id"]
        if src not in clean_cache:
            clean_cache.clear()
            clean_cache[src] = cv.imread(str(data / clean_of[src]["path"]))
        clean = clean_cache[src]
        if kind_of(row) == "clean":
            raw = cv.imread(str(source_dir / f"{src}.jpg"))
            if raw is not None and mean_abs_diff(clean, raw).mean() > 3.0:
                flags.append(("stage_a", row["path"], "clean", "clean variant differs from its source photo"))
            continue
        image = cv.imread(str(data / row["path"]))
        active = [c for c in EFFECT_NAMES if int(row[c])]
        for name in active:
            if name == "scratch":
                if src not in raw_cache:
                    raw_cache.clear()
                    raw_cache[src] = cv.imread(str(source_dir / f"{src}.jpg"))
                mask = regenerated_scratch_mask(raw_cache[src], src, row["variant_id"], 0)
            elif len(active) == 1:
                mask = np.ones(image.shape[:2], dtype=np.float32)
            else:
                continue
            count = visible_pixel_count(clean, image, mask)
            changes[name].append(count)
            if count < MIN_VISIBLE_PIXELS:
                flags.append(("stage_a", row["path"], "invisible", f"{name}: only {count} visibly changed pixels"))
        if i % 2000 == 0:
            print(f"  stage_a {i}/{len(rows)}", flush=True)
    stats["stage_a"]["visible_pixels_min"] = {k: int(min(v)) for k, v in sorted(changes.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-a", default="data/processed/stage_a")
    parser.add_argument("--stage-b", default="data/processed/stage_b")
    parser.add_argument("--source", default="data/raw/mio_tcd/images")
    parser.add_argument("--out", default=None, help="CSV of every flagged image")
    args = parser.parse_args()

    flags, stats = [], {}
    print("auditing Stage B/C dataset ...", flush=True)
    audit_stage_b(Path(args.stage_b), Path(args.source), flags, stats)
    print("auditing Stage A dataset ...", flush=True)
    audit_stage_a(Path(args.stage_a), Path(args.source), flags, stats)

    print(json.dumps(stats, indent=1))
    by_check = Counter((f[0], f[2]) for f in flags)
    print("\nflags per (dataset, check):", dict(by_check) if by_check else "none")
    for f in flags[:40]:
        print("  ", " | ".join(f))
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "item", "check", "detail"])
            w.writerows(flags)
        print(f"wrote {len(flags)} flags to {args.out}")


if __name__ == "__main__":
    main()
