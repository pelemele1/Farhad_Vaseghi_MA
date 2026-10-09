"""
One-time, in-place label fix for datasets built before the Session 26 label
audit (scripts/audit_dataset_labels.py), applying the rules the generator now
follows (src/soiling/effects.py) without re-rendering any image:

  Stage B/C dataset (masks + tile labels):
    1. thin-water masks lose their 0.2 floor (remove_thin_water_floor) -- the
       floor marked every pixel as 20% water;
    2. a labeled class whose mask is completely empty although the image is
       distorted (a vendored-effect failure: the effect rendered, its mask came
       back all-zero) -- the whole variant is re-rendered from its source photo
       with the current generator, same kind and severities, so image, masks
       and labels agree again;
    3. a labeled class whose effect is not visible in the image
       (visible_pixel_count < MIN_VISIBLE_PIXELS, measured against the clean
       variant of the same photo) is unlabeled: flag 0, severity "none",
       mask channel cleared;
    4. tile_labels.npy is re-rasterized from the corrected masks.
  Stage A dataset (labels only, no stored masks): rule 2, with scratch masks
  regenerated from their seeds (generate_scratch_mask is exactly
  reproducible); dirt and water textures are not, so a dirt- or water-only
  image is checked over the whole frame and those classes are not checked in
  combinations.

Originals are kept next to the data (`*.before_label_fix`, and
`masks_before_label_fix/` / `images_before_label_fix/` for changed files); every change is listed in
`label_fixes.csv`. Refuses to run twice.

Usage:
    python scripts/fix_dataset_labels.py --stage-b data/processed/stage_b \
        --stage-a data/processed/stage_a --source data/raw/mio_tcd/images
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np

from src.soiling.dataset_builder import EFFECT_NAMES, METADATA_FIELDNAMES, apply_effect_combo_with_masks, derive_seed
from src.soiling.effects import (
    MIN_VISIBLE_PIXELS,
    generate_scratch_mask,
    remove_thin_water_floor,
    visible_pixel_count,
)
from src.soiling.tile_labels import rasterize_tile_label

FLAG = "label_audit_fix"
WATER = EFFECT_NAMES.index("water")
# A water mask whose minimum is this high carries the thin-water floor (thick
# water and droplet masks start at 0).
FLOOR_DETECT = 0.15


def read_rows(data):
    with open(data / "metadata.csv", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(data, rows):
    shutil.copy2(data / "metadata.csv", data / "metadata.csv.before_label_fix")
    with open(data / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def unlabel(row, name):
    row[name] = "0"
    row[f"{name}_severity"] = "none"


def clean_variants(rows):
    return {r["source_id"]: r["path"] for r in rows if not any(int(r[c]) for c in EFFECT_NAMES)}


def rerender_variant(row, raw, base_seed):
    """(image, (H, W, C) mask, labels) for `row`'s kind and severities,
    rendered from the source photo by the current generator."""
    combo = tuple(bool(int(row[c])) for c in EFFECT_NAMES)
    severities = {c: row[f"{c}_severity"] for c in EFFECT_NAMES if int(row[c])}
    seeds = {c: derive_seed(base_seed, row["source_id"], int(row["variant_id"]), c, "rerender") for c in EFFECT_NAMES}
    image, labels, masks = apply_effect_combo_with_masks(raw, combo, seeds, severities)
    return image, np.stack([masks[c] for c in EFFECT_NAMES], axis=-1), labels


def fix_stage_b(data, source_dir, base_seed, log):
    meta_path = data / "stage_b_meta.json"
    meta = json.loads(meta_path.read_text())
    if meta.get(FLAG):
        sys.exit(f"{data} was already fixed -- nothing to do")
    rows = read_rows(data)
    tiles = np.load(data / "tile_labels.npy")
    clean_of = clean_variants(rows)
    backup = data / "masks_before_label_fix"
    backup.mkdir(exist_ok=True)
    image_backup = data / "images_before_label_fix"
    image_backup.mkdir(exist_ok=True)
    cache = {}
    floors = 0
    for i, row in enumerate(rows):
        if row["path"] == clean_of[row["source_id"]]:
            continue
        mask_path = data / "masks" / Path(row["path"]).with_suffix(".png").name
        saved = cv.imread(str(mask_path), cv.IMREAD_UNCHANGED)
        mask = saved.astype(np.float32) / 255.0
        changed = False

        if any(int(row[n]) and mask[..., c].max() == 0 for c, n in enumerate(EFFECT_NAMES)):
            raw = cv.imread(str(source_dir / f"{row['source_id']}.jpg"))
            if raw is None:
                sys.exit(f"source photo {row['source_id']}.jpg not found in {source_dir} (--source)")
            image, mask, labels = rerender_variant(row, raw, base_seed)
            shutil.copy2(data / row["path"], image_backup / Path(row["path"]).name)
            cv.imwrite(str(data / row["path"]), image)
            for n in EFFECT_NAMES:
                if int(row[n]) and not labels[n]:
                    log.append(("stage_b", row["path"], n, row[f"{n}_severity"], "re-render invisible"))
                row[n], row[f"{n}_severity"] = str(labels[n]), labels[f"{n}_severity"]
            log.append(("stage_b", row["path"], "all", "", "re-rendered (empty mask)"))
            changed = True

        elif int(row["water"]) and mask[..., WATER].min() >= FLOOR_DETECT:
            mask[..., WATER] = remove_thin_water_floor(mask[..., WATER])
            changed = True
            floors += 1

        src = row["source_id"]
        if src not in cache:
            cache.clear()
            cache[src] = cv.imread(str(data / clean_of[src]))
        image = cv.imread(str(data / row["path"]))
        for c, name in enumerate(EFFECT_NAMES):
            if int(row[name]):
                visible = visible_pixel_count(cache[src], image, mask[..., c])
                if visible < MIN_VISIBLE_PIXELS:
                    log.append(("stage_b", row["path"], name, row[f"{name}_severity"], visible))
                    unlabel(row, name)
                    mask[..., c] = 0.0
                    changed = True

        if changed:
            shutil.copy2(mask_path, backup / mask_path.name)
            saved = np.clip(mask * 255.0, 0, 255).round().astype(np.uint8)
            cv.imwrite(str(mask_path), saved)
            mask = saved.astype(np.float32) / 255.0  # tiles from the mask as stored, like Stage C reads it
            tiles[i] = np.stack([
                rasterize_tile_label(mask[..., c], meta["grid_h"], meta["grid_w"], meta["thresholds"][n])
                for c, n in enumerate(EFFECT_NAMES)
            ])
        if i % 2000 == 0:
            print(f"  stage_b {i}/{len(rows)}", flush=True)

    shutil.copy2(data / "tile_labels.npy", data / "tile_labels.npy.before_label_fix")
    np.save(data / "tile_labels.npy", tiles)
    write_rows(data, rows)
    meta[FLAG] = True
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"stage_b: thin-water floor removed from {floors} masks")


def regenerated_scratch_mask(raw, source_id, variant_id, base_seed):
    seed = derive_seed(base_seed, source_id, int(variant_id), "scratch")
    return generate_scratch_mask(raw.shape[:2], seed=seed)


def fix_stage_a(data, source_dir, base_seed, log):
    marker = data / "label_fix.json"
    if marker.exists():
        sys.exit(f"{data} was already fixed -- nothing to do")
    rows = read_rows(data)
    clean_of = clean_variants(rows)
    cache = {}
    for i, row in enumerate(rows):
        if row["path"] == clean_of[row["source_id"]]:
            continue
        src = row["source_id"]
        if src not in cache:
            cache.clear()
            cache[src] = (cv.imread(str(data / clean_of[src])), cv.imread(str(source_dir / f"{src}.jpg")))
        clean, raw = cache[src]
        image = cv.imread(str(data / row["path"]))
        active = [c for c in EFFECT_NAMES if int(row[c])]
        for name in active:
            if name == "scratch":
                mask = regenerated_scratch_mask(raw, src, row["variant_id"], base_seed)
            elif len(active) == 1:  # dirt/water masks are not reproducible: whole frame, alone only
                mask = np.ones(image.shape[:2], dtype=np.float32)
            else:
                continue
            visible = visible_pixel_count(clean, image, mask)
            if visible < MIN_VISIBLE_PIXELS:
                log.append(("stage_a", row["path"], name, row[f"{name}_severity"], visible))
                unlabel(row, name)
        if i % 2000 == 0:
            print(f"  stage_a {i}/{len(rows)}", flush=True)
    write_rows(data, rows)
    marker.write_text(json.dumps({FLAG: True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-b", default=None, help="Stage B/C dataset dir")
    parser.add_argument("--stage-a", default=None, help="Stage A dataset dir")
    parser.add_argument("--source", default="data/raw/mio_tcd/images",
                        help="Source photos (re-rendering, Stage A scratch-mask regeneration)")
    parser.add_argument("--seed", type=int, default=0, help="Base seed the datasets were built with")
    args = parser.parse_args()

    for data in filter(None, (args.stage_b, args.stage_a)):
        log = []
        if data == args.stage_b:
            fix_stage_b(Path(data), Path(args.source), args.seed, log)
        else:
            fix_stage_a(Path(data), Path(args.source), args.seed, log)
        with open(Path(data) / "label_fixes.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "image", "class", "severity", "visible_pixels_or_action"])
            w.writerows(log)
        print(f"{data}: {len(log)} changes (listed in label_fixes.csv)")


if __name__ == "__main__":
    main()
