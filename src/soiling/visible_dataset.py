"""
One dataset for all three stages with visible-change ground truth (Session
27; see src/soiling/visible_labels.py for the labeling rule).

Same source photos, splits, 14 variants per photo (clean, 3 effects x 3
severities, 4 combinations) and seeds as the Stage B dataset, so results stay
comparable. Writes under `out_dir`:
  images/<source>_vNN.jpg   distorted image (native resolution)
  labels/<source>_vNN.png   uint8 class map at native resolution,
                            0 clean / 1 dirt / 2 water / 3 scratch
  tile_labels.npy           (N, grid, grid) uint8 class per tile (img_size // 32)
  metadata.csv              one row per image (see METADATA_FIELDNAMES)
  visible_meta.json         label rule parameters and class pixel counts
"""
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import cv2 as cv
import numpy as np

from src.soiling.dataset_builder import (
    ALL_VARIANT_KINDS_WITH_SEVERITY,
    apply_visible_effect_with_seed,
    assign_splits,
    assign_variant_kinds,
    derive_seed,
    _resolve_severities,
)
from src.soiling.effects import MIN_PIXEL_CHANGE, MIN_VISIBLE_PIXELS
from src.soiling.visible_labels import (
    CHANGE_BLUR_SIGMA,
    CLASS_NAMES,
    CLOSE_KERNEL,
    EFFECT_NAMES,
    MAX_HOLE_PIXELS,
    MIN_COMPONENT_PIXELS,
    class_pixel_counts,
    tile_class_grid,
    visible_label_map,
)

# Tile cutoffs: a tile takes a class when that class's visible area covers at
# least this fraction of it (same values as the Stage B dataset).
DEFAULT_TILE_THRESHOLDS = {"dirt": 0.20, "water": 0.25, "scratch": 0.015}

METADATA_FIELDNAMES = [
    "path", "source_id", "variant_id", "split",
    *EFFECT_NAMES, *[f"{n}_severity" for n in EFFECT_NAMES],
    "kind", *[f"{n}_pixels" for n in EFFECT_NAMES], "total_pixels", "dominant",
]


def render_visible_variant(image, source_id, variant_idx, kind, base_seed):
    """(distorted image, class map, metadata fields) for one variant. An
    applied effect that stays invisible after the retries is dropped; one that
    is visible when applied but hidden by a later layer keeps its place in the
    image and simply gets no pixels in the class map."""
    combo, severities = _resolve_severities(kind, base_seed, source_id, variant_idx)
    out = image
    steps, masks = [], {}
    for name, active in zip(EFFECT_NAMES, combo):
        if not active:
            continue
        seed = derive_seed(base_seed, source_id, variant_idx, name)
        out, mask, used_seed = apply_visible_effect_with_seed(out, name, seed, severities[name])
        if mask is not None:
            steps.append((name, used_seed, severities[name]))
            masks[name] = mask

    labels = visible_label_map(image, out, steps, masks)
    counts = class_pixel_counts(labels)
    applied = {name: sev for name, _, sev in steps}
    fields = {"kind": "+".join(f"{n}:{applied[n]}" for n in EFFECT_NAMES if n in applied) or "clean"}
    for name in EFFECT_NAMES:
        present = counts[name] >= MIN_VISIBLE_PIXELS
        fields[name] = int(present)
        fields[f"{name}_severity"] = applied[name] if present else "none"
        fields[f"{name}_pixels"] = counts[name]
    fields["total_pixels"] = int(labels.size)
    present = [n for n in EFFECT_NAMES if fields[n]]
    fields["dominant"] = max(present, key=lambda n: counts[n]) if present else "clean"
    return out, labels, fields


def _build_source(job):
    path, split, out_dir, variants_per_image, seed, grid, thresholds = job
    source_id = Path(path).stem
    image = cv.imread(str(path))
    kinds = assign_variant_kinds(variants_per_image, derive_seed(seed, source_id, "kind_order"),
                                 kinds=ALL_VARIANT_KINDS_WITH_SEVERITY)
    results = []
    for variant_idx, kind in enumerate(kinds):
        out, labels, fields = render_visible_variant(image, source_id, variant_idx, kind, seed)
        stem = f"{source_id}_v{variant_idx:02d}"
        cv.imwrite(str(Path(out_dir) / "images" / f"{stem}.jpg"), out)
        cv.imwrite(str(Path(out_dir) / "labels" / f"{stem}.png"), labels)
        row = {"path": f"images/{stem}.jpg", "source_id": source_id, "variant_id": variant_idx,
               "split": split, **fields}
        results.append((row, tile_class_grid(labels, grid, grid, thresholds)))
    return results


def build_visible_dataset(source_dir, out_dir, variants_per_image=14, seed=0, ratios=(0.8, 0.1, 0.1),
                          img_size=512, thresholds=None, workers=1, max_sources=None, log_every=50):
    if img_size % 32:
        raise ValueError(f"img_size must be a multiple of 32, got {img_size}")
    grid = img_size // 32
    thresholds = dict(DEFAULT_TILE_THRESHOLDS if thresholds is None else thresholds)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    source_paths = sorted(Path(source_dir).glob("*.jpg"))
    split_of = assign_splits([p.stem for p in source_paths], seed=seed, ratios=ratios)
    if max_sources is not None:
        source_paths = source_paths[:max_sources]
    jobs = [(str(p), split_of[p.stem], str(out_dir), variants_per_image, seed, grid, thresholds)
            for p in source_paths]

    rows, tiles = [], []
    pool = Pool(workers) if workers > 1 else None
    results = pool.imap(_build_source, jobs) if pool else map(_build_source, jobs)
    for i, source_results in enumerate(results, start=1):
        for row, tile_grid in source_results:
            rows.append(row)
            tiles.append(tile_grid)
        if i % log_every == 0 or i == len(jobs):
            print(f"{i}/{len(jobs)} source photos done", flush=True)
    if pool:
        pool.close()
        pool.join()

    with open(out_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tiles = np.stack(tiles).astype(np.uint8)
    np.save(out_dir / "tile_labels.npy", tiles)

    pixel_counts = {}
    for split in ("train", "val", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        total = sum(r["total_pixels"] for r in split_rows)
        per_class = {n: sum(r[f"{n}_pixels"] for r in split_rows) for n in EFFECT_NAMES}
        pixel_counts[split] = {"clean": total - sum(per_class.values()), **per_class}
    tile_counts = {split: np.bincount(tiles[[r["split"] == split for r in rows]].ravel(),
                                      minlength=len(CLASS_NAMES)).tolist()
                   for split in ("train", "val", "test")}
    meta = {
        "img_size": img_size, "grid_h": grid, "grid_w": grid,
        "class_names": list(CLASS_NAMES),
        "tile_thresholds": thresholds,
        "label_rule": "visible change: per pixel, the effect whose removal changes the image most "
                      f"(a visible scratch wins), if > {MIN_PIXEL_CHANGE} gray levels",
        "min_pixel_change": MIN_PIXEL_CHANGE,
        "change_blur_sigma": CHANGE_BLUR_SIGMA,
        "close_kernel": CLOSE_KERNEL,
        "min_component_pixels": MIN_COMPONENT_PIXELS,
        "max_hole_pixels": MAX_HOLE_PIXELS,
        "min_visible_pixels": MIN_VISIBLE_PIXELS,
        "pixel_counts": pixel_counts,
        "tile_counts": tile_counts,
        "seed": seed, "variants_per_image": variants_per_image,
    }
    with open(out_dir / "visible_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return rows, tiles
