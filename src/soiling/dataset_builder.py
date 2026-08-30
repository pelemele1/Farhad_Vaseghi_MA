"""
Builds the Stage A multi-label dataset (architecture.md §2): for each source
image, generate several distorted variants, each either clean (all-zero
label, a negative) or carrying exactly one of `{dirt, water, scratch}` --
never more than one distortion type combined on the same image (user
decision). The four kinds (clean, dirt, water, scratch) are assigned to a
source image's variant slots deterministically, cycling through the kinds
before shuffling their order, so the dataset is balanced by construction
(with `variants_per_image=4`: exactly one clean + one of each distortion per
source image) rather than left to per-variant probability, which skewed
heavily toward clean when the roll for "any distortion" failed independently
each time. Precomputed once and written to disk -- not applied on-the-fly
during training (CPU cost of the effects would bottleneck a small/frozen-
backbone model on a shared HPC GPU allocation; see docs/development_log.md).

Splits are assigned per *source* image, not per variant, so all variants of
the same source image stay in the same split (no leakage).
"""
import csv
import hashlib
import json
import random
from pathlib import Path

import cv2 as cv
import numpy as np

from src.soiling.effects import add_dirt, add_scratch, add_water
from src.soiling.tile_labels import rasterize_tile_label

EFFECT_NAMES = ("dirt", "water", "scratch")
VARIANT_KINDS = ("clean",) + EFFECT_NAMES
_MAX_SEED = 2**31 - 1

# Stage B (architecture.md §2) per-class tile-coverage thresholds. dirt/water
# are filled regions, so a mid-size threshold is fine; scratch is a thin
# line and would almost never clear a uniform threshold sized for filled
# regions (see tests/test_tile_labels.py's scratch-style case) -- these are
# starting points, meant to be re-checked against the real per-class
# tile-positive rate after the first build (same lesson as Session 5's
# dirt/water texture-pool fix for Stage A).
DEFAULT_TILE_THRESHOLDS = {"dirt": 0.15, "water": 0.15, "scratch": 0.03}


def derive_seed(*parts):
    """Deterministic seed from arbitrary parts (ints/strs) -- NOT Python's
    built-in hash(), which is randomized per-process for strings unless
    PYTHONHASHSEED is fixed, so it wouldn't reproduce across runs."""
    key = "|".join(str(p) for p in parts).encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") % _MAX_SEED


def assign_variant_kinds(variant_count, seed):
    """Deterministically assign each of `variant_count` variant slots for one
    source image to exactly one kind in VARIANT_KINDS. Cycles through the
    four kinds (repeating if variant_count > 4) so counts stay as balanced
    as possible, then shuffles the order with a seeded RNG so the sequence
    isn't always clean-dirt-water-scratch."""
    pattern = [VARIANT_KINDS[i % len(VARIANT_KINDS)] for i in range(variant_count)]
    random.Random(seed).shuffle(pattern)
    return pattern


def apply_effect_combo(image, combo, seeds):
    """combo: (dirt: bool, water: bool, scratch: bool). seeds: dict of
    per-effect seeds. Returns (distorted_image, labels dict)."""
    out = image
    labels = {}
    for name, included in zip(EFFECT_NAMES, combo):
        labels[name] = int(included)
        if not included:
            continue
        if name == "dirt":
            out, _ = add_dirt(out, seed=seeds["dirt"])
        elif name == "water":
            out, _ = add_water(out, seed=seeds["water"])
        elif name == "scratch":
            out, _ = add_scratch(out, seed=seeds["scratch"])
    return out, labels


def build_variant(image, source_id, variant_idx, kind, base_seed):
    """Deterministic single variant: `kind` (one of VARIANT_KINDS) picks which
    single effect is applied, if any; that effect's own randomness is
    derived from (base_seed, source_id, variant_idx)."""
    combo = tuple(name == kind for name in EFFECT_NAMES)
    seeds = {
        name: derive_seed(base_seed, source_id, variant_idx, name)
        for name in EFFECT_NAMES
    }
    return apply_effect_combo(image, combo, seeds)


def assign_splits(source_ids, seed, ratios=(0.8, 0.1, 0.1)):
    """Deterministic per-source split assignment (train/val/test) -- shuffle
    with a seeded RNG, then cut by ratio. All variants of a source id must
    end up on the same side, which is why this operates on source ids, not
    on individual variant records."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    ids = sorted(source_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    split_of = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            split_of[sid] = "train"
        elif i < n_train + n_val:
            split_of[sid] = "val"
        else:
            split_of[sid] = "test"
    return split_of


def build_stage_a_dataset(source_dir, out_dir, variants_per_image=4, seed=0,
                           ratios=(0.8, 0.1, 0.1)):
    """For every image in `source_dir`, generate `variants_per_image`
    distorted variants and write them + a metadata.csv under `out_dir`.
    Returns the list of metadata row dicts written."""
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    source_paths = sorted(source_dir.glob("*.jpg"))
    source_ids = [p.stem for p in source_paths]
    split_of = assign_splits(source_ids, seed=seed, ratios=ratios)

    rows = []
    for path in source_paths:
        source_id = path.stem
        image = cv.imread(str(path))
        split = split_of[source_id]
        kind_order_seed = derive_seed(seed, source_id, "kind_order")
        kinds = assign_variant_kinds(variants_per_image, kind_order_seed)
        for variant_idx, kind in enumerate(kinds):
            out_image, labels = build_variant(
                image, source_id, variant_idx, kind, base_seed=seed
            )
            out_name = f"{source_id}_v{variant_idx:02d}.jpg"
            cv.imwrite(str(images_out / out_name), out_image)
            rows.append({
                "path": f"images/{out_name}",
                "source_id": source_id,
                "variant_id": variant_idx,
                "split": split,
                **labels,
            })

    metadata_path = out_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        fieldnames = ["path", "source_id", "variant_id", "split", *EFFECT_NAMES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def apply_effect_combo_with_masks(image, combo, seeds):
    """Like `apply_effect_combo`, but also returns each class's own
    full-resolution `[0, 1]` mask (all-zero for a class not in `combo`) --
    needed by Stage B to rasterize tile labels from the *exact* mask that
    produced this image, not a separately-regenerated one (see module
    docstring / docs/development_log.md Session 6: add_dirt/add_water aren't
    pixel-reproducible across separate calls, only label-reproducible)."""
    out = image
    labels = {}
    masks = {}
    for name, included in zip(EFFECT_NAMES, combo):
        labels[name] = int(included)
        if not included:
            masks[name] = np.zeros(image.shape[:2], dtype=np.float32)
            continue
        if name == "dirt":
            out, mask = add_dirt(out, seed=seeds["dirt"])
        elif name == "water":
            out, mask = add_water(out, seed=seeds["water"])
        elif name == "scratch":
            out, mask = add_scratch(out, seed=seeds["scratch"])
        masks[name] = mask
    return out, labels, masks


def build_variant_with_mask(image, source_id, variant_idx, kind, base_seed):
    """Stage B counterpart to `build_variant`: same deterministic single-kind
    variant, but also returns the per-class masks."""
    combo = tuple(name == kind for name in EFFECT_NAMES)
    seeds = {
        name: derive_seed(base_seed, source_id, variant_idx, name)
        for name in EFFECT_NAMES
    }
    return apply_effect_combo_with_masks(image, combo, seeds)


def build_stage_b_dataset(source_dir, out_dir, variants_per_image=4, seed=0,
                           ratios=(0.8, 0.1, 0.1), img_size=512,
                           thresholds=None):
    """Stage B dataset (architecture.md §2): same balanced single-distortion
    variants as `build_stage_a_dataset` (reuses the same kind-assignment and
    split logic, so results are directly comparable), plus a per-tile label
    grid rasterized from each variant's own mask in the same call that
    produced its image -- see module-level note on why this can't be a
    post-hoc step.

    Tile grid size = img_size // 32, matching the frozen backbone's P5
    stride (architecture.md: "the feature map is treated as a grid") --
    img_size=512 gives the 16x16 architecture.md itself uses as an example.
    Writes `images/`, `metadata.csv` (same columns as Stage A), a single
    consolidated `tile_labels.npy` (shape (N, 3, grid_h, grid_w) uint8,
    row-aligned with metadata.csv), and `stage_b_meta.json` (img_size,
    grid_h, grid_w, thresholds) under `out_dir`.
    """
    if img_size % 32 != 0:
        raise ValueError(f"img_size must be a multiple of 32 (P5 stride), got {img_size}")
    grid_size = img_size // 32
    thresholds = dict(DEFAULT_TILE_THRESHOLDS if thresholds is None else thresholds)

    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    source_paths = sorted(source_dir.glob("*.jpg"))
    source_ids = [p.stem for p in source_paths]
    split_of = assign_splits(source_ids, seed=seed, ratios=ratios)

    rows = []
    tile_labels = []
    for path in source_paths:
        source_id = path.stem
        image = cv.imread(str(path))
        split = split_of[source_id]
        kind_order_seed = derive_seed(seed, source_id, "kind_order")
        kinds = assign_variant_kinds(variants_per_image, kind_order_seed)
        for variant_idx, kind in enumerate(kinds):
            out_image, labels, masks = build_variant_with_mask(
                image, source_id, variant_idx, kind, base_seed=seed
            )
            out_name = f"{source_id}_v{variant_idx:02d}.jpg"
            cv.imwrite(str(images_out / out_name), out_image)

            grid = np.stack([
                rasterize_tile_label(masks[name], grid_size, grid_size, thresholds[name])
                for name in EFFECT_NAMES
            ])
            tile_labels.append(grid)

            rows.append({
                "path": f"images/{out_name}",
                "source_id": source_id,
                "variant_id": variant_idx,
                "split": split,
                **labels,
            })

    metadata_path = out_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        fieldnames = ["path", "source_id", "variant_id", "split", *EFFECT_NAMES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tile_labels_array = np.stack(tile_labels).astype(np.uint8)
    np.save(out_dir / "tile_labels.npy", tile_labels_array)

    meta = {
        "img_size": img_size,
        "grid_h": grid_size,
        "grid_w": grid_size,
        "class_names": list(EFFECT_NAMES),
        "thresholds": thresholds,
    }
    with open(out_dir / "stage_b_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return rows, tile_labels_array
