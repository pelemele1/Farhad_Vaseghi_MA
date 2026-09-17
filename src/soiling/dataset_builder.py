"""
Builds the Stage A multi-label dataset (architecture.md §2): for each source
image, generate several distorted variants, each either clean (all-zero
label, a negative) or carrying one or more of `{dirt, water, scratch}`.

Originally (Sessions 6-18) this was strictly single-distortion-or-clean --
never more than one distortion type combined on the same image (an explicit
user decision at the time, see docs/development_log.md Session 6). Session
19+ (supervisor item 5) added an *additive* `include_combos` option: 4
additional "combo" kinds (the 3 pairs + the full triple, see COMBO_KINDS)
alongside the original 4, selected explicitly via `include_combos=True` --
the original single-distortion-or-clean behavior is still exactly what you
get by default. The kinds (4 or 8, see ALL_VARIANT_KINDS) are assigned to a
source image's variant slots deterministically, cycling through the kinds
before shuffling their order, so the dataset is balanced by construction
(with `variants_per_image` equal to the kind count: exactly one of each kind
per source image) rather than left to per-variant probability, which skewed
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
# Multi-distortion variant kinds (Session 19+, supervisor item 5): every
# 2-way combination plus the full 3-way one. Additive, not a replacement --
# VARIANT_KINDS is untouched so every existing caller/test that never passes
# `kinds=`/`include_combos=True` keeps today's single-distortion-or-clean
# behavior exactly as before. `apply_effect_combo`/`apply_effect_combo_with_
# masks`/`rasterize_tile_label` already support arbitrary multi-hot combos
# with no changes needed -- effects.py's module docstring says composability
# was the design intent from day one, it just was never exercised via the
# dataset builder until now (see docs/development_log.md Session 19).
COMBO_KINDS = ("dirt+water", "dirt+scratch", "water+scratch", "dirt+water+scratch")
ALL_VARIANT_KINDS = VARIANT_KINDS + COMBO_KINDS
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


def assign_variant_kinds(variant_count, seed, kinds=None):
    """Deterministically assign each of `variant_count` variant slots for one
    source image to exactly one kind in `kinds` (default VARIANT_KINDS, the
    original 4: clean + one of each single effect -- pass
    `kinds=ALL_VARIANT_KINDS` for the 8-kind set that also includes combos).
    Cycles through the kinds (repeating if variant_count > len(kinds)) so
    counts stay as balanced as possible, then shuffles the order with a
    seeded RNG so the sequence isn't always in the same fixed order."""
    kinds = VARIANT_KINDS if kinds is None else kinds
    pattern = [kinds[i % len(kinds)] for i in range(variant_count)]
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


def _combo_from_kind(kind):
    """Parses a kind string -- "clean", a single effect name ("dirt"), or a
    "+"-joined combo ("dirt+water", "dirt+water+scratch") -- into a
    multi-hot (dirt, water, scratch) bool tuple. Handles both VARIANT_KINDS
    and COMBO_KINDS uniformly."""
    active = set() if kind == "clean" else set(kind.split("+"))
    return tuple(name in active for name in EFFECT_NAMES)


def build_variant(image, source_id, variant_idx, kind, base_seed):
    """Deterministic single variant: `kind` (one of ALL_VARIANT_KINDS) picks
    which effect(s) are applied, if any; each active effect's own randomness
    is derived from (base_seed, source_id, variant_idx, effect_name) --
    independent per effect even when several are combined on one image."""
    combo = _combo_from_kind(kind)
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
                           ratios=(0.8, 0.1, 0.1), include_combos=False):
    """For every image in `source_dir`, generate `variants_per_image`
    distorted variants and write them + a metadata.csv under `out_dir`.
    `include_combos=False` (default): today's original behavior, unchanged
    -- 4 kinds (clean + one of each single effect). `include_combos=True`:
    8 kinds (adds the 3 pairs + the full triple, see COMBO_KINDS) --
    `variants_per_image=8` gives exact balance, same principle as the
    default 4-kind case. Returns the list of metadata row dicts written."""
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    kinds_pool = ALL_VARIANT_KINDS if include_combos else VARIANT_KINDS

    source_paths = sorted(source_dir.glob("*.jpg"))
    source_ids = [p.stem for p in source_paths]
    split_of = assign_splits(source_ids, seed=seed, ratios=ratios)

    rows = []
    for path in source_paths:
        source_id = path.stem
        image = cv.imread(str(path))
        split = split_of[source_id]
        kind_order_seed = derive_seed(seed, source_id, "kind_order")
        kinds = assign_variant_kinds(variants_per_image, kind_order_seed, kinds=kinds_pool)
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
    """Stage B counterpart to `build_variant`: same deterministic variant
    (single-effect or combo), but also returns the per-class masks."""
    combo = _combo_from_kind(kind)
    seeds = {
        name: derive_seed(base_seed, source_id, variant_idx, name)
        for name in EFFECT_NAMES
    }
    return apply_effect_combo_with_masks(image, combo, seeds)


def build_stage_b_dataset(source_dir, out_dir, variants_per_image=4, seed=0,
                           ratios=(0.8, 0.1, 0.1), img_size=512,
                           thresholds=None, include_combos=False):
    """Stage B dataset (architecture.md §2): same balanced variants as
    `build_stage_a_dataset` (reuses the same kind-assignment and split
    logic, so results are directly comparable, including the same
    `include_combos` flag -- see its docstring), plus a per-tile label grid
    rasterized from each variant's own mask in the same call that produced
    its image -- see module-level note on why this can't be a post-hoc step.

    Tile grid size = img_size // 32, matching the frozen backbone's P5
    stride (architecture.md: "the feature map is treated as a grid") --
    img_size=512 gives the 16x16 architecture.md itself uses as an example.
    Writes `images/`, `metadata.csv` (same columns as Stage A), a single
    consolidated `tile_labels.npy` (shape (N, 3, grid_h, grid_w) uint8,
    row-aligned with metadata.csv), and `stage_b_meta.json` (img_size,
    grid_h, grid_w, thresholds) under `out_dir`. A combo variant's tile grid
    can have more than one class positive in the same tile wherever the
    combined effects' masks overlap -- `rasterize_tile_label` is already
    called independently per class below, so this needs no special-casing.
    """
    if img_size % 32 != 0:
        raise ValueError(f"img_size must be a multiple of 32 (P5 stride), got {img_size}")
    grid_size = img_size // 32
    thresholds = dict(DEFAULT_TILE_THRESHOLDS if thresholds is None else thresholds)
    kinds_pool = ALL_VARIANT_KINDS if include_combos else VARIANT_KINDS

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
        kinds = assign_variant_kinds(variants_per_image, kind_order_seed, kinds=kinds_pool)
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
