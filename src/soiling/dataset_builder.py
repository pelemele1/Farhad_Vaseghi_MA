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
import random
from pathlib import Path

import cv2 as cv

from src.soiling.effects import add_dirt, add_scratch, add_water

EFFECT_NAMES = ("dirt", "water", "scratch")
VARIANT_KINDS = ("clean",) + EFFECT_NAMES
_MAX_SEED = 2**31 - 1


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
