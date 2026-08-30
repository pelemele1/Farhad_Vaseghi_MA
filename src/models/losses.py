"""
Stage A/B loss (architecture.md §4): "BCE-with-logits (multi-label). For
class imbalance (lots of 'clean') -> pos_weight or focal loss."
"""
import csv

import numpy as np
import torch
import torch.nn as nn

from src.soiling.dataset_builder import EFFECT_NAMES


def compute_pos_weight(metadata_csv, split=None):
    """pos_weight[c] = (#negatives for class c) / (#positives for class c) --
    BCEWithLogitsLoss's standard imbalance-correction formula, up-weighting
    the rarer class's positives by how outnumbered they are. `split`
    restricts the count to one of metadata.csv's split values (e.g.
    "train"); pass None to use every row."""
    counts = {name: 0 for name in EFFECT_NAMES}
    total = 0
    with open(metadata_csv, newline="") as f:
        for row in csv.DictReader(f):
            if split is not None and row["split"] != split:
                continue
            total += 1
            for name in EFFECT_NAMES:
                counts[name] += int(row[name])

    return torch.tensor(
        [(total - counts[name]) / max(counts[name], 1) for name in EFFECT_NAMES],
        dtype=torch.float32,
    )


def build_stage_a_loss(pos_weight=None):
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def compute_tile_pos_weight(tile_labels_npy, metadata_csv, split=None):
    """Stage B counterpart to `compute_pos_weight`: pos_weight[c] =
    (#negative tiles for class c) / (#positive tiles for class c), counted
    over every tile of every row (not just every row) -- tile-level
    imbalance is much more extreme than image-level (an image labeled
    "dirt" still has mostly clean tiles, since the effect only covers part
    of the frame), so this must count tiles, not rows.

    `tile_labels_npy` is the (N, C, H, W) uint8 array `build_stage_b_dataset`
    writes; `split` restricts the count to rows whose metadata.csv `split`
    column matches (pass None to use every row). Row order in
    `tile_labels_npy` is assumed to match `metadata.csv`'s row order, which
    is how `build_stage_b_dataset` writes them."""
    tile_labels = np.load(tile_labels_npy) if not hasattr(tile_labels_npy, "shape") else tile_labels_npy

    with open(metadata_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(tile_labels), (
        f"metadata.csv has {len(rows)} rows but tile_labels has {len(tile_labels)} -- "
        "they must be row-aligned (same build_stage_b_dataset call)"
    )

    if split is not None:
        mask = np.array([r["split"] == split for r in rows])
        tile_labels = tile_labels[mask]

    positives = tile_labels.sum(axis=(0, 2, 3))  # per class, over rows+H+W
    total = tile_labels.shape[0] * tile_labels.shape[2] * tile_labels.shape[3]
    return torch.tensor(
        [(total - p) / max(p, 1) for p in positives],
        dtype=torch.float32,
    )


def build_stage_b_loss(pos_weight=None):
    """Like `build_stage_a_loss`, but the target is (B, C, H, W), not
    (B, C) -- a plain (C,)-shaped pos_weight would broadcast against the
    *last* dim (W), not the channel dim (broadcasting aligns from the
    right), silently weighting the wrong axis. Reshaped to (C, 1, 1) here
    so it broadcasts against the channel dim as intended."""
    if pos_weight is not None:
        pos_weight = pos_weight.view(-1, 1, 1)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
