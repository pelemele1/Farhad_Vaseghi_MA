"""
Stage A loss (architecture.md §4): "BCE-with-logits (multi-label). For
class imbalance (lots of 'clean') -> pos_weight or focal loss."
"""
import csv

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
