"""
Stage A/B loss (architecture.md §4): "BCE-with-logits (multi-label). For
class imbalance (lots of 'clean') -> pos_weight or focal loss."
"""
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _align_per_class(vec, ndim):
    """Reshapes a (C,) tensor to broadcast against the channel dim (dim 1)
    of an (B, C) or (B, C, H, W) tensor -- (1, C) or (1, C, 1, 1). Same
    broadcasting pitfall `build_stage_b_loss`'s pos_weight reshape guards
    against: PyTorch aligns broadcast dims from the right, so a bare (C,)
    vector would land on the *last* dim, not the channel dim, unless
    explicitly reshaped."""
    return vec.view(1, -1, *([1] * (ndim - 2)))


class FocalLossWithLogits(nn.Module):
    """Binary focal loss (Lin et al. 2017, "Focal Loss for Dense Object
    Detection", RetinaNet) -- architecture.md §4's own listed alternative to
    `BCEWithLogitsLoss(pos_weight=...)` for class imbalance.

    Motivation (see docs/development_log.md Session 16): Stage B's scratch
    class needed `pos_weight` ~95 to counter its ~1% tile-positive rate.
    That's a single flat multiplier applied to *every* missed positive,
    which empirically pushed the trained head to over-predict "scratch"
    almost everywhere (recall 0.852, precision 0.063) rather than localize
    it -- a large enough pos_weight rewards blanket-guessing "positive" more
    than it rewards being selective. Focal loss instead down-weights
    already-easy/confident examples via the `(1 - p_t) ** gamma` factor and
    concentrates gradient on hard ones, without any single term dominating
    the loss the way a ~95x multiplier does.

    `alpha`: foreground/background balance weight -- a python float
    (applied to every class equally) or a (num_classes,) tensor (one alpha
    per class). Default 0.25 is the RetinaNet paper's own default, used
    as-is here rather than re-deriving a new per-class value from this
    dataset's positive rates -- reusing pos_weight's inverse-frequency
    formula for alpha too would risk reproducing the same over-triggering
    problem this loss exists to avoid.
    `gamma`: focusing parameter (RetinaNet default 2.0; 0 reduces this to
    plain alpha-weighted BCE, no focusing effect).
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)

        alpha = self.alpha
        if isinstance(alpha, torch.Tensor):
            alpha = _align_per_class(alpha.to(logits.device), logits.dim())
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


def build_stage_b_focal_loss(alpha=0.25, gamma=2.0):
    return FocalLossWithLogits(alpha=alpha, gamma=gamma)
