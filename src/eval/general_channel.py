"""
Derives the tile-wise "any distortion" (general) channel on the fly from the
3 already-trained per-class Stage B channels -- NOT a separately trained 4th
output. The supervisor's ask was framed as a reconstruction/consistency
check ("union of all the classes has to reach the initial tile-wise
structure"), and a 4th trained channel would just re-learn a deterministic
function of 3 already-well-trained channels at the cost of a new HPC
retrain, for no new information. See docs/stage_b_final_report.md for the
write-up of this design choice. Works on ground truth (0/1 tile_labels.npy-
shaped arrays) and on predicted probabilities (sigmoid outputs) alike.
"""
import numpy as np


def general_tile_labels(tile_labels):
    """(..., C, H, W) 0/1 array -> (..., 1, H, W): logical OR across the
    class axis (ground truth "any class active" per tile)."""
    return tile_labels.max(axis=-3, keepdims=True)


def general_tile_probs(tile_probs):
    """(..., C, H, W) probability array -> (..., 1, H, W): noisy-OR combine
    -- 1 - prod_c(1 - p_c), assuming per-class conditional independence (the
    same assumption each class's own independent sigmoid/BCE already makes).
    Used instead of max(p_c) because max under-reports "any" confidence when
    two classes are each moderately (not individually decisively) likely."""
    return 1.0 - np.prod(1.0 - tile_probs, axis=-3, keepdims=True)


def general_channel_consistency(tile_probs, threshold=0.5):
    """Diagnostic only, not a loss: fraction of tiles where thresholded
    per-class OR agrees with the thresholded noisy-OR general channel.
    Expected close to (not exactly) 1.0 -- a large gap flags a bug in this
    derivation, not a model-training failure."""
    per_class_active = (tile_probs >= threshold).any(axis=-3, keepdims=True)
    general_active = general_tile_probs(tile_probs) >= threshold
    return float((per_class_active == general_active).mean())
