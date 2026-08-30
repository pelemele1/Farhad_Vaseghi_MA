"""
Shared per-class binary classification metrics -- originally written for
Stage A (architecture.md plan Step 6, `scripts/evaluate_stage_a.py`), moved
here so Stage B's evaluator (`scripts/evaluate_stage_b.py`) reuses the exact
same sklearn-shape-pitfall-safe implementation instead of duplicating it.
Works on any (n_samples, n_classes) array -- Stage A calls it directly on
per-image predictions, Stage B flattens its per-tile (N, C, H, W) predictions
to (N*H*W, C) first.
"""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


@torch.no_grad()
def collect_predictions(backbone, head, loader, device):
    """Runs the whole loader through backbone -> head -> sigmoid. Returns
    (probs, labels) as numpy arrays, concatenated along the batch dim --
    works for both Stage A's (n_samples, n_classes) shape and Stage B's
    (n_samples, n_classes, grid_h, grid_w) shape."""
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = head(backbone(images))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(labels, probs, class_names, threshold=0.5):
    """labels, probs: (n_samples, n_classes) arrays. Returns one dict per
    class with precision/recall/f1 (at `threshold`) and threshold-
    independent average precision (AP), plus the positive-class support
    count. AP is left as NaN for a class with zero positives in this split
    -- it isn't a meaningful score without at least one positive example.

    Computes each class's precision/recall/f1 independently via
    average="binary" on that one column, rather than calling
    precision_recall_fscore_support(..., average=None) on the whole
    (n_samples, n_classes) array at once: sklearn infers multilabel vs.
    plain binary classification from the *array shape*, and a single-class
    evaluation (one column) gets misread as ordinary 2-class binary
    classification instead of 1-class multilabel, silently shifting what
    each output index means. Doesn't happen with >=2 classes (our actual
    3-class case), but per-column is correct regardless of class count."""
    preds = (probs >= threshold).astype(int)
    rows = []
    for i, name in enumerate(class_names):
        y_true, y_pred, y_prob = labels[:, i], preds[:, i], probs[:, i]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        support = int(y_true.sum())
        ap = average_precision_score(y_true, y_prob) if support > 0 else float("nan")
        rows.append({
            "class": name,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "ap": float(ap),
            "support": support,
        })
    return rows
