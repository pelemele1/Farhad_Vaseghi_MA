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
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


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
    class with precision/recall/f1 (at `threshold`) and two threshold-
    independent ranking metrics -- average precision (AP, i.e. AUC-PR) and
    ROC-AUC -- plus the positive-class support count. `threshold` is either
    a single float applied to every class (the original behavior) or a
    dict[class_name, float] giving each class its own decision threshold
    (see `src.eval.thresholds.tune_per_class_thresholds`) -- a lower
    threshold for a class with an unfavorable precision/recall tradeoff at
    0.5 doesn't have to drag every other class's threshold along with it.

    AP is left as NaN for a class with zero positives in this split -- it
    isn't a meaningful score without at least one positive example.
    ROC-AUC additionally needs at least one *negative* too (it isn't
    meaningful for an all-positive or all-negative split either), so its
    NaN guard is stricter than AP's.

    Computes each class's precision/recall/f1 independently via
    average="binary" on that one column, rather than calling
    precision_recall_fscore_support(..., average=None) on the whole
    (n_samples, n_classes) array at once: sklearn infers multilabel vs.
    plain binary classification from the *array shape*, and a single-class
    evaluation (one column) gets misread as ordinary 2-class binary
    classification instead of 1-class multilabel, silently shifting what
    each output index means. Doesn't happen with >=2 classes (our actual
    3-class case), but per-column is correct regardless of class count."""
    per_class_threshold = isinstance(threshold, dict)
    rows = []
    for i, name in enumerate(class_names):
        t = threshold[name] if per_class_threshold else threshold
        y_true, y_prob = labels[:, i], probs[:, i]
        y_pred = (y_prob >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        support = int(y_true.sum())
        n = len(y_true)
        ap = average_precision_score(y_true, y_prob) if support > 0 else float("nan")
        roc_auc = (
            roc_auc_score(y_true, y_prob) if 0 < support < n else float("nan")
        )
        rows.append({
            "class": name,
            "threshold": float(t),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "ap": float(ap),
            "roc_auc": float(roc_auc),
            "support": support,
        })
    return rows
