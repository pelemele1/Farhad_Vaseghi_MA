"""
Reusable per-class decision-threshold tuning -- generalized from Session 18's
one-off `scripts/threshold_sweep_stage_b.py` (`best_f1_threshold`) so both
`scripts/evaluate_stage_a.py` and `scripts/evaluate_stage_b.py` can tune and
use per-class thresholds instead of one shared 0.5 for every class. A fixed
global threshold forces every class onto the same point of its own
precision/recall curve, even though the curves themselves (and where the
best tradeoff sits) differ a lot per class -- e.g. Stage B's `scratch` class
needs a very different operating point than `dirt`/`water` (see
docs/stage_b_final_report.md's threshold-tuning discussion).
"""
import numpy as np
from sklearn.metrics import precision_recall_curve


def best_f1_threshold(y_true, y_prob):
    """Sweeps every threshold on the precision-recall curve and returns the
    one with the highest F1: (threshold, precision, recall, f1) as floats.
    `precision_recall_curve` returns one more precision/recall point than
    threshold (the last point is threshold=+inf, i.e. "predict nothing"),
    so the last point is excluded from the F1 argmax."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall + 1e-12),
        0.0,
    )
    best_idx = int(np.argmax(f1[:-1]))
    return (
        float(thresholds[best_idx]),
        float(precision[best_idx]),
        float(recall[best_idx]),
        float(f1[best_idx]),
    )


def threshold_for_recall(y_true, y_prob, target_recall):
    """Highest threshold whose recall on (y_true, y_prob) is still >=
    `target_recall` -- for a gate, whose job is to drop as many negatives
    as possible while keeping a guaranteed share of positives (best-F1 is
    the wrong criterion there: with ~93% positives it lands near 0 and lets
    almost every negative through). Returns (threshold, recall,
    negative_pass_rate)."""
    y_true = np.asarray(y_true).astype(bool)
    y_prob = np.asarray(y_prob)
    pos = np.sort(y_prob[y_true])[::-1]
    k = int(np.ceil(target_recall * len(pos)))
    threshold = float(pos[max(k, 1) - 1])
    recall = float((y_prob[y_true] >= threshold).mean())
    neg_pass = float((y_prob[~y_true] >= threshold).mean()) if (~y_true).any() else float("nan")
    return threshold, recall, neg_pass


def threshold_for(threshold, class_name):
    """A single float applies to every class; a dict gives each class its own."""
    return threshold[class_name] if isinstance(threshold, dict) else threshold


def tune_per_class_thresholds(labels, probs, class_names):
    """labels, probs: (n_samples, n_classes) arrays (same shape
    `compute_metrics` expects -- flatten Stage B's tile grid first). Returns
    dict[class_name, float]: each class's best-F1 threshold, found
    independently on the given split (call this on the val split, then feed
    the result into `compute_metrics(..., threshold=tuned)` on the test
    split -- tuning and evaluating on the same split would overstate how
    good the tuned threshold generalizes)."""
    tuned = {}
    for i, name in enumerate(class_names):
        threshold, _precision, _recall, _f1 = best_f1_threshold(labels[:, i], probs[:, i])
        tuned[name] = threshold
    return tuned
