"""
Per-severity evaluation breakdown (Session 20, Round 3): now that
`metadata.csv` records each class's severity ("low"/"medium"/"high"/"none",
see src/soiling/dataset_builder.py), evaluation can answer "does the model
do worse on subtle distortions than obvious ones" directly -- almost
certainly the actual reason balanced severity levels were asked for, not
balance for its own sake.
"""
import numpy as np

from src.eval.metrics import compute_metrics


def compute_metrics_by_severity(rows, labels, probs, class_names, threshold):
    """labels, probs: (n_samples, n_classes) arrays, same shape
    compute_metrics expects (for Stage B, pre-broadcast from image-level to
    tile-level first -- see `broadcast_rows_to_tiles`). `rows`: one dict per
    sample (or repeated per tile), each carrying f"{class}_severity".

    For every class and every severity level, evaluates JUST that level's
    positives together with every genuine negative ("none") for that class
    -- without negatives in the subset, precision/recall/AP are undefined,
    so this isolates "how well does the model separate severity-Y positives
    from true negatives" rather than mixing severity levels together.
    Returns dict[class_name, dict[severity_level, metrics_row]] (a level is
    omitted if this split has no samples at it)."""
    severity_levels = ("low", "medium", "high")
    per_class_threshold = isinstance(threshold, dict)
    results = {}
    for i, name in enumerate(class_names):
        severities = [r[f"{name}_severity"] for r in rows]
        t = threshold[name] if per_class_threshold else threshold
        by_level = {}
        for level in severity_levels:
            if not any(s == level for s in severities):
                continue  # no positives at this level in this split -- omit it, not a 0-support row
            mask = np.array([s == level or s == "none" for s in severities])
            sub_labels = labels[mask, i:i + 1]
            sub_probs = probs[mask, i:i + 1]
            by_level[level] = compute_metrics(sub_labels, sub_probs, [name], threshold=t)[0]
        results[name] = by_level
    return results


def broadcast_rows_to_tiles(rows, grid_h, grid_w):
    """Repeats each image-level metadata row grid_h*grid_w times, matching
    scripts.evaluate_stage_b.flatten_tiles's own (N,C,H,W) -> (N*H*W,C)
    transpose(0,2,3,1)+reshape order (h varies before n, w varies before h)
    -- so row i*grid_h*grid_w + j lines up with the same tile
    flatten_tiles(labels, probs)[i*grid_h*grid_w + j] does. Needed because
    severity is an image-level property even in the tile dataset."""
    broadcast = []
    for row in rows:
        broadcast.extend([row] * (grid_h * grid_w))
    return broadcast
