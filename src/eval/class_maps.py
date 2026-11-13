"""
Metrics for one-class-per-location maps (Stage B tiles, Stage C pixels;
Session 27). Per image, `map_stats` keeps a confusion matrix and, per
distortion class, histograms of that class's probability over the locations
that are / are not that class. Summed over any subset of images (a split, a
severity level, gated or not) they give precision / recall / F1 / IoU of the
argmax map and average precision of the probabilities -- without storing
millions of per-pixel probabilities.
"""
import numpy as np

N_BINS = 1000


def decide(probs, offsets=None, axis=0):
    """Class per location: argmax over `axis` of log(prob) + per-class offset
    (offsets for classes 1..C-1, clean fixed at 0; tuned on val by
    `tune_class_offsets` to undo the bias the training class weights give the
    rare classes). No offsets = plain argmax."""
    if offsets is None:
        return np.asarray(probs).argmax(axis)
    shape = [1] * np.ndim(probs)
    shape[axis] = -1
    bias = np.concatenate([[0.0], np.asarray(offsets, np.float64)]).reshape(shape)
    return (np.log(np.maximum(probs, 1e-12)) + bias).argmax(axis)


def macro_f1(gt, pred, n_classes):
    conf = np.bincount(gt * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(conf)[1:].astype(np.float64)
    denom = conf[1:, :].sum(1) + conf[:, 1:].sum(0)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def tune_class_offsets(probs, gt, grid=np.linspace(-4.0, 2.0, 31), rounds=3):
    """Per-class log-offsets (classes 1..C-1) maximizing the mean F1 of the
    distortion classes. probs (N, C), gt (N,). Coordinate search over `grid`."""
    n_classes = probs.shape[1]
    log_p = np.log(np.maximum(probs, 1e-12))
    gt = np.asarray(gt).astype(np.int64)
    offsets = np.zeros(n_classes - 1)
    best = macro_f1(gt, log_p.argmax(1), n_classes)
    for _ in range(rounds):
        for k in range(n_classes - 1):
            for value in grid:
                trial = offsets.copy()
                trial[k] = value
                score = macro_f1(gt, (log_p + np.concatenate([[0.0], trial])).argmax(1), n_classes)
                if score > best:
                    best, offsets = score, trial
    return offsets.tolist(), best


def map_stats(probs, gt, offsets=None, n_bins=N_BINS):
    """probs (C, H, W) softmax output, gt (H, W) class indices (0 = clean).
    Returns {"conf": (C, C) int64 [gt, pred], "hist_pos"/"hist_neg":
    (C-1, n_bins) int64 for classes 1..C-1}; pred uses `decide(probs, offsets)`."""
    n_classes = probs.shape[0]
    gt = np.asarray(gt).ravel().astype(np.int64)
    pred = decide(probs.reshape(n_classes, -1), offsets)
    conf = np.bincount(gt * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    bins = np.minimum((probs.reshape(n_classes, -1) * n_bins).astype(np.int64), n_bins - 1)
    hist_pos = np.zeros((n_classes - 1, n_bins), np.int64)
    hist_neg = np.zeros((n_classes - 1, n_bins), np.int64)
    for k in range(1, n_classes):
        is_k = gt == k
        hist_pos[k - 1] = np.bincount(bins[k][is_k], minlength=n_bins)
        hist_neg[k - 1] = np.bincount(bins[k][~is_k], minlength=n_bins)
    return {"conf": conf, "hist_pos": hist_pos, "hist_neg": hist_neg}


def stopped_stats(stats):
    """The stats the same image gets when the gate stops it: every location
    predicted clean, every distortion probability 0."""
    conf = np.zeros_like(stats["conf"])
    conf[:, 0] = stats["conf"].sum(axis=1)
    hist_pos = np.zeros_like(stats["hist_pos"])
    hist_neg = np.zeros_like(stats["hist_neg"])
    hist_pos[:, 0] = stats["hist_pos"].sum(axis=1)
    hist_neg[:, 0] = stats["hist_neg"].sum(axis=1)
    return {"conf": conf, "hist_pos": hist_pos, "hist_neg": hist_neg}


def sum_stats(stats_list):
    keys = ("conf", "hist_pos", "hist_neg")
    return {k: np.sum([s[k] for s in stats_list], axis=0) for k in keys}


def ap_from_histograms(hist_pos, hist_neg):
    """Average precision of ranking by probability, from binned scores
    (locations in the same bin tie). NaN without positives."""
    tp = np.cumsum(hist_pos[::-1])
    fp = np.cumsum(hist_neg[::-1])
    n_pos = tp[-1]
    if n_pos == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def metrics_from_stats(stats, class_names):
    """One dict per distortion class (class_names[1:]): precision, recall, f1,
    iou of the argmax map, ap of the probability, support (locations)."""
    conf = stats["conf"]
    rows = []
    for k in range(1, len(class_names)):
        tp = conf[k, k]
        fp = conf[:, k].sum() - tp
        fn = conf[k, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        rows.append({"class": class_names[k], "precision": float(precision), "recall": float(recall),
                     "f1": float(f1), "iou": float(iou),
                     "ap": ap_from_histograms(stats["hist_pos"][k - 1], stats["hist_neg"][k - 1]),
                     "support": int(conf[k, :].sum())})
    return rows


def metrics_by_severity(rows, stats_list, class_names, levels=("low", "medium", "high")):
    """{class: {level: metrics row}} -- for each class and severity, the images
    where that class is visible at that severity together with every image
    where it is absent (its negatives), as compute_metrics_by_severity does for
    image-level labels."""
    out = {}
    for k, name in enumerate(class_names[1:], start=1):
        by_level = {}
        for level in levels:
            keep = [s for r, s in zip(rows, stats_list) if r[f"{name}_severity"] in (level, "none")]
            if not any(r[f"{name}_severity"] == level for r in rows):
                continue
            by_level[level] = metrics_from_stats(sum_stats(keep), class_names)[k - 1]
        out[name] = by_level
    return out
