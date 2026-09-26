"""
Quantifies a problem that's only ever been described qualitatively in this
project (Session 15/16 dev-log note about scratch over-predicting on clean
images, for a checkpoint that's no longer canonical): what fraction of
genuinely clean (undistorted) test-split images does a trained Stage B
checkpoint still call positive for at least one tile?

For each class, a clean image counts as a false positive if its MAX
predicted tile probability (over the whole 16x16 grid) crosses the decision
threshold -- i.e. the model would flag at least one tile of a truly clean
image as that class. This is the "clean-image FP rate" baseline that
threshold recalibration (scripts/diagnose_tile_thresholds.py) and the
impaired-gate head are both meant to improve on.

Usage:
    python scripts/diagnose_clean_false_positives.py \
        --checkpoint checkpoints/stage_b_combo/stage_b_head.pt \
        --data data/processed/stage_b --split test --tune-thresholds
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluate_stage_b import flatten_tiles
from src.data.stage_b_dataset import StageBDataset
from src.eval.gate import apply_gate, collect_gate_probs, load_gate
from src.eval.metrics import collect_predictions
from src.eval.thresholds import tune_per_class_thresholds
from scripts.train_stage_b import load_stage_b


def clean_row_mask(rows, class_names):
    """True for every row where every class column is 0 -- same check
    scripts/visualize_stage_b_results.py::_clean_image_lookup uses."""
    return np.array([not any(int(row[c]) for c in class_names) for row in rows])


def clean_false_positive_rates(probs, labels, rows, class_names, threshold):
    """probs, labels: (N, C, H, W) arrays for one split. For each class,
    the fraction of CLEAN rows (every class column 0) whose max predicted
    tile probability for that class crosses `threshold` (float or
    dict[class_name, float]). Returns a dict[class_name] -> (fp_rate,
    n_clean)."""
    per_class_threshold = isinstance(threshold, dict)
    mask = clean_row_mask(rows, class_names)
    n_clean = int(mask.sum())
    clean_probs = probs[mask]

    rates = {}
    for i, name in enumerate(class_names):
        t = threshold[name] if per_class_threshold else threshold
        max_prob_per_image = clean_probs[:, i].max(axis=(1, 2))
        fp_rate = float((max_prob_per_image >= t).mean()) if n_clean > 0 else float("nan")
        rates[name] = (fp_rate, n_clean)
    return rates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/stage_b_combo/stage_b_head.pt")
    parser.add_argument("--data", default="data/processed/stage_b")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--tune-thresholds", action="store_true",
        help="Tune a per-class best-F1 threshold on the val split (same as evaluate_stage_b.py), "
        "then use those instead of --threshold when deciding clean-image false positives.",
    )
    parser.add_argument(
        "--gate-checkpoint", default=None,
        help="Optional checkpoints/impaired_gate/impaired_gate_head.pt -- if given, ALSO prints "
        "a second, gated FP-rate table (see src/eval/gate.py) for a direct before/after comparison.",
    )
    parser.add_argument(
        "--gate-threshold", type=float, default=0.5,
        help="P(impaired) cutoff for --gate-checkpoint: below this, an image's Stage B predictions are zeroed.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    backbone, head, class_names = load_stage_b(args.checkpoint, args.weights, device)

    threshold = args.threshold
    if args.tune_thresholds:
        val_set = StageBDataset(args.data, split="val")
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        val_probs, val_labels = collect_predictions(backbone, head, val_loader, device)
        val_labels_flat, val_probs_flat = flatten_tiles(val_labels, val_probs)
        threshold = tune_per_class_thresholds(val_labels_flat, val_probs_flat, class_names)

    dataset = StageBDataset(args.data, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    probs, labels = collect_predictions(backbone, head, loader, device)

    threshold_desc = "tuned per-class" if args.tune_thresholds else args.threshold

    def _print_rates(rates, label=None):
        n_clean = next(iter(rates.values()))[1]
        heading = f"Evaluated {n_clean} clean images from split={args.split!r}, threshold={threshold_desc}"
        if label:
            heading = f"[{label}] {heading}"
        print(heading)
        print(f"{'class':<10}{'threshold':>10}{'FP rate':>12}")
        for name in class_names:
            fp_rate, _ = rates[name]
            t = threshold[name] if isinstance(threshold, dict) else threshold
            print(f"{name:<10}{t:>10.3f}{fp_rate:>12.3%}")

    rates = clean_false_positive_rates(probs, labels, dataset.rows, class_names, threshold)
    _print_rates(rates, label="ungated" if args.gate_checkpoint else None)

    if args.gate_checkpoint:
        gate_head, gate_img_size = load_gate(args.gate_checkpoint, backbone.out_channels, device)
        gate_probs = collect_gate_probs(backbone, gate_head, loader, device, img_size=gate_img_size)
        gated_probs = apply_gate(probs, gate_probs, threshold=args.gate_threshold)

        gated_rates = clean_false_positive_rates(gated_probs, labels, dataset.rows, class_names, threshold)
        print()
        _print_rates(gated_rates, label="gated")


if __name__ == "__main__":
    main()
