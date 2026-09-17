"""
One-off analysis (Session 18): sweep the decision threshold per class on the
val split to find a better operating point than the fixed 0.5 used by
scripts/evaluate_stage_b.py, then confirm on the held-out test split. Not
part of the regular pipeline -- a quick, no-retraining first response to
Session 17's focal-loss recall collapse.

Usage:
    python scripts/threshold_sweep_stage_b.py --checkpoint checkpoints/stage_b/stage_b_head_focal.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

from src.data.stage_b_dataset import StageBDataset
from src.eval.metrics import collect_predictions, compute_metrics
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import StageBDistortionHead


def flatten_tiles(labels, probs):
    n, c, h, w = labels.shape
    return (labels.transpose(0, 2, 3, 1).reshape(-1, c),
            probs.transpose(0, 2, 3, 1).reshape(-1, c))


def best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    best_idx = int(np.argmax(f1[:-1]))
    return float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx]), float(f1[best_idx])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/stage_b/stage_b_head_focal.pt")
    parser.add_argument("--data", default="data/processed/stage_b")
    parser.add_argument("--weights", default="weights/yolo11m.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]

    backbone = FrozenYOLOBackbone(args.weights).to(device)
    head = StageBDistortionHead(in_channels=backbone.out_channels, class_names=class_names).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    print("=== Finding best-F1 threshold per class on val split ===")
    val_set = StageBDataset(args.data, split="val")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    val_probs, val_labels = collect_predictions(backbone, head, val_loader, device)
    val_labels_flat, val_probs_flat = flatten_tiles(val_labels, val_probs)

    tuned_thresholds = {}
    for i, name in enumerate(class_names):
        thr, p, r, f1 = best_f1_threshold(val_labels_flat[:, i], val_probs_flat[:, i])
        tuned_thresholds[name] = thr
        print(f"{name:<10} best_threshold={thr:.3f}  val_precision={p:.3f}  val_recall={r:.3f}  val_f1={f1:.3f}")

    print("\n=== Confirming on test split: default 0.5 vs. tuned per-class thresholds ===")
    test_set = StageBDataset(args.data, split="test")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    test_probs, test_labels = collect_predictions(backbone, head, test_loader, device)
    test_labels_flat, test_probs_flat = flatten_tiles(test_labels, test_probs)

    print("\n-- threshold=0.5 (default) --")
    for row in compute_metrics(test_labels_flat, test_probs_flat, class_names, threshold=0.5):
        print(f"{row['class']:<10}precision={row['precision']:.3f}  recall={row['recall']:.3f}  "
              f"f1={row['f1']:.3f}  AP={row['ap']:.3f}")

    print("\n-- tuned per-class thresholds (from val split) --")
    preds = np.zeros_like(test_probs_flat, dtype=int)
    for i, name in enumerate(class_names):
        preds[:, i] = (test_probs_flat[:, i] >= tuned_thresholds[name]).astype(int)
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score
    for i, name in enumerate(class_names):
        y_true = test_labels_flat[:, i]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, preds[:, i], average="binary", zero_division=0
        )
        ap = average_precision_score(y_true, test_probs_flat[:, i])
        print(f"{name:<10}threshold={tuned_thresholds[name]:.3f}  precision={precision:.3f}  "
              f"recall={recall:.3f}  f1={f1:.3f}  AP={ap:.3f}")


if __name__ == "__main__":
    main()
