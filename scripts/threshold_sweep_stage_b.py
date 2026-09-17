"""
Diagnostic wrapper (Session 18, thinned in Session 19+): compares the fixed
0.5 threshold against the per-class best-F1 thresholds tuned on the val
split, for a Stage B checkpoint. The actual tuning/evaluation logic now
lives in `src/eval/thresholds.py::tune_per_class_thresholds` and
`src/eval/metrics.py::compute_metrics` (which accepts a per-class threshold
dict directly) -- both also wired into `scripts/evaluate_stage_b.py`'s own
`--tune-thresholds` flag, so that's the normal way to get a tuned-threshold
evaluation. This script is kept only as a side-by-side comparison view
(0.5 vs. tuned, printed together) that `evaluate_stage_b.py` doesn't provide
in one run.

Usage:
    python scripts/threshold_sweep_stage_b.py --checkpoint checkpoints/stage_b/stage_b_head_focal.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.stage_b_dataset import StageBDataset
from src.eval.metrics import collect_predictions, compute_metrics
from src.eval.thresholds import tune_per_class_thresholds
from src.models.backbone import FrozenYOLOBackbone
from src.models.distortion_head import StageBDistortionHead


def flatten_tiles(labels, probs):
    n, c, h, w = labels.shape
    return (labels.transpose(0, 2, 3, 1).reshape(-1, c),
            probs.transpose(0, 2, 3, 1).reshape(-1, c))


def _print_metrics(rows):
    for row in rows:
        print(f"{row['class']:<10}threshold={row['threshold']:.3f}  precision={row['precision']:.3f}  "
              f"recall={row['recall']:.3f}  f1={row['f1']:.3f}  AP={row['ap']:.3f}  "
              f"ROC-AUC={row['roc_auc']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    print("=== Tuning per-class best-F1 thresholds on val split ===")
    val_set = StageBDataset(args.data, split="val")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    val_probs, val_labels = collect_predictions(backbone, head, val_loader, device)
    val_labels_flat, val_probs_flat = flatten_tiles(val_labels, val_probs)
    tuned_thresholds = tune_per_class_thresholds(val_labels_flat, val_probs_flat, class_names)
    for name in class_names:
        print(f"  {name:<10}{tuned_thresholds[name]:.3f}")

    print("\n=== Confirming on test split: default 0.5 vs. tuned per-class thresholds ===")
    test_set = StageBDataset(args.data, split="test")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    test_probs, test_labels = collect_predictions(backbone, head, test_loader, device)
    test_labels_flat, test_probs_flat = flatten_tiles(test_labels, test_probs)

    print("\n-- threshold=0.5 (default) --")
    _print_metrics(compute_metrics(test_labels_flat, test_probs_flat, class_names, threshold=0.5))

    print("\n-- tuned per-class thresholds (from val split) --")
    _print_metrics(compute_metrics(test_labels_flat, test_probs_flat, class_names, threshold=tuned_thresholds))


if __name__ == "__main__":
    main()
